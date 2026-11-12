"""Execute only the reviewed, exact five-arm source-R0 validation screen."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import scipy
import torch

from inference import Denoiser,get_args_parser
from experiments.looped_jet_tuev_v1.r1.prepare_r1 import sha
from experiments.looped_jet_tuev_v1.r1.recurrent_wrapper import RecurrentForward
from experiments.looped_jet_tuev_v1.r1.screen_metrics import score,continuation,write_json
from experiments.looped_jet_tuev_v1.r1.screen_probes import run_probes
from experiments.looped_jet_tuev_v1.r1.verify_k1 import model_digests


def load_model(root):
    args=get_args_parser().parse_args(["--dataset","tuev","--datasets_dir",str(root/"tuev-official"),
                                      "--resume",str(root/"baseline-seed0"),"--gen_bsz","32"])
    args.target_length,args.eeg_patch_num,args.class_num=1000,5,6
    checkpoint=torch.load(root/"baseline-seed0/checkpoint-last.pth",map_location="cpu",weights_only=True,mmap=True)
    assert checkpoint["epoch"]==199
    model=Denoiser(args).cuda().eval()
    model.load_state_dict(checkpoint["model"],strict=True)
    model.ema_params1=[checkpoint["model_ema1"][name].cuda() for name,_ in model.named_parameters()]
    model.ema_params2=[checkpoint["model_ema2"][name].cuda() for name,_ in model.named_parameters()]
    return model


def generate(model,labels,noise):
    assert noise.shape==(len(labels),16,5,200)
    original=model._noise_like
    model._noise_like=lambda tensor:noise.to(device=tensor.device,dtype=tensor.dtype).clone()
    try:
        return model.generate(labels.cuda())
    finally:
        model._noise_like=original


def flops(k):
    channels,tokens,width,mlp,patch=16,5,768,3072,200
    block=channels*(6*width*width+tokens*(4*width*width+2*width*mlp)+2*tokens*tokens*width)
    other=channels*(2*tokens*patch*width+2*width*width)+256*width+width*width
    return 2*((8+4*k)*block+other)


def profile(model,labels,noise,k,steps):
    times=[]
    with torch.no_grad():
        for _ in range(2):generate(model,labels[:32],noise[:32])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            start=time.perf_counter()
            output=generate(model,labels[:32],noise[:32])
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            if not torch.isfinite(output).all():raise FloatingPointError("Nonfinite profile output")
    return {"batch_size":32,"warmup_batches":2,"measured_batches":5,"batch_latency_s":times,
        "median_batch_latency_s":float(np.median(times)),"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20,"nfe":2*steps-1,
        "block_evaluations_per_sample":(2*steps-1)*(8+4*k),"leading_matrix_flops_per_nfe":flops(k),
        "leading_matrix_flops_per_generated_sample":flops(k)*(2*steps-1),
        "flop_convention":"2FLOPs/MAC; includes per-channel conditioning, attention, MLP, embeddings and final projection. Excludes elementwise/norm/softmax/EMA copies/solver.",
        "timing_scope":"Saved CPU noise transfer+clone and clean generation; no hidden probes or diagnostic suffix decoding"}


def run(root,approved_config_sha):
    source=root/"JET/experiments/looped_jet_tuev_v1"
    previous=root/"evidence/r1-v1"
    frozen=previous/"screen-freeze-v2/screen-config.json"
    assert sha(frozen)==approved_config_sha
    config=json.loads(frozen.read_text())
    assert torch.__version__==config["torch"] and __import__("sys").prefix==config["runtime_prefix"]
    assert np.__version__==config["numpy"] and scipy.__version__==config["scipy"]
    assert subprocess.check_output(["git","-C",str(root/"JET"),"rev-parse","HEAD"],text=True).strip()=="07f9e6491796f4f2c717d6259b1a2e24afce6a77"
    assert not subprocess.check_output(["git","-C",str(root/"JET"),"status","--porcelain","--untracked-files=no"],text=True)
    assert sha(source/"R1_PROTOCOL.md")==config["protocol_sha256"]
    for name,digest in config["source_sha256"].items():assert sha(source/"r1"/name)==digest
    for name,digest in config["operator_sha256"].items():assert sha(source/name)==digest
    for name,digest in config["fixture_files_sha256"].items():assert sha(previous/"fixtures"/name)==digest
    checkpoint=root/"baseline-seed0/checkpoint-last.pth"
    assert sha(checkpoint)==config["checkpoint_sha256"]
    assert sha(root/"evidence/calibration-b-v1/median_scales.json")==config["scale_sha256"]
    output=previous/"screen-results-v1"
    output.mkdir(mode=0o700)
    results={arm["name"]:{"status":"unrun","k":arm["k"],"steps":arm["steps"]} for arm in config["arms"]}
    write_json(output/"five-arm-status.json",results)
    torch.backends.cuda.matmul.allow_tf32=False
    write_json(output/"runtime.json",{"approved_config_sha256":approved_config_sha,"torch":torch.__version__,
        "matmul_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_tf32":torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark":torch.backends.cudnn.benchmark,"cudnn_deterministic":torch.backends.cudnn.deterministic,
        "gpu":subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.free,utilization.gpu","--format=csv,noheader"],text=True),
        "shared_gpu_processes":subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"],text=True)})
    references=torch.load(previous/"fixtures/references.pt",map_location="cpu",weights_only=True)
    noise=torch.load(previous/"fixtures/screen-noise.pt",map_location="cpu",weights_only=True)
    probes=torch.load(previous/"fixtures/probes.pt",map_location="cpu",weights_only=True)
    labels=references["labels"]
    reference=references["reference"].reshape(2020,16,1000).numpy()
    probe_dir=output/"mechanism-probes"
    probe_dir.mkdir(mode=0o700)
    probe_model=load_model(root)
    probe_failed_k=[]
    try:
        before=model_digests(probe_model)
        probe_report=run_probes(probe_model,probes,probe_dir)
        probe_failed_k=probe_report["failed_k"]
        write_json(probe_dir/"report.json",probe_report)
        assert before==model_digests(probe_model),"Probe state/EMA mutation"
    except Exception as error:
        write_json(probe_dir/"failure.json",{"error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()})
        (output/"screen.failed").touch()
        raise
    finally:
        del probe_model
        gc.collect()
        torch.cuda.empty_cache()
    for arm in config["arms"]:
        name,k,steps=arm["name"],arm["k"],arm["steps"]
        folder=output/name
        folder.mkdir(mode=0o700)
        if k in probe_failed_k:
            results[name]={"status":"failed","k":k,"steps":steps,"error":"Nonfinite/failed required core probe"}
            write_json(folder/"failure.json",results[name])
            (folder/"failed.done").touch()
            write_json(output/"five-arm-status.json",results)
            continue
        model=None
        try:
            model=load_model(root)
            model.net.forward=RecurrentForward(k).bind(model.net)
            model.steps=steps
            before=model_digests(model)
            write_json(folder/"configuration.json",{**arm,"batch_size":32,"initial_noise_file_sha256":config["fixture_files_sha256"]["screen-noise.pt"],
                "counts":config["counts"],"approved_config_sha256":approved_config_sha})
            timing=profile(model,labels,noise,k,steps)
            chunks=[]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.no_grad():
                for position in range(0,2020,32):
                    values=generate(model,labels[position:position+32],noise[position:position+32]).cpu()
                    if not torch.isfinite(values).all() or (values==0).all(dim=(1,2,3)).any():
                        torch.save({"position":position,"values":values},folder/"numerical-failure.private.pt")
                        raise FloatingPointError("Nonfinite or all-zero generated sample")
                    chunks.append(values)
                    print(f"{name} generated {min(position+32,2020)}/2020",flush=True)
            torch.cuda.synchronize()
            timing["total_generation_s"]=time.perf_counter()-start
            timing["generation_peak_allocated_mib"]=torch.cuda.max_memory_allocated()/2**20
            timing["generation_peak_reserved_mib"]=torch.cuda.max_memory_reserved()/2**20
            generated=torch.cat(chunks).reshape(2020,16,1000).numpy()
            assert model_digests(model)==before,"Arm state/EMA mutation"
            archive=folder/"eval_batch.private.npz"
            np.savez_compressed(archive,generated=generated,ground_truth=reference,labels=labels.numpy())
            write_json(folder/"timing.json",timing)
            del model
            model=None
            gc.collect()
            torch.cuda.empty_cache()
            metric,quality=score(reference,generated,labels.numpy(),root/"evidence/calibration-b-v1",folder)
            results[name]={"status":"completed","k":k,"steps":steps,"ts_fid":metric["ts_fid"]["overall"],
                "macro_families":quality["macro_families"],
                "class_families":{label:row["families"] for label,row in quality["by_class"].items()},
                "archive_sha256":sha(archive),"timing":timing}
            (folder/"completed.done").touch()
        except Exception as error:
            results[name]={"status":"failed","k":k,"steps":steps,"error_type":type(error).__name__,"error":str(error)}
            write_json(folder/"failure.json",{**results[name],"traceback":traceback.format_exc()})
            (folder/"failed.done").touch()
        finally:
            if model is not None:del model
            gc.collect()
            torch.cuda.empty_cache()
            write_json(output/"five-arm-status.json",results)
    assert sha(checkpoint)==config["checkpoint_sha256"]
    write_json(output/"continuation-filter.json",continuation(results))
    (output/"screen.finished").touch()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--approved-config-sha",required=True)
    args=parser.parse_args()
    run(args.root,args.approved_config_sha)


