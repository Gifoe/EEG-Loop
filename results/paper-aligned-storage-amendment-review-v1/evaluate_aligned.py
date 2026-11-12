"""Final-EMA1 aligned evaluation: held-out primary, then train secondary."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from data.data import build_base_dataset,get_label_names
from data.metrics import collect_ground_truth_batch,compute_ts_fid_streaming
from inference import _build_labels
from experiments.looped_jet_tuev_v1 import audit_ts_fid,reference_index
from experiments.looped_jet_tuev_v1.paper_aligned_v1.metric import features,score
from experiments.looped_jet_tuev_v1.paper_aligned_v1.model import PaperAlignedJET
from experiments.looped_jet_tuev_v1.paper_aligned_v1.run_support import json_write,read_manifest,sha
from experiments.looped_jet_tuev_v1.residual_channel_diagnostics import compare as residual_compare,serializable
from experiments.looped_jet_tuev_v1.signal_diagnostics import compare as native_compare


def matrix_flops():
    width,tokens,patch,mlp,depth=768,80,200,3072,12
    per_block=6*width*width+tokens*(4*width*width+2*width*mlp)+2*tokens*tokens*width
    projections=2*tokens*patch*width+2*width*width
    time_embedding=256*width+width*width
    return 2*(depth*per_block+projections+time_embedding)


def profile(model,labels):
    labels=labels[:64].cuda()
    initial=torch.randn((64,16,1000),generator=torch.Generator().manual_seed(0)).cuda()
    with torch.no_grad():
        for _ in range(2):
            model.generate(labels,initial_noise=initial,steps=50,cfg=1)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(5):
            start=time.perf_counter()
            model.generate(labels,initial_noise=initial,steps=50,cfg=1)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
    return {"batch":64,"warmups":2,"measured":5,"batch_latency_s":times,
        "median_batch_latency_s":float(np.median(times)),"dtype":"float32","autocast":False,
        "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20,
        "nfe":99,"block_evaluations_per_sample":99*12,
        "leading_matrix_flops_per_nfe":matrix_flops(),"leading_matrix_flops_per_sample":99*matrix_flops(),
        "flop_convention":"2FLOPs/MAC; includes joint80 attention, MLP, condition1/example, projections and time MLP; excludes elementwise/norm/softmax/solver/EMA copies.",
        "timing_scope":"Fixed synthetic seed0 Gaussian initial noise transferred before timing; generate clones it. Independent of real generation noise RNG."}


def metrics(generated,reference,labels,paper):
    function=score if paper else lambda a,b:compute_ts_fid_streaming(a,b,chunk_size=64,feature_bins=256)
    g,r=torch.from_numpy(generated),torch.from_numpy(reference)
    scores={"overall":function(g,r),"per_class":{},"feature_bins":256}
    for label in range(6):
        rows=np.flatnonzero(labels==label)
        scores["per_class"][str(label)]=function(g[rows],r[rows])
    return {"metric_id":"tsfid-paper-v1-1024" if paper else "tsfid-source-1000",
            "feature_dimensions":1024 if paper else 1000,"num_samples":len(labels),
            "label_histogram":{str(c):int((labels==c).sum()) for c in range(6)},
            "label_names":get_label_names("tuev"),"ts_fid":scores}


def numeric_audit(generated,reference,labels,metric,paper):
    g=features(generated) if paper else audit_ts_fid.features(generated,256)
    r=features(reference) if paper else audit_ts_fid.features(reference,256)
    values=metric["ts_fid"]
    checks={"overall":audit_ts_fid.compare(g,r,values["overall"])}
    for label in range(6):
        rows=labels==label
        checks[str(label)]=audit_ts_fid.compare(g[rows],r[rows],values["per_class"][str(label)])
    return {"passed":all(v["passed"] for v in checks.values()),"checks":checks}


def evaluate(root):
    config,manifest_sha=read_manifest(root)
    training=Path(config["output_dir"])
    assert (training/"training.done").exists()
    status=json.loads((training/"training-completion.json").read_text())
    checkpoint_path=training/"checkpoint-last.pth"
    assert status["final_epoch"]==199 and status["manifest_sha256"]==manifest_sha
    assert sha(checkpoint_path)==status["checkpoint_sha256"]
    assert checkpoint_path.samefile(training/"checkpoint-epoch-199.pth")
    torch.backends.cuda.matmul.allow_tf32=config["backend"]["matmul_tf32"]
    torch.backends.cudnn.allow_tf32=config["backend"]["cudnn_tf32"]
    torch.backends.cudnn.benchmark=config["backend"]["cudnn_benchmark"]
    torch.backends.cudnn.deterministic=config["backend"]["cudnn_deterministic"]
    checkpoint=torch.load(checkpoint_path,map_location="cpu",mmap=True,weights_only=True)
    assert checkpoint["variant"]=="jet-paper-aligned-v1" and checkpoint["epoch"]==199
    assert checkpoint["manifest_sha256"]==manifest_sha
    model=PaperAlignedJET().cuda().eval()
    model.load_state_dict(checkpoint["model"],strict=True)
    names=[name for name,_ in model.named_parameters()]
    model.ema_params1=[checkpoint["model_ema1"][name].cuda() for name in names]
    model.ema_params2=[checkpoint["model_ema2"][name].cuda() for name in names]
    del checkpoint
    from experiments.looped_jet_tuev_v1.paper_aligned_v1.reference_sidecar import create_sidecar,reconstruct
    from experiments.looped_jet_tuev_v1.paper_aligned_v1.storage_budget import check,occupied_checkpoints
    check(root,"evaluation",config["storage"]["checkpoint_upper_each"],occupied_checkpoints(training))
    for split,expected in (("test",29421),("train",68445)):
        output=root/f"paper-aligned-v1-eval-{split}-seed0"
        if output.exists():
            assert (output/"evaluation.done").exists(),"Partial evaluation requires explicit engineering recovery"
            previous=json.loads((output/"completion.json").read_text())
            assert previous["manifest_sha256"]==manifest_sha and previous["checkpoint_sha256"]==status["checkpoint_sha256"]
            continue
        output.mkdir(mode=0o700)
        torch.manual_seed(0)
        np.random.seed(0)
        dataset=build_base_dataset("tuev",split,Path(config["data_dir"]))
        assert len(dataset)==expected
        labels=np.asarray(_build_labels(0,6,"match_gt",dataset),dtype=np.int64)
        raw=root.parent/"data/TUH-EEG/tuh_eeg_events/v2.0.1/edf"
        assert raw.is_dir(), "Audited TUEV raw root is missing"
        index=reference_index.build_index(raw,Path(config["data_dir"]),split,0)
        index_path=output/"reference-index.private.npz"
        np.savez_compressed(index_path,**index)
        assert np.array_equal(index["labels"],labels)
        json_write(output/"configuration.json",{
            "manifest_sha256":manifest_sha,"checkpoint_sha256":status["checkpoint_sha256"],
            "split":split,"seed":0,"batch":64,"ema":"ema1","solver":"heun","steps":50,"final_euler":True,
            "nfe":99,"cfg":1,"noise":"unit Gaussian","reference_index_sha256":sha(index_path),
            "backend":config["backend"],"primary":split=="test"})
        chunks=[]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        beginning=time.perf_counter()
        with torch.no_grad():
            for start in range(0,len(labels),64):
                conditioning=torch.tensor(labels[start:start+64],device="cuda")
                values=model.generate(conditioning,steps=50,cfg=1).reshape(-1,16,1000).cpu()
                if not torch.isfinite(values).all() or (values==0).all(dim=(1,2)).any():
                    raise FloatingPointError("Invalid generated batch; all rows retained for failure evidence")
                chunks.append(values)
                print(f"{split} generated {min(start+64,len(labels))}/{len(labels)}",flush=True)
        torch.cuda.synchronize()
        generation={"elapsed_s":time.perf_counter()-beginning,
                    "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                    "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20}
        generated=torch.cat(chunks).numpy()
        del chunks
        mapping_path=root/"evidence/paper-aligned-v1-data-view/view-manifest.json"
        mapping=json.loads(mapping_path.read_text())
        sidecar=output/"references.private.json"
        sidecar_hash=create_sidecar(Path(config["data_dir"]),split,index,mapping_path,
            mapping_path.parent/mapping["mapping_ledger"],
            {"input_manifest_sha256":manifest_sha,"source_hashes":config["source_sha256"],
             "environment_hashes":config["environment_sha256"],"montage":__import__("data.preprocess_tuev",fromlist=["BIPOLAR_PAIRS"]).BIPOLAR_PAIRS},sidecar)
        reference=reconstruct(Path(config["data_dir"]),sidecar,sidecar_hash,labels)
        json_write(output/"archive-schema.json",{"schema":"jet-generated-only-v1","reference_sidecar_sha256":sidecar_hash,
                    "all_reference_rows_verified":len(reference),"reference_dtype":str(reference.dtype)})
        archive=output/"eval_batch.npz"
        np.savez_compressed(archive,generated=generated,labels=labels)
        source_scores=metrics(generated,reference,labels,False)
        paper_scores=metrics(generated,reference,labels,True)
        json_write(output/"metrics-source1000.json",source_scores)
        json_write(output/"metrics-paper1024.json",paper_scores)
        for name,values,paper in (("source1000",source_scores,False),("paper1024",paper_scores,True)):
            audit=numeric_audit(generated,reference,labels,values,paper)
            json_write(output/f"numeric-audit-{name}.json",audit)
            assert audit["passed"],f"Numeric {name} audit failed"
        json_write(output/"native-quality.json",{
            str(label):native_compare(generated,reference,np.flatnonzero(labels==label)) for label in range(6)})
        json_write(output/"residual-quality.private.json",serializable(residual_compare(reference,generated,labels)))
        json_write(output/"generation-profile.json",profile(model,torch.from_numpy(labels)))
        assert len(reference)==expected and np.array_equal(index["labels"],labels)
        assert all(reference_index.waveform_hash(row)==digest for row,digest in zip(reference,index["ground_truth_sha256"],strict=True))
        (output/"integrity.verified").touch()
        assert sha(checkpoint_path)==status["checkpoint_sha256"]
        completion={"manifest_sha256":manifest_sha,"checkpoint_sha256":status["checkpoint_sha256"],
                    "split":split,"rows":expected,"generation":generation,
                    "files_sha256":{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
        json_write(output/"completion.json",completion)
        (output/"evaluation.done").touch()
        for path in output.iterdir():path.chmod(0o400)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    evaluate(parser.parse_args().root)
