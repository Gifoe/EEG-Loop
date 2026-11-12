"""Fresh own-seed paper-aligned training with atomic, resumable checkpoints."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data.data import build_dataloaders
from experiments.looped_jet_tuev_v1.paper_aligned_v1.model import PaperAlignedJET
from experiments.looped_jet_tuev_v1.paper_aligned_v1.run_support import json_write,read_manifest,sha
from util.lr_sched import adjust_learning_rate


def rng_state():
    state=np.random.get_state()
    return {"python":random.getstate(),"numpy":{"algorithm":state[0],"keys":torch.from_numpy(state[1].astype(np.int64)),
            "position":state[2],"has_gauss":state[3],"cached_gaussian":state[4]},
            "torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    value=state["numpy"]
    np.random.set_state((value["algorithm"],value["keys"].numpy().astype(np.uint32),
                         value["position"],value["has_gauss"],value["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(model,optimizer,epoch,config,manifest_sha,output):
    names=[name for name,_ in model.named_parameters()]
    payload={"variant":"jet-paper-aligned-v1","epoch":epoch,"args":config,"manifest_sha256":manifest_sha,
        "model":{key:value.detach().cpu() for key,value in model.state_dict().items()},
        "model_ema1":{name:value.detach().cpu() for name,value in zip(names,model.ema_params1,strict=True)},
        "model_ema2":{name:value.detach().cpu() for name,value in zip(names,model.ema_params2,strict=True)},
        "optimizer":optimizer.state_dict(),"rng":rng_state(),
        "resume_semantics":"Resume at the next epoch. Weighted sampler and DataLoader use restored global Torch CPU RNG. Workers receive its next iterator seed; no stochastic data transform. GPU backend nondeterminism is not eliminated."}
    temporary=output/"checkpoint-last.pth.tmp"
    assert not temporary.exists(),"Unresolved checkpoint temporary file"
    with temporary.open("xb") as stream:
        torch.save(payload,stream)
        stream.flush()
        os.fsync(stream.fileno())
    final=output/"checkpoint-last.pth"
    os.replace(temporary,final)
    directory=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    if epoch in (0,100,199):
        snapshot=output/f"checkpoint-epoch-{epoch:03d}.pth"
        assert not snapshot.exists()
        os.link(final,snapshot)
        snapshot.chmod(0o400)
    json_write(output/"checkpoint-status.json",{"epoch":epoch,"bytes":final.stat().st_size,"sha256":sha(final),
                                               "manifest_sha256":manifest_sha})


def train(root,resume):
    config,manifest_sha=read_manifest(root)
    output=Path(config["output_dir"])
    assert output.resolve()==(root/"paper-aligned-v1-seed0").resolve()
    if resume:
        assert output.is_dir() and not (output/"training.done").exists()
    else:
        output.mkdir(mode=0o700)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=config["backend"]["matmul_tf32"]
    torch.backends.cudnn.allow_tf32=config["backend"]["cudnn_tf32"]
    torch.backends.cudnn.benchmark=config["backend"]["cudnn_benchmark"]
    torch.backends.cudnn.deterministic=config["backend"]["cudnn_deterministic"]
    loaders,classes=build_dataloaders("tuev",config["data_dir"],256,config["num_workers"],use_weighted_sampler=True)
    assert classes==6 and len(loaders["train"])==267
    model=PaperAlignedJET().cuda().train()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==config["parameter_count"]
    model.initialize_ema()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.95),weight_decay=0)
    args=SimpleNamespace(lr=5e-5,warmup_epochs=5,lr_schedule="constant",epochs=200,min_lr=0)
    start_epoch=0
    if resume:
        path=output/"checkpoint-last.pth"
        checkpoint=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        assert checkpoint["variant"]=="jet-paper-aligned-v1" and checkpoint["manifest_sha256"]==manifest_sha
        model.load_state_dict(checkpoint["model"],strict=True)
        names=[name for name,_ in model.named_parameters()]
        model.ema_params1=[checkpoint["model_ema1"][name].cuda() for name in names]
        model.ema_params2=[checkpoint["model_ema2"][name].cuda() for name in names]
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch=checkpoint["epoch"]+1
        restore_rng(checkpoint["rng"])
        del checkpoint
    assert not (output/"checkpoint-last.pth.tmp").exists(),"Interrupted checkpoint temporary requires explicit recovery review"
    unique_checkpoints={}
    if resume:
        for path in [output/"checkpoint-last.pth",*output.glob("checkpoint-epoch-*.pth")]:
            assert path.is_file() and not path.is_symlink()
            info=path.stat()
            unique_checkpoints[info.st_dev,info.st_ino]=info.st_size
    occupied=sum(unique_checkpoints.values())
    from experiments.looped_jet_tuev_v1.paper_aligned_v1.storage_budget import check
    storage_check=check(root,"training",config["storage"]["checkpoint_upper_each"],occupied)
    required_free=storage_check["required_free_bytes"]
    json_write(output/"runtime.json",{"manifest_sha256":manifest_sha,"resume":resume,"start_epoch":start_epoch,
        "torch":torch.__version__,"numpy":np.__version__,"backend":config["backend"],
        "storage_check":{"required_free_bytes":required_free,"existing_unique_checkpoint_bytes":occupied},
        "loss_coefficients":{"reconstruction":1,"statistics":1,"tv":.1,"pearson":.1},
        "moment_axis":"all1000timepoints","std_correction":1,"tv_normalization":1000,
        "class_drop":"training only","sampler":"inverse-frequency replacement N draws; drop_last",
        "initialization":"Fresh own seed0" if not resume else "Explicit own-run epoch-boundary RNG restoration"})
    writer=SummaryWriter(str(output/"tensorboard"))
    overall=time.perf_counter()
    for epoch in range(start_epoch,200):
        model.train()
        losses=[]
        beginning=time.perf_counter()
        for step,(x,labels) in enumerate(loaders["train"]):
            lr=adjust_learning_rate(optimizer,epoch+step/len(loaders["train"]),args)
            x=x.cuda(non_blocking=True).float()
            labels=labels.cuda(non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=model(x,labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss epoch{epoch}/step{step}")
            loss.backward()
            optimizer.step()
            model.update_ema()
            torch.cuda.synchronize()
            losses.append(float(loss))
            if step%20==0:
                print(f"Epoch{epoch} step{step}/267 loss={float(loss):.8g} lr={lr:.8g}",flush=True)
            if step%100==0:
                writer.add_scalar("train_loss",float(loss),epoch*267+step)
                writer.add_scalar("lr",lr,epoch*267+step)
        row={"epoch":epoch,"steps":len(losses),"mean_loss":float(np.mean(losses)),
             "epoch_seconds":time.perf_counter()-beginning,"lr":optimizer.param_groups[0]["lr"]}
        assert row["steps"]==267
        with (output/"epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+chr(10))
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row),flush=True)
        if epoch%config["checkpoint_retention"]["last_save_every_epochs"]==0 or epoch==199:
            save_checkpoint(model,optimizer,epoch,config,manifest_sha,output)
        writer.flush()
    writer.close()
    final=output/"checkpoint-last.pth"
    snapshot=output/"checkpoint-epoch-199.pth"
    assert final.samefile(snapshot)
    final.chmod(0o400)
    json_write(output/"training-completion.json",{"epochs":200,"final_epoch":199,"elapsed_seconds":time.perf_counter()-overall,
        "checkpoint_sha256":sha(final),"checkpoint_bytes":final.stat().st_size,"manifest_sha256":manifest_sha,
        "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20})
    (output/"training.done").touch()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    train(args.root,args.resume)


