"""Synthetic batch256 CUDA compatibility smoke from fresh own seed0 weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from experiments.looped_jet_tuev_v1.paper_aligned_v1.model import PaperAlignedJET


def run(output):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    report = {
        "seed": 0, "batch_size": 256, "warmup_updates": 2, "timed_updates": 5,
        "initialization": "Fresh own random model; no checkpoint loaded",
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "backend": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "gpu_before": subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.free",
                                               "--format=csv,noheader"],text=True).strip(),
        "gpu_processes_before": subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory",
                                                        "--format=csv,noheader"],text=True).strip(),
    }
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tiny = torch.randn(32,32,device="cuda",requires_grad=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        test = (tiny @ tiny).square().mean()
    test.backward()
    assert torch.isfinite(test) and torch.isfinite(tiny.grad).all()
    del tiny, test
    model = PaperAlignedJET().cuda().train()
    model.initialize_ema()
    original_position = model.net.pos_embed.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.95),weight_decay=0)
    class_counts = torch.tensor([832,480,256,128,64,32])
    pool_labels = torch.repeat_interleave(torch.arange(6),class_counts)
    weights = 1 / class_counts[pool_labels].double()
    sampler = torch.utils.data.WeightedRandomSampler(weights,1792,replacement=True,
                                                    generator=torch.Generator().manual_seed(0))
    selected_labels = pool_labels[torch.tensor(list(sampler))]
    x = torch.randn(256,16,1000,device="cuda")
    measurements, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(7):
        labels = selected_labels[step*256:(step+1)*256].cuda()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss = model(x,labels)
        assert torch.isfinite(loss), "Nonfinite smoke loss"
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "Nonfinite gradient"
        optimizer.step()
        model.update_ema()
        assert all(torch.isfinite(p).all() for p in model.ema_params1+model.ema_params2), "Nonfinite EMA"
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        losses.append(float(loss))
        if step>=2:
            measurements.append(seconds)
        print(f"Update {step+1}/7 loss={float(loss):.8g} elapsed={seconds:.4f}s",flush=True)
    change = float((model.net.pos_embed-original_position).abs().max())
    assert change>0 and torch.isfinite(model.net.pos_embed).all()
    report.update(
        training={"losses":losses,"timed_update_seconds":measurements,
                  "median_update_seconds":float(np.median(measurements)),
                  "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                  "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20,
                  "position_max_change":change,"all_loss_gradient_ema_checks_finite":True,
                  "timing_includes":"forward/backward/optimizer/EMA plus explicit finite-value safety checks"},
        sampler={"synthetic_pool_counts":class_counts.tolist(),"draws":1792,"replacement":True,
                 "weighting":"inverse class count","drawn_class_counts":torch.bincount(selected_labels,minlength=6).tolist()},
    )
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval()
    calls = []
    hook = model.net.register_forward_hook(lambda module, inputs, output: calls.append(1))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(torch.tensor([0,5],device="cuda"),steps=50,cfg=1)
    torch.cuda.synchronize()
    hook.remove()
    assert len(calls)==99 and generated.shape==(2,16,1000)
    assert torch.isfinite(generated).all() and not (generated==0).all()
    report["inference"]={"batch_size":2,"dtype":str(generated.dtype),"autocast":False,"nfe":len(calls),
                         "elapsed_seconds":time.perf_counter()-start,"finite":True,
                         "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                         "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20}
    report.update(passed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return report


output = Path(sys.argv[1])
output.mkdir(mode=0o700)
try:
    report = run(output)
except Exception as error:
    (output/"smoke-failure.json").write_text(json.dumps({
        "passed":False,"error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+"\n")
    raise
(output/"smoke-report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
(output/"smoke.passed").touch()
print(json.dumps(report,indent=2))
