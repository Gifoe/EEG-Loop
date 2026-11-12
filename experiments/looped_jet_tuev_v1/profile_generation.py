"""Measure synchronized generation latency and count backbone matmul FLOPs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from inference import Denoiser, get_args_parser
from models.raw_vit import RAW_VIT_SPECS


def matmul_flops(args):
    spec = RAW_VIT_SPECS[args.model]
    d, c, p, l = spec.hidden_size, args.num_eeg_channels, args.eeg_patch_size, args.eeg_patch_num
    m = int(d * spec.mlp_ratio)
    per_block = c * (6 * d * d + l * (4 * d * d + 2 * d * m) + 2 * l * l * d)
    input_and_final = c * (2 * l * p * d + 2 * d * d)
    timestep_mlp = 256 * d + d * d
    return 2 * (spec.depth * per_block + input_and_final + timestep_mlp)


def main(checkpoint, output):
    args = get_args_parser().parse_args([
        "--dataset", "tuev", "--datasets_dir", ".", "--resume", str(checkpoint),
    ])
    args.target_length, args.class_num = 1000, 6
    args.eeg_patch_num = args.target_length // args.eeg_patch_size
    torch.manual_seed(args.seed)
    model = Denoiser(args).cuda().eval()
    weights = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(weights["model"])
    model.ema_params1 = [weights["model_ema1"][name].cuda() for name, _ in model.named_parameters()]
    model.ema_params2 = [weights["model_ema2"][name].cuda() for name, _ in model.named_parameters()]
    del weights
    labels = torch.arange(args.gen_bsz, device="cuda") % args.class_num
    measurements = []
    with torch.no_grad():
        for _ in range(2):
            model.generate(labels)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            start = time.perf_counter()
            model.generate(labels)
            torch.cuda.synchronize()
            measurements.append(time.perf_counter() - start)
    flops = matmul_flops(args)
    nfe = 2 * args.num_sampling_steps - 1
    result = {
        "warmup_batches": 2,
        "measured_batches": 5,
        "batch_size": args.gen_bsz,
        "cuda_device": torch.cuda.get_device_name(),
        "dtype": str(next(model.parameters()).dtype),
        "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        "batch_latency_s": measurements,
        "median_batch_latency_s": float(np.median(measurements)),
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / 2**20,
        "nfe": nfe,
        "matmul_flops_per_sample_per_nfe": flops,
        "matmul_flops_per_generated_sample": flops * nfe,
        "flop_convention": "two FLOPs per MAC; excludes elementwise, softmax, norms and solver arithmetic",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.checkpoint, args.output)
