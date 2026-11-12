"""Run only the frozen R1 K=1 engineering equivalence gate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state_digest(items):
    digest = hashlib.sha256()
    keys = []
    for name, tensor in sorted(items):
        keys.append(name)
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {"keys": keys, "sha256": digest.hexdigest()}


def model_digests(model):
    names = [name for name, _ in model.named_parameters()]
    return {
        "model": state_digest(model.state_dict().items()),
        "ema1": state_digest(zip(names, model.ema_params1)),
        "ema2": state_digest(zip(names, model.ema_params2)),
    }


def compare(original, recurrent):
    shape_ok = original.shape == recurrent.shape
    if not shape_ok:
        return {"passed": False, "shape_match": False}
    flat = original.double().flatten(1)
    error = recurrent.double().flatten(1) - flat
    finite = bool(torch.isfinite(original).all() and torch.isfinite(recurrent).all())
    return {
        "shape": list(original.shape), "shape_match": True, "finite": finite,
        "passed": finite and bool(torch.allclose(original, recurrent, atol=1e-6, rtol=1e-5)),
        "max_absolute_error_by_example": error.abs().max(dim=1).values.tolist(),
        "relative_l2_error_by_example": (error.norm(dim=1) / flat.norm(dim=1).clamp_min(1e-12)).tolist(),
    }


def main(root):
    repo = root / "JET"
    assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == "07f9e6491796f4f2c717d6259b1a2e24afce6a77"
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code"], check=True)
    subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--exit-code"], check=True)
    sys.path.insert(0, str(repo))
    from inference import Denoiser, get_args_parser
    from experiments.looped_jet_tuev_v1.r1.recurrent_wrapper import RecurrentForward

    output = root / "evidence/r1-v1"
    fixtures = output / "fixtures"
    config_path = output / "engineering-config-v3.json"
    config = json.loads(config_path.read_text())
    source = repo / "experiments/looped_jet_tuev_v1/r1"
    for name, digest in config["source_sha256"].items():
        assert sha(source / name) == digest
    for name, digest in config["fixture_files_sha256"].items():
        assert sha(fixtures / name) == digest
    checkpoint_path = root / "baseline-seed0/checkpoint-last.pth"
    assert sha(checkpoint_path) == config["checkpoint_sha256"]
    assert sha(repo / "experiments/looped_jet_tuev_v1/R1_PROTOCOL.md") == config["protocol_sha256"]
    gate = output / "k1-equivalence"
    gate.mkdir(mode=0o700)
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
    }
    args = get_args_parser().parse_args([
        "--dataset", "tuev", "--datasets_dir", str(root / "tuev-official"),
        "--resume", str(checkpoint_path), "--gen_bsz", "32",
    ])
    args.target_length, args.eeg_patch_num, args.class_num = 1000, 5, 6
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    assert checkpoint["epoch"] == 199
    model = Denoiser(args).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    model.ema_params1 = [checkpoint["model_ema1"][name].cuda() for name, _ in model.named_parameters()]
    model.ema_params2 = [checkpoint["model_ema2"][name].cuda() for name, _ in model.named_parameters()]
    del checkpoint
    original_forward = model.net.forward
    wrapper_forward = RecurrentForward(1).bind(model.net)
    before = model_digests(model)
    probe = torch.load(fixtures / "probes.pt", map_location="cpu", weights_only=True)
    x, labels, noise = (probe[name].cuda() for name in ("reference", "labels", "noise"))
    forward_checks = []
    start = time.perf_counter()
    with torch.no_grad():
        for label_mode in ("conditioned", "null"):
            conditioning = labels if label_mode == "conditioned" else torch.full_like(labels, 6)
            for t in config["t_grid"]:
                z = t * x + (1 - t) * noise
                times = torch.full((12,), t, device="cuda", dtype=torch.float32)
                model.net.forward = original_forward
                actual = model._run_net(z, times, conditioning).cpu()
                model.net.forward = wrapper_forward
                recurrent = model._run_net(z, times, conditioning).cpu()
                row = {"label_mode": label_mode, "t": t, **compare(actual, recurrent)}
                forward_checks.append(row)
                assert row["passed"], f"Forward gate failed at {label_mode}/{t}"
                print(f"Forward gate passed {label_mode} t={t}", flush=True)

        original_run_net = model._run_net
        original_noise_like = model._noise_like
        initial_noise = noise[:6].clone()

        def trajectory(forward):
            calls = []
            model.net.forward = forward
            model._noise_like = lambda tensor: initial_noise.to(device=tensor.device, dtype=tensor.dtype).clone()

            def record(z, t, conditioning, use_ema=True):
                prediction = original_run_net(z, t, conditioning, use_ema=use_ema)
                calls.append((float(t[0]), prediction.detach().cpu()))
                return prediction

            model._run_net = record
            final = model.generate(labels[:6]).cpu()
            model._run_net = original_run_net
            model._noise_like = original_noise_like
            return final, calls

        actual_final, actual_calls = trajectory(original_forward)
        recurrent_final, recurrent_calls = trajectory(wrapper_forward)
        assert len(actual_calls) == len(recurrent_calls) == 99
        trajectory_checks = []
        for i, ((t, actual), (other_t, recurrent)) in enumerate(zip(actual_calls, recurrent_calls, strict=True)):
            assert t == other_t
            row = {"nfe_zero_based": i, "t": t, **compare(actual, recurrent)}
            trajectory_checks.append(row)
            assert row["passed"], f"Trajectory gate failed at NFE {i}"
        final_check = compare(actual_final, recurrent_final)
        assert final_check["passed"]
        model.net.forward = original_forward
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    after = model_digests(model)
    assert before == after, "State keys or parameter/EMA hashes changed"
    assert sha(checkpoint_path) == config["checkpoint_sha256"]
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code"], check=True)
    report = {
        "passed": True, "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "backend": backend, "config_sha256": sha(config_path),
        "checkpoint_sha256": config["checkpoint_sha256"], "checkpoint_epoch": 199,
        "source_sha256": config["source_sha256"], "forward_checks": forward_checks,
        "trajectory_probe_labels": labels[:6].cpu().tolist(), "trajectory_probe_classes": sorted(set(labels[:6].cpu().tolist())),
        "trajectory_nfe": 99, "trajectory_checks": trajectory_checks, "final_waveform_check": final_check,
        "state_before": before, "state_after": after, "state_and_ema_unchanged": before == after,
        "engineering_probe_elapsed_s": elapsed,
        "scope": "Only K1 equivalence probes. No candidate screen generation or clean-generation timing.",
        "candidate_generation_authorized": False,
    }
    (gate / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (gate / "passed.done").touch()
    for path in gate.iterdir():
        path.chmod(0o400)
    print(json.dumps({"passed": True, "trajectory_nfe": 99, "engineering_probe_elapsed_s": elapsed,
                      "report_sha256": sha(gate / "report.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(**vars(parser.parse_args()))




