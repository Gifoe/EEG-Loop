"""Freeze private R1 validation fixtures before any candidate generation."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np
import torch

PROTOCOL_SHA = "102f426f91163c7b9fbe359f0eba050ec47819bb6429c43befb1fb522e46d300"
CHECKPOINT_SHA = "7047396663de22f02af0ae6676b579cb030be4c9effdd66729b093c22473f418"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main(root):
    assert subprocess.check_output(["git", "-C", str(root / "JET"), "rev-parse", "HEAD"], text=True).strip() == "07f9e6491796f4f2c717d6259b1a2e24afce6a77"
    subprocess.run(["git", "-C", str(root / "JET"), "diff", "--exit-code"], check=True)
    subprocess.run(["git", "-C", str(root / "JET"), "diff", "--cached", "--exit-code"], check=True)
    source = root / "JET/experiments/looped_jet_tuev_v1"
    protocol = source / "R1_PROTOCOL.md"
    assert sha(protocol) == PROTOCOL_SHA
    checkpoint = root / "baseline-seed0/checkpoint-last.pth"
    assert sha(checkpoint) == CHECKPOINT_SHA
    assert sha(root / "evidence/reference-index/val-seed0.npz") == "a1c100f89a1712749974609ba0fe2dc1b0a1553ec15e3dbbac5ec4b28e353b0e"
    calibration_path = root / "evidence/calibration-b-v1/calibration_config.json"
    assert sha(calibration_path) == "ade2110650e655ec1b4e6417a2eec0db35131bd9485c320bf97043436b1500da"
    output = root / "evidence/r1-v1"
    output.mkdir(mode=0o700)
    fixtures = output / "fixtures"
    fixtures.mkdir(mode=0o700)
    index_path = root / "evidence/reference-index/val-seed0.npz"
    with np.load(index_path, allow_pickle=False) as index:
        by_file = {
            str(name): (int(label), str(digest), str(directory))
            for name, label, digest, directory in zip(index["processed_files"], index["labels"],
                                                       index["ground_truth_sha256"], index["subject_directories"], strict=True)
        }
    pool = root / "tuev-official/val"
    files = sorted(path.name for path in pool.glob("*.pkl"))
    assert set(files) == by_file.keys()
    labels = np.array([by_file[name][0] for name in files])
    rng = np.random.default_rng(20261112)
    chosen = []
    counts = []
    for label in range(6):
        eligible = np.flatnonzero(labels == label)
        selected = np.sort(rng.choice(eligible, min(512, len(eligible)), replace=False))
        counts.append(len(selected))
        chosen.extend(selected.tolist())
    assert counts == [119, 291, 512, 86, 500, 512] and len(chosen) == 2020
    selection = np.array(chosen)
    x = []
    for idx in selection:
        with (pool / files[idx]).open("rb") as stream:
            item = pickle.load(stream)
        signal = np.ascontiguousarray(item["signal"] / 100.0, dtype="<f4")
        assert hashlib.sha256(signal).hexdigest() == by_file[files[idx]][1]
        assert int(item["label"][0]) - 1 == int(labels[idx])
        x.append(torch.from_numpy(signal.reshape(16, 5, 200)))
    references = torch.stack(x)
    selected_labels = torch.from_numpy(labels[selection].copy()).long()
    components = json.loads((root / "evidence/calibration-b-v1/calibration_config.json").read_text())["components_by_directory_id"]
    membership = {directory: i for i, group in enumerate(components) for directory in group}
    support = {}
    for label in range(6):
        positions = np.flatnonzero(selected_labels.numpy() == label)
        names = [files[selection[pos]] for pos in positions]
        support[str(label)] = {
            "rows": len(names), "unique_waveforms": len({by_file[name][1] for name in names}),
            "directory_ids": len({by_file[name][2] for name in names}),
            "payload_linked_components": len({membership[by_file[name][2]] for name in names}),
        }
    probe_positions = np.concatenate([np.flatnonzero(selected_labels.numpy() == label)[:2] for label in range(6)])
    screen_noise = torch.randn((2020, 16, 5, 200), generator=torch.Generator(device="cpu").manual_seed(20261112), dtype=torch.float32)
    probe_noise = torch.randn((12, 16, 5, 200), generator=torch.Generator(device="cpu").manual_seed(20261113), dtype=torch.float32)
    torch.save({"reference": references, "labels": selected_labels}, fixtures / "references.pt")
    torch.save(screen_noise, fixtures / "screen-noise.pt")
    torch.save({"reference": references[probe_positions], "labels": selected_labels[probe_positions],
                "noise": probe_noise, "selection_positions": torch.from_numpy(probe_positions)}, fixtures / "probes.pt")
    np.savez_compressed(fixtures / "selection.private.npz", sorted_dataset_indices=selection,
                        processed_files=np.array(files)[selection], labels=selected_labels.numpy(),
                        waveform_hashes=np.array([by_file[files[idx]][1] for idx in selection]),
                        probe_selection_positions=probe_positions)
    config = {
        "protocol_sha256": PROTOCOL_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
        "upstream_commit": subprocess.check_output(["git", "-C", str(root / "JET"), "rev-parse", "HEAD"], text=True).strip(),
        "calibration_config_sha256": sha(calibration_path),
        "validation_reference_index_sha256": sha(index_path), "counts": counts, "total": 2020,
        "support": support, "numpy_version": np.__version__, "torch_version": torch.__version__,
        "selection_seed": 20261112, "screen_noise_seed": 20261112, "probe_noise_seed": 20261113,
        "screen_noise_tensor_sha256": tensor_sha(screen_noise), "probe_noise_tensor_sha256": tensor_sha(probe_noise),
        "probe_count": 12, "trajectory_probe_count": 6, "t_grid": [0, 0.1, 0.5, 0.9, 0.98],
        "batch_size": 32, "dtype": "float32", "autocast": False, "matmul_tf32": False,
        "solver": "heun", "steps": 50, "final_euler": True, "cfg": 1, "noise_scale": 1, "t_eps": 0.05,
        "source_sha256": {path.name: sha(path) for path in sorted((source / "r1").glob("*.py"))},
        "fixture_files_sha256": {path.name: sha(path) for path in sorted(fixtures.iterdir())},
        "candidate_generation_authorized": False,
    }
    (output / "engineering-config.json").write_text(json.dumps(config, indent=2) + "\n")
    for path in fixtures.iterdir():
        path.chmod(0o400)
    (output / "engineering-config.json").chmod(0o400)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(**vars(parser.parse_args()))



