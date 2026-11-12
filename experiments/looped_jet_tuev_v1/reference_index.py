"""Freeze and verify the released full-split match_gt reference order."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from data.data import build_base_dataset
from data.metrics import collect_ground_truth_batch
from inference import _build_labels


def waveform_hash(signal):
    return hashlib.sha256(np.ascontiguousarray(signal, dtype="<f4")).hexdigest()


class IndexedLabels:
    """Give the upstream selector the original labels with row IDs as payloads."""

    def __init__(self, labels):
        self.labels = labels
        if len(labels) > 2**24:
            raise ValueError("Reference indices would exceed exact float32 precision")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor([[index]], dtype=torch.float32), self.labels[index]


def build_index(raw, processed, split, seed):
    source = raw / ("eval" if split == "test" else "train")
    recordings = {}
    for edf in sorted(source.rglob("*.edf")):
        if edf.stem in recordings:
            raise ValueError(f"Ambiguous EDF basename in {split}")
        recordings[edf.stem] = edf.relative_to(source)

    dataset = build_base_dataset("tuev", split, processed)
    if not dataset:
        raise ValueError(f"Empty reference split: {split}")
    files = dataset.files
    labels, waveform_hashes, sources, subjects = [], [], [], []
    for idx, name in enumerate(files):
        signal, label = dataset[idx]
        labels.append(label)
        waveform_hashes.append(waveform_hash(signal))
        recording = recordings[Path(name).stem.rsplit("-", 1)[0]]
        sources.append(recording.as_posix())
        subjects.append(recording.parent.name)

    indexed = IndexedLabels(labels)
    label_schedule = np.asarray(_build_labels(0, 6, "match_gt", indexed), dtype=np.int64)
    order = collect_ground_truth_batch(indexed, label_schedule, 1, 1, seed).reshape(-1).numpy().astype(np.int64)
    if len(np.unique(order)) != len(dataset) or not np.array_equal(np.asarray(labels)[order], label_schedule):
        raise ValueError("Upstream full-split reference order is invalid")
    return {
        "split": split,
        "seed": seed,
        "torch_version": str(torch.__version__),
        "labels": label_schedule,
        "processed_files": np.asarray(files)[order],
        "source_edfs": np.asarray(sources)[order],
        "subject_directories": np.asarray(subjects)[order],
        "ground_truth_sha256": np.asarray(waveform_hashes)[order],
    }


def verify_metrics(labels, metrics_path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    histogram = {str(label): count for label, count in Counter(labels.tolist()).items()}
    if metrics["num_samples"] != len(labels) or metrics["label_histogram"] != histogram:
        raise ValueError("Reported metrics counts differ from the reference schedule")
    scores = metrics["ts_fid"]
    overall = scores["overall"]
    if scores["feature_bins"] != 256 or overall is None or not math.isfinite(overall):
        raise ValueError("Unexpected TS-FID setting or invalid overall score")
    per_class = scores["per_class"]
    if per_class.keys() != histogram.keys() or not all(
        per_class[label] is not None and math.isfinite(per_class[label])
        for label, count in histogram.items() if count >= 2
    ):
        raise ValueError("Missing or invalid per-class TS-FID")
    return overall


def verify_index(index, archive, metrics_path=None):
    with np.load(index, allow_pickle=False) as manifest, np.load(archive, allow_pickle=False) as actual:
        labels = manifest["labels"]
        references = actual["ground_truth"]
        if not np.array_equal(labels, actual["labels"]):
            raise ValueError("Archive labels differ from the frozen reference schedule")
        if references.shape != (len(labels), 16, 1000):
            raise ValueError("Archive reference shape differs from the frozen schedule")
        for row, (reference, digest) in enumerate(zip(references, manifest["ground_truth_sha256"], strict=True)):
            if waveform_hash(reference) != digest:
                raise ValueError(f"Reference waveform hash mismatch at row {row}")
        if metrics_path is not None:
            del references
            generated = actual["generated"]
            if generated.shape != (len(labels), 16, 1000):
                raise ValueError("Generated waveforms have an unexpected shape")
            for start in range(0, len(generated), 256):
                chunk = generated[start : start + 256]
                if not np.isfinite(chunk).all():
                    raise ValueError(f"Nonfinite generated waveform in batch starting at row {start}")
                if np.any(np.all(chunk == 0, axis=(1, 2))):
                    raise ValueError(f"Zero generated waveform in batch starting at row {start}")
            score = verify_metrics(labels, metrics_path)
            print(f"Verified generated waveforms and metric metadata. Reported TS-FID: {score:.6f}")
    return len(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--raw", required=True, type=Path)
    build.add_argument("--processed", required=True, type=Path)
    build.add_argument("--split", required=True, choices=("train", "val", "test"))
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--index", required=True, type=Path)
    verify.add_argument("--archive", required=True, type=Path)
    verify.add_argument("--metrics", type=Path, help="Also verify the generated data and released metrics metadata")
    args = parser.parse_args()
    if args.command == "verify":
        print(f"Verified {verify_index(args.index, args.archive, args.metrics)} reference rows")
        return

    os.umask(0o077)
    manifest = build_index(args.raw, args.processed, args.split, args.seed)
    with args.output.open("xb") as output:
        np.savez_compressed(output, **manifest)
    print(f"Froze {len(manifest['labels'])} {args.split} reference rows at seed {args.seed}")


if __name__ == "__main__":
    main()
