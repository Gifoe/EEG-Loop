"""Inventory local TUEV sources and trusted pickles produced by upstream."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def audit_raw(root):
    recording_subjects = {}
    splits, subjects = {}, {}
    for split in ("train", "eval"):
        base = root / split
        recordings = defaultdict(list)
        missing, invalid = [], []
        event_counts = Counter()
        subjects[split] = set()
        for edf in sorted(base.rglob("*.edf")):
            recordings[edf.stem].append(str(edf.relative_to(base)))
            subjects[split].add(edf.parent.name)
            recording_subjects[split, edf.stem] = edf.parent.name
            rec = edf.with_suffix(".rec")
            if not rec.is_file():
                missing.append(str(edf.relative_to(base)))
                continue
            try:
                events = np.loadtxt(rec, delimiter=",", ndmin=2)
                if events.shape[1] != 4 or not np.isfinite(events).all():
                    raise ValueError("Malformed annotation")
            except ValueError:
                invalid.append(str(rec.relative_to(base)))
                continue
            with edf.open("rb") as stream:
                header = stream.read(256)
            try:
                duration = int(header[236:244]) * float(header[244:252])
            except ValueError:
                event_counts["unreadable_duration"] += 1
            else:
                if duration > 0:
                    event_counts["wrap_left"] += int((events[:, 1] < 2).sum())
                    event_counts["wrap_right"] += int((events[:, 2] + 2 > duration).sum())
                    event_counts["out_of_record"] += int(
                        ((events[:, 1] < 0) | (events[:, 2] >= duration)).sum()
                    )
                else:
                    event_counts["unreadable_duration"] += 1
            event_counts["all"] += len(events)
            event_counts["one_row_recordings"] += int(len(events) == 1)
            event_counts["non_one_second"] += int((~np.isclose(events[:, 2] - events[:, 1], 1)).sum())
            event_counts.update(f"label_{label:g}" for label in events[:, 3])
        splits[split] = dict(
            recordings=sum(map(len, recordings.values())),
            subjects=len(subjects[split]), events=dict(event_counts),
            collisions={key: paths for key, paths in recordings.items() if len(paths) > 1},
            missing_rec=missing, invalid_rec=invalid,
        )
    return dict(splits=splits, overlap=sorted(subjects["train"] & subjects["eval"]),
                train_prefix_mismatches=sum(stem.split("_")[0] != subject
                    for (split, stem), subject in recording_subjects.items() if split == "train")), recording_subjects


def audit_processed(root, recording_subjects):
    splits, subjects, waveforms = {}, {}, {}
    for split in ("train", "val", "test"):
        files = sorted((root / split).glob("*.pkl"))
        source_split = "eval" if split == "test" else "train"
        subjects[split] = {recording_subjects[source_split, path.stem.rsplit("-", 1)[0]] for path in files}
        shapes, labels = Counter(), Counter()
        invalid, rms, maxima = [], [], []
        waveforms[split] = set()
        digest = hashlib.sha256()
        for path in files:
            content = path.read_bytes()
            digest.update(path.name.encode() + b"\0" + hashlib.sha256(content).digest())
            try:
                sample = pickle.loads(content)
                signal = sample["signal"]
                label = int(sample["label"][0])
                shapes[str(signal.shape)] += 1
                labels[label] += 1
                if signal.shape != (16, 1000) or not np.isfinite(signal).all() or label not in range(1, 7):
                    invalid.append(path.name)
                    continue
                waveforms[split].add(hashlib.sha256(signal.tobytes()).digest())
                rms.append(float(np.sqrt(np.mean(signal ** 2))))
                maxima.append(float(np.max(np.abs(signal))))
            except (ValueError, TypeError, KeyError, IndexError, pickle.UnpicklingError, EOFError):
                invalid.append(path.name)
        splits[split] = dict(
            samples=len(files), subjects=len(subjects[split]), shapes=dict(shapes),
            labels=dict(labels), invalid=invalid, manifest_sha256=digest.hexdigest(),
            rms_uv=np.percentile(rms, [0, 25, 50, 75, 95, 100]).tolist() if rms else [],
            peak_uv=np.percentile(maxima, [0, 25, 50, 75, 95, 100]).tolist() if maxima else [],
        )
    overlap = {
        f"{left}/{right}": sorted(subjects[left] & subjects[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return dict(splits=splits, overlap=overlap, waveform_hash_overlap={
        f"{left}/{right}": len(waveforms[left] & waveforms[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw, recording_subjects = audit_raw(args.raw)
    result = {"raw": raw}
    if args.processed:
        result["processed"] = audit_processed(args.processed, recording_subjects)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")



