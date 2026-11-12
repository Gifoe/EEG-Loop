"""Audit EDF and labeled event window reuse across the upstream validation split."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def audit(raw, processed):
    splits = {name: set() for name in ("train", "val", "test")}
    for split, stems in splits.items():
        for path in (processed / split).glob("*.pkl"):
            stems.add(path.stem.rsplit("-", 1)[0])
    train_records = defaultdict(list)
    for path in (raw / "train").rglob("*.edf"):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        train_records[digest].append(path)
    groups = [(digest, paths) for digest, paths in train_records.items() if len(paths) > 1]
    report = {"duplicate_edf_groups": len(groups), "different_directory_groups": 0,
              "cross_train_val_edf_groups": 0, "shared_annotation_centers": 0,
              "shared_five_second_windows": 0, "shared_labeled_samples": 0,
              "overlapping_window_pairs": 0}
    for digest, paths in groups:
        del digest
        report["different_directory_groups"] += len({path.parent for path in paths}) > 1
        by_split = {split: set() for split in ("train", "val")}
        for path in paths:
            for split, events in by_split.items():
                if path.stem in splits[split]:
                    for line in path.with_suffix(".rec").read_text().splitlines():
                        channel, start, end, label = map(float, line.split(","))
                        events.add((round(start * 200), round(end * 200), channel, label))
        train, val = by_split["train"], by_split["val"]
        if not train or not val:
            continue
        report["cross_train_val_edf_groups"] += 1
        train_coords = {(event[0], event[1]) for event in train}
        val_coords = {(event[0], event[1]) for event in val}
        report["shared_annotation_centers"] += len(train_coords & val_coords)
        report["shared_five_second_windows"] += len(train_coords & val_coords)
        report["shared_labeled_samples"] += len(train & val)
        report["overlapping_window_pairs"] += sum(
            left_start - 400 < right_end + 400 and right_start - 400 < left_end + 400
            for left_start, left_end in train_coords for right_start, right_end in val_coords
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("processed", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.processed), indent=2))
