"""Report aggregate EDF header identity and byte-duplicate checks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(base):
    header_ids, edf_digests, pair_digests = Counter(), Counter(), Counter()
    unidentified = 0
    for edf in sorted(base.rglob("*.edf")):
        with edf.open("rb") as stream:
            header = stream.read(256)
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        patient = header[8:88].strip().split()
        if patient and patient[0].upper() not in (b"X", b"UNKNOWN", b"ANONYMOUS"):
            header_ids[patient[0].lower()] += 1
        else:
            unidentified += 1
        edf_digests[digest] += 1
        rec = edf.with_suffix(".rec")
        if rec.exists():
            pair_digests[digest, hashlib.sha256(rec.read_bytes()).hexdigest()] += 1
    return {"patient_code": header_ids, "edf_sha256": edf_digests,
            "edf_rec_sha256_pair": pair_digests}, unidentified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    train, train_missing = audit(args.root / "train")
    test, test_missing = audit(args.root / "eval")
    report = {"unidentified_headers": {"train": train_missing, "eval": test_missing}}
    for key in train:
        report[key] = {
            "distinct_train": len(train[key]), "distinct_eval": len(test[key]),
            "duplicate_train_records": sum(n - 1 for n in train[key].values()),
            "duplicate_eval_records": sum(n - 1 for n in test[key].values()),
            "cross_split_unique_matches": len(train[key].keys() & test[key].keys()),
        }
    print(json.dumps(report, indent=2))
