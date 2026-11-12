"""Cross-check released TS-FID using a symmetric covariance eigendecomposition."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from data.metrics import _prepare_fid_features


def features(eeg, bins):
    # Match the public evaluator's float32 input and spectral feature extractor.
    return np.concatenate([
        _prepare_fid_features(torch.as_tensor(eeg[start : start + 256], dtype=torch.float32), bins)
        for start in range(0, len(eeg), 256)
    ])


def symmetric_fid(generated, reference):
    delta = generated.mean(axis=0) - reference.mean(axis=0)
    cov_g = np.cov(generated, rowvar=False)
    cov_r = np.cov(reference, rowvar=False)
    eigenvalues, vectors = np.linalg.eigh(cov_r)
    root_r = (vectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ vectors.T
    product = root_r @ cov_g @ root_r
    spectrum = np.linalg.eigvalsh((product + product.T) * 0.5)
    trace_root = np.sqrt(np.maximum(spectrum, 0.0)).sum()
    return float(delta @ delta + np.trace(cov_g) + np.trace(cov_r) - 2.0 * trace_root)


def compare(generated, reference, upstream):
    audited = symmetric_fid(generated, reference)
    tolerance = 1e-3 + 1e-5 * abs(upstream)
    difference = audited - upstream
    return {
        "samples_per_collection": len(reference),
        "upstream": upstream,
        "symmetric": audited,
        "difference": difference,
        "absolute_tolerance": tolerance,
        "passed": math.isfinite(audited) and abs(difference) <= tolerance,
    }


def audit(archive, metrics):
    scores = json.loads(metrics.read_text(encoding="utf-8"))["ts_fid"]
    bins = scores["feature_bins"]
    with np.load(archive, allow_pickle=False) as arrays:
        labels = arrays["labels"]
        generated = features(arrays["generated"], bins)
        reference = features(arrays["ground_truth"], bins)
    checks = {"overall": compare(generated, reference, scores["overall"])}
    for label, upstream in scores["per_class"].items():
        mask = labels == int(label)
        if mask.sum() >= 2:
            checks[label] = compare(generated[mask], reference[mask], upstream)
    return {"effective_feature_dimensions": reference.shape[1], "checks": checks,
            "passed": all(check["passed"] for check in checks.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "metrics", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = audit(args.archive, args.metrics)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("Independent TS-FID cross-check exceeded the frozen tolerance")
