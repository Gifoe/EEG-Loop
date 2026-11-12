"""Frozen CPU-only 2-population by 2-metric control on our R0 archives."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy import linalg
import torch
from data.metrics import _prepare_fid_features

torch.set_num_threads(1)
ARCHIVES = {
    "B-test": ("eval-heldout-test-seed0", 29421, [567, 4677, 1998, 329, 2204, 19646],
               "9870912868b6fbcd812bcb43684a9fb4d6a6e0556cae2124a34fa48bc7bbf73c",
               "25d401bb8b0d233a54b6f29f65d47663c2ff36c07e4872e8aad6aa0031d80745",
               "test", "8fc319346fcfc97b6964aef6e7ee2490961a16842c67ee73baedf0a28607b884"),
    "A-train": ("eval-released-train-seed0", 68445, [526, 10963, 5515, 984, 10553, 39904],
                "a1381b1557c4adb62b75efbafc67e874fb8e496a42dc7eace6bf08e851f96387",
                "3b138d0bc3b689cf900a43a027c98b9f9d0bba3e023a62147d2c1c8d53b3a015",
                "train", "15965df6f48e05c80a3792b31ef1c8da27b6949b62a008f45614bbe741845125"),
}
OPERATORS = {"tsfid-source-1000": (0.5, 1000), "tsfid-paper-v1-1024": (None, 1024)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + chr(10))


class Moments:
    """Raw sums for source formula, independent centered Chan covariance for audit."""
    def __init__(self, dimensions):
        self.n = 0
        self.sum = np.zeros(dimensions)
        self.outer = np.zeros((dimensions, dimensions))
        self.mean = np.zeros(dimensions)
        self.m2 = np.zeros((dimensions, dimensions))

    def add(self, values):
        if not len(values):
            return
        count = len(values)
        average = values.mean(axis=0)
        centered = values - average
        delta = average - self.mean
        total = self.n + count
        self.m2 += centered.T @ centered + np.outer(delta, delta) * (self.n * count / total)
        self.mean += delta * (count / total)
        self.sum += values.sum(axis=0)
        self.outer += values.T @ values
        self.n = total

    def stats(self, independent=False):
        assert self.n >= 2
        if independent:
            return self.mean, self.m2 / (self.n - 1)
        average = self.sum / self.n
        return average, (self.outer - self.n * np.outer(average, average)) / (self.n - 1)


def source_fid(generated, reference):
    mg, cg = generated
    mr, cr = reference
    delta = mg - mr
    root, _ = linalg.sqrtm(cg @ cr, disp=False)
    if not np.isfinite(root).all():
        offset = np.eye(len(cg)) * 1e-6
        root, _ = linalg.sqrtm((cg + offset) @ (cr + offset), disp=False)
    if np.iscomplexobj(root):
        root = root.real
    return float(delta @ delta + np.trace(cg + cr - 2 * root))


def symmetric_fid(generated, reference):
    mg, cg = generated
    mr, cr = reference
    eigenvalues, eigenvectors = np.linalg.eigh(cr)
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    product = root @ cg @ root
    spectrum = np.linalg.eigvalsh((product + product.T) * 0.5)
    return float((mg - mr) @ (mg - mr) + np.trace(cg) + np.trace(cr)
                 - 2 * np.sqrt(np.maximum(spectrum, 0)).sum())


def moments(eeg, labels, cutoff, dimensions):
    groups = {name: Moments(dimensions) for name in ["overall", *map(str, range(6))]}
    for start in range(0, len(eeg), 256):
        values = _prepare_fid_features(torch.as_tensor(eeg[start:start + 256], dtype=torch.float32),
                                       feature_bins=256, spatial_bins=4, max_frequency_ratio=cutoff)
        assert values.shape == (min(256, len(eeg) - start), dimensions) and np.isfinite(values).all()
        groups["overall"].add(values)
        local = labels[start:start + 256]
        for label in range(6):
            groups[str(label)].add(values[local == label])
    return groups


def self_test():
    values = np.random.default_rng(125).normal(size=(23, 8))
    original, shifted = Moments(8), Moments(8)
    for part in np.array_split(values, [2, 9, 20]):
        original.add(part)
        shifted.add(part + 0.1)
    for independent in (False, True):
        mean, cov = original.stats(independent)
        assert np.allclose(mean, values.mean(axis=0), atol=1e-13, rtol=1e-13)
        assert np.allclose(cov, np.cov(values, rowvar=False), atol=1e-13, rtol=1e-13)
    identity = source_fid(original.stats(), original.stats())
    known = symmetric_fid(shifted.stats(True), original.stats(True))
    assert abs(identity) < 1e-10 and abs(known - 0.08) < 1e-10
    return {"passed": True, "independent_covariance_matches_numpy": True,
            "identity_fid": identity, "analytic_mean_shift_fid": known, "expected": 0.08}


def freeze(root):
    repo = root / "JET"
    source = repo / "experiments/looped_jet_tuev_v1"
    env = root / "evidence/paper-aligned-v1-environment"
    assert (env / "engineering-smokes.done").exists()
    assert not torch.cuda.is_available() and torch.__version__ == "2.7.1+cu128"
    assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == "07f9e6491796f4f2c717d6259b1a2e24afce6a77"
    assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    records = {}
    for name, (folder, count, histogram, archive_sha, metric_sha, split, index_sha) in ARCHIVES.items():
        archive = root / folder / "eval_batch.npz"
        metrics = root / folder / "metrics.json"
        index = root / f"evidence/reference-index/{split}-seed0.npz"
        assert (root / folder / "evaluation.done").exists()
        assert (root / f"evidence/evaluation-integrity/{split}-seed0.verified").exists()
        assert sha(archive) == archive_sha and sha(metrics) == metric_sha and sha(index) == index_sha
        records[name] = {"archive": str(archive), "archive_sha256": archive_sha, "rows": count,
                         "histogram": histogram, "metrics": str(metrics), "metrics_sha256": metric_sha,
                         "index": str(index), "index_sha256": index_sha}
    output = root / "evidence/own-r0-rescore-v1"
    output.mkdir(mode=0o700)
    record = {"protocol_sha256": sha(source / "OWN_R0_RESCORING_PROTOCOL.md"),
              "source_sha256": {"rescore_own.py": sha(Path(__file__)), "data/metrics.py": sha(repo / "data/metrics.py")},
              "environment_sha256": {n: sha(env / n) for n in ["pip-freeze.txt", "wheel-lock.json", "import-check.json", "conda-explicit-lock.txt"]},
              "archives": records, "operators": OPERATORS, "execution_order": ["B-test", "A-train"],
              "cpu_threads": 1, "self_test": self_test(), "no_checkpoint_or_generation": True}
    write(output / "freeze.json", record)
    (output / "freeze.json").chmod(0o400)
    print(json.dumps({"freeze_sha256": sha(output / "freeze.json"), "self_test": record["self_test"]}))


def run(root):
    folder = root / "evidence/own-r0-rescore-v1"
    config_path = folder / "freeze.json"
    config = json.loads(config_path.read_text())
    source = root / "JET/experiments/looped_jet_tuev_v1"
    assert sha(Path(__file__)) == config["source_sha256"]["rescore_own.py"]
    assert sha(root / "JET/data/metrics.py") == config["source_sha256"]["data/metrics.py"]
    assert sha(source / "OWN_R0_RESCORING_PROTOCOL.md") == config["protocol_sha256"]
    env = root / "evidence/paper-aligned-v1-environment"
    assert all(sha(env / n) == digest for n, digest in config["environment_sha256"].items())
    assert not torch.cuda.is_available() and torch.__version__ == "2.7.1+cu128"
    results = {}
    for population in config["execution_order"]:
        record = config["archives"][population]
        archive_path = Path(record["archive"])
        assert sha(archive_path) == record["archive_sha256"]
        assert sha(record["metrics"]) == record["metrics_sha256"] and sha(record["index"]) == record["index_sha256"]
        old_scores = json.loads(Path(record["metrics"]).read_text())["ts_fid"]
        with np.load(record["index"], allow_pickle=False) as index:
            expected_labels = index["labels"].copy()
            expected_hashes = index["ground_truth_sha256"].copy()
            assert len(set(index["processed_files"].tolist())) == record["rows"]
        with np.load(archive_path, allow_pickle=False) as archive:
            labels, generated, reference = archive["labels"], archive["generated"], archive["ground_truth"]
        assert np.array_equal(labels, expected_labels)
        assert [Counter(labels.tolist())[c] for c in range(6)] == record["histogram"]
        assert generated.shape == reference.shape == (record["rows"], 16, 1000)
        for start in range(0, len(labels), 128):
            assert np.isfinite(generated[start:start + 128]).all() and np.isfinite(reference[start:start + 128]).all()
            assert not np.all(generated[start:start + 128] == 0, axis=(1, 2)).any()
        for row, digest in zip(reference, expected_hashes, strict=True):
            assert hashlib.sha256(np.ascontiguousarray(row, dtype="<f4")).hexdigest() == digest
        population_results = {}
        for metric_id, (cutoff, dimensions) in OPERATORS.items():
            gs, rs = moments(generated, labels, cutoff, dimensions), moments(reference, labels, cutoff, dimensions)
            checks = {}
            for group in ["overall", *map(str, range(6))]:
                original = source_fid(gs[group].stats(), rs[group].stats())
                audited = symmetric_fid(gs[group].stats(True), rs[group].stats(True))
                tolerance = 1e-3 + 1e-5 * abs(original)
                passed = math.isfinite(original) and math.isfinite(audited) and abs(original - audited) <= tolerance
                checks[group] = {"rows_per_collection": gs[group].n, "score": original,
                                 "symmetric_audit": audited, "difference": audited - original,
                                 "absolute_tolerance": tolerance, "passed": passed}
                if metric_id == "tsfid-source-1000":
                    old = old_scores["overall"] if group == "overall" else old_scores["per_class"][group]
                    checks[group].update(old_environment_score=old, runtime_delta=original - old,
                                         runtime_within_frozen_tolerance=abs(original - old) <= 1e-3 + 1e-5 * abs(old))
            result = {"metric_id": metric_id, "dimensions": dimensions, "checks": checks,
                      "passed": all(v["passed"] for v in checks.values()),
                      "independent_covariance": "Centered Chan accumulation versus raw sums; symmetric eigensquare root versus source sqrtm"}
            write(folder / f"{population}-{metric_id}.json", result)
            assert result["passed"], f"Independent numeric audit failed: {population}/{metric_id}"
            population_results[metric_id] = result
            print(f"{population} {metric_id}: {checks['overall']['score']:.10f}; all numeric checks passed", flush=True)
            del gs, rs
        assert sha(archive_path) == record["archive_sha256"]
        results[population] = population_results
        del generated, reference
    write(folder / "results.json", {"freeze_sha256": sha(config_path), "results": results,
          "interpretation": "Our fixed R0 archives only. Metric/runtime control, no new generation/training. No author checkpoint. No model-quality improvement is implied by changing the metric."})
    (folder / "scoring.done").touch()
    for path in folder.iterdir():
        if path.is_file():
            path.chmod(0o400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["self-test", "freeze", "run"])
    parser.add_argument("root", type=Path, nargs="?")
    args = parser.parse_args()
    if args.action == "self-test":
        print(json.dumps(self_test()))
    elif args.action == "freeze":
        freeze(args.root.resolve())
    else:
        run(args.root.resolve())
