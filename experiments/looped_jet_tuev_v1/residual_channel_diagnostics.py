"""Describe residual channel dependence after removing class mean waveforms."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

VARIANCE_FLOOR = 1e-12
UPPER = np.triu_indices(16, k=1)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def channel_names(source):
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    pairs = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "BIPOLAR_PAIRS" for target in node.targets))
    return [f"{left.removeprefix('EEG ').removesuffix('-REF')}-{right.removeprefix('EEG ').removesuffix('-REF')}"
            for left, right in pairs]


def domain(eeg, rows):
    if not len(rows):
        raise ValueError("Cannot estimate a class mean from zero windows")
    if not np.isfinite(eeg).all():
        raise ValueError("Nonfinite raw EEG inputs")
    mean = np.zeros((16, 1000), dtype=np.float64)
    for start in range(0, len(rows), 128):
        mean += eeg[rows[start:start + 128]].astype(np.float64).sum(axis=0) * 100
    mean /= len(rows)
    cov_sum = np.zeros((16, 16), dtype=np.float64)
    corr_sum = np.zeros_like(cov_sum)
    counts = np.zeros((16, 16), dtype=np.int64)
    covariance_counts = np.zeros_like(counts)
    invalid_covariance = np.zeros_like(counts)
    nearconstant = np.zeros(16, dtype=np.int64)
    for start in range(0, len(rows), 128):
        residual = eeg[rows[start:start + 128]].astype(np.float64) * 100 - mean
        residual -= residual.mean(axis=-1, keepdims=True)
        cov = residual @ residual.transpose(0, 2, 1) / (residual.shape[-1] - 1)
        variances = np.diagonal(cov, axis1=1, axis2=2)
        nearconstant += (variances <= VARIANCE_FLOOR).sum(axis=0)
        valid_channels = np.isfinite(variances) & (variances > VARIANCE_FLOOR)
        valid = valid_channels[:, :, None] & valid_channels[:, None, :] & np.isfinite(cov)
        invalid_covariance += (~np.isfinite(cov)).sum(axis=0)
        denominator = np.sqrt(variances[:, :, None] * variances[:, None, :])
        corr = np.divide(cov, denominator, where=valid, out=np.full_like(cov, np.nan))
        valid &= np.isfinite(corr)
        cov_valid = np.isfinite(cov)
        cov_sum += np.where(cov_valid, cov, 0).sum(axis=0)
        covariance_counts += cov_valid.sum(axis=0)
        corr_sum += np.where(valid, corr, 0).sum(axis=0)
        counts += valid.sum(axis=0)
    covariance = np.divide(cov_sum, covariance_counts, where=covariance_counts > 0, out=np.full_like(cov_sum, np.nan))
    correlation = np.divide(corr_sum, counts, where=counts > 0, out=np.full_like(corr_sum, np.nan))
    return {
        "mean_estimation_windows": len(rows), "class_mean_waveform_uv": mean,
        "mean_residual_covariance_uv2": covariance, "mean_residual_correlation": correlation,
        "valid_correlation_window_counts_by_pair": counts, "valid_covariance_window_counts_by_pair": covariance_counts,
        "nearconstant_channel_window_counts": nearconstant,
        "nonfinite_covariance_window_counts_by_pair": invalid_covariance,
        "correlation_pairs_without_valid_windows": int((counts[UPPER] == 0).sum()),
        "covariance_pairs_without_valid_windows": int((covariance_counts[UPPER] == 0).sum()),
        "nonfinite_raw_input_values": 0,
    }


def compare(reference, generated, labels):
    if reference.shape != generated.shape or reference.shape != (len(labels), 16, 1000):
        raise ValueError("Expected matched N by 16 by 1000 arrays")
    if labels.ndim != 1 or set(labels) != set(range(6)):
        raise ValueError("Expected all six class labels")
    classes = {}
    for label in range(6):
        rows = np.flatnonzero(labels == label)
        real, fake = domain(reference, rows), domain(generated, rows)
        corr_defined = real["correlation_pairs_without_valid_windows"] == fake["correlation_pairs_without_valid_windows"] == 0
        cov_defined = real["covariance_pairs_without_valid_windows"] == fake["covariance_pairs_without_valid_windows"] == 0
        classes[str(label)] = {
            "reference": real, "generated": fake, "correlation_defined": corr_defined, "covariance_defined": cov_defined,
            "covariance_mae_uv2": float(np.abs(real["mean_residual_covariance_uv2"][UPPER] -
                                               fake["mean_residual_covariance_uv2"][UPPER]).mean()) if cov_defined else None,
            "correlation_mae": float(np.abs(real["mean_residual_correlation"][UPPER] -
                                            fake["mean_residual_correlation"][UPPER]).mean()) if corr_defined else None,
        }
    cov_defined = all(row["covariance_defined"] for row in classes.values())
    corr_defined = all(row["correlation_defined"] for row in classes.values())
    return {
        "by_class": classes, "macro_covariance_defined": cov_defined, "macro_correlation_defined": corr_defined,
        "macro_covariance_mae_uv2": float(np.mean([row["covariance_mae_uv2"] for row in classes.values()])) if cov_defined else None,
        "macro_correlation_mae": float(np.mean([row["correlation_mae"] for row in classes.values()])) if corr_defined else None,
    }


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, dict):
        return {key: serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [serializable(val) for val in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(archive, index, output):
    archive_sha = sha(archive)
    with np.load(index, allow_pickle=False) as data:
        labels = data["labels"]
        if str(data["split"]) != "test":
            raise ValueError("Expected held-out index")
    with np.load(archive, allow_pickle=False) as data:
        if not np.array_equal(labels, data["labels"]):
            raise ValueError("Archive label schedule differs from held-out index")
        reference, generated = data["ground_truth"], data["generated"]
    source = Path(__file__).resolve().parents[2] / "data" / "preprocess_tuev.py"
    result = compare(reference, generated, labels)
    result.update(
        channel_order=channel_names(source), archive_sha256=archive_sha, reference_index_sha256=sha(index),
        script_sha256=sha(Path(__file__)), montage_source_sha256=sha(source), variance_floor_uv2=VARIANCE_FLOOR,
        interpretation="Descriptive mechanism diagnostic. Each domain estimates its own class mean from all evaluated rows. Finite-N mean estimation shrinks residual variation. Real/generated class counts match. No quality gate change.",
    )
    if sha(archive) != archive_sha:
        raise ValueError("Archive changed during analysis")
    output.mkdir()
    (output / "residual-channel.json").write_text(
        json.dumps(serializable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "residual.done").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "index", "output"):
        parser.add_argument(arg, type=Path)
    main(**vars(parser.parse_args()))

