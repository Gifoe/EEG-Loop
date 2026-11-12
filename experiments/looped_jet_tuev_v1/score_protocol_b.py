"""Freeze or score fixed Protocol B held-out signal diagnostic subsets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrate_signal import REPLICATES, SEED, discrepancy, json_output
from signal_diagnostics import BANDS, batch_primitives

FAMILIES = {
    "psd": ("psd_mean_log10_mae_by_channel_0_5_to_75_hz",),
    "band_power": tuple(f"band_{name}" for name in BANDS),
    "morphology": ("peak_to_peak", "rms", "excess_kurtosis", "p95_abs_first_difference",
                   "centered_zero_crossing_rate"),
    "temporal": ("envelope_slope_uv_per_s", "last_minus_first_block_mean", "last_minus_first_block_rms"),
    "spatial": ("off_diagonal_correlation_mean_mae", "off_diagonal_correlation_distribution"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def label_rows(labels):
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or set(labels) != set(range(6)):
        raise ValueError("Expected one-dimensional labels covering all six TUEV classes")
    return {label: np.flatnonzero(labels == label) for label in range(6)}


def metadata(index, calibration):
    with np.load(index, allow_pickle=False) as rows:
        if str(rows["split"]) != "test" or int(rows["seed"]) != 0:
            raise ValueError("Expected frozen test reference index at seed 0")
        labels = rows["labels"]
    config_path = calibration / "calibration_config.json"
    scales_path = calibration / "median_scales.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    counts = config["matched_class_counts"]
    if (config["seed"] != SEED or config["replicates"] != REPLICATES
            or config["pool"] != ["train", "val"] or config["unreliable_classes"]
            or len(counts) != 6 or any(type(n) is not int or not 32 <= n <= 512 for n in counts)):
        raise ValueError("Calibration metadata does not meet frozen Protocol B")
    source = {
        "reference_index_sha256": digest(index), "calibration_config_sha256": digest(config_path),
        "median_scales_sha256": digest(scales_path), "seed_scheme": [SEED, 2, "replicate", "class", "side"],
        "side_order": ["reference", "generated"], "replicates": REPLICATES, "class_counts": counts,
    }
    return labels, counts, source


def fixed_positions(labels, counts):
    positions = {}
    for label, rows in label_rows(labels).items():
        n = counts[label]
        if n > len(rows):
            raise ValueError(f"Class {label} has {len(rows)} rows, fewer than the frozen count {n}")
        positions[str(label)] = np.stack([
            [np.sort(np.random.default_rng(np.random.SeedSequence([SEED, 2, rep, label, side]))
                     .choice(len(rows), n, replace=False)) for side in range(2)]
            for rep in range(REPLICATES)
        ])
    return positions


def freeze(index, calibration, plan):
    labels, counts, source = metadata(index, calibration)
    checked_scales(calibration)
    positions = fixed_positions(labels, counts)
    plan.mkdir()
    np.savez_compressed(plan / "class_relative_positions.npz", **positions)
    source["positions_sha256"] = digest(plan / "class_relative_positions.npz")
    json_output(plan / "plan.json", source)


def checked_scales(calibration):
    metrics = json.loads((calibration / "median_scales.json").read_text(encoding="utf-8"))
    keys = {key for family in FAMILIES.values() for key in family}
    if set(metrics) != {str(c) for c in range(6)}:
        raise ValueError("Calibration missing or adding classes")
    scales = {}
    for label, group in metrics.items():
        if set(group) != keys:
            raise ValueError(f"Unexpected primitive names in calibration class {label}")
        scales[label] = {}
        for key, evidence in group.items():
            shape = () if key.startswith("off_diagonal_") else (16,)
            scale = np.asarray(evidence["scale"], dtype=np.float64)
            finite = np.asarray(evidence["finite_replicates"])
            if (scale.shape != shape or finite.shape != shape or not np.isfinite(scale).all()
                    or (scale < 0).any() or (finite != REPLICATES).any()):
                raise ValueError(f"Invalid scale or replicate count for {label}/{key}")
            scales[label][key] = np.maximum(scale, 1e-8)
    return scales


def extract(eeg, indices):
    feature_chunks = {}
    for start in range(0, len(indices), 128):
        psd, primitive, zero = batch_primitives(eeg[indices[start : start + 128]])
        primitive.update(log_psd=psd, all_zero=zero)
        for key, values in primitive.items():
            feature_chunks.setdefault(key, []).append(values)
    return {key: np.concatenate(values) for key, values in feature_chunks.items()}


def score_class(reference, generated, rows, positions, scales):
    needed_r = np.unique(positions[:, 0])
    needed_g = np.unique(positions[:, 1])
    r = extract(reference, rows[needed_r])
    g = extract(generated, rows[needed_g])
    combined = {key: np.concatenate((values, g[key])) for key, values in r.items()}
    left = np.searchsorted(needed_r, positions[:, 0])
    right = np.searchsorted(needed_g, positions[:, 1]) + len(needed_r)
    scores = [discrepancy(combined, x, y) for x, y in zip(left, right)]
    native = {key: np.stack([row[key] for row in scores]) for key in scales}
    if any(not np.isfinite(values).all() for values in native.values()):
        raise ValueError("Undefined native discrepancy in a frozen held-out subset")
    ratio = {key: values / scales[key] for key, values in native.items()}
    families = {}
    for family, keys in FAMILIES.items():
        primitive_means = [ratio[key].reshape(len(scores), -1).mean(axis=1) for key in keys]
        families[family] = np.stack(primitive_means).mean(axis=0)
    return native, ratio, families, int(g["all_zero"].sum())


def score(index, calibration, plan, archive, output):
    labels, counts, source = metadata(index, calibration)
    expected = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
    positions_path = plan / "class_relative_positions.npz"
    source["positions_sha256"] = digest(positions_path)
    if source != expected:
        raise ValueError("Frozen positions or their provenance changed")
    with np.load(positions_path, allow_pickle=False) as loaded:
        positions = dict(loaded)
    reconstructed = fixed_positions(labels, counts)
    if positions.keys() != reconstructed.keys() or any(
        not np.array_equal(positions[k], v) for k, v in reconstructed.items()
    ):
        raise ValueError("Frozen positions differ from their declared sampling rule")
    scales = checked_scales(calibration)
    input_sha256 = digest(archive)
    with np.load(archive, allow_pickle=False) as data:
        if not np.array_equal(data["labels"], labels):
            raise ValueError("Held-out labels do not match frozen reference index")
        reference, generated = data["ground_truth"], data["generated"]
    for name, eeg in (("reference", reference), ("generated", generated)):
        if eeg.shape != (len(labels), 16, 1000) or not np.issubdtype(eeg.dtype, np.floating):
            raise ValueError(f"Invalid {name} EEG shape or dtype")
        if not np.isfinite(eeg).all():
            raise ValueError(f"Nonfinite {name} EEG")
    if not np.any(generated):
        raise ValueError("Generated archive contains only zeros")

    raw_replicates, by_class, family_arrays = {}, {}, {}
    for label, rows in label_rows(labels).items():
        key = str(label)
        native, ratio, families, zeros = score_class(reference, generated, rows, positions[key], scales[key])
        family_arrays[key] = families
        raw_replicates[key] = {
            "native": {name: values.tolist() for name, values in native.items()},
            "ratios": {name: values.tolist() for name, values in ratio.items()},
            "families": {name: values.tolist() for name, values in families.items()},
        }
        by_class[key] = {
            "native_means": {name: values.mean(axis=0).tolist() for name, values in native.items()},
            "primitive_ratio_means": {name: values.mean(axis=0).tolist() for name, values in ratio.items()},
            "family_means": {name: float(values.mean()) for name, values in families.items()},
            "all_zero_generated_windows_in_selected_union": zeros,
        }
        print(f"Scored held-out class {label}", flush=True)
    macro = {name: np.stack([group[name] for group in family_arrays.values()]).mean(axis=0)
             for name in FAMILIES}
    if input_sha256 != digest(archive):
        raise ValueError("Archive changed while being scored")
    output.mkdir()
    json_output(output / "replicates.json", {
        "by_class": raw_replicates, "macro": {name: values.tolist() for name, values in macro.items()},
    })
    json_output(output / "summary.json", {
        "archive_sha256": input_sha256, "frozen_plan": source, "by_class": by_class,
        "macro_family_means": {name: float(values.mean()) for name, values in macro.items()},
        "interpretation": "Single-checkpoint descriptive subset repetitions. No population interval or seed uncertainty.",
    })
    (output / "scoring.done").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("freeze", freeze), ("score", score)):
        sub = subparsers.add_parser(name)
        for arg in ("index", "calibration", "plan"):
            sub.add_argument(arg, type=Path)
        if name == "score":
            sub.add_argument("archive", type=Path)
            sub.add_argument("output", type=Path)
        sub.set_defaults(handler=handler)
    args = vars(parser.parse_args())
    handler = args.pop("handler")
    args.pop("command")
    handler(**args)

