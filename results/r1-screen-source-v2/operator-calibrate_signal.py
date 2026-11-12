"""Estimate Protocol B real-vs-real scales from official TUEV train and val."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import wasserstein_distance

from signal_diagnostics import batch_primitives

SEED = 20261111
REPLICATES = 200


def directory_components(raw):
    parents = {}
    digest_subjects = {}
    for path in sorted((raw / "train").rglob("*.edf")):
        directory = path.parent.name
        parents.setdefault(directory, directory)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").digest()
        previous = digest_subjects.setdefault(digest, directory)
        parents[root(parents, directory)] = root(parents, previous)
    components = {}
    for directory in sorted(parents):
        components.setdefault(root(parents, directory), []).append(directory)
    return list(components.values())


def root(parents, node):
    while node != parents[node]:
        node = parents[node]
    return node


def inventory(processed, components, reference_indices):
    component_by_dir = {directory: i for i, dirs in enumerate(components) for directory in dirs}
    paths = []
    labels = []
    component_ids = []
    index_digests = {}
    for split in ("train", "val"):
        index_path = reference_indices / f"{split}-seed0.npz"
        index_digests[split] = hashlib.sha256(index_path.read_bytes()).hexdigest()
        with np.load(index_path, allow_pickle=False) as index:
            if str(index["split"]) != split or int(index["seed"]) != 0:
                raise ValueError(f"Unexpected provenance for {split} reference index")
            rows = sorted(zip(index["processed_files"], index["labels"], index["subject_directories"]))
        if [row[0] for row in rows] != sorted(path.name for path in (processed / split).glob("*.pkl")):
            raise ValueError(f"{split} reference index does not cover the exact official pickle pool")
        for filename, label, directory in rows:
            if filename.split("_", 1)[0] != directory or not 0 <= label < 6:
                raise ValueError(f"Invalid filename/group or class provenance in {split}")
            paths.append(processed / split / filename)
            labels.append(label)
            component_ids.append(component_by_dir[directory])
    return paths, np.asarray(labels), np.asarray(component_ids), index_digests


def split_plan(labels, component_ids, num_components):
    partition_rng = np.random.default_rng(np.random.SeedSequence([SEED, 0]))
    left = np.zeros((REPLICATES, num_components), dtype=bool)
    for row in left:
        row[partition_rng.permutation(num_components)[:num_components // 2]] = True
    eligible = np.zeros((REPLICATES, 6, 2), dtype=int)
    supporting = np.zeros_like(eligible)
    for label in range(6):
        counts = np.bincount(component_ids[labels == label], minlength=num_components)
        eligible[:, label, 0] = left @ counts
        eligible[:, label, 1] = (~left) @ counts
        supporting[:, label, 0] = left @ (counts > 0).astype(int)
        supporting[:, label, 1] = (~left) @ (counts > 0).astype(int)
    matched = np.minimum(512, eligible.min(axis=(0, 2)))
    samples = {}
    for label, n in enumerate(matched):
        if not n:
            continue
        label_indices = np.flatnonzero(labels == label)
        pairs = []
        for rep, partition in enumerate(left):
            membership = partition[component_ids[label_indices]]
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, rep, label]))
            pairs.append((rng.choice(label_indices[membership], n, replace=False),
                          rng.choice(label_indices[~membership], n, replace=False)))
        samples[label] = np.asarray(pairs)
    return left, eligible, supporting, matched, samples


def feature_cache(paths, needed, cache):
    # Use float64 caches so real-vs-real and generated-vs-real use identical primitives.
    arrays = {}
    for start in range(0, len(needed), 128):
        indices = needed[start : start + 128]
        batch = []
        for index in indices:
            with paths[index].open("rb") as stream:
                signal = pickle.load(stream)["signal"]
            batch.append((signal / 100.0).astype(np.float32))
        psd, metrics, zero = batch_primitives(np.stack(batch))
        metrics.update(log_psd=psd, all_zero=zero)
        for name, values in metrics.items():
            if name not in arrays:
                arrays[name] = np.lib.format.open_memmap(
                    cache / f"{name}.npy", mode="w+", dtype=values.dtype,
                    shape=(len(needed), *values.shape[1:]),
                )
            arrays[name][start : start + len(indices)] = values
        if start % 4096 == 0:
            print(f"Cached {start + len(indices)}/{len(needed)} windows", flush=True)
    for array in arrays.values():
        array.flush()
    return arrays


def w1(left, right):
    x, y = left[np.isfinite(left)], right[np.isfinite(right)]
    if len(x) == len(y) and len(x):
        return np.abs(np.sort(x) - np.sort(y)).mean()
    if len(x) and len(y):
        return wasserstein_distance(x, y)
    return np.nan


def discrepancy(features, left, right):
    out = {}
    for key, values in features.items():
        x, y = values[left], values[right]
        if key == "log_psd":
            out["psd_mean_log10_mae_by_channel_0_5_to_75_hz"] = np.abs(
                x.mean(axis=0) - y.mean(axis=0)).mean(axis=-1)
        elif key == "off_diagonal_correlations":
            x_count = np.isfinite(x).sum(axis=0)
            y_count = np.isfinite(y).sum(axis=0)
            x_mean = np.divide(np.nansum(x, axis=0), x_count, where=x_count > 0,
                               out=np.full(x_count.shape, np.nan))
            y_mean = np.divide(np.nansum(y, axis=0), y_count, where=y_count > 0,
                               out=np.full(y_count.shape, np.nan))
            diffs = np.abs(x_mean - y_mean)
            out["off_diagonal_correlation_mean_mae"] = (
                np.nanmean(diffs) if np.isfinite(diffs).any() else np.nan)
            out["off_diagonal_correlation_distribution"] = w1(x.ravel(), y.ravel())
        elif key == "all_zero":
            continue
        else:
            out[key] = np.asarray([w1(x[:, ch], y[:, ch]) for ch in range(16)])
    return out


def json_output(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(raw, processed, manifest, reference_indices, output):
    output.mkdir()
    cache = output / "cache"
    cache.mkdir()
    components = directory_components(raw)
    paths, labels, ids, index_digests = inventory(processed, components, reference_indices)
    left, eligible, supporting, matched, samples = split_plan(labels, ids, len(components))
    needed = np.unique(np.concatenate([pair.ravel() for pair in samples.values()]))
    config = {
        "seed": SEED, "replicates": REPLICATES, "pool": ["train", "val"],
        "reference_index_sha256": index_digests,
        "processed_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "partition_seed_sequence": [SEED, 0], "sample_seed_sequence": [SEED, 1, "replicate", "class"],
        "components_by_directory_id": components,
        "matched_class_counts": matched.tolist(),
        "minimum_supporting_components_per_half": supporting.min(axis=(0, 2)).tolist(),
        "unreliable_classes": np.flatnonzero((matched < 32) | (supporting.min(axis=(0, 2)) < 5)).tolist(),
        "available_class_count_ranges": np.stack((eligible.min(axis=0), eligible.max(axis=0))).tolist(),
        "weighting": "Official windows retained. EDF-linked directory IDs assigned wholesale. Uniform windows per class and half.",
        "needed_cache_windows": len(needed),
        "primitive_source_sha256": hashlib.sha256(Path(__file__).with_name("signal_diagnostics.py").read_bytes()).hexdigest(),
        "calibrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "paths_are_restricted": True,
    }
    json_output(output / "calibration_config.json", config)
    json_output(output / "partitions.json", {
        "left_component_indices": [np.flatnonzero(row).tolist() for row in left],
        "available_class_counts": eligible.tolist(), "supporting_component_counts": supporting.tolist(),
    })
    json_output(output / "window_index.private.json", [str(path.relative_to(processed)) for path in paths])
    np.savez_compressed(output / "sample_indices.npz", **{str(label): p for label, p in samples.items()})
    np.save(cache / "window_indices.npy", needed)
    features = feature_cache(paths, needed, cache)
    cache_counts = {}
    for label in range(6):
        local_rows = np.flatnonzero(labels[needed] == label)
        cache_counts[str(label)] = {}
        for name, array in features.items():
            if name == "all_zero":
                cache_counts[str(label)]["all_zero_windows"] = int(array[local_rows].sum())
                continue
            finite_count = sum(np.isfinite(array[rows]).sum().item()
                               for rows in np.array_split(local_rows, max(1, len(local_rows) // 256)))
            cache_counts[str(label)][name] = {
                "finite_values": finite_count,
                "undefined_values": int(len(local_rows) * np.prod(array.shape[1:])) - finite_count,
            }
    json_output(output / "cache_primitive_counts.json", cache_counts)
    raw_metrics = {}
    medians = {}
    for label, pairs in samples.items():
        class_metrics = [discrepancy(features, *pair) for pair in np.searchsorted(needed, pairs)]
        raw_metrics[str(label)] = {}
        medians[str(label)] = {}
        for name in class_metrics[0]:
            values = np.stack([row[name] for row in class_metrics])
            finite = np.isfinite(values)
            raw_metrics[str(label)][name] = np.where(finite, values, None).tolist()
            median = np.nanmedian(values, axis=0)
            medians[str(label)][name] = {
                "scale": np.where(np.isfinite(median), median, None).tolist(),
                "finite_replicates": finite.sum(axis=0).tolist(),
                "undefined_replicates": (~finite).sum(axis=0).tolist(),
            }
        print(f"Completed class {label} ({matched[label]} windows per half)", flush=True)
    json_output(output / "native_discrepancies.json", raw_metrics)
    json_output(output / "median_scales.json", medians)
    (output / "calibration.done").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "processed", "manifest", "reference_indices", "output"):
        parser.add_argument(name, type=Path)
    main(**vars(parser.parse_args()))


