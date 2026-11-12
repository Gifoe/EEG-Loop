"""Measure raw Protocol B signal-quality discrepancies for full upstream inference."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, wasserstein_distance

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
FLOOR = 1e-15
OFF_DIAGONAL = np.triu_indices(16, k=1)
BLOCK_TIME = np.arange(5) - 2


def batch_primitives(x):
    """Return per-window diagnostics for loader-space EEG in one batch."""
    x = x.astype(np.float64) * 100
    all_zero = (x == 0).all(axis=(1, 2))
    freq, psd = welch(x, fs=200, window="hann", nperseg=400,
                      noverlap=200, nfft=400, detrend="constant", axis=-1)
    log_psd = np.log10(np.maximum(psd[..., (freq >= 0.5) & (freq <= 75)], FLOOR))
    centered = x - x.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    unit = np.divide(centered, norm, where=norm > 0, out=np.full_like(centered, np.nan))
    corr = unit @ unit.transpose(0, 2, 1)
    blocks = x.reshape(-1, 16, 5, 200)
    envelope = np.sqrt(np.mean(blocks**2, axis=-1))
    metrics = {
        f"band_{name}": np.log10(np.maximum(
            psd[..., (freq >= left) & ((freq <= right) if name == "gamma" else (freq < right))]
            .sum(axis=-1) * (freq[1] - freq[0]), FLOOR))
        for name, (left, right) in BANDS.items()
    }
    metrics.update(
        peak_to_peak=np.ptp(x, axis=-1),
        rms=np.sqrt(np.mean(x**2, axis=-1)),
        excess_kurtosis=kurtosis(x, axis=-1, fisher=True),
        p95_abs_first_difference=np.quantile(np.abs(np.diff(x, axis=-1)), 0.95, axis=-1),
        centered_zero_crossing_rate=((centered[..., 1:] >= 0) != (centered[..., :-1] >= 0)).mean(axis=-1),
        envelope_slope_uv_per_s=(envelope @ BLOCK_TIME) / (BLOCK_TIME @ BLOCK_TIME),
        last_minus_first_block_mean=blocks[..., -1, :].mean(axis=-1) - blocks[..., 0, :].mean(axis=-1),
        last_minus_first_block_rms=envelope[..., -1] - envelope[..., 0],
        off_diagonal_correlations=corr[:, OFF_DIAGONAL[0], OFF_DIAGONAL[1]],
    )
    return log_psd, metrics, all_zero


def primitives(eeg, indices):
    log_psd_sum = np.zeros((16, 150), dtype=np.float64)
    chunks = {}
    all_zero = 0
    for start in range(0, len(indices), 128):
        log_psd, metrics, zeros = batch_primitives(eeg[indices[start : start + 128]])
        log_psd_sum += log_psd.sum(axis=0)
        all_zero += int(zeros.sum())
        for name, values in metrics.items():
            chunks.setdefault(name, []).append(values)
    return log_psd_sum / len(indices), {name: np.concatenate(values) for name, values in chunks.items()}, all_zero


def distance(generated, reference):
    g = generated[np.isfinite(generated)]
    r = reference[np.isfinite(reference)]
    return {
        "wasserstein_1": float(wasserstein_distance(g, r)) if len(g) and len(r) else None,
        "finite_counts": {"generated": len(g), "reference": len(r)},
        "undefined_counts": {"generated": generated.size - len(g), "reference": reference.size - len(r)},
    }


def compare(generated, reference, indices):
    g_psd, g_values, g_zeros = primitives(generated, indices)
    r_psd, r_values, r_zeros = primitives(reference, indices)
    g_corr = g_values.pop("off_diagonal_correlations")
    r_corr = r_values.pop("off_diagonal_correlations")
    corr_difference = np.abs(np.nanmean(g_corr, axis=0) - np.nanmean(r_corr, axis=0))
    usable_corr = corr_difference[np.isfinite(corr_difference)]
    return {
        "samples_per_group": len(indices),
        "all_zero_windows": {"generated": g_zeros, "reference": r_zeros},
        "psd_mean_log10_mae_by_channel_0_5_to_75_hz": np.abs(g_psd - r_psd).mean(axis=-1).tolist(),
        "distribution_distances_by_channel": {
            name: [distance(values[:, ch], r_values[name][:, ch]) for ch in range(16)]
            for name, values in g_values.items()
        },
        "off_diagonal_correlation_mean_mae": float(usable_corr.mean()) if len(usable_corr) else None,
        "off_diagonal_correlation_distribution": distance(g_corr, r_corr),
        "valid_mean_correlation_edges": len(usable_corr),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.archive) as archive:
        generated = archive["generated"]
        reference = archive["ground_truth"]
        labels = archive["labels"]
    if generated.shape != reference.shape or generated.shape != (len(labels), 16, 1000):
        raise ValueError("Expected matched N×16×1000 TUEV arrays")
    if not np.isfinite(generated).all() or not np.isfinite(reference).all():
        raise ValueError("Evaluation archive has nonfinite signal values")
    result = {
        "signal_units": "microvolts",
        "per_class": {
            str(label): compare(generated, reference, np.flatnonzero(labels == label))
            for label in np.unique(labels)
        },
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

