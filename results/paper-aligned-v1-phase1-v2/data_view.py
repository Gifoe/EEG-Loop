"""Pure float32 storage-view conversion; no real-data materialization in phase 1."""
import hashlib
import numpy as np


def waveform_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def convert_sample(sample):
    signal = np.asarray(sample["signal"])
    if signal.shape != (16, 1000) or not np.isfinite(signal).all():
        raise ValueError("Expected finite 16 by 1000 source signal")
    converted = dict(sample)
    converted["signal"] = signal.astype(np.float32)
    return converted


def loader_array(sample):
    # Match the unchanged released NumPy division before Torch.float conversion.
    normalized = sample["signal"].reshape(16, 5, 200) / 100.0
    if normalized.dtype != np.float32:
        raise ValueError("The float32 view must preserve float32 division in this NumPy environment")
    return normalized.astype(np.float32, copy=False)


def synthetic_mapping(samples):
    mapping = []
    for name, sample in samples:
        view = convert_sample(sample)
        before = np.asarray(sample["signal"]).reshape(16, 5, 200) / 100.0
        after = loader_array(view)
        difference = after.astype(np.float64) - before.astype(np.float32).astype(np.float64)
        mapping.append({
            "source_name": name, "view_name": name, "label": np.asarray(sample["label"]).tolist(),
            "source_dtype": str(sample["signal"].dtype), "view_dtype": str(view["signal"].dtype),
            "division_dtype": str(after.dtype), "source_raw_sha256": waveform_sha(sample["signal"]),
            "view_raw_sha256": waveform_sha(view["signal"]), "source_loader_sha256": waveform_sha(before.astype(np.float32)),
            "view_loader_sha256": waveform_sha(after), "loader_max_abs_difference": float(np.abs(difference).max()),
            "loader_rms_difference": float(np.sqrt(np.mean(difference**2))),
        })
    if len({row["source_name"] for row in mapping}) != len(mapping):
        raise ValueError("Repeated mapping key")
    return mapping
