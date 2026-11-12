"""New named full-spectrum 1024-dimensional metric; released metric untouched."""
import torch
from data.metrics import _prepare_fid_features, compute_ts_fid_streaming

METRIC_ID = "tsfid-paper-v1-1024"


def features(eeg):
    eeg = torch.as_tensor(eeg)
    if eeg.ndim != 3 or eeg.shape[1:] != (16, 1000):
        raise ValueError("Expected [N,16,1000]")
    if not torch.isfinite(eeg).all():
        raise ValueError("Nonfinite metric input")
    return _prepare_fid_features(eeg, feature_bins=256, spatial_bins=4, max_frequency_ratio=None)


def score(generated, reference, chunk_size=64):
    return compute_ts_fid_streaming(generated, reference, chunk_size=chunk_size,
                                    feature_bins=256, spatial_bins=4, max_frequency_ratio=None)
