"""Score the full fixed R1 validation screen using frozen R0 primitives/scales."""
import json
from pathlib import Path
import sys

# Frozen R0 helpers were authored as sibling-importing standalone scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.metrics import compute_ts_fid_streaming
from experiments.looped_jet_tuev_v1 import audit_ts_fid
from experiments.looped_jet_tuev_v1.calibrate_signal import discrepancy
from experiments.looped_jet_tuev_v1.score_protocol_b import FAMILIES, checked_scales, extract
from experiments.looped_jet_tuev_v1.signal_diagnostics import compare as native_compare


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def score(reference, generated, labels, calibration, output):
    if reference.shape != generated.shape or reference.shape != (2020, 16, 1000):
        raise ValueError("Fixed screen shape mismatch")
    if not np.isfinite(generated).all() or np.any((generated == 0).all(axis=(1, 2))):
        raise ValueError("Nonfinite or all-zero generated window")
    scales = checked_scales(calibration)
    metric = {"num_samples": 2020, "label_histogram": {}, "ts_fid": {"feature_bins": 256, "per_class": {}}}
    g = torch.from_numpy(generated)
    r = torch.from_numpy(reference)
    metric["ts_fid"]["overall"] = compute_ts_fid_streaming(g, r, chunk_size=32, feature_bins=256)
    raw, weighted = {}, {}
    for label in range(6):
        rows = np.flatnonzero(labels == label)
        metric["label_histogram"][str(label)] = len(rows)
        metric["ts_fid"]["per_class"][str(label)] = compute_ts_fid_streaming(g[rows], r[rows], chunk_size=32, feature_bins=256)
        real_features, fake_features = extract(reference, rows), extract(generated, rows)
        if any(not np.isfinite(values).all() for values in (*real_features.values(), *fake_features.values())):
            raise ValueError(f"Undefined required primitive in class {label}")
        combined = {key: np.concatenate((values, fake_features[key])) for key, values in real_features.items()}
        n = len(rows)
        native = discrepancy(combined, np.arange(n), np.arange(n, 2 * n))
        ratios = {key: np.asarray(value) / scales[str(label)][key] for key, value in native.items()}
        if any(not np.isfinite(value).all() for value in ratios.values()):
            raise ValueError(f"Undefined required discrepancy in class {label}")
        families = {name: float(np.mean([np.asarray(ratios[key]).mean() for key in keys]))
                    for name, keys in FAMILIES.items()}
        weighted[str(label)] = {
            "native": {key: np.asarray(value).tolist() for key, value in native.items()},
            "ratios": {key: value.tolist() for key, value in ratios.items()}, "families": families,
        }
        raw[str(label)] = native_compare(generated, reference, rows)
    write_json(output / "metrics.json", metric)
    features_g, features_r = audit_ts_fid.features(generated, 256), audit_ts_fid.features(reference, 256)
    checks = {"overall": audit_ts_fid.compare(features_g, features_r, metric["ts_fid"]["overall"])}
    for label in range(6):
        mask = labels == label
        checks[str(label)] = audit_ts_fid.compare(features_g[mask], features_r[mask], metric["ts_fid"]["per_class"][str(label)])
    audit = {"passed": all(row["passed"] for row in checks.values()), "checks": checks}
    write_json(output / "numeric-audit.json", audit)
    if not audit["passed"]:
        raise ValueError("Independent TS-FID audit failed")
    result = {
        "by_class": weighted,
        "macro_families": {name: float(np.mean([row["families"][name] for row in weighted.values()])) for name in FAMILIES},
        "scope": "Exploratory validation-weighted scores using all fixed screen rows once; class counts differ from calibration. Not count-matched Protocol B.",
    }
    write_json(output / "native-diagnostics.json", raw)
    write_json(output / "weighted-diagnostics.json", result)
    return metric, result


def continuation(results):
    decisions = {}
    baseline = results.get("baseline")
    for arm, control in (("loop2", "flow67"), ("loop3", "flow83")):
        if not all(results.get(key, {}).get("status") == "completed" for key in ("baseline", arm, control)):
            decisions[arm] = {"passes": False, "reason": "Required arm failed or incomplete"}
            continue
        candidate, extra = results[arm], results[control]
        fid_pass = candidate["ts_fid"] < baseline["ts_fid"] and candidate["ts_fid"] < extra["ts_fid"]
        macro = {key: candidate["macro_families"][key] <= 1.10 * baseline["macro_families"][key]
                 for key in FAMILIES}
        per_class = {str(label): {key: candidate["class_families"][str(label)][key] <= 1.25 * baseline["class_families"][str(label)][key]
                                  for key in FAMILIES} for label in range(6)}
        decisions[arm] = {"passes": fid_pass and all(macro.values()) and all(all(row.values()) for row in per_class.values()),
                          "fid_strictly_below_both": fid_pass, "macro_filters": macro, "class_filters": per_class}
    passing = [name for name, row in decisions.items() if row["passes"]]
    selected = min(passing, key=lambda key: (results[key]["ts_fid"], int(key[-1]))) if passing else None
    return {"loops": decisions, "provisional_selected_arm": selected, "further_runs_authorized": False}

