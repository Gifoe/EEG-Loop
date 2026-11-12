"""Independently reconcile frozen aggregate R0 quality evidence; no EEG access."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

FAMILIES = {
    "psd": ["psd_mean_log10_mae_by_channel_0_5_to_75_hz"],
    "band_power": [f"band_{x}" for x in ["delta", "theta", "alpha", "beta", "gamma"]],
    "morphology": ["peak_to_peak", "rms", "excess_kurtosis", "p95_abs_first_difference", "centered_zero_crossing_rate"],
    "temporal": ["envelope_slope_uv_per_s", "last_minus_first_block_mean", "last_minus_first_block_rms"],
    "spatial": ["off_diagonal_correlation_mean_mae", "off_diagonal_correlation_distribution"],
}
COUNTS = [567, 4677, 1998, 329, 2204, 19646]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flat(value):
    if isinstance(value, list):
        return [x for child in value for x in flat(child)]
    return [value]


def close(left, right):
    assert math.isfinite(left) and math.isfinite(right)
    assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10), (left, right)


def review(packet):
    hashes = {}
    for line in (packet / "manifest.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        actual = digest(packet / name)
        assert actual == expected, name
        hashes[name] = actual
    load = lambda name: json.loads((packet / name).read_text())
    summary = load("calibrated-summary.json")
    reps = load("calibrated-replicates.json")
    scales = load("median_scales.json")
    assert digest(packet / "median_scales.json") == summary["frozen_plan"]["median_scales_sha256"]
    assert summary["frozen_plan"]["replicates"] == 200
    ratio_checks = 0
    for label in map(str, range(6)):
        row = reps["by_class"][label]
        reported = summary["by_class"][label]
        assert reported["all_zero_generated_windows_in_selected_union"] == 0
        primitive_means = {}
        for name, values in row["native"].items():
            evidence = scales[label][name]
            raw_scale = flat(evidence["scale"])
            assert all(math.isfinite(v) and v >= 0 for v in raw_scale)
            assert all(v == 200 for v in flat(evidence["finite_replicates"]))
            scale = [max(v, 1e-8) for v in raw_scale]
            assert len(values) == len(row["ratios"][name]) == 200
            primitive_means[name] = []
            for values_i, ratios_i in zip(values, row["ratios"][name]):
                a, b = flat(values_i), flat(ratios_i)
                assert len(a) == len(b) == len(scale)
                for x, y, s in zip(a, b, scale):
                    close(x / s, y)
                    ratio_checks += 1
                primitive_means[name].append(mean(b))
            for key, source in [("native_means", values), ("primitive_ratio_means", row["ratios"][name])]:
                actual = [mean(flat(v)[j] for v in source) for j in range(len(scale))]
                for a, b in zip(actual, flat(reported[key][name])):
                    close(a, b)
        for family, keys in FAMILIES.items():
            expected = [mean(primitive_means[k][r] for k in keys) for r in range(200)]
            assert len(row["families"][family]) == 200
            for a, b in zip(expected, row["families"][family]):
                close(a, b)
            close(mean(expected), reported["family_means"][family])
    for family in FAMILIES:
        expected = [mean(reps["by_class"][str(k)]["families"][family][r] for k in range(6)) for r in range(200)]
        for a, b in zip(expected, reps["macro"][family]):
            close(a, b)
        close(mean(expected), summary["macro_family_means"][family])

    native = load("signal-diagnostics.json")
    assert native["signal_units"] == "microvolts"
    native_review = {}
    for label, row in native["per_class"].items():
        n = COUNTS[int(label)]
        assert row["samples_per_group"] == n
        assert row["all_zero_windows"] == {"generated": 0, "reference": 0}
        assert row["valid_mean_correlation_edges"] == 120
        psd = row["psd_mean_log10_mae_by_channel_0_5_to_75_hz"]
        assert len(psd) == 16 and all(math.isfinite(v) and v >= 0 for v in psd)
        distances = {}
        for name, leads in row["distribution_distances_by_channel"].items():
            assert len(leads) == 16
            for lead in leads:
                assert lead["undefined_counts"] == {"generated": 0, "reference": 0}
                assert lead["finite_counts"] == {"generated": n, "reference": n}
                assert math.isfinite(lead["wasserstein_1"]) and lead["wasserstein_1"] >= 0
            distances[name] = mean(v["wasserstein_1"] for v in leads)
        c = row["off_diagonal_correlation_distribution"]
        assert c["undefined_counts"] == {"generated": 0, "reference": 0}
        assert c["finite_counts"] == {"generated": n * 120, "reference": n * 120}
        assert math.isfinite(row["off_diagonal_correlation_mean_mae"])
        native_review[label] = {"psd_log10_mae_mean": mean(psd), "psd_log10_mae_min": min(psd),
                               "psd_log10_mae_max": max(psd), "largest_psd_lead_index": psd.index(max(psd)),
                               "native_mean_w1_across_leads": distances,
                               "raw_correlation_mean_mae": row["off_diagonal_correlation_mean_mae"]}

    residual = load("residual-summary.json")
    residual_review = {}
    for label, row in residual["by_class"].items():
        assert row["correlation_defined"] and row["covariance_defined"]
        for domain in ["reference", "generated"]:
            d = row[domain]
            assert "class_mean_waveform_uv" not in d
            assert d["mean_estimation_windows"] == COUNTS[int(label)]
            assert d["nearconstant_channel_window_counts"] == [0] * 16
            assert d["nonfinite_raw_input_values"] == 0
            for field in ["valid_correlation_window_counts_by_pair", "valid_covariance_window_counts_by_pair"]:
                assert all(v == COUNTS[int(label)] for v in flat(d[field]))
            assert all(v == 0 for v in flat(d["nonfinite_covariance_window_counts_by_pair"]))
        r, g = row["reference"], row["generated"]
        for field, score in [("mean_residual_covariance_uv2", "covariance_mae_uv2"), ("mean_residual_correlation", "correlation_mae")]:
            actual = mean(abs(r[field][i][j] - g[field][i][j]) for i in range(16) for j in range(i + 1, 16))
            close(actual, row[score])
        residual_review[label] = {k: row[k] for k in ["covariance_mae_uv2", "correlation_mae"]}
        for domain in ["reference", "generated"]:
            matrix = row[domain]["mean_residual_correlation"]
            residual_review[label][domain + "_mean_abs_residual_corr"] = mean(abs(matrix[i][j]) for i in range(16) for j in range(i + 1, 16))
    for field in ["covariance_mae_uv2", "correlation_mae"]:
        close(mean(row[field] for row in residual["by_class"].values()), residual["macro_" + field])
    profile = load("generation-profile.json")
    assert profile["nfe"] == 99 and profile["batch_size"] == 64
    assert profile["warmup_batches"] == 2 and profile["measured_batches"] == 5
    close(median(profile["batch_latency_s"]), profile["median_batch_latency_s"])
    assert profile["matmul_flops_per_sample_per_nfe"] * 99 == profile["matmul_flops_per_generated_sample"] == 1490120146944
    return {"passed": True, "scope": "Aggregate artifact, scalar arithmetic, counts and quality reconciliation; no EEG accessed or new generation.",
            "packet_hashes": hashes, "ratio_scalar_checks": ratio_checks,
            "calibrated_macro_families": summary["macro_family_means"], "native_by_class": native_review,
            "residual_by_class": residual_review, "profile": profile,
            "sampling_repetitions_are_independent_training_replicates": False,
            "review_script_sha256": digest(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = review(args.packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "ratio_scalar_checks": result["ratio_scalar_checks"]}))
