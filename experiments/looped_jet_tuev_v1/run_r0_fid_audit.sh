#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$(cd "$1" && pwd)"
verify_pid="$2"
integrity="$root/evidence/evaluation-integrity"
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2
cd "$root/JET"

failed=0
for split in train test; do
    case "$split" in
        train) output="$root/eval-released-train-seed0" ;;
        test) output="$root/eval-heldout-test-seed0" ;;
    esac
    while [[ ! -f "$integrity/$split-seed0.verified" ]]; do
        state="$(ps -p "$verify_pid" -o stat=)" || state=
        if [[ -z "$state" || "$state" == Z* ]]; then
            echo "Archive verification ended before confirming $split" >&2
            exit 1
        fi
        sleep 60
    done
    if nice -n 10 "$root/conda-r0/bin/python" -m experiments.looped_jet_tuev_v1.audit_ts_fid \
        "$output/eval_batch.npz" "$output/metrics.json" "$integrity/$split-seed0.ts-fid-audit.json"; then
        echo "Independent TS-FID cross-check passed for $split"
    else
        failed=1
        echo "Independent TS-FID cross-check failed for $split" >&2
    fi
done
exit "$failed"
