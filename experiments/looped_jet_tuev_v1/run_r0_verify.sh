#!/usr/bin/env bash
set -euo pipefail
umask 077

root="$(cd "$1" && pwd)"
eval_pid="$2"
index_dir="$root/evidence/reference-index"
verify_dir="$root/evidence/evaluation-integrity"
mkdir -p "$verify_dir"
(cd "$index_dir" && sha256sum -c sha256.txt)
cd "$root/JET"

for split in train test; do
    case "$split" in
        train) output="$root/eval-released-train-seed0" ;;
        test) output="$root/eval-heldout-test-seed0" ;;
    esac
    while [[ ! -f "$output/evaluation.done" ]]; do
        state="$(ps -p "$eval_pid" -o stat=)" || state=
        if [[ -z "$state" || "$state" == Z* ]]; then
            echo "Evaluation coordinator exited before finishing $split" >&2
            exit 1
        fi
        sleep 60
    done
    nice -n 10 "$root/conda-r0/bin/python" -m experiments.looped_jet_tuev_v1.reference_index verify \
        --index "$index_dir/$split-seed0.npz" \
        --archive "$output/eval_batch.npz" \
        --metrics "$output/metrics.json" > "$verify_dir/$split-seed0.log"
    sha256sum "$output/metrics.json" > "$verify_dir/$split-seed0.metrics.sha256"
    touch "$verify_dir/$split-seed0.verified"
    echo "Verified $split evaluation"
done
