#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$1" && pwd)"
eval_pid="$2"
evidence="$root/evidence"
held_out="$root/eval-heldout-test-seed0"
export PATH="$root/conda-r0/bin:$PATH"
export PYTHONUNBUFFERED=1
cd "$root/JET"
until [[ -f "$held_out/evaluation.done" ]]; do
    if ! kill -0 "$eval_pid" 2>/dev/null; then
        echo "Evaluation exited without held-out completion sentinel" >&2
        exit 1
    fi
    sleep 60
done
scripts=experiments/looped_jet_tuev_v1
/usr/bin/time -v python -m experiments.looped_jet_tuev_v1.profile_generation \
    "$root/baseline-seed0/checkpoint-last.pth" "$held_out/generation-profile.json"
/usr/bin/time -v python "$scripts/signal_diagnostics.py" "$held_out/eval_batch.npz" "$held_out/signal-diagnostics.json"
touch "$held_out/diagnostics.done"
