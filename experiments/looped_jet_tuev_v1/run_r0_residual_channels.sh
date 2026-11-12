#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$(cd "$1" && pwd)"
evidence="$root/evidence"
verify_pid="$(cat "$evidence/evaluation-integrity.pid")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
until [[ -f "$evidence/evaluation-integrity/test-seed0.verified" ]]; do
    state="$(ps -p "$verify_pid" -o stat=)" || state=
    if [[ -z "$state" || "$state" == Z* ]]; then
        echo "Held-out integrity worker exited without verified archive" >&2
        exit 1
    fi
    sleep 60
done
(cd "$evidence/reference-index"; grep '  test-seed0.npz$' sha256.txt | sha256sum -c -)
cd "$root/JET"
nice -n 15 ionice -c 3 "$root/conda-r0/bin/python" experiments/looped_jet_tuev_v1/residual_channel_diagnostics.py \
    "$root/eval-heldout-test-seed0/eval_batch.npz" "$evidence/reference-index/test-seed0.npz" "$evidence/residual-channel-seed0"
