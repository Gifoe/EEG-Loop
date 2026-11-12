#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$(cd "$1" && pwd)"
evidence="$root/evidence"
integrity="$evidence/evaluation-integrity"
verifier_pid="$(cat "$evidence/evaluation-integrity.pid")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
while [[ ! -f "$integrity/test-seed0.verified" ]]; do
    state="$(ps -p "$verifier_pid" -o stat=)" || state=
    if [[ -z "$state" || "$state" == Z* ]]; then
        echo "Integrity coordinator exited before verifying held-out archive" >&2
        exit 1
    fi
    sleep 60
done
(cd "$evidence/reference-index"; grep '  test-seed0.npz$' sha256.txt | sha256sum -c -)
(cd "$evidence/calibration-b-v1"; sha256sum -c output.sha256)
cd "$root/JET"
nice -n 15 ionice -c 3 "$root/conda-r0/bin/python" experiments/looped_jet_tuev_v1/score_protocol_b.py score \
    "$evidence/reference-index/test-seed0.npz" "$evidence/calibration-b-v1" "$evidence/protocol-b-subsets-v1" \
    "$root/eval-heldout-test-seed0/eval_batch.npz" "$evidence/protocol-b-scores-seed0"
