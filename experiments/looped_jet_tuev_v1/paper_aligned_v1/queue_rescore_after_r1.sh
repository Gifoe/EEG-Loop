#!/usr/bin/env bash
set -euo pipefail
task_root=/home/luansiyu/EEG-Loop
cd "$task_root/JET"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf '%s\n' 'c3c5c4c54b7dbd6a7e189ea5a8e79ba39c3d9780b28fbd23079dbeef1d6a1237  experiments/looped_jet_tuev_v1/paper_aligned_v1/rescore_own.py' | sha256sum -c -
while true; do
    state="$(ps -p 210092 -o stat= 2>/dev/null)" || state=
    case "$state" in ''|Z*) break ;; esac
    if [[ -f "$task_root/evidence/r1-v1/screen-results-v1/screen.finished" ]]; then break; fi
    sleep 15
done
date -u +%FT%TZ
exec nice -n 15 ionice -c 3 "$task_root/conda-paper-aligned-v1/bin/python" -u -m experiments.looped_jet_tuev_v1.paper_aligned_v1.rescore_own run "$task_root"
