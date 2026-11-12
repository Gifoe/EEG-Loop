#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$1" && pwd)"
evidence="$root/evidence"
output="$root/baseline-seed0"
export PATH="$root/conda-r0/bin:$PATH"
export PYTHONUNBUFFERED=1
cd "$root/JET"
mkdir "$output"
python experiments/looped_jet_tuev_v1/snapshot_r0.py "$output" > "$evidence/baseline-snapshots.log" 2>&1 &
snapshot_pid=$!
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 5 > "$evidence/baseline-gpu.csv" &
gpu_monitor=$!
bash experiments/looped_jet_tuev_v1/monitor_gpu.sh 5 > "$evidence/baseline-gpu-processes.csv" &
process_monitor=$!
trap 'kill "$gpu_monitor" "$process_monitor" "$snapshot_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if /usr/bin/time -v bash scripts/train_tuev.sh "$root/tuev-official" "$output"; then
    deadline=$((SECONDS + 180))
    while kill -0 "$snapshot_pid" 2>/dev/null && (( SECONDS < deadline )); do
        sleep 2
    done
    if kill -0 "$snapshot_pid" 2>/dev/null; then
        echo "Timed out waiting for the final checkpoint snapshot" >&2
        touch "$output/training.failed"
        exit 1
    fi
    wait "$snapshot_pid"
    sha256sum "$output/checkpoint-last.pth" "$output/snapshots/checkpoint-epoch-199.pth" > "$evidence/baseline-final.sha256"
    touch "$output/training.done"
else
    touch "$output/training.failed"
    exit 1
fi
