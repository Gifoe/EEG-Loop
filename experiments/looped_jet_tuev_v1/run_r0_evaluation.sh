#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$1" && pwd)"
train_pid="$2"
evidence="$root/evidence"
train="$root/baseline-seed0"
export PATH="$root/conda-r0/bin:$PATH"
export PYTHONUNBUFFERED=1
cd "$root/JET"
until [[ -f "$train/training.done" ]]; do
    if [[ -f "$train/training.failed" ]] || ! kill -0 "$train_pid" 2>/dev/null; then
        echo "Training exited without completion sentinel" >&2
        exit 1
    fi
    sleep 60
done
sha256sum -c "$evidence/baseline-final.sha256"
test "$(head -1 "$evidence/baseline-final.sha256" | cut -d ' ' -f 1)" = "$(tail -1 "$evidence/baseline-final.sha256" | cut -d ' ' -f 1)"
data="$root/tuev-official"
scripts=experiments/looped_jet_tuev_v1
bash "$scripts/monitor_gpu.sh" 5 > "$evidence/evaluation-gpu-processes.csv" &
process_monitor=$!
trap 'kill "$process_monitor" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
released="$root/eval-released-train-seed0"
held_out="$root/eval-heldout-test-seed0"
mkdir "$released" "$held_out"
/usr/bin/time -v bash scripts/infer_tuev.sh "$data" "$train" "$released" > "$evidence/eval-released-train-seed0.log" 2>&1
test -s "$released/metrics.json" && test -s "$released/eval_batch.npz"
touch "$released/evaluation.done"
/usr/bin/time -v python inference.py --dataset tuev --datasets_dir "$data" \
    --resume "$train" --output_dir "$held_out" --num_images 0 --gen_bsz 64 \
    --eval_split test --eval_label_mode match_gt --seed 0 > "$evidence/eval-heldout-test-seed0.log" 2>&1
test -s "$held_out/metrics.json" && test -s "$held_out/eval_batch.npz"
touch "$held_out/evaluation.done"
