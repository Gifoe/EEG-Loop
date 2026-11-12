#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$1" && pwd)"
data="$root/tuev-official"
evidence="$root/evidence"
export PATH="$root/conda-r0/bin:$PATH"
export PYTHONUNBUFFERED=1
cd "$root/JET"
until [[ -s "$evidence/conda-r0.freeze.txt" ]] && grep -q '^Done!$' "$evidence/preprocessing.log"; do
    if grep -Eq 'Traceback|^ERROR:' "$evidence/preprocessing.log" "$evidence/conda-dependencies.log"; then
        echo "Preprocessing or dependency setup failed"
        exit 1
    fi
    sleep 30
done
scripts=experiments/looped_jet_tuev_v1
raw="$HOME/data/TUH-EEG/tuh_eeg_events/v2.0.1/edf"
python "$scripts/audit_tuev.py" --raw "$raw" --processed "$data" --output "$evidence/processed-manifest.json"
python "$scripts/audit_duplicate_windows.py" "$raw" "$data" > "$evidence/duplicate-windows.json"
python - "$evidence/processed-manifest.json" <<'PY'
import json
import sys
from pathlib import Path
audit = json.loads(Path(sys.argv[1]).read_text())
assert not any(audit["processed"]["overlap"].values())
assert all(split["samples"] and not split["invalid"] for split in audit["processed"]["splits"].values())
assert sum(s["samples"] for s in audit["processed"]["splits"].values()) == sum(
    s["events"]["all"] for s in audit["raw"]["splits"].values()
)
PY
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 2 > "$evidence/smoke-gpu.csv" &
gpu_monitor=$!
trap 'kill "$gpu_monitor" 2>/dev/null || true' EXIT
/usr/bin/time -v python train_eeg.py --dataset tuev --datasets_dir "$data" \
    --output_dir "$root/smoke" --model JiT-B/16 --num_eeg_channels 16 \
    --target_length 1000 --eeg_patch_size 200 --batch_size 256 --epochs 1 \
    --blr 5e-5 --loss_type mix --loss_weight_stat 1.0 --loss_weight_tv 0.1 --loss_weight_corr 0.1
