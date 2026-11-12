#!/usr/bin/env bash
set -euo pipefail
interval="$1"
while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
        awk -v timestamp="$(date -u +%FT%TZ)" '{print timestamp "," $0; fflush()}'
    sleep "$interval"
done
