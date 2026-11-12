#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$(cd "$1" && pwd)"
prefix="$root/conda-paper-aligned-v1"
evidence="$root/evidence/paper-aligned-v1-environment"
requirements="$root/JET/experiments/looped_jet_tuev_v1/paper_aligned_v1/requirements-py310.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$HOME/miniconda3/bin/conda" create --yes --json --prefix "$prefix" python=3.10.19 pip > "$evidence/conda-create.json"
python="$prefix/bin/python"
"$python" -c 'import sys; assert sys.version_info[:3] == (3,10,19); print(sys.version)' > "$evidence/python-build.txt"
"$python" -m pip install --dry-run --report "$evidence/torch-dryrun-report.json" 'torch==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
"$python" -m pip install --report "$evidence/torch-install-report.json" 'torch==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
"$python" -m pip install --dry-run --report "$evidence/direct-dryrun-report.json" -r "$requirements"
"$python" -m pip install --report "$evidence/direct-install-report.json" -r "$requirements"
"$python" -m pip check > "$evidence/pip-check.txt"
"$python" -m pip freeze --all > "$evidence/pip-freeze.txt"
"$python" -m pip inspect --local > "$evidence/pip-inspect.json"
"$HOME/miniconda3/bin/conda" list --prefix "$prefix" --explicit > "$evidence/conda-explicit-lock.txt"
df -B1 "$root" > "$evidence/disk-after-install.txt"
touch "$evidence/install.done"
