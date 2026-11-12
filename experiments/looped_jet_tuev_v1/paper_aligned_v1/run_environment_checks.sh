#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$(cd "$1" && pwd)"
evidence="$root/evidence/paper-aligned-v1-environment"
installer="$(cat "$evidence/install-attempt2.pid")"
until [[ -f "$evidence/install.done" ]]; do
    state="$(ps -p "$installer" -o stat=)" || state=
    if [[ -z "$state" || "$state" == Z* ]]; then
        echo "Environment installer exited without completion sentinel" >&2
        exit 1
    fi
    sleep 20
done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
python="$root/conda-paper-aligned-v1/bin/python"
"$python" -I - "$root" <<'PY'
import hashlib,importlib,importlib.metadata as m,json,pathlib,site,sys
root=pathlib.Path(sys.argv[1])
prefix=root/"conda-paper-aligned-v1"
assert sys.version_info[:3]==(3,10,19) and not site.ENABLE_USER_SITE
versions={"torch":"2.7.1+cu128","numpy":"2.2.6","scipy":"1.15.3","mne":"1.10.2","scikit-learn":"1.7.2","matplotlib":"3.10.8","tensorboard":"2.21.0","tqdm":"4.70.1","einops":"0.8.2"}
paths={}
for name,version in versions.items():
    assert m.version(name)==version,(name,m.version(name))
    module=importlib.import_module("sklearn" if name=="scikit-learn" else name)
    paths[name]=str(pathlib.Path(module.__file__).resolve())
    assert pathlib.Path(paths[name]).is_relative_to(prefix)
folder=root/"evidence/paper-aligned-v1-environment"
(folder/"import-check.json").write_text(json.dumps({"passed":True,"python":sys.version,"versions":versions,"module_paths":paths,"user_site_enabled":site.ENABLE_USER_SITE},indent=2)+chr(10))
wheels=[]
for filename in ("torch-install-report.json","direct-install-report.json"):
    data=json.loads((folder/filename).read_text())
    for item in data["install"]:
        wheels.append({"name":item["metadata"]["name"],"version":item["metadata"]["version"],"download_info":item["download_info"]})
(folder/"wheel-lock.json").write_text(json.dumps(wheels,indent=2)+chr(10))
PY
cd "$root/JET"
CUDA_VISIBLE_DEVICES= "$python" -m experiments.looped_jet_tuev_v1.paper_aligned_v1.synthetic_checks "$root/evidence/paper-aligned-v1-newenv-synthetic"
"$python" -m experiments.looped_jet_tuev_v1.paper_aligned_v1.cuda_smoke "$root/evidence/paper-aligned-v1-cuda-smoke"
touch "$evidence/engineering-smokes.done"
