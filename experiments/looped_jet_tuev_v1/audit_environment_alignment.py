"""Audit explicit released environment requirements against preserved R0 runtime."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

import torch

root=Path(sys.argv[1])
repo=root/"JET"
output=root/"evidence/environment-alignment-v1"
output.mkdir(mode=0o700)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=("torch","numpy","scipy","tensorboard","mne","tqdm","einops","scikit-learn","matplotlib")
installed={name:{"version":metadata.version(name),"requires_python":metadata.metadata(name).get("Requires-Python")} for name in names}
requirements=(repo/"requirements.txt").read_text().splitlines()
source_files=subprocess.check_output(["git","-C",str(repo),"ls-files"],text=True).splitlines()
environment_files=[p for p in source_files if any(s in Path(p).name.lower() for s in ("requirement","environment","pyproject","setup.cfg","setup.py","lockfile","pipfile","conda"))]
jobs=[]
for line in subprocess.check_output(["ps","-u",str(__import__("os").getuid()),"-o","pid=,comm=,args="],text=True).splitlines():
    parts=line.split(None,2)
    if len(parts)==3 and parts[1].startswith("python") and any(
        word.endswith(("/prepare_r1.py","/verify_k1.py","/run_screen.py")) for word in parts[2].split()):
        jobs.append({"pid":int(parts[0]),"command":parts[2]})
report={
 "r1_live_python_jobs":jobs,
 "r1_state":"K1 engineering complete. No screen arms generated. Partial screen_metrics.py and screen_probes.py preserved locally; runner never implemented/deployed.",
 "upstream_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
 "tracked_status":subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],text=True),
 "source_sha256":{"README.md":sha(repo/"README.md"),"requirements.txt":sha(repo/"requirements.txt")},
 "tracked_environment_files":environment_files,
 "installed":{"python":sys.version.split()[0],"packages":installed,"torch_cuda_runtime":torch.version.cuda},
 "explicit_release_requirements":{"python":"3.10 (README, patch unpinned)","torch_minimum":"2.0","torch_distribution_index":"https://download.pytorch.org/whl/cu128","other_packages_version_pins":None,"requirements_lines":requirements},
 "alignment":[
   {"setting":"Python minor","prescribed":"3.10","actual":sys.version.split()[0],"status":"mismatch"},
   {"setting":"Torch CUDA distribution","prescribed":"cu128 index","actual":installed["torch"]["version"],"status":"mismatch"},
   {"setting":"Torch minimum","prescribed":">=2.0","actual":installed["torch"]["version"],"status":"satisfied"},
   {"setting":"Other eight direct dependencies","prescribed":"package presence only; unpinned","actual":"all installed","status":"presence_satisfied_author_versions_unknown"},
   {"setting":"Paper runtime hardware","prescribed":"B200 for reported efficiency","actual":"RTX A6000","status":"different_hardware"},
 ],
 "python310_incompatible_current_packages":[name for name,record in installed.items() if record["requires_python"]==">=3.11"],
 "observed_gpu":subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip(),
 "paper_sources":{"paper":"https://arxiv.org/html/2605.21280v1#A2","software_versions":"No Python/PyTorch/CUDA or direct-package version prescription found in paper main text/Appendix B.2. README supplies Python3.10/cu128.","hardware":"Appendix B.3 says single NVIDIA B200"},
 "driver_constraint":{"source":"https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html","interpretation":"Driver560.35.03 is within CUDA12.x minor-compatibility range>=525. CUDA12.8 PTX/new-driver-feature requirements can still fail. No cu128 smoke observed yet; do not infer mandatory driver upgrade or compatibility from cu124 success."},
 "proposed_isolated_target":{"prefix":str(root/"conda-readme-align-py310-cu128"),
     "constraints":["CPython3.10","Torch>=2.0 distributed from officialcu128 wheel index","all direct requirements with Python3.10-compatible versions"],
     "release_author_exact_lock_known":False,
     "creation_status":"not_created","review_status":"concrete resolved pins and runtime smoke require review before new full training/evaluation",
     "candidate_resolution_commands":["conda create --prefix PREFIX python=3.10 pip","PREFIX/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128","PREFIX/bin/python -m pip install -r requirements.txt","PREFIX/bin/python -m pip freeze"],
     "caveat":"Resolver choice is a new reproducible environment satisfying released constraints, not evidence of author package versions. Existing numpy/MNE/sklearn/matplotlib builds require Python>=3.11 and cannot all be retained."},
 "no_new_jobs":"No full training, evaluation, sensitivity run, installation or global driver/package modification performed.",
 "auditor_sha256":sha(Path(__file__)),
}
assert not jobs
(output/"environment-alignment.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"jobs":jobs,"environment_files":environment_files,"python310_incompatible":report["python310_incompatible_current_packages"],"alignment":report["alignment"]},indent=2))
