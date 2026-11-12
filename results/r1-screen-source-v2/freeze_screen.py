"""Freeze the reviewed five-arm source-R0 screen without generating outputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import scipy
import torch

from experiments.looped_jet_tuev_v1.r1.prepare_r1 import sha,tensor_sha

ARMS=[{"name":"baseline","k":1,"steps":50},{"name":"loop2","k":2,"steps":50},
      {"name":"loop3","k":3,"steps":50},{"name":"flow67","k":1,"steps":67},{"name":"flow83","k":1,"steps":83}]


def freeze(root):
    repo=root/"JET"
    source=repo/"experiments/looped_jet_tuev_v1"
    assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()=="07f9e6491796f4f2c717d6259b1a2e24afce6a77"
    assert not subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],text=True)
    assert sha(source/"R1_PROTOCOL.md")=="102f426f91163c7b9fbe359f0eba050ec47819bb6429c43befb1fb522e46d300"
    previous=root/"evidence/r1-v1"
    engineering=previous/"engineering-config-v3.json"
    gate=previous/"k1-equivalence/report.json"
    assert sha(engineering)=="3939c4c9da3ba2ed54c79c01daa18212e8e41d09d607d30d7ea4aa96bd0e186b"
    assert sha(gate)=="47feafdbb01721dbf3f409e156c26e2458866c96d18eea5c31ad21d9f3cf3b47"
    config=json.loads(engineering.read_text())
    report=json.loads(gate.read_text())
    assert report["passed"] and report["state_and_ema_unchanged"]
    assert sha(root/"baseline-seed0/checkpoint-last.pth")==config["checkpoint_sha256"]=="7047396663de22f02af0ae6676b579cb030be4c9effdd66729b093c22473f418"
    for name,digest in config["source_sha256"].items():
        assert sha(source/"r1"/name)==digest
    for name,digest in config["fixture_files_sha256"].items():
        assert sha(previous/"fixtures"/name)==digest
    noise=torch.load(previous/"fixtures/screen-noise.pt",map_location="cpu",weights_only=True)
    assert noise.shape==(2020,16,5,200) and noise.dtype==torch.float32
    assert tensor_sha(noise)==config["screen_noise_tensor_sha256"]
    refs=torch.load(previous/"fixtures/references.pt",map_location="cpu",weights_only=True)
    assert refs["reference"].shape==noise.shape and torch.isfinite(refs["reference"]).all()
    assert torch.bincount(refs["labels"],minlength=6).tolist()==[119,291,512,86,500,512]
    probes=torch.load(previous/"fixtures/probes.pt",map_location="cpu",weights_only=True)
    assert tensor_sha(probes["noise"])==config["probe_noise_tensor_sha256"]
    assert sys.version_info[:3]==(3,11,5) and torch.__version__=="2.6.0+cu124"
    assert np.__version__==config["numpy_version"]
    assert sha(root/"evidence/calibration-b-v1/median_scales.json")=="e0699d6becc15936b84e671b43609ef2d0a706d3ebbf75323cc7f8d075eaa3f8"
    required=3*1024**3
    assert shutil.disk_usage(root).free>=required
    output=previous/"screen-freeze-v2"
    output.mkdir(mode=0o700)
    operators=["audit_ts_fid.py","calibrate_signal.py","signal_diagnostics.py","score_protocol_b.py"]
    names=["run_screen.py","freeze_screen.py","screen_metrics.py","screen_probes.py","recurrent_wrapper.py","prepare_r1.py","verify_k1.py"]
    record={"protocol_sha256":sha(source/"R1_PROTOCOL.md"),"engineering_config_sha256":sha(engineering),
        "k1_report_sha256":sha(gate),"checkpoint_sha256":config["checkpoint_sha256"],"fixture_files_sha256":config["fixture_files_sha256"],
        "source_sha256":{name:sha(source/"r1"/name) for name in names},
        "operator_sha256":{name:sha(source/name) for name in operators},
        "scale_sha256":sha(root/"evidence/calibration-b-v1/median_scales.json"),
        "python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"runtime_prefix":sys.prefix,
        "counts":config["counts"],"support":config["support"],"arms":ARMS,"batch_size":32,
        "profile_noise_rows":list(range(32)),"profile_warmups":2,"profile_measurements":5,
        "dtype":"float32","autocast":False,"matmul_tf32":False,"screen_seed":20261112,
        "solver":"official Denoiser.generate, uniform Heun with final Euler, CFG1,EMA1",
        "required_free_bytes":required,"candidate_gate":"Requires explicit Astra release of this exact config hash"}
    path=output/"screen-config.json"
    path.write_text(json.dumps(record,indent=2)+chr(10))
    for name in names:shutil.copyfile(source/"r1"/name,output/name)
    for name in operators:shutil.copyfile(source/name,output/("operator-"+name))
    (output/"manifest.sha256").write_text("".join(sha(p)+"  "+p.name+chr(10) for p in sorted(output.iterdir()) if p.is_file()))
    for p in output.iterdir():p.chmod(0o400)
    print(json.dumps({"config_sha256":sha(path),"counts":record["counts"],"support":record["support"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    freeze(parser.parse_args().root)


