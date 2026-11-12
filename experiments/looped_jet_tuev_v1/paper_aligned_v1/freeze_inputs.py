"""Freeze reviewed environment, data view and drivers before execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PROTOCOL="a05b1ac74bfa490ff5cd822e430bcfe2582983d585d34a72ee46df31089d3ed3"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(root):
    repo=root/"JET"
    source=repo/"experiments/looped_jet_tuev_v1"
    assert sha(source/"PAPER_ALIGNED_V1_PROTOCOL.md")==PROTOCOL
    amendment=source/"PAPER_ALIGNED_STORAGE_AMENDMENT.md"
    assert sha(amendment)=="07d739c0e910dbdec8e10d12ca038d712ec7fd05419307f1db936736c76fdf5a"
    assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()=="07f9e6491796f4f2c717d6259b1a2e24afce6a77"
    assert not subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],text=True).strip()
    environment=root/"evidence/paper-aligned-v1-environment"
    assert (environment/"engineering-smokes.done").exists()
    assert (root/"evidence/paper-aligned-v1-newenv-synthetic/checks.passed").exists()
    smoke_path=root/"evidence/paper-aligned-v1-cuda-smoke/smoke-report.json"
    smoke=json.loads(smoke_path.read_text())
    assert smoke["passed"] and smoke["batch_size"]==256 and smoke["torch_version"]=="2.7.1+cu128"
    assert smoke["training"]["all_loss_gradient_ema_checks_finite"]
    view=root/"evidence/paper-aligned-v1-data-view"
    assert (view/"view.done").exists()
    data=json.loads((view/"view-manifest.json").read_text())
    assert [data["splits"][split]["rows"] for split in ("train","val","test")]==[68445,15487,29421]
    assert sha(view/data["mapping_ledger"])==data["mapping_ledger_sha256"]
    output=root/"evidence/paper-aligned-v1-input-freeze"
    output.mkdir(mode=0o700)
    code=source/"paper_aligned_v1"
    synthetic_path=root/"evidence/paper-aligned-v1-newenv-synthetic/synthetic-report.json"
    synthetic=json.loads(synthetic_path.read_text())
    assert synthetic["passed"] and sha(code/"model.py")==synthetic["source_sha256"]["model.py"]
    parameter_count=synthetic["checks"]["joint_sequence"]["trainable_parameters"]
    checkpoint_upper=2_650_000_000
    archive_upper=int((68445+29421)*16*1000*4*1.02)+20_000_000
    required=4*checkpoint_upper+5*1024**3
    free=shutil.disk_usage(root).free
    assert free>=required,"Storage below checkpoint/archives/atomic-write/safety budget"
    manifest={"storage_amendment_sha256":sha(amendment),"variant":"jet-paper-aligned-v1","protocol_sha256":PROTOCOL,
        "source_sha256":{p.name:sha(p) for p in code.iterdir() if p.is_file() and p.suffix in (".py",".sh",".txt")},
        "environment_sha256":{name:sha(environment/name) for name in ("pip-freeze.txt","wheel-lock.json","conda-explicit-lock.txt","import-check.json")},
        "synthetic_report_sha256":sha(synthetic_path),
        "smoke_report_sha256":sha(smoke_path),"data_manifest_sha256":sha(view/"view-manifest.json"),
        "data_mapping_sha256":data["mapping_ledger_sha256"],"data_dir":data["view"],
        "output_dir":str(root/"paper-aligned-v1-seed0"),"seed":0,"epochs":200,"batch_size":256,
        "lr":5e-5,"warmup_epochs":5,"lr_schedule":"constant","betas":[.9,.95],"weight_decay":0,
        "ema_decays":[.9999,.9996],"label_drop":.1,"weighted_sampler":True,"drop_last":True,
        "num_workers":4,"dtype_training":"bfloat16","dtype_inference":"float32",
        "backend":smoke["backend"],"parameter_count":parameter_count,
        "checkpoint_retention":{"full_snapshots":[0,100,199],"last_save_every_epochs":1,
              "atomic_last":True,"final_snapshot_is_hardlink_to_last":True},
        "storage":{"free_bytes":free,"checkpoint_upper_each":checkpoint_upper,"checkpoint_peak_copies":4,
              "archive_schema":"jet-generated-only-v1","evaluation_checkpoint_copies":3,"archives_upper_bytes":archive_upper,"safety_bytes":5*1024**3,"required_bytes":required},
        "initialization":"Fresh own seed0; no checkpoint warmstart",
        "primary_evaluation":{"split":"test","rows":29421,"seed":0,"batch_size":64,"ema":"ema1","heun_steps":50,"nfe":99,"cfg":1},
        "secondary_evaluation":{"split":"train","rows":68445},
        "metrics":["tsfid-source-1000","tsfid-paper-v1-1024"]}
    path=output/"manifest.json"
    path.write_text(json.dumps(manifest,indent=2)+chr(10))
    (output/"manifest.sha256").write_text(sha(path)+"  manifest.json"+chr(10))
    for p in output.iterdir():p.chmod(0o400)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    freeze(parser.parse_args().root)


