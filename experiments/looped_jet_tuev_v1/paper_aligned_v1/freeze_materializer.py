"""Read-only materializer input/source/storage freeze, before data conversion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(root):
    source=root/"JET/experiments/looped_jet_tuev_v1"
    protocol=source/"PAPER_ALIGNED_V1_PROTOCOL.md"
    assert sha(protocol)=="a05b1ac74bfa490ff5cd822e430bcfe2582983d585d34a72ee46df31089d3ed3"
    amendment=source/"PAPER_ALIGNED_STORAGE_AMENDMENT.md"
    assert sha(amendment)=="07d739c0e910dbdec8e10d12ca038d712ec7fd05419307f1db936736c76fdf5a"
    smoke=root/"evidence/paper-aligned-v1-cuda-smoke/smoke-report.json"
    assert json.loads(smoke.read_text())["passed"]
    indexes={
        "reference-index/train-seed0.npz":"15965df6f48e05c80a3792b31ef1c8da27b6949b62a008f45614bbe741845125",
        "reference-index/val-seed0.npz":"a1c100f89a1712749974609ba0fe2dc1b0a1553ec15e3dbbac5ec4b28e353b0e",
        "reference-index/test-seed0.npz":"8fc319346fcfc97b6964aef6e7ee2490961a16842c67ee73baedf0a28607b884",
        "calibration-b-v1/calibration_config.json":"ade2110650e655ec1b4e6417a2eec0db35131bd9485c320bf97043436b1500da",
        "all-edf-payload-v1/heldout-component-map.private.json":"3c163e22183211050d271e1f09ba92e1040dfd9c4187011a9f29f133bbf7960e",
    }
    for path,digest in indexes.items():assert sha(root/"evidence"/path)==digest
    free=shutil.disk_usage(root).free
    # Measure representative serialized file allocations, including block rounding.
    import os,pickle
    from experiments.looped_jet_tuev_v1.paper_aligned_v1.data_view import convert_sample
    block=os.statvfs(root).f_frsize
    samples=[]
    for split in ("train","val","test"):
        for path in list((root/"tuev-official"/split).glob("*.pkl"))[:16]:
            with path.open("rb") as stream:
                encoded=pickle.dumps(convert_sample(pickle.load(stream)),protocol=4)
            samples.append(((len(encoded)+block-1)//block)*block)
    view_upper=int(max(samples)*113353*1.02)+20_000_000
    budget={"view_upper_bytes":view_upper,"safety_bytes":5*1024**3}
    required=sum(budget.values())
    assert free>=required,"Insufficient materialization headroom"
    output=root/"evidence/paper-aligned-v1-materializer-freeze"
    output.mkdir(mode=0o700)
    record={"storage_amendment_sha256":sha(amendment),"protocol_sha256":sha(protocol),
        "source_sha256":{name:sha(source/"paper_aligned_v1"/name) for name in ("materialize_view.py","data_view.py","freeze_materializer.py")},
        "inputs_sha256":indexes,"smoke_sha256":sha(smoke),"disk_budget":budget,"required_bytes":required,
        "free_bytes":free,"destination":str(root/"tuev-paper-aligned-v1-f32"),
        "scope":"Source/input freeze only. Actual conversion requires lead release."}
    path=output/"manifest.json"
    path.write_text(json.dumps(record,indent=2)+chr(10))
    (output/"manifest.sha256").write_text(sha(path)+"  manifest.json"+chr(10))
    for p in output.iterdir():p.chmod(0o400)
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    freeze(parser.parse_args().root)
