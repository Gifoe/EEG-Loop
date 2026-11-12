"""Package aggregate R0 quality evidence and exclude residual mean waveforms."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

root=Path(sys.argv[1])
out=root/"evidence/r0-quality-evidence-v1"
out.mkdir(mode=0o700)
sources={
 "signal-diagnostics.json":("eval-heldout-test-seed0/signal-diagnostics.json","d1754e549691c7c6578e6f9efd3d5ab25d01df636bb3e91339168b739674c77b"),
 "generation-profile.json":("eval-heldout-test-seed0/generation-profile.json","c84d53b6db7b5da359c1dda3573460e7ed7cdd9ed07ddd30e56cc6fb5cc555f5"),
 "calibrated-summary.json":("evidence/protocol-b-scores-seed0/summary.json","c491938812641aa953812d16b2ccfaa22a50322f0501b628c30cbdd58214076d"),
 "calibrated-replicates.json":("evidence/protocol-b-scores-seed0/replicates.json","25610ee7285b2f0f5bda183e877d3236d7c714a10c1e163de6e6933169c2f20c"),
 "median_scales.json":("evidence/calibration-b-v1/median_scales.json",None),
}
provenance={}
for target,(source,expected) in sources.items():
    source=root/source
    actual=sha(source)
    if expected and expected!=actual:
        raise ValueError(f"Unexpected source hash for {target}")
    shutil.copyfile(source,out/target)
    provenance[target]={"source":str(source.relative_to(root)),"source_sha256":actual}

residual=root/"evidence/residual-channel-seed0/residual-channel.json"
expected="7bf8116afb328f95d78f171d1353a29de7cc1835835e19590b3b256706d30b1a"
assert sha(residual)==expected
full=json.loads(residual.read_text())
removed=[]
def reduce(value,path=""):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            location=path+"/"+key
            if key=="class_mean_waveform_uv":
                removed.append(location)
            else:
                result[key]=reduce(item,location)
        return result
    if isinstance(value,list):
        return [reduce(v,path+f"/{i}") for i,v in enumerate(value)]
    return value
reduced=reduce(full)
assert len(removed)==12
assert all("class_mean_waveform_uv" not in reduced["by_class"][str(c)][side]
           for c in range(6) for side in ("reference","generated"))
reduced["reduction_provenance"]={
    "full_source_sha256":expected,"rule":"Recursively remove only dictionary entries named class_mean_waveform_uv. Preserve every other field and value.",
    "removed_paths":removed,"reduction_script_sha256":sha(Path(__file__)),
}
(out/"residual-summary.json").write_text(json.dumps(reduced,indent=2,allow_nan=False)+"\n")
provenance["residual-summary.json"]={"source":str(residual.relative_to(root)),"source_sha256":expected,
                                   "removed_mean_waveform_fields":12}
sentinels=[
 "eval-heldout-test-seed0/diagnostics.done","evidence/protocol-b-scores-seed0/scoring.done",
 "evidence/residual-channel-seed0/residual.done","evidence/calibration-b-v1/calibration.done",
 "evidence/evaluation-integrity/test-seed0.verified",
]
logs=["diagnostics-coordinator.log","protocol-b-score-coordinator.log","residual-channel-coordinator.log"]
statuses={}
for source in sentinels:
    path=root/source
    assert path.is_file()
    statuses[source]={"bytes":path.stat().st_size,"sha256":sha(path)}
for name in logs:
    path=root/"evidence"/name
    text=path.read_text()
    assert "Traceback" not in text
    shutil.copyfile(path,out/name)
    provenance[name]={"source":str(path.relative_to(root)),"source_sha256":sha(path)}
shutil.copyfile(Path(__file__),out/"package_quality_evidence.py")
record={"source_provenance":provenance,"successful_sentinels":statuses,
        "packet_scope":"Aggregate quality evidence only. Excludes raw/generated EEG, residual mean waveforms, cached features and sample indices.",
        "files_sha256":{path.name:sha(path) for path in out.iterdir() if path.is_file()}}
(out/"artifact-manifest.json").write_text(json.dumps(record,indent=2)+"\n")
(out/"manifest.sha256").write_text("".join(sha(path)+"  "+path.name+"\n" for path in sorted(out.iterdir()) if path.is_file() and path.name!="manifest.sha256"))
for path in out.iterdir():
    path.chmod(0o400)
for source,_ in sources.values():
    (root/source).chmod(0o400)
residual.chmod(0o400)
print(json.dumps({path.name:sha(path) for path in out.iterdir()},indent=2))
