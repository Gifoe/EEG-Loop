"""Generated-only archive reference sidecars with row-by-row reconstruction."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wave(x):
    return hashlib.sha256(np.ascontiguousarray(x,dtype="<f4").tobytes()).hexdigest()


def safe(root,name):
    relative=Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe reference path")
    path=root/relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Reference escapes immutable view")
    return path


def create_sidecar(data_root,split,index,mapping_manifest,mapping_ledger,provenance,output):
    data_root=Path(data_root)
    mapping={}
    for line in Path(mapping_ledger).read_text().splitlines():
        row=json.loads(line)
        if row["split"]==split:
            mapping[row["view_name"]]=row
    sorted_files=sorted(p.name for p in (data_root/split).glob("*.pkl"))
    dataset_index={name:i for i,name in enumerate(sorted_files)}
    rows=[]
    for position,(name,label,digest) in enumerate(zip(index["processed_files"],index["labels"],index["ground_truth_sha256"],strict=True)):
        record=mapping[str(name)]
        path=safe(data_root,split+"/"+str(name))
        if sha(path)!=record["view_file_sha256"] or int(label)!=record["label"] or str(digest)!=record["view_loader_sha256"]:
            raise ValueError("Frozen reference mapping mismatch")
        rows.append({"position":position,"dataset_index":dataset_index[str(name)],"path":split+"/"+str(name),"label":int(label),
                     "source_file_sha256":record["source_file_sha256"],"view_file_sha256":record["view_file_sha256"],
                     "loader_sha256":str(digest),"source_name":record["source_name"]})
    result={"schema":"jet-generated-only-v1","split":split,"dtype":"float32","shape":[len(rows),16,1000],
            "normalization":"stored float32 /100.0, then float32","seed":0,
            "selection":"released match_gt class schedule and without-replacement seed0 reference ordering",
            "mapping_manifest":json.loads(Path(mapping_manifest).read_text()),
            "mapping_manifest_sha256":sha(mapping_manifest),"mapping_ledger_sha256":sha(mapping_ledger),
            "provenance":provenance,"rows":rows}
    Path(output).write_text(json.dumps(result,indent=2)+chr(10))
    Path(output).chmod(0o400)
    return sha(output)


def reconstruct(data_root,sidecar,expected_sha,labels):
    if sha(sidecar)!=expected_sha:
        raise ValueError("Sidecar modified")
    record=json.loads(Path(sidecar).read_text())
    if record["schema"]!="jet-generated-only-v1" or len(record["rows"])!=len(labels):
        raise ValueError("Invalid sidecar schema/count")
    sorted_files=sorted(p.name for p in (Path(data_root)/record["split"]).glob("*.pkl"))
    values=[]
    for i,row in enumerate(record["rows"]):
        if row["position"]!=i or row["label"]!=int(labels[i]):
            raise ValueError("Reference order/label mismatch")
        original=safe(Path(record["mapping_manifest"]["source"]),row["path"])
        if sha(original)!=row["source_file_sha256"]:
            raise ValueError("Original source file modified")
        if sorted_files[row["dataset_index"]]!=Path(row["path"]).name:
            raise ValueError("Dataset index changed")
        path=safe(Path(data_root),row["path"])
        if sha(path)!=row["view_file_sha256"]:
            raise ValueError("View file modified")
        with path.open("rb") as stream:
            item=pickle.load(stream)
        signal=item["signal"]
        if signal.dtype!=np.float32 or signal.shape!=(16,1000) or int(item["label"][0])-1!=row["label"]:
            raise ValueError("Reference dtype/shape/label mismatch")
        normalized=(signal/100.0).astype(np.float32)
        if not np.isfinite(normalized).all() or wave(normalized)!=row["loader_sha256"]:
            raise ValueError("Reference waveform modified")
        values.append(normalized)
    return np.stack(values)
