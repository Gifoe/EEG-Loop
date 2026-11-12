"""Materialize the reviewed float32 storage view; run only after lead release."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import uuid

import numpy as np

from experiments.looped_jet_tuev_v1.paper_aligned_v1.data_view import convert_sample, loader_array, waveform_sha


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(root):
    original=root/"tuev-official"
    target=root/"tuev-paper-aligned-v1-f32"
    evidence=root/"evidence/paper-aligned-v1-data-view"
    frozen_path=root/"evidence/paper-aligned-v1-materializer-freeze/manifest.json"
    frozen=json.loads(frozen_path.read_text())
    assert frozen["protocol_sha256"]=="a05b1ac74bfa490ff5cd822e430bcfe2582983d585d34a72ee46df31089d3ed3"
    assert sha(Path(__file__).parent.parent/"PAPER_ALIGNED_STORAGE_AMENDMENT.md")==frozen["storage_amendment_sha256"]
    for name,digest in frozen["source_sha256"].items():
        assert sha(Path(__file__).parent/name)==digest
    assert original.resolve()!=target.resolve() and not target.is_symlink()
    assert original.resolve().parent==root.resolve() and target.resolve().parent==root.resolve()
    evidence.mkdir(mode=0o700,exist_ok=True)
    assert not (evidence/"view.done").exists()
    target.mkdir(mode=0o700,exist_ok=True)
    expected_indexes={"train":"15965df6f48e05c80a3792b31ef1c8da27b6949b62a008f45614bbe741845125",
                      "val":"a1c100f89a1712749974609ba0fe2dc1b0a1553ec15e3dbbac5ec4b28e353b0e",
                      "test":"8fc319346fcfc97b6964aef6e7ee2490961a16842c67ee73baedf0a28607b884"}
    for split,digest in expected_indexes.items():
        assert sha(root/f"evidence/reference-index/{split}-seed0.npz")==digest
    assert sha(root/"evidence/calibration-b-v1/calibration_config.json")=="ade2110650e655ec1b4e6417a2eec0db35131bd9485c320bf97043436b1500da"
    assert sha(root/"evidence/all-edf-payload-v1/heldout-component-map.private.json")=="3c163e22183211050d271e1f09ba92e1040dfd9c4187011a9f29f133bbf7960e"
    metadata={}
    index_hashes={}
    for split in ("train","val","test"):
        index=root/f"evidence/reference-index/{split}-seed0.npz"
        index_hashes[split]=sha(index)
        with np.load(index,allow_pickle=False) as data:
            metadata[split]={str(name):(int(label),str(digest),str(directory))
                             for name,label,digest,directory in zip(data["processed_files"],data["labels"],
                                 data["ground_truth_sha256"],data["subject_directories"],strict=True)}
    calibration=root/"evidence/calibration-b-v1/calibration_config.json"
    groups=json.loads(calibration.read_text())["components_by_directory_id"]
    source_components={directory:f"trainval-{i}" for i,group in enumerate(groups) for directory in group}
    test_map=root/"evidence/all-edf-payload-v1/heldout-component-map.private.json"
    test_components=json.loads(test_map.read_text())
    assert isinstance(test_components,dict) and len(test_components)==80 and all(isinstance(v,str) for v in test_components.values())
    assert len(groups)==274 and len(source_components)==290
    ledger=evidence/f"mapping-{uuid.uuid4().hex}.private.jsonl"
    summary={"splits":{},"source_index_sha256":index_hashes,
             "component_mapping_sha256":{"trainval":sha(calibration),"test":sha(test_map)},
             "source":str(original),"view":str(target),"normalization":"Unchanged source NumPy float32 /100.0 then Torch.float32",
             "materializer_freeze_sha256":sha(frozen_path),
             "source_code_sha256":sha(Path(__file__)),"real_data_view":True}
    loader_hashes={split:set() for split in ("train","val","test")}
    with ledger.open("x") as stream:
        for split,expected in (("train",68445),("val",15487),("test",29421)):
            source=original/split
            dest=target/split
            assert not source.is_symlink() and not dest.is_symlink()
            dest.mkdir(mode=0o700,exist_ok=True)
            files=sorted(source.glob("*.pkl"))
            assert len(files)==expected and {p.name for p in files}==metadata[split].keys()
            classes=Counter()
            raw_sq=loader_sq=0.0
            raw_max=loader_max=0.0
            changed_raw=changed_loader=storage_numeric_changed=0
            for path in files:
                assert not path.is_symlink() and path.resolve().is_relative_to(original.resolve())
                with path.open("rb") as source_stream:
                    sample=pickle.load(source_stream)
                label,digest,directory=metadata[split][path.name]
                before=np.asarray(sample["signal"]).reshape(16,5,200)/100.0
                before=before.astype(np.float32)
                assert waveform_sha(before)==digest and int(sample["label"][0])-1==label
                view=convert_sample(sample)
                assert np.isfinite(view["signal"]).all(),"Float32 storage overflow"
                after=loader_array(view)
                assert np.isfinite(after).all(),"Nonfinite normalized view"
                content=pickle.dumps(view,protocol=4)
                content_hash=hashlib.sha256(content).hexdigest()
                final=dest/path.name
                if final.exists():
                    assert not final.is_symlink() and sha(final)==content_hash
                else:
                    temp=dest/f".{path.name}.{uuid.uuid4().hex}.tmp"
                    with temp.open("xb") as view_stream:
                        view_stream.write(content)
                        view_stream.flush()
                        os.fsync(view_stream.fileno())
                    os.replace(temp,final)
                assert sha(final)==content_hash,"Written or reused view file hash mismatch"
                with final.open("rb") as verify_stream:
                    stored=pickle.load(verify_stream)
                assert set(stored)==set(view)
                assert stored["signal"].dtype==np.float32 and stored["signal"].shape==(16,1000)
                assert np.array_equal(stored["signal"],view["signal"])
                for key in sample:
                    if key!="signal":
                        assert np.array_equal(stored[key],sample[key]),f"View metadata mismatch: {key}"
                assert np.array_equal(loader_array(stored),after)
                final.chmod(0o400)
                raw_delta=view["signal"].astype(np.float64)-sample["signal"].astype(np.float64)
                load_delta=after.astype(np.float64)-before.astype(np.float64)
                raw_sq+=float(np.square(raw_delta).sum())
                loader_sq+=float(np.square(load_delta).sum())
                raw_max=max(raw_max,float(np.abs(raw_delta).max()))
                loader_max=max(loader_max,float(np.abs(load_delta).max()))
                source_raw_hash=waveform_sha(sample["signal"])
                view_raw_hash=waveform_sha(view["signal"])
                after_hash=waveform_sha(after)
                storage_numeric_changed+=bool(np.any(raw_delta!=0))
                loader_hashes[split].add(after_hash)
                changed_raw+=source_raw_hash!=view_raw_hash
                changed_loader+=digest!=after_hash
                classes[label]+=1
                row={"split":split,"source_name":path.name,"view_name":path.name,
                     "source_file_sha256":sha(path),"view_file_sha256":content_hash,
                     "source_raw_sha256":source_raw_hash,"view_raw_sha256":view_raw_hash,
                     "source_loader_sha256":digest,"view_loader_sha256":after_hash,"label":label,
                     "directory_id":directory,"component":(test_components[directory] if split=="test" else source_components[directory]),
                     "raw_max_abs_change_uv":float(np.abs(raw_delta).max()),
                     "loader_max_abs_change":float(np.abs(load_delta).max()),"division_dtype":str(after.dtype)}
                stream.write(json.dumps(row)+chr(10))
            assert {p.name for p in dest.glob("*.pkl")}==metadata[split].keys()
            summary["splits"][split]={"rows":expected,"class_counts":dict(classes),
                "raw_max_abs_change_uv":raw_max,"raw_rms_change_uv":float(np.sqrt(raw_sq/(expected*16000))),
                "loader_max_abs_change":loader_max,"loader_rms_change":float(np.sqrt(loader_sq/(expected*16000))),
                "storage_raw_hash_changed_rows":changed_raw,"storage_numeric_changed_rows":storage_numeric_changed,
                "loader_hash_changed_rows":changed_loader,"one_to_one_mapping":True}
            stream.flush()
            os.fsync(stream.fileno())
            print(f"Verified float32 view {split}: {expected} rows",flush=True)
    summary["cross_split_view_loader_hash_overlaps"]={f"{left}/{right}":len(loader_hashes[left]&loader_hashes[right])
        for left,right in (("train","val"),("train","test"),("val","test"))}
    summary["mapping_ledger_sha256"]=sha(ledger)
    summary["mapping_ledger"]=ledger.name
    (evidence/"view-manifest.json").write_text(json.dumps(summary,indent=2)+chr(10))
    assert not any(summary["cross_split_view_loader_hash_overlaps"].values()),"New view loader-hash leakage detected"
    (evidence/"view.done").touch()
    ledger.chmod(0o400)
    (evidence/"view-manifest.json").chmod(0o400)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    materialize(parser.parse_args().root)



