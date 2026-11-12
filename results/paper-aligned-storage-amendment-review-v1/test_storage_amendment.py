"""Private temporary synthetic and real-row sidecar integrity regression checks."""
import json
from pathlib import Path
import pickle
import tempfile
import numpy as np

from experiments.looped_jet_tuev_v1.paper_aligned_v1.reference_sidecar import create_sidecar,reconstruct,sha,wave,safe
from experiments.looped_jet_tuev_v1.paper_aligned_v1.storage_budget import required,SAFETY,GENERATED_UPPER

def check(root):
    outcomes=[]
    for domain in ("synthetic","private"):
        with tempfile.TemporaryDirectory(dir=root/"evidence") as temp:
            base=Path(temp);original=base/"source";view=base/"view"
            (original/"val").mkdir(parents=True);(view/"val").mkdir(parents=True)
            labels=[];names=[];hashes=[];rows=[]
            real=sorted((root/"tuev-official/val").glob("*.pkl"))[:3]
            for i in range(3):
                if domain=="private":
                    with real[i].open("rb") as stream:item=pickle.load(stream)
                else:item={"signal":np.random.default_rng(i).normal(size=(16,1000)),"label":np.array([i+1]),"offending_channel":np.array([0])}
                name=f"fixture-{i}.pkl";name_path=original/"val"/name
                name_path.write_bytes(pickle.dumps(item))
                converted=dict(item);converted["signal"]=item["signal"].astype(np.float32)
                path=view/"val"/name;path.write_bytes(pickle.dumps(converted))
                label=int(item["label"][0])-1;digest=wave(converted["signal"]/100.0)
                labels.append(label);names.append(name);hashes.append(digest)
                rows.append({"split":"val","view_name":name,"source_name":name,"label":label,"source_file_sha256":sha(name_path),"view_file_sha256":sha(path),"view_loader_sha256":digest})
            mapping=base/"mapping.jsonl";mapping.write_text("".join(json.dumps(r)+chr(10) for r in rows))
            manifest=base/"manifest.json";manifest.write_text(json.dumps({"source":str(original)}))
            index={"processed_files":names,"labels":labels,"ground_truth_sha256":hashes}
            sidecar=base/"sidecar.json";expected=create_sidecar(view,"val",index,manifest,mapping,{},sidecar)
            actual=reconstruct(view,sidecar,expected,labels)
            assert actual.dtype==np.float32 and actual.shape==(3,16,1000)
            assert [wave(row) for row in actual]==hashes
            for action in ("label","sidecar","view","source","path"):
                try:
                    if action=="label":reconstruct(view,sidecar,expected,[labels[0]+1,*labels[1:]])
                    elif action=="sidecar":reconstruct(view,sidecar,"0"*64,labels)
                    elif action=="path":safe(view,"../escape.pkl")
                    else:
                        path=(view if action=="view" else original)/"val"/names[0]
                        old=path.read_bytes();path.write_bytes(old+b"tamper")
                        try:reconstruct(view,sidecar,expected,labels)
                        finally:path.write_bytes(old)
                except ValueError:pass
                else:raise AssertionError(f"{action} tamper accepted")
            outcomes.append({"domain":domain,"rows":3,"all_row_hashes_verified":True,"tamper_cases_rejected":5})
    c=2650000000
    assert required("training",c,2*c)==2*c+SAFETY
    assert required("evaluation",c,3*c)==GENERATED_UPPER+SAFETY
    assert required("materialization",c,view_upper=8000000000)==8000000000+SAFETY
    return {"passed":True,"fixtures":outcomes,"phase_budget_checks":True}

if __name__=="__main__":
    import sys
    root=Path(sys.argv[1])
    result=check(root)
    out=root/"evidence/paper-aligned-storage-tests-v1.json"
    out.write_text(json.dumps(result,indent=2)+chr(10))
    print(json.dumps(result,indent=2))
