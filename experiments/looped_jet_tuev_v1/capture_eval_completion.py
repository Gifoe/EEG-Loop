"""Capture completed released-source A engineering evidence without loading EEG."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main(root):
    output=root/"evidence/r0-source-a-completion-v1"
    output.mkdir(mode=0o700)
    folder=root/"eval-released-train-seed0"
    evidence=root/"evidence"
    log=evidence/"eval-released-train-seed0.log"
    text=log.read_text()
    assert (folder/"evaluation.done").exists() and "Exit status: 0" in text
    integrity=evidence/"evaluation-integrity"
    assert (integrity/"train-seed0.verified").exists()
    metrics=json.loads((folder/"metrics.json").read_text())
    audit=json.loads((integrity/"train-seed0.ts-fid-audit.json").read_text())
    assert audit["passed"] and all(item["passed"] for item in audit["checks"].values())
    final=json.loads((evidence/"final-checkpoints.audit.json").read_text())
    assert final["status"]=="verified" and final["live_and_final_snapshot_identical"]
    checkpoint=next(row for row in final["checkpoint_records"] if row["epoch"]==199)
    process_memory=[]
    for line in (evidence/"evaluation-gpu-processes.csv").read_text().splitlines():
        parts=line.split(",")
        if len(parts)==3 and parts[1].strip()=="109127":
            process_memory.append(int(parts[2]))
    repo=root/"JET"
    sys.path.insert(0,str(repo))
    from inference import get_args_parser
    args=get_args_parser().parse_args([
        "--dataset","tuev","--datasets_dir",str(root/"tuev-official"),
        "--resume",str(root/"baseline-seed0"),"--output_dir",str(folder),
        "--num_images","0","--gen_bsz","64","--eval_split","train","--eval_label_mode","match_gt",
    ])
    settings=vars(args)
    settings.update(target_length=1000,eeg_patch_num=5,class_num=6,weights_used="model_ema1",
                    nfe_per_sample=99,model_parameter_dtype="float32",autocast=False,
                    feature_dimensions_effective=1000)
    sources=("inference.py","denoiser.py","models/raw_vit.py","models/common.py","data/metrics.py",
             "data/data.py","data/tuev_dataset.py","scripts/infer_tuev.sh",
             "experiments/looped_jet_tuev_v1/run_r0_evaluation.sh",
             "experiments/looped_jet_tuev_v1/reference_index.py",
             "experiments/looped_jet_tuev_v1/audit_ts_fid.py")
    artifacts={"eval_batch.npz":folder/"eval_batch.npz","metrics.json":folder/"metrics.json",
               "evaluation.log":log,"reference-index.npz":evidence/"reference-index/train-seed0.npz",
               "processed-manifest.json":evidence/"processed-manifest.json",
               "environment.freeze.txt":evidence/"conda-r0.freeze.txt",
               "integrity.log":integrity/"train-seed0.log",
               "ts-fid-independent-audit.json":integrity/"train-seed0.ts-fid-audit.json",
               "checkpoint-independent-audit.json":evidence/"final-checkpoints.audit.json"}
    artifact_records={name:{"bytes":path.stat().st_size,"sha256":sha(path)} for name,path in artifacts.items()}
    record={"upstream_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
            "evaluation_python_pid":109127,"exit_status":0,"samples":metrics["num_samples"],
            "metrics":metrics,"settings":settings,"checkpoint":checkpoint,
            "command_wall_hms":re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([^\n]+)",text)[1],
            "peak_host_rss_kib":int(re.search(r"Maximum resident set size \(kbytes\): (\d+)",text)[1]),
            "sampled_process_nvml_peak_mib":max(process_memory),"nvml_samples":len(process_memory),
            "nvml_interval_s":5,"source_sha256":{source:sha(repo/source) for source in sources},
            "artifacts":artifact_records,"independent_numeric_audit":audit,
            "limits":"Command elapsed includes generation, archive compression and metrics; not isolated sampling latency. NVML process peak sampled every5seconds. Paper comparability requires separate protocol adjudication."}
    target=output/"completion.json"
    target.write_text(json.dumps(record,indent=2)+"\n")
    (output/"manifest.sha256").write_text(sha(target)+"  completion.json\n")
    print(json.dumps({key:record[key] for key in ("samples","command_wall_hms","peak_host_rss_kib",
        "sampled_process_nvml_peak_mib","nvml_samples")},indent=2))
    print(json.dumps(artifact_records,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    main(**vars(parser.parse_args()))
