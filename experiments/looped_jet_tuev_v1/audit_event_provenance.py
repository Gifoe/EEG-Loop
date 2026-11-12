"""Attribute repeated frozen waveform hashes to original TUEV REC provenance."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recording(path):
    with path.open("rb") as stream:
        header = stream.read(256)
    n = round(int(header[236:244]) * float(header[244:252]) * 200)
    if n <= 0:
        raise ValueError("Cannot derive resampled recording length")
    annotations = np.loadtxt(path.with_suffix(".rec"), delimiter=",", ndmin=2)
    return {"digest": sha(path), "rec_digest": sha(path.with_suffix(".rec")),
            "annotations": annotations, "times": np.arange(n) / 200, "n": n}


def summarize(groups):
    multiplicities = [len(rows) for rows in groups.values()]
    repeated = [value for value in multiplicities if value > 1]
    result = {
        "rows": sum(multiplicities), "unique_waveforms": len(groups),
        "repeated_waveform_groups": len(repeated),
        "multiplicity_quantiles_0_25_50_75_95_99_100": np.percentile(multiplicities, [0,25,50,75,95,99,100]).tolist(),
        "repeated_only_multiplicity_quantiles": np.percentile(repeated, [0,25,50,75,95,99,100]).tolist() if repeated else [],
        "redundant_rows": sum(multiplicities)-len(groups),
        "same_recording_same_extraction_redundancy": 0,
        "identical_edf_copy_same_extraction_redundancy": 0,
        "other_coordinate_or_recording_redundancy": 0,
        "multilabel_waveform_groups": 0, "rows_in_multilabel_groups": 0,
        "same_extraction_groups_with_multiple_offending_channels": 0,
        "same_extraction_groups_with_multiple_labels": 0,
        "same_extraction_groups_with_repeated_identical_channel_and_label": 0,
        "cross_path_waveform_groups": 0,
    }
    labels = defaultdict(set)
    label_rows = Counter()
    combinations = Counter()
    for digest, rows in groups.items():
        by_coord = defaultdict(list)
        content_coords = set()
        for row in rows:
            by_coord[(row["source"], row["begin"], row["end"], row["length"])].append(row)
            content_coords.add((row["edf_sha"], row["begin"], row["end"], row["length"]))
            labels[row["label"]].add(digest)
            label_rows[row["label"]] += 1
        result["same_recording_same_extraction_redundancy"] += len(rows)-len(by_coord)
        result["identical_edf_copy_same_extraction_redundancy"] += len(by_coord)-len(content_coords)
        result["other_coordinate_or_recording_redundancy"] += len(content_coords)-1
        result["cross_path_waveform_groups"] += len({row["source"] for row in rows}) > 1
        distinct_labels = sorted({row["label"] for row in rows})
        if len(distinct_labels)>1:
            result["multilabel_waveform_groups"] += 1
            result["rows_in_multilabel_groups"] += len(rows)
            combinations[",".join(map(str,distinct_labels))] += 1
        for same in by_coord.values():
            result["same_extraction_groups_with_multiple_offending_channels"] += len({r["channel"] for r in same})>1
            result["same_extraction_groups_with_multiple_labels"] += len({r["label"] for r in same})>1
            result["same_extraction_groups_with_repeated_identical_channel_and_label"] += (
                len({(r["channel"],r["label"]) for r in same}) < len(same))
    result["per_class"] = {str(label): {"rows": label_rows[label], "unique_waveforms": len(labels[label])} for label in range(6)}
    result["multilabel_combinations"] = dict(combinations)
    assert result["redundant_rows"] == sum(result[key] for key in (
        "same_recording_same_extraction_redundancy", "identical_edf_copy_same_extraction_redundancy",
        "other_coordinate_or_recording_redundancy"))
    return result


def main(raw, processed, indices, output):
    output.mkdir(mode=0o700)
    recordings = {}
    extraction_hashes = defaultdict(set)
    report = {"splits": {}, "index_sha256": {}, "source_sha256": {}, "mapping_mismatches": [],
              "spot_checked_pickle_metadata": 0, "left_wrapped_rows": 0, "right_wrapped_rows": 0}
    for source in ("data/preprocess_tuev.py","data/tuev_dataset.py"):
        report["source_sha256"][source] = sha(Path(__file__).resolve().parents[2]/source)
    report["auditor_sha256"] = sha(Path(__file__))
    with (output/"event-provenance.private.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=("split","file","source","edf_sha","rec_sha","event_index","channel","start_s","end_s","label","begin","end","length","resampled_n","waveform_sha"))
        writer.writeheader()
        for split in ("train","val","test"):
            index_path = indices/f"{split}-seed0.npz"
            report["index_sha256"][split] = sha(index_path)
            groups = defaultdict(list)
            checked = set()
            with np.load(index_path,allow_pickle=False) as data:
                for file,source,label,digest in zip(data["processed_files"],data["source_edfs"],data["labels"],data["ground_truth_sha256"],strict=True):
                    source = ("eval/" if split == "test" else "train/") + str(source)
                    path=raw/source
                    if source not in recordings:
                        recordings[source]=recording(path)
                    rec=recordings[source]
                    stem,event = Path(str(file)).stem.rsplit("-",1)
                    event=int(event)
                    if stem != path.stem or not 0<=event<len(rec["annotations"]):
                        raise ValueError("Filename/event mapping mismatch")
                    channel,start,end,raw_label=rec["annotations"][event]
                    if int(raw_label)-1 != int(label):
                        report["mapping_mismatches"].append("index-label-vs-rec")
                    left=int(np.searchsorted(rec["times"],start))
                    right=int(np.searchsorted(rec["times"],end))
                    if left>=rec["n"] or right>=rec["n"]:
                        raise ValueError("REC boundary exceeds official time array")
                    if right-left+800!=1000:
                        raise ValueError("Unexpected extraction length")
                    report["left_wrapped_rows"] += left<400
                    report["right_wrapped_rows"] += right+400>rec["n"]
                    row={"split":split,"file":str(file),"source":source,"edf_sha":rec["digest"],"rec_sha":rec["rec_digest"],
                         "event_index":event,"channel":int(channel),"start_s":float(start),"end_s":float(end),
                         "label":int(label),"begin":left-400,"end":right+400,"length":1000,
                         "resampled_n":rec["n"],"waveform_sha":str(digest)}
                    writer.writerow(row)
                    groups[str(digest)].append(row)
                    extraction_hashes[(rec["digest"],left-400,right+400,rec["n"])].add(str(digest))
                    # Deterministically verify actual pickle metadata at REC endpoints and midpoint.
                    if event in (0,len(rec["annotations"])//2,len(rec["annotations"])-1) and (source,event) not in checked:
                        with (processed/split/str(file)).open("rb") as sample_stream:
                            sample=pickle.load(sample_stream)
                        if int(sample["label"][0])!=int(raw_label) or int(sample["offending_channel"][0])!=int(channel):
                            report["mapping_mismatches"].append("pickle-metadata-vs-rec")
                        report["spot_checked_pickle_metadata"] += 1
                        checked.add((source,event))
            report["splits"][split]=summarize(groups)
            print(f"Audited {split}: {report['splits'][split]['rows']} rows",flush=True)
    report["identical_edf_extraction_keys_with_different_waveform_hashes"]=sum(len(h)>1 for h in extraction_hashes.values())
    report["coordinate_rule"]="Searchsorted on arange(N)/200 exactly reproduces first times>=start/end. Slice in triple-concatenated signal is [N+start-400:N+end+400]. N derived from EDF record count and duration at 200Hz."
    report["limits"]="All REC labels verified against frozen index; pickle channel+label checked at first/middle/last event per recording. Frozen waveform hashes rely on existing reference-index integrity. No waveform reconstruction or modified preprocessing."
    (output/"aggregate.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"audit.done").touch()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("raw","processed","indices","output"):
        parser.add_argument(arg,type=Path)
    main(**vars(parser.parse_args()))

