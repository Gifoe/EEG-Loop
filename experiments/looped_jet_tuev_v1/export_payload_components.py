"""Reconcile frozen calibration groups and export held-out signal-copy groups."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def components(rows):
    parents={row["directory_id"]:row["directory_id"] for row in rows}
    def root(key):
        while key!=parents[key]:
            key=parents[key]
        return key
    grouped=defaultdict(list)
    for row in rows:
        grouped[row["payload_sha256"],row["signal_metadata_sha256"]].append(row["directory_id"])
    for keys in grouped.values():
        for key in keys[1:]:
            parents[root(key)]=root(keys[0])
    result=defaultdict(set)
    for key in parents:
        result[root(key)].add(key)
    return sorted((sorted(group) for group in result.values()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(evidence,calibration):
    ledger_path=evidence/"signal-metadata-ledger.private.json"
    rows=json.loads(ledger_path.read_text())
    config_path=calibration/"calibration_config.json"
    config=json.loads(config_path.read_text())
    original={frozenset(group) for group in config["components_by_directory_id"]}
    actual=components([row for row in rows if row["split"]!="test"])
    heldout=components([row for row in rows if row["split"]=="test"])
    report={"payload_metadata_components":len(actual),"frozen_calibration_components":len(original),
            "exact_component_membership_match":{frozenset(group) for group in actual}==original,
            "calibration_config_sha256":sha(config_path),"payload_metadata_ledger_sha256":sha(ledger_path)}
    (evidence/"calibration-component-reconciliation.json").write_text(json.dumps(report,indent=2)+"\n")
    mapping={directory:f"component-{i:03d}" for i,group in enumerate(heldout) for directory in group}
    path=evidence/"heldout-component-map.private.json"
    path.write_text(json.dumps(mapping,indent=2,sort_keys=True)+"\n")
    summary={"directory_ids":len(mapping),"components":len(heldout),
             "component_size_histogram":dict(Counter(map(len,heldout))),
             "map_sha256":sha(path),"payload_metadata_ledger_sha256":sha(ledger_path),
             "auditor_sha256":sha(Path(__file__)),
             "interpretation":"Directory IDs connected by identical stored sample payload and interpretation metadata. This groups copied signals and does not identify people."}
    (evidence/"heldout-component-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"calibration":report,"heldout":summary},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence",type=Path)
    parser.add_argument("calibration",type=Path)
    main(**vars(parser.parse_args()))
