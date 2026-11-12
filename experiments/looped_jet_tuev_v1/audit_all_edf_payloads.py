"""Audit EDF sample-payload hashes across every official TUEV recording."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def root(parents, key):
    while parents[key]!=key:
        key=parents[key]
    return key


def main(raw, indices, output):
    output.mkdir(mode=0o700)
    membership={}
    index_hashes={}
    for split in ("train","val","test"):
        path=indices/f"{split}-seed0.npz"
        index_hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as index:
            for source in set(index["source_edfs"]):
                source=("eval/" if split=="test" else "train/")+str(source)
                if source in membership and membership[source]!=split:
                    raise ValueError("Source recording crosses processed partitions")
                membership[source]=split
    ledger=[]
    groups=defaultdict(list)
    for path in sorted(raw.rglob("*.edf")):
        source=path.relative_to(raw).as_posix()
        with path.open("rb") as stream:
            header=stream.read(256)
            length=int(header[184:192])
            channels=int(header[252:256])
            if length != 256+256*channels:
                raise ValueError("Unexpected EDF header layout")
            stream.seek(length)
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        row={"source":source,"split":membership.get(source),"directory_id":path.parent.name,
             "payload_sha256":digest,"header_bytes":length,"payload_bytes":path.stat().st_size-length}
        if row["split"] is None:
            raise ValueError("Raw EDF missing from processed provenance")
        ledger.append(row)
        groups[digest].append(row)
    splits={}
    for split in ("train","val","test"):
        selected=[r for r in ledger if r["split"]==split]
        hashes={r["payload_sha256"] for r in selected}
        parents={r["directory_id"]:r["directory_id"] for r in selected}
        duplicate_groups=[]
        for digest in hashes:
            rows=[r for r in groups[digest] if r["split"]==split]
            if len(rows)>1:
                duplicate_groups.append(rows)
                anchor=rows[0]["directory_id"]
                for row in rows[1:]:
                    parents[root(parents,row["directory_id"])]=root(parents,anchor)
        component_sizes=Counter(root(parents,k) for k in parents)
        linked=[size for size in component_sizes.values() if size>1]
        splits[split]={"recordings":len(selected),"unique_payloads":len(hashes),
                       "duplicate_payload_groups":len(duplicate_groups),
                       "records_in_duplicate_payload_groups":sum(map(len,duplicate_groups)),
                       "directory_ids":len(parents),"payload_linked_components":len(component_sizes),
                       "components_with_multiple_directory_ids":len(linked),
                       "linked_component_size_histogram":dict(Counter(linked))}
    cross={}
    for left,right in (("train","val"),("train","test"),("val","test")):
        matches=[rows for rows in groups.values() if {left,right}<= {r["split"] for r in rows}]
        cross[f"{left}/{right}"]={"shared_payload_hashes":len(matches),
                                   "left_recordings":sum(sum(r["split"]==left for r in rows) for rows in matches),
                                   "right_recordings":sum(sum(r["split"]==right for r in rows) for rows in matches)}
    summary={"total_recordings":len(ledger),"total_unique_payload_hashes":len(groups),"splits":splits,
             "cross_split":cross,"index_sha256":index_hashes,"auditor_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "method":"Hash bytes from exact EDF header-length offset to EOF. These are stored sample payloads, before calibration/filtering/resampling. Equality does not establish patient identity.",
             "cache_note":"The earlier18 selected-recording payload hashes were not persisted individually; all518 were hashed directly for this complete audit."}
    (output/"payload-ledger.private.json").write_text(json.dumps(ledger,indent=2)+"\n")
    (output/"aggregate.json").write_text(json.dumps(summary,indent=2)+"\n")
    (output/"audit.done").touch()
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("raw","indices","output"):
        parser.add_argument(arg,type=Path)
    main(**vars(parser.parse_args()))
