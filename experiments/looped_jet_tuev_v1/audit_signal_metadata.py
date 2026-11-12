"""Compare interpretation metadata for every equal EDF sample payload."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def metadata(path):
    with path.open("rb") as stream:
        header=stream.read(256)
        n=int(header[252:256])
        length=int(header[184:192])
        signal_header=stream.read(length-256)
    fields={}
    offset=0
    for name,width,convert in (
        ("labels",16,str),("transducers",80,str),("physical_dimensions",8,str),
        ("physical_min",8,float),("physical_max",8,float),("digital_min",8,int),("digital_max",8,int),
        ("prefiltering",80,str),("samples_per_record",8,int),("reserved",32,str)):
        values=[signal_header[offset+i*width:offset+(i+1)*width].decode("latin1").strip() for i in range(n)]
        if name not in ("transducers","prefiltering","reserved"):
            fields[name]=[convert(v) for v in values]
        offset+=n*width
    fields.update(num_channels=n,num_records=int(header[236:244]),record_duration=float(header[244:252]))
    return fields


def main(raw,evidence):
    ledger=json.loads((evidence/"payload-ledger.private.json").read_text())
    groups=defaultdict(list)
    for row in ledger:
        fields=metadata(raw/row["source"])
        row["signal_metadata_sha256"]=hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()
        groups[row["payload_sha256"]].append(row)
    repeated=[rows for rows in groups.values() if len(rows)>1]
    cross={}
    for left,right in (("train","val"),("train","test"),("val","test")):
        payload_groups=[rows for rows in repeated if {left,right}<={r["split"] for r in rows}]
        compatible=0
        for rows in payload_groups:
            l={r["signal_metadata_sha256"] for r in rows if r["split"]==left}
            r={r["signal_metadata_sha256"] for r in rows if r["split"]==right}
            compatible+=bool(l&r)
        cross[f"{left}/{right}"]={"shared_payload_groups":len(payload_groups),
                                  "shared_payload_with_matching_signal_metadata_groups":compatible}
    report={"repeated_payload_groups":len(repeated),
            "groups_with_uniform_signal_metadata":sum(len({r["signal_metadata_sha256"] for r in rows})==1 for rows in repeated),
            "groups_with_differing_signal_metadata":sum(len({r["signal_metadata_sha256"] for r in rows})>1 for rows in repeated),
            "cross_split":cross,"metadata_fields":["channel count/order/labels","physical dimensions","physical and digital min/max",
                                               "samples per record","record duration","record count"],
            "excluded":"Patient codes, recording identities and dates are excluded from interpretation metadata.",
            "auditor_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (evidence/"signal-metadata-ledger.private.json").write_text(json.dumps(ledger,indent=2)+"\n")
    (evidence/"signal-metadata-aggregate.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw",type=Path)
    parser.add_argument("evidence",type=Path)
    main(**vars(parser.parse_args()))
