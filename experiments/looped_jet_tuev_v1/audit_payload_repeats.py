"""Inspect raw sample payload equality for cross-path frozen waveform groups."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


def main(raw, evidence):
    groups=defaultdict(list)
    with (evidence/"event-provenance.private.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["split"]=="test":
                groups[row["waveform_sha"]].append(row)
    cross=[rows for rows in groups.values() if len({r["source"] for r in rows})>1]
    payload={}
    for rows in cross:
        for row in rows:
            source=row["source"]
            if source not in payload:
                with (raw/source).open("rb") as stream:
                    header=stream.read(256)
                    stream.seek(int(header[184:192]))
                    payload[source]=hashlib.file_digest(stream,"sha256").hexdigest()
    result={
        "cross_path_test_groups":len(cross),
        "identical_entire_edf_data_payload_groups":sum(
            len({payload[r["source"]] for r in rows})==1 for rows in cross),
        "identical_extraction_coordinates_groups":sum(
            len({(r["begin"],r["end"],r["resampled_n"]) for r in rows})==1 for rows in cross),
        "same_directory_id_groups":sum(
            len({Path(r["source"]).parent for r in rows})==1 for rows in cross),
        "distinct_source_files_checked":len(payload),
        "interpretation":"EDF data payload excludes variable-length EDF header. Existing normalized waveform hashes establish extracted-signal equality. This does not establish patient identity.",
        "auditor_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (evidence/"test-cross-path-followup.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw",type=Path)
    parser.add_argument("evidence",type=Path)
    main(**vars(parser.parse_args()))
