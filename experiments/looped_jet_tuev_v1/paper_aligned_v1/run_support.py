"""Shared immutable-input checks for paper-aligned execution drivers."""
import hashlib
import json
from pathlib import Path
import site
import sys

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(root):
    path=root/"evidence/paper-aligned-v1-input-freeze/manifest.json"
    assert sha(path)==path.with_name("manifest.sha256").read_text().split()[0]
    prefix=(root/"conda-paper-aligned-v1").resolve()
    assert Path(sys.executable).resolve().is_relative_to(prefix)
    assert sys.version_info[:2]==(3,10) and site.ENABLE_USER_SITE is False
    assert torch.__version__=="2.7.1+cu128" and np.__version__=="2.2.6"
    record=json.loads(path.read_text())
    assert record["variant"]=="jet-paper-aligned-v1" and record["initialization"]=="Fresh own seed0; no checkpoint warmstart"
    source=root/"JET/experiments/looped_jet_tuev_v1/paper_aligned_v1"
    assert sha(source.parent/"PAPER_ALIGNED_V1_PROTOCOL.md")==record["protocol_sha256"]
    for name,digest in record["source_sha256"].items():
        assert sha(source/name)==digest, f"Frozen source changed: {name}"
    view=root/"evidence/paper-aligned-v1-data-view"
    assert sha(view/"view-manifest.json")==record["data_manifest_sha256"]
    environment=root/"evidence/paper-aligned-v1-environment"
    for name,digest in record["environment_sha256"].items():
        assert sha(environment/name)==digest
    return record,sha(path)


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+chr(10))
