"""Capture auditable upstream and experiment code snapshots without data."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def sha(content):
    return hashlib.sha256(content).hexdigest()


def archive_files(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o600
            archive.addfile(member, io.BytesIO(content))
    with tarfile.open(path) as archive:
        actual = {member.name: sha(archive.extractfile(member).read()) for member in archive}
    expected = {name: sha(content) for name, content in files.items()}
    if actual != expected:
        raise ValueError("Archive verification failed")
    return expected


def capture(repo, output):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", commit], check=True)
    tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"]).decode().split("\0")
    upstream = {name: (repo / name).read_bytes() for name in tracked if name}
    exp = repo / "experiments" / "looped_jet_tuev_v1"
    tooling = {}
    for path in sorted(exp.iterdir()):
        if path.is_file() and path.suffix in (".py", ".sh", ".md", ".csv"):
            tooling[path.relative_to(repo).as_posix()] = path.read_bytes()
    output.mkdir(mode=0o700)
    manifest = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": commit,
        "upstream_files": archive_files(output / "upstream-tracked.tar.gz", upstream),
        "tooling_files": archive_files(output / "experiment-tooling.tar.gz", tooling),
    }
    manifest["archive_sha256"] = {
        path.name: sha(path.read_bytes()) for path in output.glob("*.tar.gz")
    }
    path = output / "source-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.sha256").write_text(sha(path.read_bytes()) + "  source-manifest.json\n")
    print(json.dumps({"upstream_commit": commit, "upstream_files": len(upstream),
                      "tooling_files": len(tooling), "archive_sha256": manifest["archive_sha256"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    capture(**vars(parser.parse_args()))
