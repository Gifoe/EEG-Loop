"""Retain designated completed upstream checkpoint writes without changing training."""
import argparse
import os
from pathlib import Path
import shutil
import time
import zipfile

import torch


def main(root):
    checkpoint = root / "checkpoint-last.pth"
    snapshots = root / "snapshots"
    snapshots.mkdir(exist_ok=True)
    wanted = {0, 25, 50, 100, 150, 199}
    observed = None
    while True:
        if checkpoint.is_file():
            stat = checkpoint.stat()
            version = (stat.st_size, stat.st_mtime_ns)
            if version != observed and zipfile.is_zipfile(checkpoint):
                try:
                    epoch = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["epoch"]
                except (OSError, RuntimeError, EOFError):
                    time.sleep(2)
                    continue
                if epoch in wanted:
                    dest = snapshots / f"checkpoint-epoch-{epoch:03d}.pth"
                    partial = dest.with_suffix(".partial")
                    shutil.copyfile(checkpoint, partial)
                    final_stat = checkpoint.stat()
                    if (final_stat.st_size, final_stat.st_mtime_ns) != version:
                        partial.unlink()
                        continue
                    os.replace(partial, dest)
                    print(f"Preserved epoch {epoch} ({stat.st_size} bytes)", flush=True)
                    if epoch == 199:
                        return
                observed = version
        if (root / "training.failed").exists():
            raise SystemExit("Training failed")
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
