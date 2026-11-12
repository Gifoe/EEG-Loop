"""Version the engineering input manifest after explicit pinned-HEAD guard review."""
import json
from pathlib import Path
import subprocess
import sys

from prepare_r1 import sha

root = Path(sys.argv[1])
repo = root / "JET"
actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert actual == "07f9e6491796f4f2c717d6259b1a2e24afce6a77"
subprocess.run(["git", "-C", str(repo), "diff", "--exit-code"], check=True)
subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--exit-code"], check=True)
folder = root / "evidence/r1-v1"
previous = folder / "engineering-config-v2.json"
config = json.loads(previous.read_text())
assert sha(root / "baseline-seed0/checkpoint-last.pth") == config["checkpoint_sha256"]
assert sha(root / "evidence/reference-index/val-seed0.npz") == config["validation_reference_index_sha256"]
assert sha(root / "evidence/calibration-b-v1/calibration_config.json") == config["calibration_config_sha256"]
for name, digest in config["fixture_files_sha256"].items():
    assert sha(folder / "fixtures" / name) == digest
config["initial_engineering_config_sha256"] = sha(folder / "engineering-config.json")
config["previous_engineering_config_sha256"] = sha(previous)
config["previous_source_sha256"] = config["source_sha256"]
config["source_sha256"] = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
config["actual_verified_upstream_commit"] = actual
config["guard_amendment"] = "Added explicit pinned Git HEAD and staged/unstaged cleanliness guards after fixture preparation. Original fixtures/selection/noise preserved byte-identically and reverified before gate."
path = folder / "engineering-config-v3.json"
assert not path.exists()
path.write_text(json.dumps(config, indent=2) + "\n")
path.chmod(0o400)
print(json.dumps({"config_sha256": sha(path), "support": config["support"], "source_sha256": config["source_sha256"]}, indent=2))

