"""Read-only verification of the isolated prefix, without inline Python."""
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import site
import sys


root = Path(sys.argv[1])
prefix = root / "conda-paper-aligned-v1"
assert Path(sys.prefix) == prefix
assert sys.version_info[:3] == (3, 10, 19) and not site.ENABLE_USER_SITE
expected = {
    "torch": "2.7.1+cu128", "numpy": "2.2.6", "scipy": "1.15.3",
    "mne": "1.10.2", "scikit-learn": "1.7.2", "matplotlib": "3.10.8",
    "tensorboard": "2.21.0", "tqdm": "4.70.1", "einops": "0.8.2",
}
paths = {}
for name, version in expected.items():
    assert metadata.version(name) == version
    module = importlib.import_module("sklearn" if name == "scikit-learn" else name)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(prefix)
    paths[name] = str(path)
folder = root / "evidence/paper-aligned-v1-environment"
previous = json.loads((folder / "import-check.json").read_text())
assert previous["versions"] == expected and previous["module_paths"] == paths
result = {"passed": True, "versions": expected, "module_paths": paths,
          "user_site_enabled": site.ENABLE_USER_SITE,
          "previous_import_record_matches": True,
          "scope": "Read-only imports and metadata. No CUDA smoke repeated."}
(folder / "verify-environment-file.json").write_text(json.dumps(result, indent=2) + chr(10))
print(json.dumps(result, indent=2))
