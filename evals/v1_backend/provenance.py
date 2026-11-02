"""Safe code fingerprints without importing offline sandbox composition."""

import hashlib
import subprocess
from pathlib import Path


def code_state() -> dict:
    """Include dirty source bytes, never secret configs or raw Git diffs."""
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    names = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], cwd=root, text=True).splitlines()
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        path = root / name
        if path.is_file() and (
            name in {"pyproject.toml", "uv.lock"}
            or name.startswith(("app/", "tests/", "scripts/", "evals/"))
            and path.suffix in {".py", ".json"}
        ):
            digest.update(name.encode())
            digest.update(path.read_bytes())
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    ).splitlines()
    return {"head": head, "source_hash": digest.hexdigest(), "dirty_paths": [x[3:] for x in dirty]}
