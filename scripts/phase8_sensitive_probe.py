"""Strict offline scan; reviewed fingerprints are independent of mutable baseline."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console


class ScanSummary(BaseModel):
    """No secret values or unrestricted detector stdout are returned."""

    scan_exit: int
    scanned_files: int
    candidate_identities: int
    unexplained: int
    baseline_unchanged: bool


def identities(data: dict) -> set[tuple[str, str, str]]:
    """Normalize detector identities without examining/outputting candidate values."""
    return {
        (name.replace("\\", "/"), row["type"], row["hashed_secret"])
        for name, rows in data["results"].items()
        for row in rows
    }


def reviewed_identities(root: Path, head: dict) -> set[tuple[str, str, str]]:
    """Accept only identities already recorded in the committed baseline."""
    del root
    return identities(head)


def scan(root: Path, reports: list[Path], diagnostics: Path | None = None) -> ScanSummary:
    """Temporary HEAD baseline plus explicit Git files; no private env traversal."""
    baseline = root / ".secrets.baseline"
    before = hashlib.sha256(baseline.read_bytes()).digest()
    head = subprocess.check_output(["git", "show", "HEAD:.secrets.baseline"], cwd=root)
    files = (
        subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    files = [name for name in files if name and (root / name).is_file()]
    for report in reports:
        resolved = report.resolve()
        # Explicit sanitized JSON reports only; arbitrary env paths are rejected.
        if resolved.parent != (root / "evals/reports").resolve() or resolved.suffix != ".json":
            raise ValueError("synthetic_report_path_required")
        if not resolved.is_file():
            raise ValueError("synthetic_report_missing")
        files.append(resolved.relative_to(root).as_posix())
    with tempfile.TemporaryDirectory(prefix="phase8-sensitive-") as temporary:
        copy = Path(temporary) / "baseline.json"
        copy.write_bytes(head)
        process = subprocess.run(
            ["uv", "run", "--frozen", "detect-secrets", "scan", "--baseline", str(copy), "--no-verify", *files],
            cwd=root,
            capture_output=True,
            timeout=600,
            check=False,
        )
        result = json.loads(copy.read_text(encoding="utf-8"))
        current = identities(result)
        if diagnostics:
            diagnostics.parent.mkdir(parents=True, exist_ok=True)
            # Detector baseline contains hashes/line numbers, not raw secrets.
            diagnostics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return ScanSummary(
        scan_exit=process.returncode,
        scanned_files=len(files),
        candidate_identities=len(current),
        unexplained=len(current - reviewed_identities(root, json.loads(head))),
        baseline_unchanged=before == hashlib.sha256(baseline.read_bytes()).digest(),
    )


def main() -> int:
    """Fail on unexplained candidates, tool failure or any baseline mutation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = scan(root, args.report, args.diagnostics)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    Console().print(result.model_dump())
    return 0 if result.scan_exit == 0 and result.unexplained == 0 and result.baseline_unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
