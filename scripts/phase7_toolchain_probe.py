"""Compare pytest entry points without environment or global path patches."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console


def probe(output: Path) -> bool:
    """Save bounded interpreter provenance and exact collected node sets."""
    root = Path(__file__).resolve().parents[1]
    commands = [
        [str(Path(sys.executable).with_name("pytest.exe" if sys.platform == "win32" else "pytest"))],
        [sys.executable, "-m", "pytest"],
    ]
    results = []
    for command in commands:
        process = subprocess.run(
            [*command, "--collect-only", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        nodes = sorted(
            line.strip() for line in process.stdout.splitlines() if "::" in line and line.strip().startswith("tests/")
        )
        results.append({"entry": command, "exit_code": process.returncode, "nodes": nodes})
    matched = bool(results[0]["nodes"]) and results[0]["nodes"] == results[1]["nodes"]
    passed = matched and all(value["exit_code"] == 0 for value in results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "same_nonempty_set": passed,
                "collections": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    Console().print({"same_nonempty_set": passed, "counts": [len(value["nodes"]) for value in results]})
    return passed


def main() -> int:
    """Run a read-only collection comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if probe(args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
