"""Configure the existing isolated stack before importing any application settings."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from scripts.v11_stack import configuration


def main() -> int:
    """No migrations, cleanup or real provider switches."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["seed", "verify"])
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--port", type=int, default=36439)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, values = configuration(args.compose_command)
    environment = {**os.environ, **values, "LOCAL_MVP_ONLY": "true", "LOG_FORMAT": "json"}
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "scripts.inheritance_probe",
            args.action,
            "--port",
            str(args.port),
            "--private",
            str(args.compose_command.parent),
            "--output",
            str(args.output),
        ],
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
