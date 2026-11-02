"""Launch a loopback browser fixture with real auth/PG/Graph and synthetic boundaries."""

import argparse
import os
import secrets
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from scripts.v11_stack import configuration


def main() -> int:
    """Validate isolation before migration or starting the test-only HTTP app."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--port", type=int, default=36439)
    parser.add_argument("--migrate", action="store_true")
    args = parser.parse_args()
    _, values = configuration(args.compose_command)
    private = args.compose_command.resolve().parent
    password_path = private / "browser-password.private.txt"
    if not password_path.exists():
        password_path.write_text("Rg1!" + secrets.token_urlsafe(20), encoding="utf-8")
    environment = {
        **os.environ,
        **values,
        "V11_BROWSER_PROBE": "1",
        "V11_BROWSER_PASSWORD": password_path.read_text(encoding="utf-8"),
        "V11_COMPLETE_EVIDENCE_PROBE": "1",
        "OPENAI_API_KEY": "synthetic-not-a-credential",  # pragma: allowlist secret
        "LLM_API_KEY": "synthetic-not-a-credential",  # pragma: allowlist secret
        "LLM_ENDPOINT": "http://127.0.0.1:1/v1",
        "CHAT_LLM_API_KEY": "synthetic-not-a-credential",  # pragma: allowlist secret
        "CHAT_LLM_ENDPOINT": "http://127.0.0.1:1/v1",
    }
    if args.migrate:
        with (private / "browser-migration.private.log").open("wb") as log:
            subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                env=environment,
                stdout=log,
                stderr=log,
                check=True,
            )
    Console().print(
        {
            "url": f"http://127.0.0.1:{args.port}",
            "credentials_file": str(password_path),
            "provenance": "synthetic_sources_models_real_auth_pg_graph",
        }
    )
    with (private / "browser-api.private.log").open("ab") as log:
        return subprocess.call(
            [sys.executable, "-m", "scripts.v11_probe_app", "--port", str(args.port)],
            env=environment,
            stdout=log,
            stderr=log,
        )


if __name__ == "__main__":
    raise SystemExit(main())
