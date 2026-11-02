"""Create a new loopback PostgreSQL probe stack using an already installed image."""

import argparse
import json
import os
import re
import secrets
import socket
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]


def initialize(port: int) -> Path:
    """Never reuse an existing project, volume or configuration; never pull an image."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
    subprocess.run(
        ["docker", "image", "inspect", "pgvector/pgvector:pg16"], capture_output=True, check=True, timeout=15
    )
    project = "releaseguard-v11-" + uuid4().hex[:12]
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "label=com.docker.compose.project=" + project, "--format", "{{.ID}}"],
        capture_output=True,
        check=True,
        timeout=15,
    )
    if result.stdout.strip():
        raise ValueError("new_project_required")
    private = Path(tempfile.mkdtemp(prefix="releaseguard-v11-private-"))
    env = private / "runtime.env"
    values = {
        "APP_ENV": "test",
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": str(port),
        "POSTGRES_DB": "v11_isolated",
        "POSTGRES_USER": "v11_probe",
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "JWT_SECRET_KEY": secrets.token_urlsafe(48),
        "MEMORY_ENABLED": "false",
        "LANGFUSE_TRACING_ENABLED": "false",
        "LOCAL_MVP_ENABLED": "true",
        "V1_BACKEND_ENABLED": "true",
        "V11_BACKEND_ENABLED": "true",
        "V1_REAL_MODELS_ENABLED": "false",
        "V11_REAL_MODELS_ENABLED": "false",
        "V11_REAL_READS_ENABLED": "false",
        "SESSION_NAMING_ENABLED": "false",
        "V11_PROBE_PROJECT": project,
    }
    env.write_text("\n".join(key + "=" + json.dumps(value) for key, value in values.items()) + "\n", encoding="utf-8")
    compose = private / "compose.yaml"
    compose.write_text(
        "services:\n  db:\n    image: pgvector/pgvector:pg16\n    pull_policy: never\n    environment:\n      POSTGRES_DB: ${POSTGRES_DB}\n      POSTGRES_USER: ${POSTGRES_USER}\n      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}\n    ports:\n      - '127.0.0.1:${POSTGRES_PORT}:5432'\n    volumes:\n      - v11-data:/var/lib/postgresql/data\nvolumes:\n  v11-data:\n",
        encoding="utf-8",
    )
    command = ["docker", "compose", "--project-name", project, "--env-file", str(env), "-f", str(compose)]
    path = private / "runtime-command.json"
    path.write_text(json.dumps(command), encoding="utf-8")
    return path


def configuration(path: Path, *, offline: bool = True) -> tuple[list[str], dict[str, str]]:
    """Strictly validate the new probe namespace before any database operation."""
    path = path.resolve()
    args = json.loads(path.read_text(encoding="utf-8"))
    if (
        len(args) != 8
        or args[:3] != ["docker", "compose", "--project-name"]
        or not re.fullmatch(r"releaseguard-v11-[0-9a-f]{12}", args[3])
    ):
        raise ValueError("v11_isolated_stack_required")
    if (
        args[4] != "--env-file"
        or args[6] != "-f"
        or any(Path(args[index]).resolve().parent != path.parent for index in (5, 7))
    ):
        raise ValueError("v11_private_config_required")
    values = {key: value for key, value in dotenv_values(args[5]).items() if value is not None}
    if (
        values.get("V11_PROBE_PROJECT") != args[3]
        or values.get("POSTGRES_DB") != "v11_isolated"
        or values.get("POSTGRES_HOST") != "127.0.0.1"
    ):
        raise ValueError("v11_isolated_database_required")
    if offline and any(
        values.get(key) != "false"
        for key in ("V1_REAL_MODELS_ENABLED", "V11_REAL_MODELS_ENABLED", "V11_REAL_READS_ENABLED")
    ):
        raise ValueError("v11_offline_probe_required")
    return args, values


def main() -> int:
    """Only create/up/stop this new stack; no prune, drop, down or volume removal."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "up", "stop"])
    parser.add_argument("--port", type=int, default=35432)
    parser.add_argument("--compose-command", type=Path)
    args = parser.parse_args()
    if args.action == "init":
        Console().print({"compose_command": str(initialize(args.port))})
        return 0
    command, values = configuration(args.compose_command)
    step = ["up", "-d", "--pull", "never", "db"] if args.action == "up" else ["stop", "db"]
    result = subprocess.run([*command, *step], env={**os.environ, **values}, capture_output=True, timeout=60)
    (args.compose_command.parent / (args.action + ".private.log")).write_bytes(result.stdout + result.stderr)
    Console().print({"action": args.action, "exit_code": result.returncode})
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
