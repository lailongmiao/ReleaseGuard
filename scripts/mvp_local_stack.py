"""Create a new private, loopback-only MVP stack without touching old volumes.

Only development synthetic mode is enabled. Secrets and raw command logs stay
outside the repository. Never run down -v, prune, downgrade or an existing DB.
"""

import argparse
import json
import os
import secrets
import socket
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from rich.console import Console
from tenacity import retry, retry_if_exception_type, stop_after_delay, wait_exponential


@retry(
    stop=stop_after_delay(60),
    wait=wait_exponential(multiplier=0.5, max=3),
    retry=retry_if_exception_type((httpx.HTTPError, AssertionError)),
    reraise=True,
)
def ready(port: int) -> None:
    """Require actual migration/Graph readiness after container startup."""
    with httpx.Client(trust_env=False, timeout=5) as client:
        response = client.get(f"http://127.0.0.1:{port}/ready")
        response.raise_for_status()
        assert response.json()["ready"] and response.json()["production_execution"] == "disabled"


def initialize(root: Path, port: int) -> Path:
    """Create fresh private files and an unoccupied project, never reuse data."""
    ports = [port + offset for offset in range(5)]
    for value in ports:
        with socket.socket() as connection:
            connection.bind(("127.0.0.1", value))
    project = "releaseguard-mvp-" + uuid4().hex[:12]
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "label=com.docker.compose.project=" + project, "--format", "{{.ID}}"],
        capture_output=True,
        timeout=15,
        check=True,
    )
    if result.stdout.strip():
        raise ValueError("new_project_required")
    private = Path(tempfile.mkdtemp(prefix="releaseguard-mvp-private-"))
    env = private / "runtime.env"
    values = {
        "APP_ENV": "development",
        "POSTGRES_DB": "mvp_isolated",
        "POSTGRES_USER": "mvp_local",
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "JWT_SECRET_KEY": secrets.token_urlsafe(48),
        "GRAFANA_ADMIN_PASSWORD": secrets.token_urlsafe(32),
        "MEMORY_ENABLED": "false",
        "LANGFUSE_TRACING_ENABLED": "false",
        "SESSION_NAMING_ENABLED": "false",
        "LOCAL_MVP_ENABLED": "true",
        "LOCAL_MVP_ONLY": "true",
        "DEBUG": "false",
        "VALKEY_HOST": "valkey",
        "VALKEY_PORT": "6379",
        "RATE_LIMIT_DEFAULT": '["10000 per day","10000 per hour"]',
        "RUNTIME_ENV_FILE": env.as_posix(),
        **dict(
            zip(
                (
                    "LOCAL_API_PORT",
                    "LOCAL_DB_PORT",
                    "LOCAL_VALKEY_PORT",
                    "LOCAL_PROMETHEUS_PORT",
                    "LOCAL_GRAFANA_PORT",
                ),
                map(str, ports),
                strict=True,
            )
        ),
    }
    env.write_text("\n".join(key + "=" + json.dumps(value) for key, value in values.items()) + "\n", encoding="utf-8")
    override = private / "runtime.override.yaml"
    override.write_text(
        "services:\n  app:\n    image: " + project + "-app:local\n"
        "    volumes:\n      - mvp-logs:/app/logs\nvolumes:\n  mvp-logs:\n",
        encoding="utf-8",
    )
    command = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--env-file",
        str(env),
        "-f",
        str(root / "docker-compose.yml"),
        "-f",
        str(override),
    ]
    path = private / "runtime-command.json"
    path.write_text(json.dumps(command), encoding="utf-8")
    return path


def operate(path: Path, action: str, port: int) -> int:
    """Accept only this script's isolated command; stop preserves all volumes."""
    root = Path(__file__).resolve().parents[1]
    args = json.loads(path.read_text(encoding="utf-8"))
    if not (
        len(args) == 10
        and args[:3] == ["docker", "compose", "--project-name"]
        and args[3].startswith("releaseguard-mvp-")
        and Path(args[7]).resolve() == root / "docker-compose.yml"
        and Path(args[5]).resolve().parent == path.resolve().parent
        and Path(args[9]).resolve().parent == path.resolve().parent
    ):
        raise ValueError("private_isolated_command_required")
    steps = {
        "up": [
            ["config", "--quiet"],
            ["build", "app"],
            ["up", "-d", "db", "valkey"],
            ["run", "--rm", "--no-deps", "app", "/app/.venv/bin/alembic", "upgrade", "head"],
            ["up", "-d", "app", "prometheus", "grafana"],
        ],
        "start": [["start"]],
        "stop": [["stop"]],
    }[action]
    # Explicit private stack values take precedence over the caller's dotenv.
    # Compose normally prioritizes process env over --env-file interpolation.
    child_env = {**os.environ, **{k: v for k, v in dotenv_values(args[5]).items() if v is not None}}
    results = []
    for number, step in enumerate(steps):
        result = subprocess.run([*args, *step], env=child_env, capture_output=True, timeout=1200, check=False)
        (path.parent / f"{action}-{number}.private.log").write_bytes(result.stdout + result.stderr)
        results.append({"step": step, "exit_code": result.returncode})
        (path.parent / "stack-result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        Console().print({"step": step[:3], "exit_code": result.returncode})
        if result.returncode:
            return result.returncode
    if action != "stop":
        ready(port)
    return 0


def main() -> int:
    """Prepare or operate a private stack; never print its secret values."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "up", "start", "stop"])
    parser.add_argument("--port", type=int, default=28080, help="API port; next four ports must also be free")
    parser.add_argument("--compose-command", type=Path)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65531:
        parser.error("five_unprivileged_ports_required")
    try:
        if args.action == "init":
            path = initialize(Path(__file__).resolve().parents[1], args.port)
            Console().print({"private_command_file": str(path), "port": args.port, "production_execution": "disabled"})
            return 0
        if args.compose_command is None:
            parser.error("compose_command_required")
        return operate(args.compose_command, args.action, args.port)
    except Exception as error:
        Console().print({"failure": type(error).__name__, "raw_logs": "private_only"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
