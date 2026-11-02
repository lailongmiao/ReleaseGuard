"""Offline two-process UNKNOWN hold proof for the versioned business wait policy."""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import httpx
from jose import jwt
from rich.console import Console

from app.core.config import settings
from scripts.v11_probe import configure, ready
from scripts.v11_probe_fixtures import create_body
from scripts.v11_process_environment import offline_environment


async def phase(port: int, number: int, saved: dict) -> list[str]:
    """Every command traverses authentication, HTTP, Graph and isolated PostgreSQL."""
    token = jwt.encode(
        {"sub": "v11-synthetic-session-1", "exp": datetime.now(UTC).timestamp() + 1800},
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    prefix = "/api/v1/releaseguard/reviews"
    checks = []
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=125, trust_env=False) as client:

        async def call(method: str, path: str, body: dict | None = None) -> dict:
            response = await client.request(
                method, prefix + path, json=body, headers={"Authorization": "Bearer " + token}
            )
            assert response.status_code == 200, response.status_code
            return response.json()

        async def view() -> dict:
            return await call("GET", "/" + saved["object"])

        async def analyze(key: str) -> dict:
            v = await view()
            return await call(
                "POST",
                "/" + saved["object"] + "/analyze",
                {
                    "command_id": key,
                    "expected_object_version": v["object_version"],
                    "expected_state_version": v["state"]["state_version"],
                    "snapshot_id": v["active_snapshot_id"],
                    "generation_id": v["active_generation_id"],
                    "query": "probe-unknown",
                },
            )

        async def diagnostic() -> dict:
            return await call("GET", "/" + saved["object"] + "/commands/" + saved["command"])

        if number == 1:
            await call("POST", "/repositories/checkout/verify")
            for rule in (await call("GET", "/rules"))["items"]:
                r = await call(
                    "POST",
                    "/rules/" + rule["rule_set_id"] + "/sync",
                    {"command_id": saved["nonce"] + rule["rule_set_id"], "expected_object_version": rule["version"]},
                )
                assert r["status"] == "completed"
            obj = await call("POST", "", create_body(saved["nonce"]))
            saved["object"] = obj["object_id"]
            result = await call(
                "POST",
                "/" + saved["object"] + "/sync",
                {"command_id": saved["nonce"] + "sync", "expected_object_version": obj["object_version"]},
            )
            assert result["status"] == "completed"
            v = await view()
            saved["body"] = {
                "command_id": saved["command"],
                "expected_object_version": v["object_version"],
                "expected_state_version": v["state"]["state_version"],
                "snapshot_id": v["active_snapshot_id"],
                "generation_id": v["active_generation_id"],
                "query": "probe-unknown",
            }
            result = await call("POST", "/" + saved["object"] + "/analyze", saved["body"])
            assert result["status"] == "unknown"
            saved["diagnostic"] = await diagnostic()
            checks.append("sent_failure_unknown")
        same = await call("POST", "/" + saved["object"] + "/analyze", saved["body"])
        assert same["status"] == "unknown"
        other = await analyze(saved["nonce"] + f"new-key-{number}")
        assert other.get("reason") == "analysis_unknown_hold"
        assert await diagnostic() == saved["diagnostic"]
        attempts = saved["diagnostic"]["attempts"]
        model = [a for a in attempts if a["attempt"] == "analysis-1"]
        assert len(model) == 1 and Decimal(str(model[0]["charge"])) == Decimal(str(model[0]["reserved"])) > 0
        checks += [
            f"process_{number}_same_key_no_resend",
            f"process_{number}_new_key_denied",
            f"process_{number}_reservation_unchanged",
        ]
        if number == 2:
            v = await view()
            r = await call(
                "POST",
                "/" + saved["object"] + "/revision",
                {
                    "command_id": saved["nonce"] + "revision",
                    "expected_object_version": v["object_version"],
                    "expected_state_version": v["state"]["state_version"],
                    "snapshot_id": v["active_snapshot_id"],
                    "generation_id": v["active_generation_id"],
                    "change": create_body("unused")["change"],
                },
            )
            assert r["status"] == "completed"
            assert (await analyze(saved["nonce"] + "after-revision"))["reason"] == "analysis_unknown_hold"
            assert await diagnostic() == saved["diagnostic"]
            checks.append("restart_new_revision_same_intent_denied")
    return checks


def main() -> int:
    """Run only on a validated new synthetic stack and retain every result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=35987)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("report_already_exists")
    _, values = configure(args.compose_command)
    saved = {"nonce": "wait-policy-" + uuid4().hex}
    saved["command"] = saved["nonce"] + "analyze"
    result = {"external_calls": 0, "checks": [], "processes": []}
    try:
        for number in (1, 2):
            with (args.compose_command.parent / f"wait-{number}.private.log").open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "scripts.v11_probe_app", "--port", str(args.port)],
                    env={**offline_environment(), **values, "LOCAL_MVP_ONLY": "true"},
                    stdout=log,
                    stderr=log,
                )
                result["processes"].append(process.pid)
                try:
                    ready(args.port)
                    with asyncio.Runner(
                        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None
                    ) as runner:
                        result["checks"].extend(runner.run(phase(args.port, number, saved)))
                finally:
                    process.terminate()
                    process.wait(timeout=20)
        result["passed"] = True
    except Exception as error:
        result.update(passed=False, failure_type=type(error).__name__)
    result["saved"] = saved
    args.report.write_text(json.dumps(result, indent=2), encoding="utf8")
    Console().print({k: v for k, v in result.items() if k != "saved"})
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
