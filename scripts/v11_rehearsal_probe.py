"""Three-process rehearsal through real authentication, HTTP, Graph and isolated PG."""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from copy import deepcopy
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
    """Only synthetic sources/models; actors authenticate against distinct PG sessions."""
    checks = []
    prefix = "/api/v1/releaseguard/reviews"
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=120) as client:

        async def call(method: str, path: str, body: dict | None = None, actor: int = 1, expected: int = 200) -> dict:
            saved["last_action"] = {"method": method, "path": path, "command": (body or {}).get("command_id")}
            token = jwt.encode(
                {"sub": f"v11-synthetic-session-{actor}", "exp": datetime.now(UTC).timestamp() + 3600},
                settings.JWT_SECRET_KEY,
                algorithm=settings.JWT_ALGORITHM,
            )
            response = await client.request(
                method,
                path if path.startswith("/api/") else prefix + path,
                json=body,
                headers={"Authorization": "Bearer " + token},
            )
            if response.status_code != expected:
                raise AssertionError("unexpected_http_status:" + str(response.status_code) + ":" + response.text[:300])
            return response.json()

        async def view() -> dict:
            return await call("GET", "/" + saved["object"])

        async def command(route: str, key: str, extra: dict | None = None, actor: int = 1) -> dict:
            current = await view()
            return await call(
                "POST",
                "/" + saved["object"] + "/" + route,
                {
                    "command_id": saved["nonce"] + key,
                    "expected_object_version": current["object_version"],
                    **({"expected_state_version": current["state"]["state_version"]} if route != "sync" else {}),
                    **(extra or {}),
                },
                actor,
            )

        if number == 1:
            await call("POST", "/repositories/checkout/verify")
            versions = {row["rule_set_id"]: row["version"] for row in (await call("GET", "/rules"))["items"]}
            for rule in ("application-policy", "checkout-reference"):
                result = await call(
                    "POST",
                    "/rules/" + rule + "/sync",
                    {
                        "command_id": saved["nonce"] + rule,
                        "expected_object_version": versions[rule],
                    },
                )
                assert result["status"] == "completed", result
            current = await call("POST", "", create_body(saved["nonce"]))
            saved["object"] = current["object_id"]
            result = await command("sync", "sync")
            assert result["status"] == "completed", result
            current = await view()
            result = await command(
                "analyze",
                "analyze",
                {
                    "query": "Describe the changed line.",
                    "snapshot_id": current["active_snapshot_id"],
                    "generation_id": current["active_generation_id"],
                },
            )
            assert result["status"] == "completed", result
            saved["analysis"] = result["analysis_id"]
            current = await view()
            assert current["rehearsal_eligibility"]["eligible"], current["rehearsal_eligibility"]
            result = await command("prepare", "prepare", {"analysis_id": saved["analysis"]})
            assert result["status"] == "completed", result
            saved["prepared"] = result["result"]
            denied = await command("run", "no-approval")
            assert denied["status"] == "blocked", denied
            checks.append("no_approval_denied")
            checks.append("authenticated_owner_strict_gate_prepare")
        elif number == 2:
            current = await view()
            assert current["state"]["state_version"] == saved["prepared"]["state"]["state_version"]
            approval = saved["prepared"]["approval"]["request"]
            extra = {
                "approval_request_id": approval["approval_request_id"],
                "authorization_hash": approval["authorization_hash"],
                "decision": "APPROVED",
                "reason_code": "synthetic_reviewed",
            }
            denied = await command("approval", "self-approval", extra)
            assert denied["status"] == "blocked", denied
            result = await command("approval", "approve", extra, actor=2)
            assert result["status"] == "completed", result
            current = await view()
            saved["run_body"] = {
                "command_id": saved["nonce"] + "run",
                "expected_object_version": current["object_version"],
                "expected_state_version": current["state"]["state_version"],
            }
            result = await call("POST", "/" + saved["object"] + "/run", saved["run_body"])
            assert result["status"] == "completed", result
            saved["receipt"] = (await call("GET", "/" + saved["object"] + "/receipt"))["receipt"]
            assert saved["receipt"] is not None
            assert saved["receipt"]["final_status"] == "SIMULATED", saved["receipt"]
            checks += [
                "restart_preserves_preparation",
                "self_approval_denied",
                "distinct_authenticated_reviewer",
                "dry_run_receipt",
            ]
        else:
            receipt = await call("GET", "/" + saved["object"] + "/receipt")
            assert receipt["receipt"] == saved["receipt"] and receipt["target_write_calls"] == 0, receipt
            prior = await call("GET", "/" + saved["object"] + "/commands/" + saved["nonce"] + "self-approval")
            assert prior["result"]["status"] == "blocked", prior
            replayed = await call("POST", "/" + saved["object"] + "/run", saved["run_body"])
            assert replayed["status"] == "completed", replayed
            conflict = deepcopy(saved["run_body"])
            conflict["expected_state_version"] += 1
            response = await call("POST", "/" + saved["object"] + "/run", conflict, expected=409)
            assert response.get("code") == "idempotency_conflict", response
            checks += [
                "third_process_receipt_unchanged_zero_writes",
                "rejection_persisted",
                "identical_replay_no_target_write",
                "conflicting_replay_denied",
            ]
            original = saved["object"]
            for name, repository, selector, query in (
                ("ci-failure", "checkout", {"kind": "pr", "pr_number": 2}, "Describe the changed line."),
                ("commit", "checkout", {"kind": "commit", "commit_sha": "a" * 40}, "Describe the changed line."),
                (
                    "branch",
                    "checkout",
                    {"kind": "branch_compare", "base_branch": "main", "head_branch": "topic"},
                    "Describe the changed line.",
                ),
                ("unknown", "checkout", {"kind": "pr", "pr_number": 1}, "probe-unknown"),
                ("expired", "checkout", {"kind": "pr", "pr_number": 1}, "Describe the changed line."),
            ):
                current = await call("POST", "", create_body(saved["nonce"] + name, repository, selector))
                saved["object"] = current["object_id"]
                result = await command("sync", name + "sync")
                assert result["status"] == "completed", result
                current = await view()
                result = await command(
                    "analyze",
                    name + "analyze",
                    {
                        "query": query,
                        "snapshot_id": current["active_snapshot_id"],
                        "generation_id": current["active_generation_id"],
                    },
                )
                analysis = result["analysis_id"]
                prepared = await command(
                    "prepare",
                    name + "prepare",
                    {"analysis_id": analysis, "approval_ttl_seconds": 30 if name == "expired" else 600},
                )
                if name == "expired":
                    assert prepared["status"] == "completed", prepared
                    request = prepared["result"]["approval"]["request"]
                    await asyncio.sleep(
                        max(
                            0,
                            (
                                datetime.fromisoformat(request["expires_at"].replace("Z", "+00:00"))
                                - datetime.now(UTC)
                            ).total_seconds(),
                        )
                        + 0.1
                    )
                    result = await command(
                        "approval",
                        name + "approve",
                        {
                            "approval_request_id": request["approval_request_id"],
                            "authorization_hash": request["authorization_hash"],
                            "decision": "APPROVED",
                            "reason_code": "synthetic_reviewed",
                        },
                        actor=2,
                    )
                    result = await command("run", name + "run")
                    assert result["status"] == "blocked", result
                else:
                    assert prepared["status"] == "blocked", prepared
                assert (await call("GET", "/" + saved["object"] + "/receipt"))["receipt"] is None
                checks.append(name + "_http_denied_zero_receipt")
            saved["object"] = original
            denied = await call("GET", "/" + original, actor=3, expected=404)
            assert denied["code"] == "local_object_not_found"
            source = next(
                row
                for row in (await call("GET", "/api/v1/releaseguard/reviews/sources"))["items"]
                if row["source_id"] == "review-github-checkout"
            )
            await call(
                "POST",
                "/api/v1/releaseguard/reviews/sources/review-github-checkout/revoke",
                {
                    "command_id": saved["nonce"] + "revoke-source",
                    "expected_version": source["version"],
                    "user_id": 2,
                },
            )
            denied = await call("GET", "/" + original, actor=2, expected=404)
            assert denied["code"] == "local_object_not_found"
            checks += ["unrelated_authenticated_subject_denied", "source_revocation_denies_reviewer"]
    return checks


def main() -> int:
    """Keep every failure and shut down only the child processes created here."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=35981)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("report_already_exists")
    _, values = configure(args.compose_command)
    saved = {"nonce": "rehearsal-" + uuid4().hex}
    result = {
        "provenance": "synthetic_sources_models_real_http_pg_graph",
        "external_calls": 0,
        "checks": [],
        "processes": [],
    }
    try:
        for number in (1, 2, 3):
            with (args.compose_command.parent / f"{args.report.stem}-{number}.private.log").open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "scripts.v11_probe_app", "--port", str(args.port)],
                    env={**offline_environment(), **values},
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
        (args.compose_command.parent / (args.report.stem + ".failure.private.txt")).write_text(
            str(error), encoding="utf-8"
        )
    result["saved"] = saved
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print({key: value for key, value in result.items() if key != "saved"})
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
