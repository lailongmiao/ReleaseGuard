"""Original-contract legacy fixtures and before/after PG invariants; synthetic only."""

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

import httpx
from rich.console import Console

from app.core.config import settings
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.v1_storage import V1Store


async def fingerprint(ids: list[str]) -> str:
    """Digest all retained facts, approvals, receipts and command/budget rows."""
    async with await LocalPostgresStore().connect() as db:
        records = await (
            await db.execute(
                "SELECT change_id,payload,version FROM local_mvp_records WHERE change_id=ANY(%s) ORDER BY change_id",
                (ids,),
            )
        ).fetchall()
        commands = await (
            await db.execute("SELECT * FROM v1_commands WHERE resource=ANY(%s) ORDER BY owner_id,command_id", (ids,))
        ).fetchall()
        budgets = await (
            await db.execute(
                "SELECT b.* FROM v1_budget b JOIN v1_commands c USING(owner_id,command_id) WHERE c.resource=ANY(%s) ORDER BY b.owner_id,b.command_id,b.attempt",
                (ids,),
            )
        ).fetchall()
        return canonical_hash(json.loads(json.dumps([records, commands, budgets], default=str)))


async def run(args: argparse.Namespace) -> None:
    """Generate new records only; verify mode performs business GETs and SQL SELECTs."""
    if (
        not re.fullmatch(r"releaseguard-v11-[0-9a-f]{12}", os.environ.get("V11_PROBE_PROJECT", ""))
        or settings.POSTGRES_DB != "v11_isolated"
        or settings.POSTGRES_HOST != "127.0.0.1"
        or settings.V1_REAL_MODELS_ENABLED
        or settings.V11_REAL_MODELS_ENABLED
        or settings.V11_REAL_READS_ENABLED
    ):
        raise RuntimeError("isolated_synthetic_environment_required")
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", timeout=90) as client:
        password = (args.private / "browser-password.private.txt").read_text(encoding="utf-8")
        tokens = {}
        for user in (1, 2, 3):
            login = await client.post(
                "/api/v1/auth/login", data={"email": f"v11-synthetic-{user}@example.invalid", "password": password}
            )
            login.raise_for_status()
            session = await client.post(
                "/api/v1/auth/session", headers={"Authorization": "Bearer " + login.json()["access_token"]}
            )
            session.raise_for_status()
            tokens[user] = session.json()["token"]["access_token"]

        async def call(method: str, path: str, data: dict | None = None, user: int = 1) -> dict:
            response = await client.request(
                method, "/api/v1" + path, json=data, headers={"Authorization": "Bearer " + tokens[user]}
            )
            if response.status_code >= 300:
                raise AssertionError(f"{method} {path}: {response.status_code} {response.text[:400]}")
            return response.json()

        root = "/releaseguard/changes"
        if args.action == "seed":
            sources = await call("GET", "/evidence/sources")
            for source in sources["items"]:
                if 2 not in source["acl"]:
                    await call(
                        "POST",
                        f"/evidence/sources/{source['source_id']}/grant",
                        {"command_id": str(uuid4()), "expected_version": source["version"], "user_id": 2},
                    )
                    source = await call("GET", f"/evidence/sources/{source['source_id']}")
                result = await call(
                    "POST",
                    f"/evidence/sources/{source['source_id']}/sync",
                    {"command_id": str(uuid4()), "expected_version": source["version"], "pr_number": 1},
                )
                assert result["status"] == "completed", result
            fixtures = {}
            for scenario in ("pending", "receipt", "revised", "rejected"):
                now = datetime.now(UTC)
                change = {
                    "requester_id": "local-user:1",
                    "service": "inheritance-synthetic",
                    "environment": "staging",
                    "title": "历史合成审阅 / " + scenario,
                    "description": "Original-contract inheritance fixture",
                    "change_window": {
                        "start": now.isoformat(),
                        "end": (now + timedelta(hours=2)).isoformat(),
                        "timezone": "UTC",
                    },
                    "rollback_plan": "Retain prior facts",
                    "verification_plan": "Inspect original simulation receipt",
                }
                view = await call(
                    "POST",
                    root,
                    {
                        "change": change,
                        "approver_user_id": 2,
                        "idempotency_key": str(uuid4()),
                        "product": {"pr_number": 1},
                    },
                )
                identifier = view["state"]["change_id"]
                base = root + "/" + identifier
                command = str(uuid4())
                analyzed = await call(
                    "POST",
                    base + "/analyze",
                    {
                        "command_id": command,
                        "expected_state_version": view["state"]["state_version"],
                        "query": "simulation approval",
                    },
                )
                assert analyzed["status"] == "completed", analyzed
                view = await call("GET", base)
                view = await call(
                    "POST",
                    base + "/prepare",
                    {
                        "command_id": str(uuid4()),
                        "expected_state_version": view["state"]["state_version"],
                        "mode": "offline_simulation",
                        "approval_ttl_seconds": 3600,
                        "analysis_id": analyzed["analysis_id"],
                    },
                )
                if scenario in {"receipt", "rejected"}:
                    request = view["approval"]["request"]
                    view = await call(
                        "POST",
                        base + "/approval",
                        {
                            "command_id": str(uuid4()),
                            "expected_state_version": view["state"]["state_version"],
                            "approval_request_id": request["approval_request_id"],
                            "authorization_hash": request["authorization_hash"],
                            "decision": "APPROVED" if scenario == "receipt" else "REJECTED",
                            "reason_code": "synthetic_reviewed",
                        },
                        user=2,
                    )
                if scenario == "receipt":
                    view = await call(
                        "POST",
                        base + "/run",
                        {"command_id": str(uuid4()), "expected_state_version": view["state"]["state_version"]},
                    )
                    assert view["state"]["receipt"]["final_status"] == "SIMULATED"
                if scenario == "revised":
                    change["description"] = "Revised, old report body may be absent"
                    view = await call(
                        "POST",
                        base + "/revision",
                        {
                            "command_id": str(uuid4()),
                            "expected_state_version": view["state"]["state_version"],
                            "change": change,
                        },
                    )
                fixtures[scenario] = {"id": identifier, "analysis_command": command}
            # Explicit fault boundary: claim persisted, no external call and no completion.
            store = V1Store()
            async with await store.connection.connect() as db:
                row = await (
                    await db.execute("SELECT id FROM session WHERE user_id=1 ORDER BY id DESC LIMIT 1")
                ).fetchone()
            actor = await store.connection.authenticate(row["id"], datetime.now(UTC).timestamp() + 3600)
            unknown = str(uuid4())
            await store.claim(actor, unknown, canonical_hash({"synthetic": unknown}), fixtures["pending"]["id"])
            await store.reserve(actor, unknown, "synthetic-loss", Decimal("0.001"), Decimal("10"))
            fixtures["pending"]["unknown_command"] = unknown
            failed = str(uuid4())
            await store.claim(actor, failed, canonical_hash({"synthetic": failed}), fixtures["pending"]["id"])
            await store.complete(actor, failed, {"status": "failed", "reason": "synthetic_failure"})
            fixtures["pending"]["failed_command"] = failed
            ids = [value["id"] for value in fixtures.values()]
            payload = {
                "fixtures": fixtures,
                "facts_digest": await fingerprint(ids),
                "provenance": "synthetic_original_contract_real_pg",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            Console().print({"seeded": len(ids), "output": str(args.output)})
        else:
            payload = json.loads(args.output.read_text(encoding="utf-8"))
            checks = []
            for scenario, fixture in payload["fixtures"].items():
                base = root + "/" + fixture["id"]
                for suffix in ("", "/analysis", "/evidence", "/receipt", "/audit", "/approval-history"):
                    value = await call("GET", base + suffix)
                    if suffix == "/approval-history":
                        assert value["items"] and value["next_cursor"] is None
                        assert all(
                            item["change_id"] == fixture["id"]
                            and item["authorization_hash"]
                            and item["binding"]["invocation"]["change_id"] == fixture["id"]
                            for item in value["items"]
                        )
                        assert "session_id" not in json.dumps(value)
                        if scenario in {"receipt", "rejected"}:
                            assert value["items"][0]["decision"]["decision"] == (
                                "APPROVED" if scenario == "receipt" else "REJECTED"
                            )
                        if scenario == "revised":
                            assert value["items"][0]["invalidations"]
                    if suffix == "/receipt" and scenario == "receipt":
                        assert value["receipt"]["final_status"] == "SIMULATED"
                        assert value["receipt"]["change_id"] == fixture["id"]
                    denied = await client.get(
                        "/api/v1" + base + suffix, headers={"Authorization": "Bearer " + tokens[3]}
                    )
                    assert denied.status_code == 404
                result = await call("GET", base + "/commands/" + fixture["analysis_command"])
                assert result["status"] == "completed"
                denied = await client.get(
                    "/api/v1" + base + "/commands/" + fixture["analysis_command"],
                    headers={"Authorization": "Bearer " + tokens[2]},
                )
                assert denied.status_code == 404
                checks.append("object_and_original_actor_authorization")
            pending = payload["fixtures"]["pending"]
            result = await call("GET", root + "/" + pending["id"] + "/commands/" + pending["unknown_command"])
            assert result["status"] == "unknown" and result["attempts"]
            checks.append("unknown_retains_budget_no_resend")
            result = await call("GET", root + "/" + pending["id"] + "/commands/" + pending["failed_command"])
            assert result["result"]["status"] == "failed" and result["result"]["reason"] == "synthetic_failure"
            checks.extend(["saved_decisions_invalidations_receipt_bindings", "saved_failed_command_reason"])
            ids = [value["id"] for value in payload["fixtures"].values()]
            assert payload["facts_digest"] == await fingerprint(ids)
            payload["verification"] = {
                "checks": checks,
                "facts_commands_budget_unchanged": True,
                "checked_at": datetime.now(UTC).isoformat(),
            }
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            Console().print(payload["verification"])


def main() -> None:
    """CLI output contains only synthetic IDs and invariant results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["seed", "verify"])
    parser.add_argument("--port", type=int, default=36439)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run(args))


if __name__ == "__main__":
    main()
