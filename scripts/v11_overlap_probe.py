"""Three-process synthetic PG/HTTP proof of the four-position formal exception."""

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from jose import jwt
from rich.console import Console

from app.core.config import settings
from app.schemas.local_mvp import LocalActor
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_backend import V11Backend
from app.services.v11_evaluation_policy import (
    FORMAL_BATCH,
    LINK_CONTRACT,
    OLD_COMMANDS,
    OVERLAP_CASES,
    READONLY_REASON,
    RETIRED_BATCH,
    RETIRED_BATCHES,
    DATASET_HASH,
)
from app.services.v11_storage import V11Store
from evals.v1_backend.provenance import code_state
from evals.v11_backend.http_evidence import EvidencePermit
from scripts.v11_probe import configure, ready
from scripts.v11_probe_fixtures import create_body

ACTOR = LocalActor(user_id=1, session_id="v11-synthetic-session-1", expires_at=4102444800)


async def old_digest(store: V11Store) -> str:
    """Include unknown command bytes and full reservations, not just an in-memory count."""
    async with await store.connection.connect() as db:
        rows = await (
            await db.execute(
                "SELECT row_to_json(c) AS row FROM v1_commands c WHERE command_id=ANY(%s) ORDER BY command_id",
                (list(OLD_COMMANDS),),
            )
        ).fetchall()
        budget = await (
            await db.execute(
                "SELECT row_to_json(b) AS row FROM v1_budget b WHERE command_id=ANY(%s) ORDER BY command_id,attempt",
                (list(OLD_COMMANDS),),
            )
        ).fetchall()
    return canonical_hash([rows, budget])


async def phase(port: int, number: int, saved: dict, permit: EvidencePermit) -> list[str]:
    """Real authentication, revision routes, Graph, storage and private evidence reads."""
    prefix = "/api/v1/releaseguard/reviews"
    store = V11Store(LocalPostgresStore())
    token = jwt.encode(
        {"sub": ACTOR.session_id, "exp": ACTOR.expires_at}, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False, headers={"Authorization": "Bearer " + token}
    ) as client:

        async def call(method: str, path: str, body: dict | None = None, status: int = 200) -> dict:
            response = await client.request(method, path, json=body)
            assert response.status_code == status, (response.status_code, response.text[:250])
            return response.json()

        if number == 1:
            await call("POST", prefix + "/repositories/checkout/verify")
            versions = {r["rule_set_id"]: r["version"] for r in (await call("GET", prefix + "/rules"))["items"]}
            for rule in ("application-policy", "checkout-reference"):
                result = await call(
                    "POST",
                    prefix + "/rules/" + rule + "/sync",
                    {
                        "command_id": "overlap-probe-rule-" + rule,
                        "expected_object_version": versions[rule],
                    },
                )
                assert result["status"] == "completed", result
            old = await call("POST", prefix, create_body("overlap-probe-old"))
            saved["old_object"] = old["object_id"]
            # Synthetic historical fixtures only, in the new dedicated database.
            for command in OLD_COMMANDS:
                await store.claim_review(ACTOR, command, canonical_hash(command), old["object_id"], "acceptance")
                await store.reserve(ACTOR, command, "analysis-1", Decimal("0.001"), Decimal("100"))
                await store.complete(ACTOR, command, {"status": "unknown", "synthetic_fixture": True})
            saved["old_digest"] = await old_digest(store)
            await asyncio.gather(*(store.retire_formal_batch(ACTOR) for _ in range(3)))

        if number in (1, 2):
            for case in sorted(OVERLAP_CASES):
                command = f"overlap-probe:{case}:{number}"
                if number == 1:
                    obj = await call("POST", prefix, create_body(command))
                    saved[case] = {"object_id": obj["object_id"]}
                identifier = saved[case]["object_id"]
                obj = await call("GET", prefix + "/" + identifier)
                synced = await call(
                    "POST",
                    prefix + "/" + identifier + "/sync",
                    {
                        "command_id": command + ":sync",
                        "expected_object_version": obj["object_version"],
                    },
                )
                assert synced["status"] == "completed", synced
                obj = await call("GET", prefix + "/" + identifier)
                if number == 2:
                    assert synced["generation_id"] == saved[case]["generation"]
                    revised = await call(
                        "POST",
                        prefix + "/" + identifier + "/revision",
                        {
                            "command_id": command + ":revision",
                            "expected_object_version": obj["object_version"],
                            "expected_state_version": obj["state"]["state_version"],
                            "snapshot_id": synced["snapshot_id"],
                            "generation_id": synced["generation_id"],
                            "change": create_body("unused")["change"],
                        },
                    )
                    assert revised["status"] == "completed", revised
                    obj = await call("GET", prefix + "/" + identifier)
                saved[case]["generation"] = synced["generation_id"]
                body = {
                    "command_id": command,
                    "query": permit.positions[command]["query"],
                    "expected_object_version": obj["object_version"],
                    "expected_state_version": obj["state"]["state_version"],
                    "snapshot_id": synced["snapshot_id"],
                    "generation_id": synced["generation_id"],
                }
                path = prefix + "/" + identifier + "/analyze"
                if number == 1 and case == "real-03-en":
                    responses = await asyncio.gather(*(client.post(path, json=body) for _ in range(4)))
                    assert sum(r.status_code == 200 and r.json().get("status") == "completed" for r in responses) == 1
                    assert sum(r.status_code == 409 for r in responses) == 3
                else:
                    result = await call("POST", path, body)
                    assert result["status"] == "completed", result
                replay = await call("POST", path, body, status=409)
                assert replay["code"] == "evaluation_position_consumed_no_resend", replay
                saved[case]["body"] = body
                answer = await call("GET", "/_evaluation/" + command + "/evidence/answer")
                assert answer["status"] == "unvalidated" and answer["review_complete"]
                view = await call("GET", prefix + "/" + identifier)
                assert READONLY_REASON in view["rehearsal_eligibility"]["reason_codes"]

        if number == 3:
            backend = V11Backend(SimpleNamespace(local=SimpleNamespace()), store, Mock(), Mock())
            for case in sorted(OVERLAP_CASES):
                identifier = saved[case]["object_id"]
                obj = await call("GET", prefix + "/" + identifier)
                assert READONLY_REASON in obj["rehearsal_eligibility"]["reason_codes"]
                body = saved[case]["body"]
                replay = await call("POST", prefix + "/" + identifier + "/analyze", body, status=409)
                assert replay["code"] == "evaluation_position_consumed_no_resend"
                with patch("app.services.v11_backend.require_rehearsal", lambda: None):
                    for kind in ("prepare", "decide", "execute"):
                        blocked = await backend.command(identifier, ACTOR, SimpleNamespace(), kind)
                        assert blocked == {"status": "blocked", "reason": READONLY_REASON}
                    try:
                        await backend.fresh(identifier, ACTOR, {})
                    except DomainError as error:
                        assert error.code == READONLY_REASON
                    else:
                        raise AssertionError("future_recovery_bypassed_permanent_restriction")
                with_ordinary_backend = dict(body, command_id="another-key")
                try:
                    await backend.analyze(identifier, ACTOR, SimpleNamespace(**with_ordinary_backend))
                except DomainError as error:
                    assert error.code == "evaluation_position_not_authorized"
                else:
                    raise AssertionError("ordinary_analysis_bypassed_evaluation")
            source = next(
                s
                for s in (await call("GET", prefix + "/sources"))["items"]
                if s["source_id"] == "review-github-checkout"
            )
            await call(
                "POST",
                prefix + "/sources/review-github-checkout/revoke",
                {
                    "command_id": "overlap-probe-revoke",
                    "expected_version": source["version"],
                    "user_id": 2,
                },
            )
            command = "overlap-probe:real-03-zh:2"
            await call("GET", "/_evaluation/" + command + "/evidence/answer", status=404)
            await store.claim_review(
                ACTOR, "unapproved-unknown", canonical_hash("unapproved"), saved["old_object"], "acceptance"
            )
            await store.complete(ACTOR, "unapproved-unknown", {"status": "unknown", "synthetic_fixture": True})
            try:
                await store.check_evaluation_overlap(saved["real-03-zh"]["object_id"], ACTOR, permitted=True)
            except DomainError as error:
                assert error.code == "evaluation_unapproved_unknown_overlap"
            else:
                raise AssertionError("extra_unknown_operation_was_not_blocked")
        assert await old_digest(store) == saved["old_digest"]
        async with await store.connection.connect() as db:
            links = await (
                await db.execute("SELECT result FROM v1_commands WHERE contract_version=%s", (LINK_CONTRACT,))
            ).fetchall()
        assert len(links) == (2 if number == 1 else 4)
        assert all(r["result"]["actual_input_hash"] and r["result"]["permanently_readonly"] for r in links)
    retired_command = "analysis:" + canonical_hash([RETIRED_BATCH, DATASET_HASH, "real-03-zh", 1])
    try:
        await store.reserve(ACTOR, retired_command, "analysis-1", Decimal("0.001"), Decimal("100"))
    except DomainError as error:
        assert error.code == "evaluation_batch_retired"
    else:
        raise AssertionError("retired_send_permitted")
    if number == 3:
        # Exercise the earlier batch too, not only the most recent failed batch.
        retired_command = "analysis:" + canonical_hash([RETIRED_BATCHES[0], DATASET_HASH, "real-03-en", 2])
        await store.claim_review(
            ACTOR, retired_command, canonical_hash(retired_command), saved["old_object"], "acceptance"
        )
        try:
            await store.retire_formal_batch(ACTOR)
        except DomainError as error:
            assert error.code == "evaluation_retired_overlap_consumed"
        else:
            raise AssertionError("consumed_old_overlap_accepted")
    return [
        f"process_{number}_old_unknown_and_reservations_unchanged",
        f"process_{number}_durable_links_verified",
        f"process_{number}_retired_send_denied",
    ]


def main() -> int:
    """Use only a new isolated synthetic stack and always stop each child process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("report_already_exists")
    _, values = configure(args.compose_command)
    private = args.compose_command.parent / "formal-overlap-evidence"
    private.mkdir(mode=0o700)
    if sys.platform == "win32":
        script = (
            "$ErrorActionPreference='Stop';$p=$env:V11_NEW_EVIDENCE_DIR;"
            "$sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
            "& icacls $p /grant:r \"*${sid}:(OI)(CI)F\" '*S-1-5-18:(OI)(CI)F';if($LASTEXITCODE){exit 1};"
            "& icacls $p /inheritance:r;if($LASTEXITCODE){exit 1};"
            "& icacls $p /remove:g '*S-1-3-4' '*S-1-5-32-544';exit $LASTEXITCODE"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            check=True,
            env={**os.environ, "V11_NEW_EVIDENCE_DIR": str(private)},
        )
    positions = {
        f"overlap-probe:{case}:{repeat}": {
            "case_id": case,
            "repeat": repeat,
            "query": "probe-success",
            "binding": create_body("unused")["binding"],
            "sync_command": f"overlap-probe:{case}:{repeat}:sync",
            "comparison": {
                "kind": "pr",
                "base_kind": "commit",
                "base_tip_sha": "b" * 40,
                "head_sha": "a" * 40,
                "diff_base_sha": "c" * 40,
                "parent_sha": None,
                "merge_base_sha": "c" * 40,
                "comparison": "merge_base",
            },
        }
        for case in sorted(OVERLAP_CASES)
        for repeat in (1, 2)
    }
    permit = EvidencePermit(
        batch_id=FORMAL_BATCH,
        source_hash=code_state()["source_hash"],
        dataset_hash=canonical_hash(positions),
        owner_id=1,
        expires_at=datetime.now(UTC) + timedelta(hours=2),
        mode="synthetic",
        designated_ledger_id="synthetic-offline-only",
        ledger_charge_at_start=0,
        maximum_batch_charge=19,
        positions=positions,
        overlap_exception=LINK_CONTRACT,
    )
    key = secrets.token_bytes(32)
    (private / "key.private").write_bytes(key)
    (private / "permit.private.json").write_text(
        json.dumps(
            {
                "permit": permit.model_dump(mode="json"),
                "signature": hmac.new(key, permit.model_dump_json().encode(), hashlib.sha256).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    result: dict = {"provenance": "new_isolated_synthetic_pg_http", "real_sends": 0, "processes": [], "checks": []}
    saved: dict = {}
    try:
        for number in (1, 2, 3):
            with (private / f"phase-{number}.private.log").open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "scripts.v11_probe_app", "--port", str(args.port)],
                    env={**os.environ, **values, "V11_SYNTHETIC_EVIDENCE_DIR": str(private), "LOCAL_MVP_ONLY": "true"},
                    stdout=log,
                    stderr=log,
                )
                result["processes"].append(process.pid)
                try:
                    ready(args.port)
                    with asyncio.Runner(
                        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None
                    ) as runner:
                        result["checks"].extend(runner.run(phase(args.port, number, saved, permit)))
                finally:
                    process.terminate()
                    process.wait(timeout=20)
        result["passed"] = True
    except Exception as error:
        result.update(passed=False, failure_type=type(error).__name__)
        (private / "failure.private.txt").write_text(str(error), encoding="utf-8")
    result["private_directory"] = str(private)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
