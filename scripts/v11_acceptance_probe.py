"""New isolated PG/HTTP evidence and budget probe; no real source or model capability."""

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
from uuid import uuid4

import httpx
from jose import jwt
from rich.console import Console

from app.core.config import settings
from app.schemas.local_mvp import LocalActor
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_storage import V11Store
from evals.v1_backend.provenance import code_state
from evals.v11_backend.http_evidence import EvidencePermit
from scripts.v11_probe import configure, ready
from scripts.v11_probe_fixtures import create_body, seed


async def budgets() -> dict:
    """Actual PG locking admits one 17th reservation and rejects every 18th contender."""
    store = V11Store(LocalPostgresStore())
    await seed(store)
    actor = LocalActor(user_id=1, session_id="v11-synthetic-session-1", expires_at=4102444800)
    nonce = "engineering-budget-" + uuid4().hex
    for number in range(16):
        command = nonce + str(number)
        await store.claim_review(actor, command, canonical_hash(command), "synthetic", "development")
        await store.reserve(actor, command, "analysis-1", Decimal("0.001"), Decimal("100"))

    async def compete(number: int) -> str:
        command = nonce + "race" + str(number)
        await store.claim_review(actor, command, canonical_hash(command), "synthetic", "development")
        try:
            await store.reserve(actor, command, "analysis-1", Decimal("0.001"), Decimal("100"))
            return "reserved"
        except DomainError as error:
            return error.code

    outcomes = await asyncio.gather(*(compete(n) for n in range(8)))
    assert outcomes.count("reserved") == 1 and outcomes.count("development_attempt_limit") == 7
    assert await compete(8) == "development_attempt_limit"
    async with await store.connection.connect() as db:
        row = await (
            await db.execute(
                "SELECT COUNT(*) AS count,SUM(b.charge) AS charge,BOOL_AND(b.charge=b.reserved AND (b.usage IS NULL OR b.usage='{}'::jsonb)) AS held "
                "FROM v1_budget b JOIN v1_commands c USING(owner_id,command_id) "
                "WHERE c.budget_category='development'"
            )
        ).fetchone()
    assert row["count"] == 17 and row["held"]
    for category, amount in (("development", Decimal("3")), ("acceptance", Decimal("25.01"))):
        command = nonce + category
        await store.claim_review(actor, command, canonical_hash(command), "synthetic", category)
        try:
            await store.reserve(actor, command, "embedding-limit", amount, Decimal("100"))
        except DomainError as error:
            assert error.code == "model_budget_exhausted"
        else:
            raise AssertionError("category_limit_not_enforced")
    store.evaluation_ceiling = Decimal("0.00000001")
    command = nonce + "batch-ceiling"
    await store.claim_review(actor, command, canonical_hash(command), "synthetic", "acceptance")
    try:
        await store.reserve(actor, command, "analysis-1", Decimal("0.001"), Decimal("100"))
    except DomainError as error:
        assert error.code == "model_budget_exhausted"
    else:
        raise AssertionError("batch_ceiling_not_enforced")
    return {
        "race_17_one_winner": True,
        "attempt_18_denied": True,
        "unknown_full_reservations": True,
        "category_amount_caps": True,
        "signed_batch_ceiling": True,
    }


async def phase(port: int, number: int, saved: dict, private: Path, permit: EvidencePermit) -> list[str]:
    """Every request traverses real HTTP, authentication, one Graph and PostgreSQL."""
    checks = []
    prefix = "/api/v1/releaseguard/reviews"
    token = jwt.encode(
        {"sub": "v11-synthetic-session-1", "exp": datetime.now(UTC).timestamp() + 3600},
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=120, headers={"Authorization": "Bearer " + token}
    ) as client:

        async def call(method: str, path: str, body: dict | None = None, status: int = 200) -> dict:
            response = await client.request(method, path, json=body)
            assert response.status_code == status, (response.status_code, response.text[:250])
            return response.json()

        async def evidence(command: str, artifact: str, status: int = 200) -> dict:
            return await call("GET", "/_evaluation/" + command + "/evidence/" + artifact, status=status)

        async def analyze(name: str, repeat: int = 1) -> dict:
            command = saved["nonce"] + name + str(repeat)
            if repeat == 1:
                obj = await call("POST", prefix, create_body(command))
                saved[name] = obj["object_id"]
            identifier = saved[name]
            obj = await call("GET", prefix + "/" + identifier)
            synced = await call(
                "POST",
                prefix + "/" + identifier + "/sync",
                {"command_id": command + "sync", "expected_object_version": obj["object_version"]},
            )
            assert synced["status"] == "completed", synced
            obj = await call("GET", prefix + "/" + identifier)
            if repeat == 2:
                assert synced["generation_id"] == saved["generation"]
                revised = await call(
                    "POST",
                    prefix + "/" + identifier + "/revision",
                    {
                        "command_id": command + "revision",
                        "expected_object_version": obj["object_version"],
                        "expected_state_version": obj["state"]["state_version"],
                        "snapshot_id": synced["snapshot_id"],
                        "generation_id": synced["generation_id"],
                        "change": create_body("unused")["change"],
                    },
                )
                assert revised["status"] == "completed", revised
                obj = await call("GET", prefix + "/" + identifier)
                assert obj["state"]["revision"] == 2
            if name == "success":
                saved["generation"] = synced["generation_id"]
            if name in {"savefail", "rejectsave"}:
                artifact = "answer" if name == "savefail" else "validation"
                (private / (canonical_hash([permit.batch_id, command]) + "." + artifact + ".json")).mkdir()
            body = {
                "command_id": command,
                "expected_object_version": obj["object_version"],
                "expected_state_version": obj["state"]["state_version"],
                "snapshot_id": synced["snapshot_id"],
                "generation_id": synced["generation_id"],
                "query": permit.positions[command]["query"],
            }
            saved[name + "body"] = body
            result = await call("POST", prefix + "/" + identifier + "/analyze", body)
            saved[name + "result"] = result
            saved[name + "body"] = body
            if name == "inputbad":
                captured = await evidence(command, "input")
                assert not captured["review_complete"] and captured["input_redacted"]
                assert (await evidence(command, "answer"))["status"] == "not_retained"
                store = V11Store(LocalPostgresStore())
                async with await store.connection.connect() as db:
                    assert not await (
                        await db.execute(
                            "SELECT 1 FROM v1_budget WHERE command_id=%s AND attempt='analysis-1'", (command,)
                        )
                    ).fetchone()
                return result
            if name not in {"savefail", "unknown"}:
                answer = await evidence(command, "answer")
                assert answer["status"] == "unvalidated" and answer["answer"]["observations"]
                if name in {"redact", "truncate"}:
                    assert not answer["review_complete"] and "synthetic-private-example" not in json.dumps(answer)
                else:
                    assert answer["review_complete"]
                if name == "success":
                    captured = await evidence(command, "input")
                    assert captured["review_complete"] and len(captured["user"]) > 4000
                    assert "https://registry.npmjs.org/example" in captured["user"]
                    assert "sha512-" in json.dumps(answer)
                    assert canonical_hash(captured["user"]) == captured["input_hash"]
                    assert canonical_hash(answer["answer"]) == answer["original_parsed_hash"]
                    assert ". 中文（https://" in answer["answer"]["observations"][0]["text"]
            if name not in {"rejectsave"}:
                validation = await evidence(command, "validation")
                assert validation["budget"]["reserved"]
            return result

        if number == 1:
            for repository in ("checkout", "setup-python"):
                await call("POST", prefix + "/repositories/" + repository + "/verify")
            versions = {r["rule_set_id"]: r["version"] for r in (await call("GET", prefix + "/rules"))["items"]}
            for rule in ("application-policy", "checkout-reference"):
                result = await call(
                    "POST",
                    prefix + "/rules/" + rule + "/sync",
                    {"command_id": saved["nonce"] + rule, "expected_object_version": versions[rule]},
                )
                assert result["status"] == "completed", result
            result = await analyze("success")
            assert result["status"] == "completed", result
            result = await analyze("success", 2)
            assert result["status"] == "completed", result
            checks += ["v2_http_single_graph_storage", "same_generation_two_revisions", "complete_answer_and_usage"]
            for artifact in ("sync_timing", "analyze_timing"):
                measured = await evidence(saved["nonce"] + "success1", artifact)
                assert measured["elapsed_seconds"] >= measured["covered_union_seconds"] >= 0
                assert measured["unattributed_seconds"] >= 0
                assert measured["spans"] and not measured["truncated"]
            checks.append("private_monotonic_timing_readback")
            interrupted = asyncio.create_task(analyze("interrupt"))
            try:
                store = V11Store(LocalPostgresStore())
                for _ in range(100):
                    async with await store.connection.connect() as db:
                        row = await (
                            await db.execute(
                                "SELECT 1 FROM v1_budget WHERE command_id=%s AND attempt='analysis-1'",
                                (saved["nonce"] + "interrupt1",),
                            )
                        ).fetchone()
                    if row:
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("synthetic_interruption_reservation_missing")
            finally:
                interrupted.cancel()
                try:
                    await interrupted
                except asyncio.CancelledError:
                    pass
            checks.append("interrupt_after_durable_model_reservation")
        elif number == 2:
            replay = await call(
                "POST", prefix + "/" + saved["interrupt"] + "/analyze", saved["interruptbody"], status=409
            )
            assert replay["code"] == "evaluation_position_consumed_no_resend"
            checks.append("process_interruption_restart_no_resend")
            assert (await evidence(saved["nonce"] + "success2", "answer"))["review_complete"]
            for name in ("rejected", "schema", "redact", "truncate", "inputbad"):
                result = await analyze(name)
                assert result["status"] == "failed", result
            checks += [
                "restart_reads_private_evidence",
                "contract_rejection_complete_answer",
                "schema_failure_complete_answer",
                "redaction_disclosed",
            ]
        else:
            for name in ("unknown", "savefail", "rejectsave"):
                result = await analyze(name)
                assert result["status"] == ("unknown" if name == "unknown" else "failed"), result
                replay = await call("POST", prefix + "/" + saved[name] + "/analyze", saved[name + "body"], status=409)
                assert replay["code"] == "evaluation_position_consumed_no_resend", replay
            assert saved["rejectsaveresult"]["reason"] == saved["rejectedresult"]["reason"]
            other_token = jwt.encode(
                {"sub": "v11-synthetic-session-2", "exp": datetime.now(UTC).timestamp() + 3600},
                settings.JWT_SECRET_KEY,
                algorithm=settings.JWT_ALGORITHM,
            )
            denied = await client.get(
                "/_evaluation/" + saved["nonce"] + "success2/evidence/answer",
                headers={"Authorization": "Bearer " + other_token},
            )
            assert denied.status_code == 404
            source = next(
                s
                for s in (await call("GET", prefix + "/sources"))["items"]
                if s["source_id"] == "review-github-checkout"
            )
            await call(
                "POST",
                prefix + "/sources/review-github-checkout/revoke",
                {"command_id": saved["nonce"] + "revoke", "expected_version": source["version"], "user_id": 2},
            )
            # A changed source grant invalidates the previously captured dependency version.
            await evidence(saved["nonce"] + "success2", "answer", status=404)
            checks += [
                "third_process_recovery",
                "unknown_reservation_no_resend",
                "answer_save_failure_no_resend",
                "diagnostic_save_failure_preserves_original_rejection",
                "foreign_identity_denied",
                "grant_revocation_denies_saved_evidence",
            ]
    return checks


def main() -> int:
    """Only an explicitly named new synthetic DB is used; every child is closed in finally."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=35684)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("report_already_exists")
    _, values = configure(args.compose_command)
    private = args.compose_command.parent / ("evidence-" + uuid4().hex[:10])
    private.mkdir(mode=0o700)
    if sys.platform == "win32":
        acl_script = (
            "$ErrorActionPreference='Stop';$p=$env:V11_NEW_EVIDENCE_DIR;"
            "$sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
            "& icacls $p /grant:r \"*${sid}:(OI)(CI)F\" '*S-1-5-18:(OI)(CI)F';if($LASTEXITCODE){exit 1};"
            "& icacls $p /inheritance:r;if($LASTEXITCODE){exit 1};"
            "& icacls $p /remove:g '*S-1-3-4' '*S-1-5-32-544';exit $LASTEXITCODE"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", acl_script],
            env={**os.environ, "V11_NEW_EVIDENCE_DIR": str(private)},
            check=True,
            capture_output=True,
        )
    nonce = "engineering-" + uuid4().hex[:10]
    positions = {}
    for name in (
        "success",
        "rejected",
        "schema",
        "redact",
        "truncate",
        "inputbad",
        "unknown",
        "savefail",
        "rejectsave",
        "interrupt",
    ):
        for repeat in (1, 2) if name == "success" else (1,):
            positions[nonce + name + str(repeat)] = {
                "query": "token=synthetic-input-secret"
                if name == "inputbad"
                else "probe-rejected"
                if name == "rejectsave"
                else "probe-" + name,
                "binding": create_body("unused")["binding"],
                "case_id": name,
                "sync_command": nonce + name + str(repeat) + "sync",
                "repeat": repeat,
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
    permit = EvidencePermit(
        batch_id=nonce,
        source_hash=code_state()["source_hash"],
        dataset_hash=canonical_hash(positions),
        owner_id=1,
        expires_at=datetime.now(UTC) + timedelta(hours=2),
        mode="synthetic",
        designated_ledger_id="synthetic-offline-only",
        positions=positions,
        ledger_charge_at_start=0,
        maximum_batch_charge=19,
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
    saved = {"nonce": nonce}
    result = {
        "provenance": "new_isolated_pg_http_graph_synthetic_only",
        "real_sends": 0,
        "processes": [],
        "checks": [],
    }
    try:
        for number in (1, 2, 3):
            with (private / f"phase-{number}.private.log").open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "scripts.v11_probe_app", "--port", str(args.port)],
                    env={
                        **os.environ,
                        **values,
                        "V11_SYNTHETIC_EVIDENCE_DIR": str(private),
                        "V11_COMPLETE_EVIDENCE_PROBE": "1",
                        "LOCAL_MVP_ONLY": "true",
                    },
                    stdout=log,
                    stderr=log,
                )
                result["processes"].append(process.pid)
                try:
                    ready(args.port)
                    with asyncio.Runner(
                        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None
                    ) as runner:
                        result["checks"].extend(runner.run(phase(args.port, number, saved, private, permit)))
                finally:
                    process.terminate()
                    process.wait(timeout=20)
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            result["budget"] = runner.run(budgets())
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
