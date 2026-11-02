"""Isolated actual PG/HTTP verification; real mode is separately authorized."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import httpx
from jose import jwt
from rich.console import Console
from tenacity import retry, retry_if_exception_type, stop_after_delay, wait_exponential
from psycopg.errors import ForeignKeyViolation

from app.core.config import settings
from app.schemas.local_mvp import LocalActor
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_storage import V11Store
from app.services.v11_policy import CAPABILITIES
from scripts.v11_probe_fixtures import create_body, seed
from scripts.v11_stack import configuration
from evals.v11_backend.contracts import Dataset
from evals.v11_backend.real import Authorization, sample


def configure(path: Path) -> tuple[list[str], dict[str, str]]:
    """Only this explicitly verified new test database can receive synthetic writes."""
    command, values = configuration(path)
    settings.POSTGRES_HOST = values["POSTGRES_HOST"]
    settings.POSTGRES_PORT = int(values["POSTGRES_PORT"])
    settings.POSTGRES_DB = values["POSTGRES_DB"]
    settings.POSTGRES_USER = values["POSTGRES_USER"]
    settings.POSTGRES_PASSWORD = values["POSTGRES_PASSWORD"]
    settings.JWT_SECRET_KEY = values["JWT_SECRET_KEY"]
    settings.LANGFUSE_TRACING_ENABLED = False
    return command, values


async def pg_probe() -> dict:
    """Concurrency and conservative budget arithmetic use actual PostgreSQL locks."""
    store = V11Store(LocalPostgresStore())
    await seed(store)
    actor = LocalActor(user_id=1, session_id="v11-synthetic-session-1", expires_at=4102444800)
    nonce = "v11-pg:" + uuid4().hex
    checks = []

    async def claim() -> str:
        try:
            value = await store.claim_review(actor, nonce, canonical_hash(nonce), "synthetic-pg", "acceptance")
            return "original" if value is None else "replay"
        except DomainError as error:
            return error.code

    values = await asyncio.gather(*(claim() for _ in range(8)))
    checks.append(
        {
            "case": "eight_concurrent_command_claims",
            "passed": values.count("original") == 1 and values.count("provider_command_outcome_unknown") == 7,
        }
    )
    checks.append({"case": "persisted_unknown_claim", "passed": await claim() == "provider_command_outcome_unknown"})
    intent = canonical_hash([nonce, "same-unknown-input"])

    async def new_key(number: int) -> str:
        try:
            await store.claim_review(
                actor,
                nonce + ":intent:" + str(number),
                canonical_hash([intent, number]),
                "synthetic-shared-input",
                "acceptance",
                intent,
            )
            return "claimed"
        except DomainError as error:
            return error.code

    intents = await asyncio.gather(*(new_key(number) for number in range(8)))
    checks.append(
        {
            "case": "eight_new_keys_same_uncertain_intent",
            "passed": intents.count("claimed") == 1 and intents.count("provider_intent_unknown_hold") == 7,
        }
    )
    try:
        await store.claim_review(actor, nonce, canonical_hash("different object"), "different", "acceptance")
        conflict = False
    except DomainError as error:
        conflict = error.code == "idempotency_conflict"
    checks.append({"case": "cross_object_command_namespace", "passed": conflict})
    async with await store.connection.connect() as db:
        row = await (await db.execute("SELECT COALESCE(SUM(charge),0) AS spent FROM v1_budget")).fetchone()
        limit = row["spent"] + Decimal("0.00000002")

    async def reserve(number: int) -> str:
        command = nonce + ":" + str(number)
        await store.claim_review(actor, command, canonical_hash(command), "synthetic-budget", "acceptance")
        try:
            await store.reserve(actor, command, "synthetic-no-send", Decimal("0.00000001"), limit)
            return "reserved"
        except DomainError as error:
            return error.code

    values = await asyncio.gather(*(reserve(number) for number in range(8)))
    checks.append(
        {
            "case": "atomic_cumulative_budget",
            "passed": values.count("reserved") == 2 and values.count("model_budget_exhausted") == 6,
        }
    )
    async with await store.connection.connect() as db:
        revision = await (await db.execute("SELECT version_num FROM alembic_version")).fetchone()
        checks.append({"case": "additive_migration_head", "passed": revision["version_num"] == "v11_backend_01"})
        scoped = await (
            await db.execute(
                "SELECT object_id,active_snapshot_id,active_generation_id FROM v11_objects WHERE active_generation_id IS NOT NULL ORDER BY object_id LIMIT 2"
            )
        ).fetchall()
        if len(scoped) == 2:
            rejected = False
            try:
                async with db.transaction():
                    await db.execute(
                        "UPDATE v11_objects SET active_generation_id=%s WHERE object_id=%s",
                        (scoped[1]["active_generation_id"], scoped[0]["object_id"]),
                    )
                    raise AssertionError("cross_object_generation_accepted")
            except ForeignKeyViolation:
                rejected = True
            checks.append({"case": "database_rejects_cross_object_generation", "passed": rejected})
    return {
        "mode": "pg",
        "provenance": "actual_isolated_postgresql_synthetic_inputs_no_provider_sends",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


@retry(
    stop=stop_after_delay(40),
    wait=wait_exponential(multiplier=0.25, max=2),
    retry=retry_if_exception_type((httpx.HTTPError, AssertionError)),
    reraise=True,
)
def ready(port: int) -> None:
    """Wait only on the loopback child API readiness endpoint."""
    with httpx.Client(trust_env=False, timeout=3) as client:
        response = client.get(f"http://127.0.0.1:{port}/probe-ready")
        response.raise_for_status()
        assert response.json()["ready"]


async def http_phase(port: int, phase: int, saved: dict) -> tuple[dict, list[dict]]:
    """Each phase runs against a different API process with the same isolated facts."""
    checks = []
    prefix = "/api/v1/releaseguard/reviews"

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=120) as client:

        async def call(method: str, path: str, value: dict | None = None, actor: int = 1, expected: int = 200) -> dict:
            if value and "command_id" in value:
                value = {**value, "command_id": value["command_id"] + "-" + saved["nonce"]}
            if value and "idempotency_key" in value:
                value = {**value, "idempotency_key": value["idempotency_key"] + "-" + saved["nonce"]}
            token = jwt.encode(
                {"sub": "v11-synthetic-session-" + str(actor), "exp": datetime.now(UTC).timestamp() + 3600},
                settings.JWT_SECRET_KEY,
                algorithm=settings.JWT_ALGORITHM,
            )
            response = await client.request(
                method, prefix + path, json=value, headers={"Authorization": "Bearer " + token}
            )
            if response.status_code != expected:
                raise AssertionError(f"http_{response.status_code}_{method}_{path}:" + response.text[:400])
            return response.json()

        async def view(name: str) -> dict:
            return await call("GET", "/" + saved[name])

        async def analyze(
            name: str, query: str = "Describe the actual changed line.", command_id: str | None = None
        ) -> dict:
            current = await view(name)
            return await call(
                "POST",
                "/" + saved[name] + "/analyze",
                {
                    "command_id": command_id or "analysis-" + name,
                    "expected_object_version": current["object_version"],
                    "expected_state_version": current["state"]["state_version"],
                    "snapshot_id": current["active_snapshot_id"],
                    "generation_id": current["active_generation_id"],
                    "query": query,
                },
            )

        if phase == 1:
            for repository in ("checkout", "setup-python"):
                await call("POST", "/repositories/" + repository + "/verify")
            versions = {row["rule_set_id"]: row["version"] for row in (await call("GET", "/rules"))["items"]}
            for rule in ("application-policy", "checkout-reference"):
                result = await call(
                    "POST",
                    "/rules/" + rule + "/sync",
                    {"command_id": "rule-" + rule, "expected_object_version": versions[rule]},
                )
                assert result["status"] == "completed", result
            for name, repository, selector in (
                ("pr", "checkout", {"kind": "pr", "pr_number": 1}),
                ("failed-ci", "checkout", {"kind": "pr", "pr_number": 2}),
                ("commit", "setup-python", {"kind": "commit", "commit_sha": "a" * 40}),
                ("unknown", "checkout", {"kind": "commit", "commit_sha": "a" * 40}),
            ):
                current = await call("POST", "", create_body("v11-http-" + name, repository, selector))
                saved[name] = current["object_id"]
                result = await call(
                    "POST",
                    "/" + current["object_id"] + "/sync",
                    {"command_id": "sync-" + name, "expected_object_version": current["object_version"]},
                )
                assert result["status"] == "completed", result
                result = await analyze(
                    name, "probe-unknown" if name == "unknown" else "Describe the actual changed line."
                )
                assert result["status"] == ("unknown" if name == "unknown" else "completed"), result
                saved[name + "-analysis"] = result["analysis_id"]
            checks.append(
                {
                    "case": "multi_repository_multi_object_ci_orthogonality",
                    "passed": (await view("failed-ci"))["analysis_status"] == "completed"
                    and not (await view("failed-ci"))["rehearsal_eligibility"]["eligible"],
                }
            )
            current = await view("pr")
            result = await call(
                "POST",
                "/" + saved["pr"] + "/prepare",
                {
                    "command_id": "prepare-pr",
                    "expected_object_version": current["object_version"],
                    "expected_state_version": current["state"]["state_version"],
                    "analysis_id": saved["pr-analysis"],
                },
            )
            assert result["status"] == "blocked", result
            checks.append(
                {
                    "case": "limited_prepare_paused_before_gate",
                    "passed": result["reason"] == CAPABILITIES.pause_reason
                    and not current["rehearsal_eligibility"]["eligible"]
                    and current["capabilities"] == CAPABILITIES.model_dump(mode="json"),
                }
            )
        elif phase == 2:
            current = await view("pr")
            result = await call(
                "POST",
                "/" + saved["pr"] + "/approval",
                {
                    "command_id": "approval-pr",
                    "expected_object_version": current["object_version"],
                    "expected_state_version": current["state"]["state_version"],
                    "approval_request_id": "approval:synthetic-paused",
                    "authorization_hash": "a" * 64,
                    "decision": "APPROVED",
                    "reason_code": "synthetic_probe_reviewed",
                },
                actor=2,
            )
            assert result["status"] == "blocked" and result["reason"] == CAPABILITIES.pause_reason, result
            current = await view("pr")
            result = await call(
                "POST",
                "/" + saved["pr"] + "/run",
                {
                    "command_id": "run-pr",
                    "expected_object_version": current["object_version"],
                    "expected_state_version": current["state"]["state_version"],
                },
            )
            assert result["status"] == "blocked" and result["reason"] == CAPABILITIES.pause_reason, result
            receipt = await call("GET", "/" + saved["pr"] + "/receipt")
            checks.append(
                {
                    "case": "restart_approval_run_paused_zero_target_writes",
                    "passed": receipt["target_write_calls"] == 0 and receipt["receipt"] is None,
                }
            )
            same_key = await analyze("unknown", "probe-unknown")
            checks.append({"case": "restart_same_key_unknown_no_resend", "passed": same_key["status"] == "unknown"})
            current = await view("unknown")
            result = await call(
                "POST",
                "/" + saved["unknown"] + "/analyze",
                {
                    "command_id": "new-key-unknown",
                    "expected_object_version": current["object_version"],
                    "expected_state_version": current["state"]["state_version"],
                    "snapshot_id": current["active_snapshot_id"],
                    "generation_id": current["active_generation_id"],
                    "query": "probe-unknown",
                },
            )
            checks.append(
                {
                    "case": "restart_new_key_unknown_zero_resend",
                    "passed": result.get("reason") == "analysis_unknown_hold",
                }
            )
            denied = await call("GET", "/" + saved["pr"], actor=3, expected=404)
            checks.append(
                {
                    "case": "unrelated_subject_no_identity_or_report",
                    "passed": denied["code"] == "local_object_not_found",
                }
            )
            for repository in ("checkout", "setup-python"):
                name = "branch-" + repository
                current = await call(
                    "POST",
                    "",
                    create_body(
                        name, repository, {"kind": "branch_compare", "base_branch": "main", "head_branch": "topic"}
                    ),
                )
                saved[name] = current["object_id"]
                result = await call(
                    "POST", "/" + saved[name] + "/sync", {"command_id": "sync-" + name, "expected_object_version": 0}
                )
                assert result["status"] == "completed", result
                result = await analyze(name)
                assert result["status"] == "completed", result
                checks.append(
                    {
                        "case": name + "_readonly",
                        "passed": "object_type_readonly"
                        in (await view(name))["rehearsal_eligibility"]["reason_codes"],
                    }
                )
        else:
            receipt = await call("GET", "/" + saved["pr"] + "/receipt")
            checks.append(
                {
                    "case": "third_process_paused_run_has_no_receipt",
                    "passed": receipt["target_write_calls"] == 0 and receipt["receipt"] is None,
                }
            )
            for command_name, owner in (("prepare-pr", 1), ("approval-pr", 2), ("run-pr", 1)):
                absent = await call(
                    "GET",
                    "/" + saved["pr"] + "/commands/" + command_name + "-" + saved["nonce"],
                    actor=owner,
                    expected=404,
                )
                checks.append(
                    {
                        "case": command_name + "_paused_without_claim",
                        "passed": absent["code"] == "local_object_not_found",
                    }
                )
            current = await view("pr")
            sid = current["active_snapshot_id"]
            history = await call("GET", "/" + saved["pr"] + "/snapshots")
            files = await call("GET", "/" + saved["pr"] + "/snapshots/" + sid + "/files")
            fid = files["items"][0]["file_id"]
            actual_file = await call("GET", "/" + saved["pr"] + "/snapshots/" + sid + "/files/" + fid)
            report = await call("GET", "/" + saved["pr"] + "/analyses/" + saved["pr-analysis"])
            checks.append(
                {
                    "case": "snapshot_diff_report_history_and_experimental_metadata",
                    "passed": bool(history["items"])
                    and bool(actual_file["hunks"])
                    and bool(report["citations"])
                    and report["status"] == "completed"
                    and report["capabilities"] == CAPABILITIES.model_dump(mode="json")
                    and CAPABILITIES.pause_reason in report["rehearsal_eligibility"]["reason_codes"],
                }
            )
            foreign_snapshot = await call("GET", "/" + saved["commit"] + "/snapshots/" + sid, expected=404)
            checks.append(
                {
                    "case": "cross_object_snapshot_history_denied",
                    "passed": foreign_snapshot["code"] == "local_object_not_found",
                }
            )
            diagnostic = await call("GET", "/" + saved["unknown"] + "/commands/analysis-unknown-" + saved["nonce"])
            checks.append(
                {
                    "case": "unknown_reservation_retained",
                    "passed": any(
                        attempt["attempt"] == "analysis-1" and Decimal(str(attempt["charge"])) > 0
                        for attempt in diagnostic["attempts"]
                    ),
                }
            )
            listing = await call("GET", "?limit=2")
            checks.append(
                {"case": "bounded_keyset_page", "passed": len(listing["items"]) == 2 and bool(listing["next_cursor"])}
            )
            legacy = await client.get(
                "/api/v1/releaseguard/changes/" + saved["pr"],
                headers={
                    "Authorization": "Bearer "
                    + jwt.encode(
                        {"sub": "v11-synthetic-session-1", "exp": 4102444800},
                        settings.JWT_SECRET_KEY,
                        algorithm=settings.JWT_ALGORITHM,
                    )
                },
            )
            checks.append({"case": "legacy_contract_excludes_v3", "passed": legacy.status_code == 404})
            before_b = await view("commit")
            current = await view("failed-ci")
            unchanged = await call(
                "POST",
                "/" + saved["failed-ci"] + "/sync",
                {"command_id": "same-code-new-observation", "expected_object_version": current["object_version"]},
            )
            after_a, after_b = await view("failed-ci"), await view("commit")
            checks.append(
                {
                    "case": "same_code_reuses_generation_other_object_unchanged",
                    "passed": unchanged["status"] == "completed"
                    and current["active_generation_id"] == after_a["active_generation_id"]
                    and all(
                        before_b[key] == after_b[key]
                        for key in (
                            "object_version",
                            "active_snapshot_id",
                            "active_generation_id",
                            "state",
                            "analysis_status",
                        )
                    ),
                }
            )
            for name in ("failed-ci", "commit", "branch-checkout", "branch-setup-python"):
                current = await view(name)
                blocked = await call(
                    "POST",
                    "/" + saved[name] + "/prepare",
                    {
                        "command_id": "deny-prepare-" + name,
                        "expected_object_version": current["object_version"],
                        "expected_state_version": current["state"]["state_version"],
                        "analysis_id": saved.get(name + "-analysis"),
                    },
                )
                checks.append({"case": name + "_prepare_denied", "passed": blocked["status"] == "blocked"})
            reviewer_page = await call("GET", "?limit=1", actor=2)
            source_rows = (await call("GET", "/sources"))["items"]
            source = next(row for row in source_rows if row["source_id"] == "review-github-checkout")
            await call(
                "POST",
                "/sources/review-github-checkout/revoke",
                {"command_id": "revoke-reviewer", "expected_version": source["version"], "user_id": 2},
            )
            denied_report = await call(
                "GET", "/" + saved["pr"] + "/analyses/" + saved["pr-analysis"], actor=2, expected=404
            )
            denied_cursor = await call("GET", "?limit=1&cursor=" + reviewer_page["next_cursor"], actor=2, expected=409)
            checks.append(
                {
                    "case": "revocation_blocks_saved_report_and_cursor",
                    "passed": denied_report["code"] == "local_object_not_found"
                    and denied_cursor["code"] == "cursor_invalid_or_expired",
                }
            )
            source = next(
                row for row in (await call("GET", "/sources"))["items"] if row["source_id"] == "review-github-checkout"
            )
            await call(
                "POST",
                "/sources/review-github-checkout/grant",
                {"command_id": "restore-reviewer", "expected_version": source["version"], "user_id": 2},
            )
            current = await view("unknown")
            revised = await call(
                "POST",
                "/" + saved["unknown"] + "/revision",
                {
                    "command_id": "unknown-new-revision",
                    "expected_object_version": current["object_version"],
                    "expected_state_version": current["state"]["state_version"],
                    "snapshot_id": current["active_snapshot_id"],
                    "generation_id": current["active_generation_id"],
                    "change": create_body("unused")["change"],
                },
            )
            assert revised["status"] == "completed", revised
            held = await analyze("unknown", "probe-unknown", "unknown-after-revision")
            checks.append(
                {"case": "unknown_new_revision_zero_resend", "passed": held.get("reason") == "analysis_unknown_hold"}
            )
            current = await call(
                "POST",
                "",
                create_body("concurrent-analysis", "setup-python", {"kind": "commit", "commit_sha": "a" * 40}),
            )
            concurrent_id = current["object_id"]
            result = await call(
                "POST", "/" + concurrent_id + "/sync", {"command_id": "concurrent-sync", "expected_object_version": 0}
            )
            assert result["status"] == "completed", result
            current = await call("GET", "/" + concurrent_id)
            results = await asyncio.gather(
                *(
                    call(
                        "POST",
                        "/" + concurrent_id + "/analyze",
                        {
                            "command_id": "concurrent-" + str(number),
                            "expected_object_version": current["object_version"],
                            "expected_state_version": current["state"]["state_version"],
                            "snapshot_id": current["active_snapshot_id"],
                            "generation_id": current["active_generation_id"],
                            "query": "Describe the actual changed line.",
                        },
                    )
                    for number in range(8)
                )
            )
            diagnostics = [
                await call("GET", "/" + concurrent_id + "/commands/concurrent-" + str(number) + "-" + saved["nonce"])
                for number in range(8)
            ]
            checks.append(
                {
                    "case": "eight_concurrent_analysis_single_model_attempt",
                    "passed": sum(result["status"] == "completed" for result in results) == 1
                    and sum(
                        attempt["attempt"] == "analysis-1"
                        for diagnostic in diagnostics
                        for attempt in diagnostic["attempts"]
                    )
                    == 1,
                }
            )
        return saved, checks


def http_probe(values: dict[str, str], private: Path, port: int) -> dict:
    """Spawn three fresh API processes; prior Python memory cannot restore authority."""
    saved: dict = {"nonce": uuid4().hex[:10]}
    checks = []
    for phase in (1, 2, 3):
        with (private / f"http-phase-{phase}.private.log").open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "scripts.v11_probe_app", "--port", str(port)],
                env={**os.environ, **values},
                stdout=log,
                stderr=log,
            )
            try:
                ready(port)
                saved, current = asyncio.run(http_phase(port, phase, saved))
                checks.extend(current)
                (private / "http-progress.json").write_text(
                    json.dumps({"phase": phase, "saved": saved, "checks": checks}, indent=2), encoding="utf-8"
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    return {
        "mode": "http",
        "api_processes": 3,
        "provenance": "real_http_pg_graph_synthetic_sources_and_model",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def main() -> int:
    """Reports preserve failures and never overwrite a previous validation result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pg", "http", "real"], required=True)
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=35439)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("report_already_exists")
    if args.mode == "real":
        # Validate the isolated target descriptor without switching to synthetic settings or writing DB facts.
        _, real_values = configuration(args.compose_command, offline=False)
        if args.authorization_file is None or args.dataset is None or args.repeat != 2:
            raise ValueError("explicit_authorization_frozen_dataset_repeat_two_required")
        auth = Authorization.model_validate_json(args.authorization_file.read_text(encoding="utf-8"))
        if auth.database_port != int(real_values["POSTGRES_PORT"]) or auth.database_name != real_values["POSTGRES_DB"]:
            raise ValueError("designated_database_mismatch")
        dataset = Dataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
        result = asyncio.run(sample(dataset, auth, args.report))
        Console().print(
            {
                "positions": len(result["positions"]),
                "completed": result["completed"],
                "quality": "requires_separate_semantic_review",
            }
        )
        return 0 if result["completed"] else 1
    _, values = configure(args.compose_command)
    try:
        if args.mode == "pg":
            with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
                result = runner.run(pg_probe())
        else:
            result = http_probe(values, args.compose_command.parent, args.port)
    except Exception as error:
        (args.compose_command.parent / (args.report.stem + ".private.log")).write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        result = {
            "mode": args.mode,
            "passed": False,
            "failure_type": type(error).__name__,
            "details": str(error)[:600] if isinstance(error, (DomainError, AssertionError)) else "private_logs_only",
            "real_calls": 0,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Console().print(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
