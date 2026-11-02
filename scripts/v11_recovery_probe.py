"""Fresh synthetic PostgreSQL recovery CAS verification; no real runtime is accepted."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from psycopg.types.json import Jsonb
from rich.console import Console

from app.schemas.local_mvp import LocalActor
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_recovery_guard import MARKER, STAMPS
from app.services.v11_storage import V11Store
from app.services.v11_evaluation_policy import LINK_CONTRACT, FORMAL_BATCH
from evals.v11_backend.recovery import Manifest, activate, stop, transition
from scripts.v11_probe import configure
from scripts.v11_probe_fixtures import seed

ACTOR = LocalActor(user_id=1, session_id="v11-synthetic-session-1", expires_at=4102444800)


async def probe() -> dict:
    """All inserted history is explicitly synthetic in this newly provisioned database."""
    store = V11Store(LocalPostgresStore())
    store.evaluation_recovery_boot_id = "synthetic-process-one"
    await seed(store)
    async with await store.connection.connect() as db, db.transaction():
        if await (await db.execute("SELECT 1 FROM v1_commands LIMIT 1")).fetchone():
            raise ValueError("empty_new_synthetic_database_required")
        for source in ("markdown-rules", "review-github-checkout", "review-github-setup-python"):
            await db.execute("INSERT INTO v1_sources(source_id,owner_id,acl) VALUES(%s,1,'[1]')", (source,))
        for stamp in STAMPS[:2]:
            for prefix in ("sync:", "analysis:", "rag:"):
                await db.execute(
                    "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,result) "
                    "VALUES(1,%s,%s,'synthetic-retained','completed',%s)",
                    (prefix + stamp, canonical_hash(stamp), Jsonb({"status": "completed"})),
                )
            for attempt in ("analysis-1", "query-1-batch-0"):
                await db.execute(
                    "INSERT INTO v1_budget(owner_id,command_id,attempt,reserved,charge,usage) "
                    "VALUES(1,%s,%s,0.001,0.001,'{\"synthetic\":true}')",
                    ("analysis:" + stamp, attempt),
                )
    manifest = Manifest(
        old_report_sha256="a" * 64,
        supplemental_sha256="b" * 64,
        supplemental_hash="b" * 64,
        retained_hashes=("c" * 64, "d" * 64),
        remaining=STAMPS[2:],
        compatibility_hash="e" * 64,
        retained_result_hashes={
            prefix + s: canonical_hash({"status": "completed"})
            for s in STAMPS[:2]
            for prefix in ("sync:", "analysis:", "rag:")
        },
    )
    checks = []

    async def outcome(call) -> str:
        try:
            await call
            return "ok"
        except DomainError as error:
            return error.code

    async def start() -> None:
        await activate(store, ACTOR, manifest, datetime.now(UTC) + timedelta(hours=1))

    async with await store.connection.connect() as db:
        await db.execute(
            "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,contract_version,result) "
            "VALUES(1,'synthetic-orphan-overlap',%s,'synthetic','completed',%s,%s)",
            (canonical_hash("synthetic orphan link"), LINK_CONTRACT, Jsonb({"batch_id": FORMAL_BATCH})),
        )
    checks.append(
        {
            "case": "orphan_overlap_link_blocks_activation",
            "passed": await outcome(start()) == "recovery_overlap_link_already_consumed",
        }
    )
    async with await store.connection.connect() as db:
        await db.execute("DELETE FROM v1_commands WHERE command_id='synthetic-orphan-overlap'")
    # Deliberately committed synthetic orphan: prove the live tables override a clean manifest.
    async with await store.connection.connect() as db:
        await db.execute(
            "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,result) "
            "VALUES(1,%s,%s,'synthetic-orphan','started','{\"claimed\":true}')",
            ("analysis:" + STAMPS[2], canonical_hash("synthetic orphan")),
        )
        await db.execute(
            "INSERT INTO v1_budget(owner_id,command_id,attempt,reserved,charge) VALUES(1,%s,'analysis-1',0.01,0.01)",
            ("analysis:" + STAMPS[2],),
        )
    checks.append(
        {
            "case": "persisted_orphan_and_budget_block_activation",
            "passed": await outcome(start()) == "recovery_remaining_position_consumed_or_uncertain",
        }
    )
    # This is a new synthetic fixture only, never an operation against retained real history.
    async with await store.connection.connect() as db, db.transaction():
        await db.execute("DELETE FROM v1_budget WHERE command_id=%s", ("analysis:" + STAMPS[2],))
        await db.execute("DELETE FROM v1_commands WHERE command_id=%s", ("analysis:" + STAMPS[2],))
        await db.execute("UPDATE v1_sources SET acl='[]' WHERE source_id='markdown-rules'")
    checks.append({"case": "revocation_blocks_activation", "passed": await outcome(start()) == "source_not_found"})
    async with await store.connection.connect() as db:
        await db.execute("UPDATE v1_sources SET acl='[1]' WHERE source_id='markdown-rules'")
    checks.append(
        {
            "case": "expiry_rejected",
            "passed": await outcome(activate(store, ACTOR, manifest, datetime.now(UTC) - timedelta(seconds=1)))
            == "recovery_eight_hour_limit",
        }
    )
    bad = manifest.model_copy(update={"retained_result_hashes": {}})
    checks.append(
        {
            "case": "retained_database_result_mismatch",
            "passed": await outcome(activate(store, ACTOR, bad, datetime.now(UTC) + timedelta(hours=1)))
            == "recovery_retained_result_changed",
        }
    )
    values = await asyncio.gather(*(outcome(start()) for _ in range(8)))
    checks.append(
        {
            "case": "concurrent_activation_single_winner",
            "passed": values.count("ok") == 1 and values.count("recovery_already_claimed_no_restart") == 7,
        }
    )
    checks.append(
        {"case": "restart_rejected", "passed": await outcome(start()) == "recovery_already_claimed_no_restart"}
    )
    checks.append(
        {
            "case": "old_permit_claim_rejected",
            "passed": await outcome(store.claim_review(ACTOR, "analysis:" + STAMPS[2], "x", "synthetic", "acceptance"))
            == "recovery_old_permit_retired_or_stopped",
        }
    )
    store.evaluation_recovery_hash = canonical_hash(manifest)
    store.evaluation_recovery_boot_id = "synthetic-restarted-server"
    checks.append(
        {
            "case": "server_restart_cannot_use_pending_permission",
            "passed": await outcome(store.claim_review(ACTOR, "analysis:" + STAMPS[2], "x", "synthetic", "acceptance"))
            == "recovery_server_restarted_no_resend",
        }
    )
    store.evaluation_recovery_boot_id = "synthetic-process-one"
    for stamp in STAMPS[:2]:
        checks.append(
            {
                "case": "retained_never_reserved",
                "passed": await outcome(
                    store.reserve(ACTOR, "analysis:" + stamp, "analysis-1", Decimal("0.001"), Decimal("100"))
                )
                == "recovery_consumed_or_unlisted_position",
            }
        )
    checks.append(
        {
            "case": "future_position_not_active",
            "passed": await outcome(store.claim_review(ACTOR, "analysis:" + STAMPS[3], "x", "synthetic", "acceptance"))
            == "recovery_position_not_claimed",
        }
    )
    values = await asyncio.gather(*(outcome(transition(store, ACTOR, manifest, STAMPS[2])) for _ in range(8)))
    checks.append(
        {
            "case": "concurrent_position_single_winner",
            "passed": values.count("ok") == 1 and values.count("recovery_orphan_position_no_restart") == 7,
        }
    )
    fresh = V11Store(LocalPostgresStore())
    checks.append(
        {
            "case": "new_connection_orphan_retained",
            "passed": await outcome(transition(fresh, ACTOR, manifest, STAMPS[2]))
            == "recovery_orphan_position_no_restart",
        }
    )
    checks.append(
        {
            "case": "cannot_finish_missing_response",
            "passed": await outcome(transition(store, ACTOR, manifest, STAMPS[2], completed_hash="f" * 64))
            == "recovery_position_not_completed",
        }
    )
    async with await store.connection.connect() as db, db.transaction():
        for prefix in ("sync:", "analysis:", "rag:"):
            await db.execute(
                "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,result) "
                "VALUES(1,%s,%s,'synthetic-completion','completed','{\"status\":\"completed\"}')",
                (prefix + STAMPS[2], canonical_hash(prefix)),
            )
    checks.append(
        {
            "case": "completed_commands_without_usage_blocked",
            "passed": await outcome(transition(store, ACTOR, manifest, STAMPS[2], completed_hash="f" * 64))
            == "recovery_attempts_incomplete",
        }
    )
    async with await store.connection.connect() as db, db.transaction():
        for prefix, attempts in (("analysis:", ("analysis-1", "query-1-batch-0")), ("rag:", ("query-1-batch-0",))):
            for attempt in attempts:
                await db.execute(
                    "INSERT INTO v1_budget(owner_id,command_id,attempt,reserved,charge,usage) "
                    "VALUES(1,%s,%s,0.001,0,'{\"synthetic\":true}')",
                    (prefix + STAMPS[2], attempt),
                )
    await transition(store, ACTOR, manifest, STAMPS[2], completed_hash="f" * 64)
    checks.append(
        {
            "case": "completed_position_cannot_begin_again",
            "passed": await outcome(transition(store, ACTOR, manifest, STAMPS[2])) == "recovery_position_out_of_order",
        }
    )
    await transition(store, ACTOR, manifest, STAMPS[3])
    await stop(store, ACTOR, manifest)
    checks.append(
        {
            "case": "stop_does_not_release",
            "passed": await outcome(transition(store, ACTOR, manifest, STAMPS[2])) == "recovery_stopped_or_expired",
        }
    )
    async with await store.connection.connect() as db:
        saved = await (await db.execute("SELECT result FROM v1_commands WHERE command_id=%s", (MARKER,))).fetchone()
        totals = await (await db.execute("SELECT count(*) AS n,sum(charge) AS charge FROM v1_budget")).fetchone()
    checks.append(
        {
            "case": "orphan_and_original_budget_unchanged",
            "passed": saved["result"]["active_stamp"] == STAMPS[3]
            and totals["n"] == 7
            and totals["charge"] == Decimal("0.004"),
        }
    )
    return {
        "provenance": "new_isolated_synthetic_pg",
        "real_sends": 0,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }


def main() -> int:
    """Only a verified new stack descriptor is accepted; safe errors omit environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--restart-check", action="store_true")
    args = parser.parse_args()
    configure(args.compose_command)
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        result = runner.run(restart_probe() if args.restart_check else probe())
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    Console().print({"passed": result["passed"], "checks": len(result["checks"]), "real_sends": 0})
    return 0 if result["passed"] else 1


async def restart_probe() -> dict:
    """A fresh Python process cannot treat a persisted stopped/orphan slot as unused."""
    store = V11Store(LocalPostgresStore())
    store.evaluation_recovery_boot_id = "synthetic-process-one"
    async with await store.connection.connect() as db:
        row = await (await db.execute("SELECT result FROM v1_commands WHERE command_id=%s", (MARKER,))).fetchone()
    if not row:
        raise ValueError("synthetic_prior_phase_required")
    manifest = Manifest.model_validate(row["result"]["manifest"])
    codes = []
    for operation in (
        activate(store, ACTOR, manifest, datetime.now(UTC) + timedelta(hours=1)),
        transition(store, ACTOR, manifest, STAMPS[2]),
    ):
        try:
            await operation
            codes.append("unexpected_success")
        except DomainError as error:
            codes.append(error.code)
    checks = [
        {
            "case": "independent_process_restart_stays_blocked",
            "passed": codes == ["recovery_already_claimed_no_restart", "recovery_stopped_or_expired"],
        }
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks, "real_sends": 0}


if __name__ == "__main__":
    raise SystemExit(main())
