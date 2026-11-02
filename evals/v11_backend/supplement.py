"""Fixed B acceptance supplement: a linked exception, never a business retry."""

from decimal import Decimal

from psycopg.types.json import Jsonb

from app.schemas.local_mvp import LocalActor
from app.schemas.v11_backend import CONTRACT
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_storage import V11Store

EVALUATION_ID = "v11-contract2-B-acceptance60-01"
COMMAND_ID = "eval:v11-contract2-B-acceptance60-01"
OLD_COMMANDS = (
    "eval:v11-d1-contract2-behavior-01",
    "smoke-analysis-c77c6d75ff2e4ec8e8351cc1",
)
RESERVATION = Decimal("0.130804")


async def claim(store: V11Store, actor: LocalActor, resource: str, digest: str, baseline: Decimal) -> dict:
    """Atomically consume the sole exception with links before any Provider reservation."""
    async with await store.connection.connect() as db, db.transaction():
        await db.execute("SELECT pg_advisory_xact_lock(71800801)")
        await store.object_row(resource, actor, db, lock=True)
        old = await (
            await db.execute(
                "SELECT command_id,resource,result FROM v1_commands WHERE owner_id=%s "
                "AND command_id=ANY(%s) ORDER BY command_id FOR SHARE",
                (actor.user_id, list(OLD_COMMANDS)),
            )
        ).fetchall()
        if len(old) != 2 or any(r["resource"] != resource or r["result"].get("status") != "unknown" for r in old):
            raise DomainError("evaluation_old_unknown_link_mismatch")
        used = await (
            await db.execute(
                "SELECT 1 FROM v1_commands WHERE command_id=%s UNION ALL SELECT 1 FROM v1_budget WHERE command_id=%s",
                (COMMAND_ID, COMMAND_ID),
            )
        ).fetchone()
        if used:
            raise DomainError("evaluation_consumed_no_resend")
        total = await (
            await db.execute(
                "SELECT (SELECT COALESCE(sum(charge),0) FROM v1_budget) + "
                "(SELECT COALESCE(sum(total_charge),0) FROM v11_budget_carryovers) AS amount"
            )
        ).fetchone()
        if not total or total["amount"] != baseline:
            raise DomainError("evaluation_acceptance_baseline_changed")
        binding = {
            "evaluation_id": EVALUATION_ID,
            "not_business_analysis": True,
            "no_business_adoption": True,
            "old_commands": list(OLD_COMMANDS),
            "old_resource": resource,
            "acceptance_increment_baseline": str(baseline),
            "acceptance_increment_limit": "19",
            "acceptance_total_ceiling": str(baseline + Decimal("19")),
            "supplement_reservation_limit": str(RESERVATION),
            "call_timeout_seconds": 60,
            "command_timeout_seconds": 120,
        }
        await db.execute(
            "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,contract_version,"
            "budget_category,intent_hash,result) VALUES(%s,%s,%s,%s,'started',%s,'acceptance',%s,%s)",
            (actor.user_id, COMMAND_ID, digest, resource, CONTRACT, digest, Jsonb({"claimed": True, **binding})),
        )
        await store.event(db, actor, COMMAND_ID, resource, "acceptance_supplement_linked", binding)
    # Reserve still checks all original caps atomically. A stricter caller ceiling is preserved.
    ceiling = baseline + RESERVATION
    store.evaluation_ceiling = min(store.evaluation_ceiling, ceiling) if store.evaluation_ceiling else ceiling
    return binding
