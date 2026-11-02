"""Durable, evidence3-only retirement guard; no retries or Provider implementation."""

from datetime import UTC, datetime
from typing import Any

from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_evaluation_policy import DATASET_HASH, FORMAL_BATCH

RECOVERY_ID = "v11-evidence3-recovery-01"
MARKER = "recovery:" + FORMAL_BATCH
VERSION = "v11-evidence3-recovery.1"
POSITION_KEYS = tuple((f"real-{n:02d}-{lang}", r) for n in range(1, 21) for lang in ("zh", "en") for r in (1, 2))
STAMPS = tuple(canonical_hash([FORMAL_BATCH, DATASET_HASH, case, repeat]) for case, repeat in POSITION_KEYS)
COMMANDS = frozenset(prefix + stamp for stamp in STAMPS for prefix in ("sync:", "analysis:", "rag:"))


async def guard(db: Any, command: str, owner: int, recovery_hash: str | None, boot_id: str | None) -> None:
    """Checked in the same transaction as claims/reservations, including ordinary APIs."""
    if command not in COMMANDS and not command.startswith("rules:" + FORMAL_BATCH + ":"):
        return
    await db.execute("SELECT pg_advisory_xact_lock(71801110)")
    row = await (await db.execute("SELECT owner_id,result FROM v1_commands WHERE command_id=%s", (MARKER,))).fetchone()
    if row is None:
        if recovery_hash:
            raise DomainError("recovery_not_activated")
        return
    state = row["result"]
    if (
        row["owner_id"] != owner
        or not recovery_hash
        or state.get("manifest_hash") != recovery_hash
        or state.get("state") != "running"
        or datetime.now(UTC) >= datetime.fromisoformat(state["expires_at"])
    ):
        raise DomainError("recovery_old_permit_retired_or_stopped")
    if command not in state["allowed_commands"]:
        raise DomainError("recovery_consumed_or_unlisted_position")
    if not boot_id or boot_id != state.get("server_boot_id"):
        raise DomainError("recovery_server_restarted_no_resend")
    if command.split(":", 1)[1] != state.get("active_stamp"):
        raise DomainError("recovery_position_not_claimed")
