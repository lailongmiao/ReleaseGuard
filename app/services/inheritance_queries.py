"""Safe projections over original facts. No save, graph, provider or recovery calls."""

from datetime import UTC, datetime
from typing import Any

from app.schemas.approval import ApprovalStoreSnapshot
from app.schemas.local_mvp import LocalActor
from app.schemas.inheritance import ApprovalHistoryItem, ApprovalHistoryPage, LegacyCommandView, ReportCIView
from app.schemas.v11_backend import CIObservation
from app.services.local_mvp import LocalMVPService
from app.services.local_mvp_storage import unit
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_queries import decode_cursor, encode_cursor
from app.services.v11_storage import V11Store


def select(value: dict, keys: tuple[str, ...]) -> dict:
    """Explicit whitelist, including nested records separately."""
    return {key: value[key] for key in keys if key in value}


def record_projection(value: Any) -> dict:
    """Do not expose session identifiers or tool arguments from original records."""
    data = value.model_dump(mode="json")
    result = select(
        data,
        (
            "record_type",
            "change_id",
            "run_id",
            "revision",
            "request_hash",
            "input_hash",
            "approval_id",
            "invalidation_id",
            "plan_hash",
            "decision",
            "approver_id",
            "approver_role",
            "reason",
            "requested_at",
            "decided_at",
            "expires_at",
            "requester_id",
            "authorization_hash",
            "evidence_ids",
            "citation_ids",
            "manifest_hash",
            "scope",
        ),
    )
    result["audit"] = select(data.get("audit", {}), ("actor_id", "reason", "occurred_at", "input_hash"))
    return result


def approval_page(
    snapshot: ApprovalStoreSnapshot,
    actor: LocalActor,
    identifier: str,
    version: str,
    cursor: str | None,
    limit: int,
    at: datetime,
) -> ApprovalHistoryPage:
    """Pagination does not synthesize expiration events or alter original hashes."""
    if not 1 <= limit <= 100:
        raise DomainError("history_limit_invalid")
    entries = sorted(
        snapshot.entries, key=lambda e: (e.request.requested_at.isoformat(), e.request.approval_request_id)
    )
    scope = {"object": identifier, "version": version, "kind": "approval-history"}
    # Appending or changing history invalidates a cursor instead of skipping records.
    digest = canonical_hash(snapshot.model_dump(mode="json"))
    position = decode_cursor(cursor, actor, scope, digest, 0)
    if type(position) is not int or position < 0 or position > len(entries):
        raise DomainError("cursor_invalid_or_expired")
    items = []
    for entry in entries[position : position + limit]:
        request = entry.request
        invocation = request.binding.invocation
        if invocation.change_id != identifier:
            raise DomainError("local_store_integrity_failure")
        binding = select(
            request.binding.model_dump(mode="json"),
            (
                "plan_hash",
                "permission_policy_hash",
                "evidence_policy_hash",
                "risk_hash",
                "evidence_hash",
                "requester_id",
                "scope",
                "manifest_hash",
                "effect_ids",
                "binding_version",
                "product_hash",
            ),
        )
        binding["invocation"] = select(
            invocation.model_dump(mode="json"),
            (
                "invocation_id",
                "change_id",
                "run_id",
                "revision",
                "request_hash",
                "tool_name",
                "version",
                "input_hash",
                "descriptor_hash",
            ),
        )
        events = [
            select(
                event.model_dump(mode="json"),
                (
                    "event_id",
                    "event_type",
                    "invocation_id",
                    "change_id",
                    "run_id",
                    "revision",
                    "principal_id",
                    "input_hash",
                    "authorization_hash",
                    "approval_id",
                    "reason_code",
                    "occurred_at",
                ),
            )
            for event in snapshot.audit_events
            if event.change_id == identifier and event.invocation_id == invocation.invocation_id
        ]
        items.append(
            ApprovalHistoryItem(
                approval_request_id=request.approval_request_id,
                change_id=identifier,
                run_id=invocation.run_id,
                revision=invocation.revision,
                authorization_hash=request.authorization_hash,
                binding=binding,
                requested_at=request.requested_at,
                expires_at=request.expires_at,
                reason_code=request.reason_code,
                pending=record_projection(entry.pending),
                decision=record_projection(entry.decision) if entry.decision else None,
                invalidations=[record_projection(item) for item in entry.invalidations],
                audit_events=events,
                expired_now=at >= request.expires_at,
            )
        )
    next_position = position + len(items)
    return ApprovalHistoryPage(
        items=items,
        next_cursor=encode_cursor(actor, scope, digest, next_position) if next_position < len(entries) else None,
        evaluated_at=at,
    )


async def history(
    local: LocalMVPService, identifier: str, actor: LocalActor, cursor: str | None, limit: int
) -> ApprovalHistoryPage:
    """Both callers keep their original review scope and source checks."""
    async with local.store.open(identifier, actor):
        await local.view()
        result = approval_page(
            unit().facts.approvals, actor, identifier, unit().facts.version, cursor, limit, datetime.now(UTC)
        )
        await local.view()
        await local.store.authenticate(actor.session_id, actor.expires_at)
        return result


async def report_ci(store: V11Store, identifier: str, analysis_id: str, actor: LocalActor) -> ReportCIView:
    """Use the report's persisted FK; the latest observation is never a fallback."""
    async with await store.connection.connect() as db, db.transaction():
        await store.object_row(identifier, actor, db, lock=True)
        saved = await (
            await db.execute(
                "SELECT * FROM v11_analyses WHERE object_id=%s AND analysis_id=%s", (identifier, analysis_id)
            )
        ).fetchone()
        if not saved:
            raise DomainError("local_object_not_found")
        observation_id = saved["report"].get("ci_observation_id")
        result = ReportCIView(
            object_id=identifier,
            analysis_id=analysis_id,
            snapshot_id=saved["snapshot_id"],
            observation_id=observation_id,
            availability="not_bound",
        )
        if not observation_id:
            return result
        if observation_id != saved["observation_id"]:
            return result.model_copy(update={"availability": "binding_conflict"})
        row = await (
            await db.execute(
                "SELECT facts FROM v11_observations WHERE object_id=%s AND snapshot_id=%s AND observation_id=%s",
                (identifier, saved["snapshot_id"], observation_id),
            )
        ).fetchone()
        if not row:
            return result.model_copy(update={"availability": "missing"})
        observation = CIObservation.model_validate(row["facts"])
        comparison = saved["report"].get("resolved_comparison")
        if comparison and comparison.get("head_sha") != observation.head_sha:
            return result.model_copy(update={"availability": "binding_conflict"})
        return result.model_copy(update={"availability": "available", "observation": observation})


async def legacy_command(
    local: LocalMVPService, identifier: str, command_id: str, actor: LocalActor
) -> LegacyCommandView:
    """Legacy hash-only commands do not prove their actor: fail closed for those."""
    async with local.store.open(identifier, actor):
        await local.view()
        row = await (
            await unit().connection.execute(
                "SELECT status,result FROM v1_commands WHERE resource=%s AND command_id=%s AND owner_id=%s",
                (identifier, command_id, actor.user_id),
            )
        ).fetchone()
        if not row:
            raise DomainError("local_object_not_found")
        rows = await (
            await unit().connection.execute(
                "SELECT attempt,reserved,charge,usage FROM v1_budget WHERE owner_id=%s AND command_id=%s ORDER BY attempt LIMIT 16",
                (actor.user_id, command_id),
            )
        ).fetchall()
        attempts = [
            {
                **select(item, ("attempt", "reserved", "charge")),
                "usage": select(item.get("usage") or {}, ("input_tokens", "output_tokens", "latency_ms")),
            }
            for item in rows
        ]
        result = (
            select(
                row["result"],
                ("status", "reason", "analysis_id", "generation_id", "snapshot_id", "change_id", "binding_hash"),
            )
            if row["result"] is not None
            else None
        )
        await local.view()
        await local.store.authenticate(actor.session_id, actor.expires_at)
        return LegacyCommandView(
            command_id=command_id,
            change_id=identifier,
            status="unknown" if row["status"] == "started" else row["status"],
            result=result,
            attempts=attempts,
        )
