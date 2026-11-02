"""Lock-protected offline approval/audit substitute with isolated immutable facts."""

import asyncio

from app.core.metrics import observe_fact

from app.schemas.approval import ApprovalEntry, ApprovalStoreSnapshot, ToolAuditEvent, ToolCallEntry
from app.schemas.tooling import ToolResult
from app.schemas.releaseguard import ApprovalDecision, ApprovalInvalidation, ApprovalRecord, matches_binding
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError


def validate_entry(entry: ApprovalEntry) -> ApprovalEntry:
    """Check workflow bindings and seals at every storage/restore boundary."""
    entry = ApprovalEntry.model_validate_json(entry.model_dump_json())
    request, pending = entry.request, entry.pending
    binding, call = request.binding, request.binding.invocation
    if request.authorization_hash != canonical_hash(binding):
        raise DomainError("approval_binding_conflict")
    for record in (pending, entry.decision):
        if record is None:
            continue
        if (
            not matches_binding(record, call)
            or record.authorization_hash != request.authorization_hash
            or record.plan_hash != binding.plan_hash
            or record.input_hash != call.input_hash
            or record.descriptor_hash != call.descriptor_hash
            or record.requester_id != binding.requester_id
            or record.tenant_id != call.tenant_id
            or record.permission_policy_hash != binding.permission_policy_hash
            or record.evidence_policy_hash != binding.evidence_policy_hash
            or record.risk_hash != binding.risk_hash
            or record.scope != binding.scope
            or record.manifest_hash != binding.manifest_hash
            or record.effect_ids != binding.effect_ids
            or record.evidence_ids != sorted(set(call.evidence_ids))
            or record.citation_ids != sorted(set(call.citation_ids))
            or record.expires_at != request.expires_at
            or record.requested_at != request.requested_at
            or record.idempotency_key != request.idempotency_key
            or record.audit.policy_version != call.policy_version
            or record.audit.input_hash != call.input_hash
            or record.audit.correlation_id != call.correlation_id
            or record.audit.reason != record.reason
        ):
            raise DomainError("approval_binding_conflict")
    if (
        pending.decision is not ApprovalDecision.PENDING
        or pending.approval_id != request.approval_request_id + ":pending"
        or pending.session_id != call.session_id
        or pending.audit.actor_id != call.principal_id
        or pending.audit.occurred_at != request.requested_at
        or pending.reason != request.reason_code
        or binding.requester_id != call.principal_id
    ):
        raise DomainError("approval_pending_conflict")
    if entry.decision is not None:
        if (
            entry.decision.approval_id != request.approval_request_id + ":decision"
            or entry.decision.decision
            not in {ApprovalDecision.APPROVED, ApprovalDecision.REJECTED, ApprovalDecision.NEEDS_CHANGES}
            or entry.decision_command_hash is None
            or not entry.decision.approver_roles
            or entry.decision.decided_at is None
            or entry.decision.decided_at >= request.expires_at
            or entry.decision.decided_at < request.requested_at
            or entry.decision.audit.occurred_at != entry.decision.decided_at
            or entry.decision.approver_id != entry.decision.audit.actor_id
            or entry.decision.approver_role not in entry.decision.approver_roles
            or entry.decision.session_id is None
        ):
            raise DomainError("approval_decision_conflict")
    elif entry.decision_command_hash is not None:
        raise DomainError("approval_decision_conflict")
    if len(entry.invalidations) > 1:
        raise DomainError("approval_invalidation_conflict")
    ids: set[str] = set()
    for fact in entry.invalidations:
        original = entry.decision or pending
        if (
            fact.invalidation_id in ids
            or not matches_binding(fact, original)
            or fact.approval_id != original.approval_id
            or fact.plan_hash != original.plan_hash
            or fact.input_hash != call.input_hash
            or fact.audit.input_hash != call.input_hash
            or fact.audit.correlation_id != call.correlation_id
            or fact.audit.policy_version != call.policy_version
            or (fact.decision is ApprovalDecision.EXPIRED and fact.audit.occurred_at < request.expires_at)
        ):
            raise DomainError("approval_invalidation_conflict")
        if fact.audit.occurred_at < original.audit.occurred_at:
            raise DomainError("approval_invalidation_conflict")
        ids.add(fact.invalidation_id)
    return entry


def validate_approval_audit(entry: ApprovalEntry, event: ToolAuditEvent) -> None:
    """Join the immutable audit to its specific request/decision/invalidation."""
    call = entry.request.binding.invocation
    fact = next(
        (
            item
            for item in (entry.pending, entry.decision, *entry.invalidations)
            if item is not None
            and event.approval_id == item.approval_id
            and event.occurred_at == item.audit.occurred_at
            and event.reason_code == item.audit.reason
            and event.principal_id == item.audit.actor_id
            and (
                event.event_type == "approval_invalidated"
                and isinstance(item, ApprovalInvalidation)
                or event.event_type == "approval_requested"
                and item is entry.pending
                or event.event_type == "approval_decided"
                and item is entry.decision
            )
        ),
        None,
    )
    if (
        fact is None
        or (
            event.invocation_id,
            event.change_id,
            event.run_id,
            event.revision,
            event.thread_id,
            event.tenant_id,
            event.correlation_id,
            event.input_hash,
            event.descriptor_hash,
            event.authorization_hash,
        )
        != (
            call.invocation_id,
            call.change_id,
            call.run_id,
            call.revision,
            call.thread_id,
            call.tenant_id,
            call.correlation_id,
            call.input_hash,
            call.descriptor_hash,
            entry.request.authorization_hash,
        )
        or (isinstance(fact, ApprovalRecord) and event.session_id != fact.session_id)
    ):
        raise DomainError("approval_audit_conflict")


class InMemoryApprovalRepository:
    """Offline CAS/idempotency semantics; no cross-process durability guarantee."""

    def __init__(self) -> None:
        """Initialize independent indexes and one atomic local lock."""
        self._lock = asyncio.Lock()
        self._entries: dict[str, ApprovalEntry] = {}
        self._keys: dict[tuple[str, str, str], str] = {}
        self._audit: dict[str, ToolAuditEvent] = {}
        self._calls: dict[str, ToolCallEntry] = {}

    async def create(self, entry: ApprovalEntry, *, audit: ToolAuditEvent | None = None) -> ApprovalEntry:
        """Store only pending entries and reject conflicting key reuse."""
        entry = validate_entry(entry)
        if entry.decision or entry.invalidations:
            raise DomainError("approval_create_invalid")
        if audit is not None:
            validate_approval_audit(entry, audit)
        call = entry.request.binding.invocation
        key = (call.tenant_id, call.principal_id, entry.request.idempotency_key)
        async with self._lock:
            existing_id = self._keys.get(key, entry.request.approval_request_id)
            existing = self._entries.get(existing_id)
            if existing:
                if existing.request != entry.request or existing.pending != entry.pending:
                    raise DomainError("approval_idempotency_conflict")
                return validate_entry(existing)
            if audit is not None:
                self._append_audit_locked(audit)
            self._entries[entry.request.approval_request_id] = entry
            self._keys[key] = entry.request.approval_request_id
            return validate_entry(entry)

    async def get(self, request_id: str) -> ApprovalEntry:
        """Read an isolated authoritative entry, never a checkpoint copy."""
        async with self._lock:
            if request_id not in self._entries:
                raise DomainError("approval_not_found")
            return validate_entry(self._entries[request_id])

    async def decide(
        self, request_id: str, decision: ApprovalRecord, command_hash: str, *, audit: ToolAuditEvent | None = None
    ) -> ApprovalEntry:
        """First decision wins; only byte-equivalent command replay is accepted."""
        async with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                raise DomainError("approval_not_found")
            if entry.decision:
                previous = entry.decision.model_dump(exclude={"decided_at", "audit"})
                proposed = decision.model_dump(exclude={"decided_at", "audit"})
                old_audit = entry.decision.audit.model_dump(exclude={"occurred_at"})
                new_audit = decision.audit.model_dump(exclude={"occurred_at"})
                if previous != proposed or old_audit != new_audit or entry.decision_command_hash != command_hash:
                    raise DomainError("approval_decision_conflict")
                return validate_entry(entry)
            if entry.invalidations:
                raise DomainError("approval_invalidated")
            updated = validate_entry(
                ApprovalEntry.model_validate(
                    {**entry.model_dump(), "decision": decision, "decision_command_hash": command_hash}
                )
            )
            if audit is not None:
                validate_approval_audit(updated, audit)
                self._append_audit_locked(audit)
            self._entries[request_id] = updated
            return validate_entry(updated)

    async def invalidate(
        self, request_id: str, fact: ApprovalInvalidation, *, audit: ToolAuditEvent | None = None
    ) -> ApprovalEntry:
        """Keep decisions intact and append exact-once invalidation facts."""
        async with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                raise DomainError("approval_not_found")
            for existing in entry.invalidations:
                if existing.invalidation_id == fact.invalidation_id:
                    if existing != fact:
                        raise DomainError("approval_invalidation_conflict")
                    return validate_entry(entry)
            updated = validate_entry(
                ApprovalEntry.model_validate({**entry.model_dump(), "invalidations": (*entry.invalidations, fact)})
            )
            if audit is not None:
                validate_approval_audit(updated, audit)
                self._append_audit_locked(audit)
            self._entries[request_id] = updated
            return validate_entry(updated)

    async def append_audit(self, event: ToolAuditEvent) -> ToolAuditEvent:
        """Reject same-id/different-body audit rather than silently overwrite."""
        event = ToolAuditEvent.model_validate_json(event.model_dump_json())
        async with self._lock:
            return self._append_audit_locked(event)

    def _append_audit_locked(self, event: ToolAuditEvent) -> ToolAuditEvent:
        """Validate before mutation; caller holds the store's local transaction lock."""
        event = ToolAuditEvent.model_validate_json(event.model_dump_json())
        existing = self._audit.get(event.event_id)
        if existing and existing != event:
            raise DomainError("audit_conflict")
        self._audit[event.event_id] = event
        if existing is None:
            category = (
                "approval"
                if event.event_type.startswith("approval_")
                else "verification"
                if event.event_type == "verification_completed"
                else "receipt"
                if event.event_type == "receipt_issued"
                else "recovery"
                if event.event_type in {"execution_reconciled", "consistency_stopped"}
                else "execution"
                if event.event_type
                in {"execution_started", "execution_completed", "execution_preflight", "simulation_completed"}
                else "tool"
            )
            outcome = (
                "pending"
                if event.event_type == "approval_requested"
                else "denied"
                if event.event_type == "tool_call_rejected"
                else "unknown"
                if event.event_type == "consistency_stopped"
                else "recorded"
            )
            observe_fact(
                category,
                outcome,
                **event.model_dump(
                    include={
                        "change_id",
                        "run_id",
                        "revision",
                        "session_id",
                        "principal_id",
                        "correlation_id",
                        "approval_id",
                        "event_id",
                    }
                ),
            )
        return ToolAuditEvent.model_validate_json(event.model_dump_json())

    async def export_snapshot(self) -> ApprovalStoreSnapshot:
        """Return a validated transfer value with stable object ordering."""
        async with self._lock:
            return ApprovalStoreSnapshot(
                entries=tuple(validate_entry(self._entries[key]) for key in sorted(self._entries)),
                audit_events=tuple(sorted(self._audit.values(), key=_audit_order)),
                calls=tuple(self._calls[key] for key in sorted(self._calls)),
            )

    async def restore_snapshot(self, snapshot: ApprovalStoreSnapshot) -> None:
        """Build all indexes privately; corrupted transfers cannot partly install."""
        snapshot = ApprovalStoreSnapshot.model_validate_json(snapshot.model_dump_json())
        candidate = InMemoryApprovalRepository()
        for entry in snapshot.entries:
            entry = validate_entry(entry)
            pending = ApprovalEntry(request=entry.request, pending=entry.pending)
            await candidate.create(pending)
            if entry.decision:
                await candidate.decide(
                    entry.request.approval_request_id, entry.decision, str(entry.decision_command_hash)
                )
            for fact in entry.invalidations:
                await candidate.invalidate(entry.request.approval_request_id, fact)
        for event in snapshot.audit_events:
            await candidate.append_audit(event)
        approval_events = {"approval_requested", "approval_decided", "approval_invalidated"}
        for entry in candidate._entries.values():
            expected = ["approval_requested"]
            if entry.decision:
                expected.append("approval_decided")
            if entry.invalidations:
                expected.append("approval_invalidated")
            for kind in expected:
                events = [
                    event
                    for event in candidate._audit.values()
                    if event.event_type == kind
                    and event.authorization_hash == entry.request.authorization_hash
                    and event.approval_id
                    in {entry.pending.approval_id, entry.request.approval_request_id + ":decision"}
                ]
                if len(events) != 1:
                    raise DomainError("approval_audit_conflict")
                validate_approval_audit(entry, events[0])
        for event in candidate._audit.values():
            if event.event_type in approval_events and not any(
                event.approval_id in {entry.pending.approval_id, entry.request.approval_request_id + ":decision"}
                and event.authorization_hash == entry.request.authorization_hash
                for entry in candidate._entries.values()
            ):
                raise DomainError("approval_audit_conflict")
        for call in snapshot.calls:
            await candidate.claim_call(
                ToolCallEntry(invocation=call.invocation, authorization_hash=call.authorization_hash)
            )
            if call.result:
                await candidate.complete_call(call.invocation.invocation_id, call.result)
        if len(candidate._entries) != len(snapshot.entries) or len(candidate._audit) != len(snapshot.audit_events):
            raise DomainError("approval_snapshot_conflict")
        async with self._lock:
            if self._entries or self._audit or self._calls:
                raise DomainError("approval_restore_requires_empty")
            self._entries, self._keys, self._audit = candidate._entries, candidate._keys, candidate._audit
            self._calls = candidate._calls

    async def get_call(self, invocation_id: str) -> ToolCallEntry | None:
        """Read an isolated completed or unresolved claim."""
        async with self._lock:
            entry = self._calls.get(invocation_id)
            return ToolCallEntry.model_validate_json(entry.model_dump_json()) if entry else None

    async def claim_call(self, entry: ToolCallEntry) -> None:
        """Claim once before transport; unknown completion cannot trigger a retry."""
        entry = ToolCallEntry.model_validate_json(entry.model_dump_json())
        if entry.result:
            raise DomainError("tool_claim_invalid")
        async with self._lock:
            if entry.invocation.invocation_id in self._calls:
                raise DomainError("tool_claim_conflict")
            self._calls[entry.invocation.invocation_id] = entry

    async def complete_call(self, invocation_id: str, result: ToolResult) -> None:
        """Seal only a matching result and its known audit references."""
        result = ToolResult.model_validate_json(result.model_dump_json())
        async with self._lock:
            entry = self._calls.get(invocation_id)
            if entry is None:
                raise DomainError("tool_claim_missing")
            call = entry.invocation
            if (result.invocation_id, result.input_hash, result.descriptor_hash) != (
                call.invocation_id,
                call.input_hash,
                call.descriptor_hash,
            ) or any(identifier not in self._audit for identifier in result.audit_ids):
                raise DomainError("tool_result_conflict")
            events = [self._audit[identifier] for identifier in result.audit_ids]
            if (
                [event.event_type for event in events]
                != ["tool_invocation_requested", "permission_evaluated", "tool_call_completed"]
                or any(
                    (
                        event.invocation_id,
                        event.change_id,
                        event.run_id,
                        event.revision,
                        event.thread_id,
                        event.principal_id,
                        event.session_id,
                        event.tenant_id,
                        event.correlation_id,
                        event.input_hash,
                        event.descriptor_hash,
                    )
                    != (
                        call.invocation_id,
                        call.change_id,
                        call.run_id,
                        call.revision,
                        call.thread_id,
                        call.principal_id,
                        call.session_id,
                        call.tenant_id,
                        call.correlation_id,
                        call.input_hash,
                        call.descriptor_hash,
                    )
                    for event in events
                )
                or events[0].occurred_at != call.occurred_at
                or events != sorted(events, key=_audit_order)
                or (
                    events[-1].authorization_hash != entry.authorization_hash
                    or events[-1].approval_id != call.approval_id
                    or events[-1].reason_code != (result.error_code or "tool_read_completed")
                )
            ):
                raise DomainError("tool_result_conflict")
            if entry.result and entry.result != result:
                raise DomainError("tool_result_conflict")
            self._calls[invocation_id] = ToolCallEntry(
                invocation=call, authorization_hash=entry.authorization_hash, result=result
            )


def _audit_order(event: ToolAuditEvent) -> tuple:
    """Stable logical chronology; equal-time ties respect workflow prerequisites."""
    order = {
        "tool_invocation_requested": 0,
        "permission_evaluated": 1,
        "approval_requested": 2,
        "approval_decided": 3,
        "approval_invalidated": 4,
        "tool_call_rejected": 5,
        "tool_call_not_implemented": 5,
        "tool_call_completed": 5,
        "consistency_stopped": 6,
        "execution_preflight": 7,
        "simulation_completed": 8,
        "execution_started": 9,
        "execution_completed": 10,
        "execution_reconciled": 11,
        "verification_completed": 12,
        "receipt_issued": 13,
    }
    return event.occurred_at, order[event.event_type], event.event_id
