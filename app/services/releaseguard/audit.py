"""Safe append-only audit construction independent of external observability."""

from datetime import datetime

from app.repositories.approval import ApprovalRepository
from app.schemas.approval import AuditEventType, ToolAuditEvent
from app.schemas.tooling import ToolInvocation
from app.services.releaseguard.normalization import canonical_hash


def build_audit(
    call: ToolInvocation,
    event_type: AuditEventType,
    code: str,
    *,
    at: datetime,
    discriminator: str = "",
    authorization_hash: str | None = None,
    approval_id: str | None = None,
) -> ToolAuditEvent:
    """Derive an event id from immutable payload; no raw tool data is accepted."""
    values = dict(
        event_type=event_type,
        invocation_id=call.invocation_id,
        change_id=call.change_id,
        run_id=call.run_id,
        revision=call.revision,
        thread_id=call.thread_id,
        principal_id=call.principal_id,
        tenant_id=call.tenant_id,
        session_id=call.session_id,
        correlation_id=call.correlation_id,
        input_hash=call.input_hash,
        descriptor_hash=call.descriptor_hash,
        reason_code=code,
        occurred_at=at,
        authorization_hash=authorization_hash,
        approval_id=approval_id,
    )
    return ToolAuditEvent.model_validate(
        {
            **values,
            "event_id": "audit:" + canonical_hash({"event": values, "discriminator": discriminator}),
        }
    )


async def record_audit(
    repository: ApprovalRepository,
    call: ToolInvocation,
    event_type: AuditEventType,
    code: str,
    *,
    at: datetime,
    discriminator: str = "",
    authorization_hash: str | None = None,
    approval_id: str | None = None,
) -> ToolAuditEvent:
    """Append a safe event; immutable approval writes use the same builder."""
    return await repository.append_audit(
        build_audit(
            call,
            event_type,
            code,
            at=at,
            discriminator=discriminator,
            authorization_hash=authorization_hash,
            approval_id=approval_id,
        )
    )
