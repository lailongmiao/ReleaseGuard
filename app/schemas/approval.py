"""Approval workflow envelopes reusing Phase 2 immutable domain records."""

from typing import Annotated, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from app.schemas.releaseguard import ApprovalInvalidation, ApprovalRecord, DomainModel, Identifier, Sha256, Timestamp
from app.schemas.tooling import SafeCode, ToolInvocation, ToolResult

AuditEventType = Literal[
    "tool_invocation_requested",
    "permission_evaluated",
    "approval_requested",
    "approval_decided",
    "approval_invalidated",
    "tool_call_rejected",
    "tool_call_not_implemented",
    "tool_call_completed",
    "consistency_stopped",
    "execution_preflight",
    "simulation_completed",
    "execution_started",
    "execution_completed",
    "execution_reconciled",
    "verification_completed",
    "receipt_issued",
]


class AuthorizationBinding(DomainModel):
    """Seal all non-temporal decision inputs; every use recomputes this value."""

    invocation: ToolInvocation
    plan_hash: Sha256
    permission_policy_hash: Sha256
    evidence_policy_hash: Sha256
    risk_hash: Sha256
    evidence_hash: Sha256
    requester_id: Identifier
    scope: Literal["tool", "plan", "recovery"] = "tool"
    manifest_hash: Sha256 | None = None
    effect_ids: tuple[Identifier, ...] = ()
    binding_version: Literal["v1", "v2", "v3"] = "v1"
    product_hash: Sha256 | None = None

    @model_validator(mode="after")
    def check_scope(self) -> "AuthorizationBinding":
        """Legacy authority cannot acquire execution or recovery by default."""
        if (self.scope == "tool") != (self.manifest_hash is None):
            raise ValueError("approval_scope_conflict")
        if (self.scope == "recovery") != bool(self.effect_ids) or len(set(self.effect_ids)) != len(self.effect_ids):
            raise ValueError("approval_effect_conflict")
        if (self.binding_version in {"v2", "v3"}) != (self.product_hash is not None):
            raise ValueError("approval_product_binding_conflict")
        return self

    @model_serializer(mode="wrap")
    def serialize_binding(self, handler: SerializerFunctionWrapHandler) -> dict:
        """Keep historical tool authorization hashes byte-for-byte compatible."""
        values = handler(self)
        if self.binding_version == "v1":
            values.pop("binding_version", None)
            values.pop("product_hash", None)
        if self.scope == "tool":
            for key in ("scope", "manifest_hash", "effect_ids"):
                values.pop(key, None)
        return values


class ApprovalRequest(DomainModel):
    """Checkpoint-safe request; the authenticated actor is never stored here."""

    approval_request_id: Identifier
    idempotency_key: Identifier
    binding: AuthorizationBinding
    authorization_hash: Sha256
    expected_state_version: Annotated[int, Field(strict=True, ge=0)]
    requested_at: Timestamp
    expires_at: Timestamp
    reason_code: SafeCode

    @model_validator(mode="after")
    def check_interval(self) -> "ApprovalRequest":
        """Require an explicit bounded waiting interval."""
        if not self.requested_at < self.expires_at:
            raise ValueError("approval_interval_invalid")
        if (self.expires_at - self.requested_at).total_seconds() > 86400:
            raise ValueError("approval_interval_too_long")
        return self


class ApprovalCommand(DomainModel):
    """A resume command conveys intent, never proof of approver identity."""

    command_id: Identifier
    approval_request_id: Identifier
    authorization_hash: Sha256
    expected_state_version: Annotated[int, Field(strict=True, ge=0)]
    decision: Literal["APPROVED", "REJECTED", "NEEDS_CHANGES"]
    reason_code: SafeCode


class ApprovalEntry(DomainModel):
    """Append-only workflow history; pending and decision have distinct record ids."""

    request: ApprovalRequest
    pending: ApprovalRecord
    decision: ApprovalRecord | None = None
    decision_command_hash: Sha256 | None = None
    invalidations: tuple[ApprovalInvalidation, ...] = ()


class ToolAuditEvent(DomainModel):
    """Bounded audit attribution without raw arguments, response text or tokens."""

    event_id: Identifier
    event_type: AuditEventType
    invocation_id: Identifier
    change_id: Identifier
    run_id: Identifier
    revision: Annotated[int, Field(strict=True, ge=1)]
    thread_id: Identifier
    principal_id: Identifier
    tenant_id: Identifier
    session_id: Identifier
    correlation_id: Identifier
    input_hash: Sha256
    descriptor_hash: Sha256
    authorization_hash: Sha256 | None = None
    approval_id: Identifier | None = None
    reason_code: SafeCode
    occurred_at: Timestamp


class ApprovalStoreSnapshot(DomainModel):
    """In-memory test transfer format; not a production backup or checkpoint."""

    version: Literal["approval-store.v1"] = "approval-store.v1"
    entries: tuple[ApprovalEntry, ...] = ()
    audit_events: tuple[ToolAuditEvent, ...] = ()
    calls: tuple["ToolCallEntry", ...] = ()


class ToolCallEntry(DomainModel):
    """Claim-before-call ledger; an unresolved claim stops replay after a fault."""

    invocation: ToolInvocation
    authorization_hash: Sha256
    result: ToolResult | None = None


class ToolAuthorizationDecision(DomainModel):
    """Authorization is explicitly separate from execution success."""

    decision: Literal["allow", "deny", "approval_required", "expired", "revoked", "unavailable", "conflicting"]
    reason_codes: Annotated[tuple[SafeCode, ...], Field(min_length=1, max_length=16)]
    policy_version: Identifier
    evidence_ids: tuple[Identifier, ...]
    citation_ids: tuple[Identifier, ...]
    approval_id: Identifier | None
    approval_expires_at: Timestamp | None = None
    descriptor_hash: Sha256
    input_hash: Sha256
    authorization_hash: Sha256 | None = None
    evaluated_at: Timestamp
    audit_ids: tuple[Identifier, ...]
