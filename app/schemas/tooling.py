"""Bounded tool contracts containing references and hashes, never raw payloads."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.releaseguard import DomainModel, Identifier, RunBound, Sha256, Timestamp
from app.services.releaseguard.normalization import canonical_hash

SafeCode = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$", strict=True)]
SafeIds = Annotated[tuple[Identifier, ...], Field(max_length=64)]


class SideEffect(StrEnum):
    """Capabilities are declarations, never permission to execute."""

    READ_ONLY = "read_only"
    SIMULATION = "simulation"
    MUTATING = "mutating"
    IRREVERSIBLE = "irreversible"


class ToolStatus(StrEnum):
    """Stable, redacted outcomes shared by adapters and policy services."""

    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    PERMISSION_DENIED = "permission_denied"
    APPROVAL_REQUIRED = "approval_required"
    EXPIRED = "expired"
    REVOKED = "revoked"
    UNAVAILABLE = "unavailable"
    CONFLICTING = "conflicting"
    NOT_IMPLEMENTED = "not_implemented"


class ToolDescriptor(DomainModel):
    """Locally approved, sealed descriptor; remote discovery cannot widen it."""

    tool_name: Identifier
    version: Identifier
    capabilities: Annotated[frozenset[SideEffect], Field(min_length=1, max_length=4)]
    resource_types: Annotated[frozenset[Identifier], Field(min_length=1, max_length=64)]
    action_types: Annotated[frozenset[Identifier], Field(min_length=1, max_length=64)]
    input_schema: Identifier
    output_schema: Identifier
    side_effect_level: SideEffect
    timeout_ms: Annotated[int, Field(strict=True, ge=1, le=30000)]
    idempotency_required: bool = Field(strict=True)
    adapter_kind: Literal["internal", "mcp"]
    descriptor_hash: Sha256 = "0" * 64

    @model_validator(mode="after")
    def seal(self) -> "ToolDescriptor":
        """Reject conflicting capabilities and verify the canonical seal."""
        if self.capabilities != frozenset({self.side_effect_level}):
            raise ValueError("capability_conflict")
        if self.side_effect_level is not SideEffect.READ_ONLY and not self.idempotency_required:
            raise ValueError("idempotency_required")
        digest = canonical_hash(self.model_dump(exclude={"descriptor_hash"}, mode="python"))
        if self.descriptor_hash not in {"0" * 64, digest}:
            raise ValueError("descriptor_hash_mismatch")
        object.__setattr__(self, "descriptor_hash", digest)
        return self


class ToolInvocation(RunBound):
    """Safe input envelope; input content remains outside audit/checkpoints."""

    invocation_id: Identifier
    tool_name: Identifier
    version: Identifier
    descriptor_hash: Sha256
    input_schema: Identifier
    input_hash: Sha256
    thread_id: Identifier
    principal_id: Identifier
    session_id: Identifier
    tenant_id: Identifier
    resource_type: Identifier
    resource_id: Identifier
    action: Identifier
    environment: Identifier
    policy_version: Identifier
    correlation_id: Identifier
    approval_id: Identifier | None = None
    evidence_ids: SafeIds = ()
    citation_ids: SafeIds = ()
    occurred_at: Timestamp


class ToolResult(DomainModel):
    """No free-form response/error field can retain an external sensitive body."""

    invocation_id: Identifier
    status: ToolStatus
    input_hash: Sha256
    descriptor_hash: Sha256
    output_schema: Identifier
    output_hash: Sha256 | None = None
    item_count: Annotated[int, Field(strict=True, ge=0, le=10000)] = 0
    error_code: SafeCode | None = None
    audit_ids: SafeIds = ()

    @model_validator(mode="after")
    def check_outcome(self) -> "ToolResult":
        """Only a successful result may describe output."""
        if self.status is ToolStatus.SUCCEEDED:
            if self.output_hash is None or self.error_code is not None:
                raise ValueError("tool_result_invalid")
        elif self.output_hash is not None or self.item_count != 0 or self.error_code is None:
            raise ValueError("tool_result_invalid")
        return self
