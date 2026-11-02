"""Local product intents reuse the domain contracts and contain no authority."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from app.schemas.approval import ApprovalEntry
from app.schemas.releaseguard import ChangeRequestInput, ChangeRunState, Identifier
from app.schemas.v1_backend import ProductBinding


class LocalCreate(BaseModel):
    """Identity is assigned by the authenticated boundary, never this body."""

    model_config = ConfigDict(extra="forbid")
    change: ChangeRequestInput
    approver_user_id: Annotated[int, Field(strict=True, ge=1)]
    idempotency_key: Identifier
    evidence_scenario: Literal["complete", "missing", "expired", "conflicting", "unavailable"] = "complete"
    product: ProductBinding | None = None

    @model_serializer(mode="wrap")
    def serialize_create(self, handler: SerializerFunctionWrapHandler) -> dict:
        """Preserve existing idempotency digests when no product is requested."""
        values = handler(self)
        if self.product is None:
            values.pop("product", None)
        return values


class LocalCommand(BaseModel):
    """Server checks CAS, object scope and durable command identity."""

    model_config = ConfigDict(extra="forbid")
    command_id: Identifier
    expected_state_version: Annotated[int, Field(strict=True, ge=0)]


class LocalPrepare(LocalCommand):
    """Only finite local simulation or synthetic sandbox may be planned."""

    mode: Literal["offline_simulation", "offline_sandbox"] = "offline_simulation"
    approval_ttl_seconds: Annotated[int, Field(strict=True, ge=1, le=3600)] = 600
    analysis_id: Identifier | None = None

    @model_serializer(mode="wrap")
    def serialize_prepare(self, handler: SerializerFunctionWrapHandler) -> dict:
        """Legacy commands retain their exact body digest."""
        values = handler(self)
        if self.analysis_id is None:
            values.pop("analysis_id", None)
        return values


class LocalRevision(LocalCommand):
    """Owner supplements inputs; a new revision invalidates old approval."""

    change: ChangeRequestInput
    evidence_scenario: Literal["complete", "missing", "expired", "conflicting", "unavailable"] = "complete"


class LocalView(BaseModel):
    """Expose saved authority and mode, without presenting simulation as real."""

    model_config = ConfigDict(extra="forbid")
    state: ChangeRunState
    approval: ApprovalEntry | None = None
    synthetic: Literal[True] = True
    production_execution: Literal["disabled"] = "disabled"
    target_write_calls: int = 0
    recovery: Literal["saved_facts", "safe_hold"] = "saved_facts"
    source_provenance: Literal["synthetic", "readonly_github_markdown"] = "synthetic"
    analysis_provenance: Literal["deterministic", "provider", "unverified"] = "deterministic"
    execution_mode: str | None = None
    target_provenance: Literal["synthetic"] = "synthetic"


class LocalActor(BaseModel):
    """Private trusted session identity; not a request/checkpoint value."""

    user_id: int
    session_id: str
    expires_at: float
