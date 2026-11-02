"""Read-only public projections; never serialized back into business facts."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from app.schemas.v11_backend import CIObservation


class ApprovalHistoryItem(BaseModel):
    """Original seals and sanitized decisions, separate from live expiry."""

    approval_request_id: str
    change_id: str
    run_id: str
    revision: int
    authorization_hash: str
    binding: dict[str, Any]
    requested_at: datetime
    expires_at: datetime
    reason_code: str
    pending: dict[str, Any]
    decision: dict[str, Any] | None
    invalidations: list[dict[str, Any]]
    audit_events: list[dict[str, Any]]
    expired_now: bool


class ApprovalHistoryPage(BaseModel):
    """A bounded page, not a declaration that the whole archive was returned."""

    items: list[ApprovalHistoryItem]
    next_cursor: str | None
    evaluated_at: datetime


class ReportCIView(BaseModel):
    """Only the observation bound to the selected report."""

    object_id: str
    analysis_id: str
    snapshot_id: str
    observation_id: str | None
    availability: Literal["available", "not_bound", "missing", "binding_conflict"]
    observation: CIObservation | None = None


class LegacyCommandView(BaseModel):
    """Only commands with persisted actor attribution may be queried."""

    command_id: str
    change_id: str
    status: str
    result: dict[str, Any] | None
    attempts: list[dict[str, Any]]
