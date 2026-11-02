"""Async append-only approval/audit protocol; no production storage binding."""

from typing import Protocol

from app.schemas.approval import ApprovalEntry, ApprovalStoreSnapshot, ToolAuditEvent, ToolCallEntry
from app.schemas.tooling import ToolResult
from app.schemas.releaseguard import ApprovalInvalidation, ApprovalRecord


class ApprovalRepository(Protocol):
    """All decisions use CAS and scoped idempotency; original records survive."""

    async def create(self, entry: ApprovalEntry, *, audit: ToolAuditEvent | None = None) -> ApprovalEntry:
        """Create a pending workflow or return its exact original request."""
        ...

    async def get(self, request_id: str) -> ApprovalEntry:
        """Read a deeply isolated authoritative history."""
        ...

    async def decide(
        self, request_id: str, decision: ApprovalRecord, command_hash: str, *, audit: ToolAuditEvent | None = None
    ) -> ApprovalEntry:
        """Set the only decision, rejecting conflicts or invalidated requests."""
        ...

    async def invalidate(
        self, request_id: str, fact: ApprovalInvalidation, *, audit: ToolAuditEvent | None = None
    ) -> ApprovalEntry:
        """Append an attributed expiry/revocation/invalidation fact."""
        ...

    async def append_audit(self, event: ToolAuditEvent) -> ToolAuditEvent:
        """Append once by event id; a different body must conflict."""
        ...

    async def export_snapshot(self) -> ApprovalStoreSnapshot:
        """Export typed copies for offline serialization validation."""
        ...

    async def get_call(self, invocation_id: str) -> ToolCallEntry | None:
        """Read a prior claim/result; authorization must precede this lookup."""
        ...

    async def claim_call(self, entry: ToolCallEntry) -> None:
        """Atomically reserve one invocation; existing claims always conflict."""
        ...

    async def complete_call(self, invocation_id: str, result: ToolResult) -> None:
        """Append a matching result; never alter a committed result."""
        ...

    async def restore_snapshot(self, snapshot: ApprovalStoreSnapshot) -> None:
        """Validate every fact before installing into an empty test store."""
        ...
