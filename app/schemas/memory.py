"""Provider-neutral long-term memory degradation contracts."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class MemoryStatus(StrEnum):
    """Observable outcome of a memory lookup."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"


class MemorySearchResult(BaseModel):
    """Safe memory lookup result used by Chat and Streaming callers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: MemoryStatus
    content: str = Field(default="", max_length=12000)
    degradation_reason: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = None
    error_category: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = None
    latency_ms: float | None = Field(default=None, ge=0)

    @classmethod
    def disabled(cls, reason: str) -> "MemorySearchResult":
        """Create a disabled result without exposing query or user data."""
        return cls(status=MemoryStatus.DISABLED, degradation_reason=reason)

    @classmethod
    def empty(cls, latency_ms: float | None = None) -> "MemorySearchResult":
        """Create an explicit empty result."""
        return cls(status=MemoryStatus.EMPTY, latency_ms=latency_ms)

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        error_category: str | None = None,
        latency_ms: float | None = None,
    ) -> "MemorySearchResult":
        """Create an unavailable result suitable for degraded chat."""
        return cls(
            status=MemoryStatus.UNAVAILABLE,
            degradation_reason=reason,
            error_category=error_category,
            latency_ms=latency_ms,
        )
