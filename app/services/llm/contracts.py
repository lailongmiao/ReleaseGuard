"""Provider-neutral contracts shared by LLM integrations.

The application depends on these values instead of provider-specific model names,
exception classes, or response formats. Concrete adapters may still use a
provider SDK internally, but the rest of the application only sees these types.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from numbers import Integral, Real
from typing import Any


class LLMRole(StrEnum):
    """Independent model roles used by the application."""

    CHAT = "chat"
    MEMORY_LLM = "memory_llm"
    EMBEDDING = "embedding"
    NAMING = "naming"
    JUDGE = "judge"


class ProviderErrorCode(StrEnum):
    """Stable error categories exposed by provider adapters."""

    CONFIGURATION = "provider_configuration_error"
    AUTHENTICATION = "provider_authentication_error"
    CAPABILITY = "provider_capability_error"
    RATE_LIMITED = "provider_rate_limited"
    TIMEOUT = "provider_timeout"
    RESPONSE_INVALID = "provider_response_invalid"
    UNAVAILABLE = "provider_unavailable"
    TOKENIZER_UNKNOWN = "tokenizer_unknown"


class EmbeddingStatus(StrEnum):
    """Stable outcome categories for embedding operations."""

    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    DIMENSION_MISMATCH = "dimension_mismatch"
    INVALID_RESPONSE = "invalid_response"


class TokenCountQuality(StrEnum):
    """How a token count was obtained."""

    EXACT = "exact"
    ESTIMATE = "estimate"
    UNKNOWN = "unknown"


class Availability(StrEnum):
    """Provider availability attached to usage telemetry."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Provider-neutral result for a validated embedding request.

    ``vectors`` is populated only for successful requests.  Error messages are
    intentionally short and classified so callers can expose degradation state
    without serialising a provider response or user input.
    """

    status: EmbeddingStatus
    vectors: tuple[tuple[float, ...], ...] = ()
    provider: str | None = None
    model: str | None = None
    dimensions: int | None = None
    latency_ms: float | None = None
    error_category: ProviderErrorCode | None = None

    def __post_init__(self) -> None:
        """Validate result consistency at the adapter boundary."""
        if self.status is EmbeddingStatus.SUCCEEDED and not self.vectors:
            raise ValueError("successful embedding result must contain vectors")
        if self.status is not EmbeddingStatus.SUCCEEDED and self.vectors:
            raise ValueError("non-success embedding result cannot contain vectors")
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool) or not isinstance(self.dimensions, Integral) or self.dimensions <= 0
        ):
            raise ValueError("embedding dimensions must be positive")
        if self.latency_ms is not None and (self.latency_ms < 0 or not isfinite(self.latency_ms)):
            raise ValueError("embedding latency must be finite and non-negative")

    @classmethod
    def empty(cls, *, provider: str, model: str, dimensions: int | None, latency_ms: float) -> "EmbeddingResult":
        """Create a valid result for an empty input batch."""
        return cls(
            status=EmbeddingStatus.EMPTY,
            provider=provider,
            model=model,
            dimensions=dimensions,
            latency_ms=latency_ms,
        )

    @classmethod
    def failure(
        cls,
        status: EmbeddingStatus,
        *,
        provider: str,
        model: str,
        dimensions: int | None,
        latency_ms: float,
        error_category: ProviderErrorCode,
    ) -> "EmbeddingResult":
        """Create a classified failure result without provider payloads."""
        return cls(
            status=status,
            provider=provider,
            model=model,
            dimensions=dimensions,
            latency_ms=latency_ms,
            error_category=error_category,
        )


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Declared capabilities for one provider/model pair."""

    supports_chat: bool = False
    supports_streaming: bool = False
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_reasoning: bool = False
    supports_usage: bool = False
    supports_embeddings: bool = False
    context_window: int | None = None
    embedding_dimensions: int | None = None
    tokenizer: str | None = None

    def __post_init__(self) -> None:
        """Validate capability values before an adapter consumes them."""
        for name in (
            "supports_chat",
            "supports_streaming",
            "supports_tools",
            "supports_structured_output",
            "supports_reasoning",
            "supports_usage",
            "supports_embeddings",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

        if self.context_window is not None and (
            isinstance(self.context_window, bool)
            or not isinstance(self.context_window, Integral)
            or self.context_window <= 0
        ):
            raise ValueError("context_window must be positive")
        if self.embedding_dimensions is not None and (
            isinstance(self.embedding_dimensions, bool)
            or not isinstance(self.embedding_dimensions, Integral)
            or self.embedding_dimensions <= 0
        ):
            raise ValueError("embedding_dimensions must be positive")
        if self.tokenizer is not None and (not isinstance(self.tokenizer, str) or not self.tokenizer.strip()):
            raise ValueError("tokenizer must not be empty")

    def supports(self, capability: str) -> bool:
        """Return whether a named capability is declared for this model."""
        field_name = capability if capability.startswith("supports_") else f"supports_{capability}"
        value = getattr(self, field_name, None)
        if value is None:
            raise ValueError(f"unknown provider capability: {capability}")
        return bool(value)

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a serializable capability summary with no credentials."""
        return {
            "supports_chat": self.supports_chat,
            "supports_streaming": self.supports_streaming,
            "supports_tools": self.supports_tools,
            "supports_structured_output": self.supports_structured_output,
            "supports_reasoning": self.supports_reasoning,
            "supports_usage": self.supports_usage,
            "supports_embeddings": self.supports_embeddings,
            "context_window": self.context_window,
            "embedding_dimensions": self.embedding_dimensions,
            "tokenizer": self.tokenizer,
        }


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Configuration and capabilities for one independent application role."""

    role: LLMRole
    provider: str
    model: str
    endpoint: str | None = None
    credentials_ref: str | None = None
    api_key: str | None = field(default=None, repr=False, compare=False)
    capabilities: ProviderCapabilities = ProviderCapabilities()
    fallback_models: tuple[str, ...] = ()
    parse_mode: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    api_key_env: str | None = None
    allow_global_api_key_fallback: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration invariants before an adapter is created."""
        if not isinstance(self.role, LLMRole):
            try:
                object.__setattr__(self, "role", LLMRole(self.role))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown provider role: {self.role!r}") from exc
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.endpoint is not None and (not isinstance(self.endpoint, str) or not self.endpoint.strip()):
            raise ValueError("endpoint must not be empty")
        if self.credentials_ref is not None and (
            not isinstance(self.credentials_ref, str) or not self.credentials_ref.strip()
        ):
            raise ValueError("credentials_ref must not be empty")
        if self.api_key_env is not None and (not isinstance(self.api_key_env, str) or not self.api_key_env.strip()):
            raise ValueError("api_key_env must not be empty")
        if self.api_key is not None and (not isinstance(self.api_key, str) or not self.api_key.strip()):
            raise ValueError("api_key must not be empty")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, Real)
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, Integral) or self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if isinstance(self.fallback_models, str):
            raise ValueError("fallback_models must be a sequence of model names")
        if not isinstance(self.fallback_models, tuple):
            try:
                object.__setattr__(self, "fallback_models", tuple(self.fallback_models))
            except TypeError as exc:
                raise ValueError("fallback_models must be a sequence of model names") from exc
        if any(not isinstance(model, str) or not model.strip() for model in self.fallback_models):
            raise ValueError("fallback_models must contain non-empty model names")
        if self.parse_mode is not None and (not isinstance(self.parse_mode, str) or not self.parse_mode.strip()):
            raise ValueError("parse_mode must not be empty")
        if not isinstance(self.capabilities, ProviderCapabilities):
            raise ValueError("capabilities must be a ProviderCapabilities instance")
        if self.role is LLMRole.EMBEDDING and not self.capabilities.supports_embeddings:
            raise ValueError("embedding role requires supports_embeddings")

    def to_safe_dict(self) -> dict[str, Any]:
        """Return configuration metadata suitable for logs and diagnostics."""
        return {
            "role": self.role.value,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "credentials_ref": self.credentials_ref,
            "api_key_env": self.api_key_env,
            "capabilities": self.capabilities.to_safe_dict(),
            "fallback_models": list(self.fallback_models),
            "parse_mode": self.parse_mode,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }

    def safe_summary(self) -> dict[str, Any]:
        """Alias for :meth:`to_safe_dict` used by bootstrap diagnostics."""
        return self.to_safe_dict()


@dataclass(frozen=True, slots=True)
class UsageInfo:
    """Token usage with an explicit unknown state."""

    status: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    quality: TokenCountQuality | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    provider: str | None = None
    model: str | None = None
    availability: Availability = Availability.UNKNOWN
    error_category: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed token counts instead of reporting false precision."""
        if self.status not in {"known", "unknown"}:
            raise ValueError("usage status must be known or unknown")
        if self.quality is None:
            object.__setattr__(
                self,
                "quality",
                TokenCountQuality.EXACT if self.status == "known" else TokenCountQuality.UNKNOWN,
            )
        elif not isinstance(self.quality, TokenCountQuality):
            try:
                object.__setattr__(self, "quality", TokenCountQuality(self.quality))
            except (TypeError, ValueError) as exc:
                raise ValueError("usage quality must be exact, estimate, or unknown") from exc
        if self.status == "known" and self.quality is TokenCountQuality.UNKNOWN:
            raise ValueError("known usage cannot have unknown quality")
        if self.status == "unknown" and self.quality is not TokenCountQuality.UNKNOWN:
            raise ValueError("unknown usage must have unknown quality")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.status == "unknown" and any(
            value is not None for value in (self.input_tokens, self.output_tokens, self.total_tokens)
        ):
            raise ValueError("unknown usage cannot contain token counts")
        if self.cost_usd is not None and (
            isinstance(self.cost_usd, bool)
            or not isinstance(self.cost_usd, Real)
            or not isfinite(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be finite and non-negative")
        if self.latency_ms is not None and (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, Real)
            or not isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be finite and non-negative")
        if not isinstance(self.availability, Availability):
            try:
                object.__setattr__(self, "availability", Availability(self.availability))
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid availability") from exc

    @classmethod
    def unknown(cls) -> "UsageInfo":
        """Create a usage value for providers that do not report token counts."""
        return cls(status="unknown", quality=TokenCountQuality.UNKNOWN)


class ProviderError(RuntimeError):
    """Provider-neutral exception carrying retry and routing information."""

    def __init__(
        self,
        code: ProviderErrorCode,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        retryable: bool = False,
        cause: BaseException | None = None,
    ) -> None:
        """Initialize an error with stable routing metadata and an optional cause."""
        super().__init__(message)
        try:
            self.code = code if isinstance(code, ProviderErrorCode) else ProviderErrorCode(code)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown provider error code: {code!r}") from exc
        self.provider = provider
        self.model = model
        self.retryable = retryable
        self.cause = cause
        self.call_diagnostic: dict[str, Any] | None = None

    @property
    def category(self) -> str:
        """Return the stable wire value used for logs and API errors."""
        return self.code.value

    def to_safe_dict(self) -> dict[str, Any]:
        """Return error metadata without serializing the underlying exception."""
        return {
            "code": self.code.value,
            "message": str(self),
            "provider": self.provider,
            "model": self.model,
            "retryable": self.retryable,
        }


def capability_error(message: str, config: ProviderConfig) -> ProviderError:
    """Build a consistent capability error for a provider configuration."""
    return ProviderError(
        ProviderErrorCode.CAPABILITY,
        message,
        provider=config.provider,
        model=config.model,
    )


def usage_from_response(response: Any, capabilities: ProviderCapabilities) -> UsageInfo:
    """Extract usage metadata without assuming a provider response shape."""
    if not capabilities.supports_usage:
        return UsageInfo.unknown()

    metadata = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {})
    if isinstance(metadata, dict):
        usage = metadata.get("token_usage") or metadata.get("usage") or metadata
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
            total_tokens = usage.get("total_tokens")
            if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
                total_tokens = input_tokens + output_tokens
            values = (input_tokens, output_tokens, total_tokens)
            if any(value is not None for value in values) and all(
                value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)
                for value in values
            ):
                return UsageInfo(
                    status="known",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
    return UsageInfo.unknown()
