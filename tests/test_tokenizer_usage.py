"""Exact/estimate/unknown token and usage telemetry contracts."""

from app.services.llm.contracts import (
    Availability,
    ProviderCapabilities,
    TokenCountQuality,
    usage_from_response,
)
from app.services.llm.tokenizer import ProviderNeutralTokenizer


class _Encoding:
    """Small deterministic encoder fixture."""

    def encode(self, value: str) -> list[int]:
        """Return one token per whitespace-separated word."""
        return list(range(len(value.split())))


class _Response:
    """Response fixture exposing LangChain usage metadata."""

    usage_metadata = {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}


def test_tokenizer_reports_exact_estimate_and_unknown() -> None:
    """Keep quality visible for real, conservative, and unavailable counters."""
    exact = ProviderNeutralTokenizer(provider="fixture", model="exact", encoding=_Encoding())
    estimate = ProviderNeutralTokenizer(provider="fixture", model="estimate", allow_estimate=True)
    unknown = ProviderNeutralTokenizer(provider="fixture", model="unknown", allow_estimate=False)

    assert exact.count_text("one two").quality is TokenCountQuality.EXACT
    assert estimate.count_text("12345").quality is TokenCountQuality.ESTIMATE
    assert unknown.count_text("12345").quality is TokenCountQuality.UNKNOWN
    assert unknown.count_text("12345").value is None


def test_tokenizer_handles_structured_messages() -> None:
    """Count text blocks without serializing reasoning or provider payloads."""
    tokenizer = ProviderNeutralTokenizer(provider="fixture", model="estimate", allow_estimate=True)

    result = tokenizer.count_messages(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "reasoning"}]}]
    )

    assert result.value is not None
    assert result.quality is TokenCountQuality.ESTIMATE


def test_usage_without_provider_metadata_is_unknown() -> None:
    """Never convert absent usage metadata into zero tokens or exact cost."""
    unknown = usage_from_response(object(), ProviderCapabilities(supports_usage=True))

    assert unknown.status == "unknown"
    assert unknown.quality is TokenCountQuality.UNKNOWN
    assert unknown.input_tokens is None
    assert unknown.cost_usd is None


def test_usage_metadata_is_provider_neutral_and_validated() -> None:
    """Extract validated counts while retaining unknown cost semantics."""
    usage = usage_from_response(_Response(), ProviderCapabilities(supports_usage=True))

    assert usage.status == "known"
    assert usage.quality is TokenCountQuality.EXACT
    assert usage.total_tokens == 7
    assert usage.availability is Availability.UNKNOWN
