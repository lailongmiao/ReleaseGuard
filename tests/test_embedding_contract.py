"""Contract tests for provider-neutral embedding validation and degradation."""

import pytest
from langchain_core.embeddings import Embeddings

from app.services.llm.contracts import (
    EmbeddingStatus,
    LLMRole,
    ProviderCapabilities,
    ProviderConfig,
    ProviderErrorCode,
)
from app.services.llm.embeddings import DimensionValidatedEmbeddings


def _config(*, retries: int = 1) -> ProviderConfig:
    return ProviderConfig(
        role=LLMRole.EMBEDDING,
        provider="fixture",
        model="fixture-embed",
        capabilities=ProviderCapabilities(supports_embeddings=True, embedding_dimensions=3),
        max_retries=retries,
    )


class _FakeEmbeddings(Embeddings):
    def __init__(self, documents: list[list[float]], query: list[float] | None = None) -> None:
        self.documents = documents
        self.query = query or [0.1, 0.2, 0.3]
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return self.documents

    def embed_query(self, text: str) -> list[float]:
        return self.query


def test_embedding_vectors_validate_dimension_and_finite_numbers() -> None:
    """Accept a vector that matches the declared finite dimensions."""
    adapter = DimensionValidatedEmbeddings(_FakeEmbeddings([[0.1, 0.2, 0.3]]), _config())

    result = adapter.embed_documents_result(["hello"])

    assert result.status is EmbeddingStatus.SUCCEEDED
    assert result.vectors == ((0.1, 0.2, 0.3),)


@pytest.mark.parametrize(
    "vector, expected_status",
    [
        ([0.1, 0.2], EmbeddingStatus.DIMENSION_MISMATCH),
        ([0.1, float("nan"), 0.3], EmbeddingStatus.INVALID_RESPONSE),
        ([0.1, float("inf"), 0.3], EmbeddingStatus.INVALID_RESPONSE),
        ([0.1, "bad", 0.3], EmbeddingStatus.INVALID_RESPONSE),
    ],
)
def test_embedding_invalid_vectors_are_classified(vector: list[object], expected_status: EmbeddingStatus) -> None:
    """Classify dimension, NaN, infinity, and non-numeric responses."""
    adapter = DimensionValidatedEmbeddings(_FakeEmbeddings([vector]), _config())  # type: ignore[list-item]

    result = adapter.embed_documents_result(["hello"])

    assert result.status is expected_status
    assert result.error_category is ProviderErrorCode.RESPONSE_INVALID


def test_empty_batch_is_explicit_and_does_not_call_provider() -> None:
    """Return an explicit empty result without invoking the provider."""
    fake = _FakeEmbeddings([])
    adapter = DimensionValidatedEmbeddings(fake, _config())

    result = adapter.embed_documents_result([])

    assert result.status is EmbeddingStatus.EMPTY
    assert fake.document_calls == 0


def test_timeout_is_retried_and_classified_without_provider_payload() -> None:
    """Retry transient timeouts and return a safe classified failure."""

    class TimeoutEmbeddings(_FakeEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.document_calls += 1
            raise TimeoutError("fixture timeout")

    fake = TimeoutEmbeddings([])
    adapter = DimensionValidatedEmbeddings(fake, _config(retries=2))

    result = adapter.embed_documents_result(["hello"])

    assert result.status is EmbeddingStatus.TIMEOUT
    assert result.error_category is ProviderErrorCode.TIMEOUT
    assert fake.document_calls == 3


def test_provider_recovery_after_one_transient_failure() -> None:
    """Return success after a retry recovers from a transient outage."""

    class RecoveringEmbeddings(_FakeEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.document_calls += 1
            if self.document_calls == 1:
                raise ConnectionError("fixture unavailable")
            return [[0.1, 0.2, 0.3]]

    fake = RecoveringEmbeddings([])
    adapter = DimensionValidatedEmbeddings(fake, _config(retries=1))

    result = adapter.embed_documents_result(["hello"])

    assert result.status is EmbeddingStatus.SUCCEEDED
    assert fake.document_calls == 2
