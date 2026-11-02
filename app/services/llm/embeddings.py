"""Validated LangChain embedding adapter for provider-neutral model roles."""

from __future__ import annotations

import math
import asyncio
import logging
from collections.abc import Sequence
from numbers import Real
from time import perf_counter
from typing import Any
from typing_extensions import override

from langchain_core.embeddings import Embeddings
from pydantic import PrivateAttr
from tenacity import Retrying, before_sleep_log, retry_if_exception, stop_after_attempt, wait_exponential

from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, RateLimitError

from app.core.logging import logger
from app.services.llm.contracts import (
    EmbeddingResult,
    EmbeddingStatus,
    ProviderConfig,
    ProviderError,
    ProviderErrorCode,
)


class DimensionValidatedEmbeddings(Embeddings):
    """Wrap an embedding adapter and reject vectors that violate its contract."""

    _adapter: Embeddings = PrivateAttr()
    _config: ProviderConfig = PrivateAttr()

    def __init__(self, adapter: Embeddings, config: ProviderConfig) -> None:
        """Wrap an embedding adapter with the role's dimension contract."""
        super().__init__()
        self._adapter = adapter
        self._config = config

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch and validate each returned vector."""
        if not texts:
            return []
        vectors = self._invoke(lambda: self._adapter.embed_documents(texts))
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            self._raise_invalid_response("embedding provider returned an unexpected vector count")
        return [self._validate_vector(vector) for vector in vectors]

    @override
    def embed_query(self, text: str) -> list[float]:
        """Embed one query and validate its returned vector."""
        vector = self._invoke(lambda: self._adapter.embed_query(text))
        return self._validate_vector(vector)

    @override
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Run the synchronous provider adapter without blocking the event loop."""
        return await asyncio.to_thread(self.embed_documents, texts)

    @override
    async def aembed_query(self, text: str) -> list[float]:
        """Run a query embedding without blocking the event loop."""
        return await asyncio.to_thread(self.embed_query, text)

    def embed_documents_result(self, texts: list[str]) -> EmbeddingResult:
        """Return a classified result for a batch without leaking provider payloads."""
        started = perf_counter()
        dimensions = self._config.capabilities.embedding_dimensions
        if not texts:
            return EmbeddingResult.empty(
                provider=self._config.provider,
                model=self._config.model,
                dimensions=dimensions,
                latency_ms=_latency_ms(started),
            )
        try:
            vectors = self.embed_documents(texts)
        except ProviderError as exc:
            return EmbeddingResult.failure(
                _embedding_status(exc),
                provider=self._config.provider,
                model=self._config.model,
                dimensions=dimensions,
                latency_ms=_latency_ms(started),
                error_category=exc.code,
            )
        return EmbeddingResult(
            status=EmbeddingStatus.SUCCEEDED,
            vectors=tuple(tuple(vector) for vector in vectors),
            provider=self._config.provider,
            model=self._config.model,
            dimensions=dimensions,
            latency_ms=_latency_ms(started),
        )

    def embed_query_result(self, text: str) -> EmbeddingResult:
        """Return a classified result for one query."""
        started = perf_counter()
        dimensions = self._config.capabilities.embedding_dimensions
        try:
            vector = self.embed_query(text)
        except ProviderError as exc:
            return EmbeddingResult.failure(
                _embedding_status(exc),
                provider=self._config.provider,
                model=self._config.model,
                dimensions=dimensions,
                latency_ms=_latency_ms(started),
                error_category=exc.code,
            )
        return EmbeddingResult(
            status=EmbeddingStatus.SUCCEEDED,
            vectors=(tuple(vector),),
            provider=self._config.provider,
            model=self._config.model,
            dimensions=dimensions,
            latency_ms=_latency_ms(started),
        )

    def _invoke(self, call: Any) -> Any:
        retrying = Retrying(
            stop=stop_after_attempt(self._config.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception(lambda exc: isinstance(exc, ProviderError) and exc.retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                try:
                    return call()
                except ProviderError:
                    raise
                except AuthenticationError as exc:
                    raise self._provider_error(
                        ProviderErrorCode.AUTHENTICATION, "embedding provider authentication failed", exc
                    )
                except RateLimitError as exc:
                    raise self._provider_error(
                        ProviderErrorCode.RATE_LIMITED, "embedding provider rate limit reached", exc, retryable=True
                    )
                except (APITimeoutError, TimeoutError, asyncio.TimeoutError) as exc:
                    raise self._provider_error(
                        ProviderErrorCode.TIMEOUT, "embedding provider timed out", exc, retryable=True
                    )
                except (APIConnectionError, ConnectionError, OSError) as exc:
                    raise self._provider_error(
                        ProviderErrorCode.UNAVAILABLE, "embedding provider is unavailable", exc, retryable=True
                    )
                except APIError as exc:
                    raise self._provider_error(
                        ProviderErrorCode.UNAVAILABLE, "embedding provider request failed", exc, retryable=True
                    )
                except Exception as exc:
                    raise self._provider_error(
                        ProviderErrorCode.UNAVAILABLE, "embedding provider request failed", exc, retryable=True
                    )
        raise AssertionError("embedding retry loop completed without a result")

    def _validate_vector(self, vector: Any) -> list[float]:
        expected = self._config.capabilities.embedding_dimensions
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or len(vector) != expected:
            actual = len(vector) if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)) else 0
            self._raise_invalid_response(
                f"embedding vector dimension mismatch: expected {expected}, received {actual}"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
            for value in vector
        ):
            self._raise_invalid_response("embedding provider returned a non-numeric vector")
        return [float(value) for value in vector]

    def _raise_invalid_response(self, message: str) -> None:
        raise self._provider_error(ProviderErrorCode.RESPONSE_INVALID, message)

    def _provider_error(
        self,
        code: ProviderErrorCode,
        message: str,
        cause: BaseException | None = None,
        *,
        retryable: bool = False,
    ) -> ProviderError:
        return ProviderError(
            code,
            message,
            provider=self._config.provider,
            model=self._config.model,
            retryable=retryable,
            cause=cause,
        )


def _latency_ms(started: float) -> float:
    """Return a rounded monotonic duration for diagnostics."""
    return round((perf_counter() - started) * 1000, 3)


def _embedding_status(error: ProviderError) -> EmbeddingStatus:
    """Map the provider taxonomy to the embedding result taxonomy."""
    if error.code is ProviderErrorCode.TIMEOUT:
        return EmbeddingStatus.TIMEOUT
    if error.code in {ProviderErrorCode.UNAVAILABLE, ProviderErrorCode.RATE_LIMITED}:
        return EmbeddingStatus.UNAVAILABLE
    if error.code is ProviderErrorCode.RESPONSE_INVALID:
        if "dimension" in str(error).lower():
            return EmbeddingStatus.DIMENSION_MISMATCH
        return EmbeddingStatus.INVALID_RESPONSE
    return EmbeddingStatus.INVALID_RESPONSE
