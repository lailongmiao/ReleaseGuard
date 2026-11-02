"""Provider-neutral token counting with explicit exact/estimate/unknown quality."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import BaseMessage

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional provider tokenizer
    tiktoken = None  # type: ignore[assignment]

from app.core.logging import logger
from app.services.llm.contracts import TokenCountQuality


@dataclass(frozen=True, slots=True)
class TokenCount:
    """A token count that never hides whether it is exact."""

    value: int | None
    quality: TokenCountQuality
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        """Validate count and quality invariants."""
        if self.value is not None and (
            isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 0
        ):
            raise ValueError("token count must be a non-negative integer or None")
        if self.quality is TokenCountQuality.UNKNOWN and self.value is not None:
            raise ValueError("unknown token count cannot contain a value")
        if self.quality is not TokenCountQuality.UNKNOWN and self.value is None:
            raise ValueError("exact or estimated token count requires a value")

    @classmethod
    def unknown(cls, *, provider: str | None = None, model: str | None = None) -> "TokenCount":
        """Create an explicit unknown count."""
        return cls(None, TokenCountQuality.UNKNOWN, provider, model)


class Tokenizer(Protocol):
    """Minimal provider-neutral tokenizer interface."""

    def count_text(self, text: str) -> TokenCount:
        """Count one text value."""
        ...

    def count_messages(self, messages: Sequence[Any]) -> TokenCount:
        """Count text and structured message content."""
        ...


class ProviderNeutralTokenizer:
    """Use a configured exact encoder or a conservative character estimate."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        encoding: Any | None = None,
        allow_estimate: bool = True,
    ) -> None:
        """Initialize without requiring a provider SDK or network call."""
        self.provider = provider
        self.model = model
        self._encoding = encoding
        self._allow_estimate = allow_estimate

    @classmethod
    def from_model(
        cls,
        model: str,
        *,
        provider: str = "openai",
        tokenizer_name: str | None = None,
        allow_estimate: bool = True,
    ) -> "ProviderNeutralTokenizer":
        """Create an exact tokenizer only when a real encoding is available."""
        encoding = None
        if tiktoken is not None and (
            provider.lower() in {"openai", "openai-compatible", "openai_compatible"} or tokenizer_name
        ):
            try:
                encoding = (
                    tiktoken.get_encoding(tokenizer_name) if tokenizer_name else tiktoken.encoding_for_model(model)
                )
            except (AttributeError, KeyError, ValueError):
                logger.debug("tokenizer_exact_encoding_unavailable", provider=provider, model=model)
        return cls(provider=provider, model=model, encoding=encoding, allow_estimate=allow_estimate)

    def count_text(self, text: str) -> TokenCount:
        """Count text exactly, conservatively, or as unknown."""
        if not isinstance(text, str):
            return TokenCount.unknown(provider=self.provider, model=self.model)
        if self._encoding is not None:
            return TokenCount(len(self._encoding.encode(text)), TokenCountQuality.EXACT, self.provider, self.model)
        if self._allow_estimate:
            # Four characters per token is intentionally conservative for
            # context budgeting and is always labelled as an estimate.
            return TokenCount(max(0, math.ceil(len(text) / 4)), TokenCountQuality.ESTIMATE, self.provider, self.model)
        return TokenCount.unknown(provider=self.provider, model=self.model)

    def count_messages(self, messages: Sequence[Any]) -> TokenCount:
        """Count message text and structured content blocks."""
        if not isinstance(messages, Sequence):
            return TokenCount.unknown(provider=self.provider, model=self.model)
        total = 4 * len(messages) + 2
        quality = TokenCountQuality.EXACT if self._encoding is not None else TokenCountQuality.ESTIMATE
        for message in messages:
            for text in _message_texts(message):
                item = self.count_text(text)
                if item.quality is TokenCountQuality.UNKNOWN:
                    return TokenCount.unknown(provider=self.provider, model=self.model)
                total += item.value or 0
                if item.quality is TokenCountQuality.ESTIMATE:
                    quality = TokenCountQuality.ESTIMATE
        return TokenCount(total, quality, self.provider, self.model)


def _message_texts(message: Any) -> list[str]:
    """Extract text from dicts, LangChain messages, and content blocks."""
    value: Any
    if isinstance(message, Mapping):
        value = message.get("content", "")
    elif isinstance(message, BaseMessage):
        value = message.content
    else:
        value = str(message)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        texts: list[str] = []
        for block in value:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                texts.append(block["text"])
        return texts
    return []
