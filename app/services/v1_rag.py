"""Versioned bounded chunks and deterministic exact-search rank fusion."""

import re
from typing import Sequence

from app.schemas.v1_backend import Chunk, SourceDocument
from app.services.llm.tokenizer import ProviderNeutralTokenizer
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError

STRATEGY = "heading-cl100k-512-overlap64-rrf60-positive-lexical-cjk2.v5"


def chinese_terms(query: str) -> tuple[str, ...]:
    """Bound literal Chinese bigrams; never translate or expand the query."""
    terms = dict.fromkeys(
        span[index : index + 2] for span in re.findall(r"[\u4e00-\u9fff]+", query) for index in range(len(span) - 1)
    )
    return tuple(terms)[:64]


def chunk_documents(documents: Sequence[SourceDocument], generation: str) -> list[Chunk]:
    """Keep paragraphs together where possible, with a declared local tokenizer."""
    tokenizer = ProviderNeutralTokenizer.from_model("", provider="openai-compatible", tokenizer_name="cl100k_base")
    encoding = tokenizer._encoding
    if encoding is None:
        raise DomainError("index_tokenizer_unavailable")
    chunks = []
    for document in documents:
        windows: list[tuple[int, int]] = []
        start = 0
        content = document.content
        while start < len(content):
            tokens = encoding.encode(content[start:])
            if len(tokens) <= 512:
                windows.append((start, len(content)))
                break
            _, offsets = encoding.decode_with_offsets(tokens)
            end = start + offsets[512]
            while len(encoding.encode(content[start:end])) > 512:
                end -= 1
            # A topical heading is a stronger boundary than a URL paragraph.
            # Avoid cutting short documents or heading-only overlap windows.
            headings = [
                start + match.start()
                for match in re.finditer(r"(?m)^#{1,6} ", content[start:end])
                if len(encoding.encode(content[start : start + match.start()])) >= 128
            ]
            # Prefer a paragraph/heading boundary inside the bounded window.
            paragraph = content.rfind("\n\n", start, end)
            if headings:
                end = headings[-1]
            elif paragraph > start and len(encoding.encode(content[start:paragraph])) >= 256:
                end = paragraph + 2
            windows.append((start, end))
            window_tokens = encoding.encode(content[start:end])
            _, window_offsets = encoding.decode_with_offsets(window_tokens)
            overlap = window_offsets[max(0, len(window_tokens) - 64)]
            start = max(start + 1, start + overlap)
        for ordinal, (start, end) in enumerate(windows):
            text = content[start:end]
            digest = canonical_hash(text)
            chunks.append(
                Chunk(
                    chunk_id="chunk:"
                    + canonical_hash({"document": document.document_id, "ordinal": ordinal, "hash": digest}),
                    document=document,
                    ordinal=ordinal,
                    text=text,
                    text_hash=digest,
                    tokens=len(encoding.encode(text)),
                    vector=[],
                    generation=generation,
                    start_char=start,
                    end_char=end,
                    locator=document.locator + f"#chars={start}-{end}",
                )
            )
    if not chunks or len(chunks) > 500:
        raise DomainError("index_chunk_limit_or_empty")
    return chunks


def fuse(lexical: Sequence[Chunk], vector: Sequence[Chunk], limit: int = 5) -> list[Chunk]:
    """RRF 60, stable chunk ties, then one winning chunk per document."""
    scores: dict[str, float] = {}
    values: dict[str, Chunk] = {}
    for ranking in (lexical[:20], vector[:20]):
        for rank, chunk in enumerate(ranking, 1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (60 + rank)
            values[chunk.chunk_id] = chunk
    result, documents = [], set()
    for identifier in sorted(scores, key=lambda key: (-scores[key], key)):
        chunk = values[identifier]
        if chunk.document.document_id in documents:
            continue
        documents.add(chunk.document.document_id)
        result.append(chunk)
        if len(result) >= limit:
            break
    return result
