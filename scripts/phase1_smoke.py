"""Offline Phase 1 smoke checks using synthetic, non-sensitive fixtures."""

import asyncio
import os
import sys
from datetime import UTC, datetime
from hashlib import sha256

try:
    from rich.console import Console
except ImportError:  # pragma: no cover - minimal environments without optional rich
    Console = None  # type: ignore[assignment,misc]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.evidence import EvidenceAccessScope, EvidenceItem, EvidenceSourceType, RetrievalStatus
from app.services.evidence_store import EvidenceAccessContext, InMemoryEvidenceStore
from app.services.llm.contracts import ProviderCapabilities, TokenCountQuality, usage_from_response
from app.services.llm.tokenizer import ProviderNeutralTokenizer


def _fixture() -> EvidenceItem:
    """Build a synthetic evidence record without credentials or personal data."""
    snippet = "fixture deployment verification passed"
    digest = sha256(snippet.encode()).hexdigest()
    return EvidenceItem(
        evidence_id="fixture-1",
        source_type=EvidenceSourceType.CI,
        source_uri="fixture://ci/fixture-1",
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_hash=f"sha256:{digest}",
        access_scope=EvidenceAccessScope(public=True),
        snippet=snippet,
        confidence=1.0,
    )


async def _run() -> dict[str, str]:
    """Run local Phase 1 contract checks."""
    store = InMemoryEvidenceStore([_fixture()])
    retrieval = await store.search("deployment verification", access=EvidenceAccessContext())
    tokenizer = ProviderNeutralTokenizer(provider="fixture", model="fixture", allow_estimate=True)
    count = tokenizer.count_text("synthetic phase one")
    unknown_usage = usage_from_response(object(), ProviderCapabilities(supports_usage=True))
    return {
        "evidence": retrieval.status.value,
        "tokenizer_quality": count.quality.value,
        "usage_quality": unknown_usage.quality.value if unknown_usage.quality else "unknown",
        "memory": "disabled_supported",
        "external_verification": "unverified_without_credentials",
    }


def main() -> int:
    """Print safe status labels and return a process status."""
    result = asyncio.run(_run())
    if result["evidence"] != RetrievalStatus.SUCCEEDED.value:
        raise RuntimeError("offline evidence smoke check failed")
    if result["tokenizer_quality"] not in {TokenCountQuality.EXACT.value, TokenCountQuality.ESTIMATE.value}:
        raise RuntimeError("offline tokenizer smoke check failed")
    if Console is not None:
        Console().print(result)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
