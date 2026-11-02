"""Synthetic model boundary with the production v2 Provider validator and real PG ledger."""

import asyncio
import json
import re
from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage

from app.services.llm.contracts import ProviderConfig, LLMRole, UsageInfo, ProviderError, ProviderErrorCode
from app.services.v11_provider import ReviewProvider
from scripts.v11_probe_fixtures import ProbeProvider


class NativeProbeProvider(ReviewProvider):
    """No SDK or real model capability exists in this fixture."""

    provenance = "synthetic"

    def config(self, role: LLMRole) -> ProviderConfig:
        """Use the default pinned settings without constructing a live SDK."""
        return ProviderConfig(
            provider="openai-compatible", model="deepseek-flash", role=role, timeout_seconds=30, max_retries=0
        )

    async def reserve(
        self, actor: Any, command: str, attempt: str, role: LLMRole, input_tokens: int, output_tokens: int
    ) -> ProviderConfig:
        """Reserve only synthetic money in the isolated fixture database."""
        await self.store.reserve(actor, command, attempt, Decimal("0.001"), Decimal("100"))
        return self.config(role)

    async def embed(self, texts: list[str], actor: Any, command: str, attempt: str) -> list[list[float]]:
        """Return deterministic local vectors without an embedding service."""
        return await ProbeProvider(self.store).embed(texts, actor, command, attempt)


async def synthetic_call(messages: Any, config: Any, **kwargs: Any) -> tuple[AIMessage, UsageInfo]:
    """Only deterministic JSON is produced; external transports are not involved."""
    data = json.loads(messages[1].content)
    query = data["query"]
    if "probe-interrupt" in query:
        await asyncio.sleep(60)
    if "probe-unknown" in query:
        error = ProviderError(ProviderErrorCode.TIMEOUT, "synthetic timeout", retryable=False)
        error.call_diagnostic = {"stage": "invoke", "timeout_source": "synthetic", "provider_processing": "unknown"}
        raise error
    diffs = [row for row in data["sources"] if row.get("kind") == "diff"]
    ids = [row["citation_id"] for row in diffs]
    answer: dict[str, Any] = {
        "summary": "The limit changes from 1 to 2.",
        "observations": [{"kind": "observation", "text": "The limit changes from 1 to 2.", "citation_ids": ids}],
        "risks": [],
        "validation_suggestions": [],
        "unknowns": [],
        "candidate_steps": [],
        "model_risk": "HIGH",
        "comparison_contract": "v11-behavior-comparison.2",
        "comparisons": [
            {"section": "observations", "index": 0, "classification": "literal_edit", "before": "1", "after": "2"}
        ],
    }
    if "probe-rejected" in query:
        answer["observations"][0]["citation_ids"] = []
    if "probe-schema" in query:
        answer["model_risk"] = "invalid-enum"
    if "probe-redact" in query:
        answer["summary"] = "token=synthetic-private-example"
    if "probe-truncate" in query:
        answer["summary"] = "中" * 4001
    if diffs and diffs[0].get("path") == "package-lock.json":
        before = next(r["text"] for r in diffs if r["side"] == "LEFT")
        after = next(r["text"] for r in diffs if r["side"] == "RIGHT")
        answer["observations"][0]["text"] = before + " → " + after
        answer["comparisons"][0].update(before=before, after=after)
        url = re.search(r'"resolved": "([^"\n]+)"', after)
        if url:
            answer["observations"][0]["text"] += " English " + url[1] + ". 中文（" + url[1] + "）。"

    return AIMessage(content=json.dumps(answer), response_metadata={"finish_reason": "stop"}), UsageInfo(
        status="known", input_tokens=20, output_tokens=20, total_tokens=40, latency_ms=1.0
    )
