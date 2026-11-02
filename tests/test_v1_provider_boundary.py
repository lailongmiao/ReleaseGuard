"""Exercise the actual SDK wire shape without network or paid calls."""

import json
from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.schemas.local_mvp import LocalActor
from app.services.llm import registry
from app.services.llm.contracts import LLMRole, ProviderConfig
from app.services.llm.registry import LLMRegistry
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_provider import ANALYSIS_OUTPUT_TOKEN_LIMIT, BudgetedProvider
from app.services.v1_rag import chunk_documents
from app.services.v1_sources import MarkdownSource
from app.services.v1_storage import V1Store


@pytest.fixture
def anyio_backend() -> str:
    """Match the application's async runtime."""
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "outcome", ["success", "timeout", "budget_denied", "fenced_json", "invalid_json", "empty", "truncated"]
)
async def test_native_output_limit_matches_reservation(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    """Catch SDK parameter renaming, under-reservation and hidden retries."""
    monkeypatch.setattr(settings, "V1_CHAT_MODEL", "deepseek-flash")
    monkeypatch.setattr(settings, "V1_REAL_MODELS_ENABLED", True)
    monkeypatch.setattr(settings, "LANGFUSE_TRACING_ENABLED", False)
    monkeypatch.setattr("app.services.v1_provider.PRICE_EXPIRY", "2099-01-01T00:00:00+00:00")
    config = ProviderConfig(
        role=LLMRole.CHAT,
        provider="openai-compatible",
        model="deepseek-flash",
        endpoint="https://api.deepseek.com",
        api_key="synthetic-no-network",
    )
    monkeypatch.setattr(LLMRegistry, "get_config", lambda *args: config)
    chunks = chunk_documents((await MarkdownSource().collect()).documents[:1], "synthetic-generation")
    store = AsyncMock(spec=V1Store)
    if outcome == "budget_denied":
        store.reserve.side_effect = DomainError("model_budget_exhausted")
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload["max_tokens"] == ANALYSIS_OUTPUT_TOKEN_LIMIT
        assert "max_completion_tokens" not in payload
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "low"
        assert payload["response_format"] == {"type": "json_object"}
        if outcome == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        output = {
            "explanation": "Synthetic protocol fixture.",
            "claims": [{"text": "Synthetic protocol fixture.", "citation_ids": [chunks[0].chunk_id]}],
            "risk": "HIGH",
            "unknowns": [],
            "candidate_steps": [],
        }
        content = json.dumps(output)
        if outcome == "fenced_json":
            content = "```json\n" + content + "\n```"
        elif outcome == "invalid_json":
            content = '{"explanation": "unescaped\nnewline"}'
        elif outcome == "empty":
            content = ""
        return httpx.Response(
            200,
            json={
                "id": "synthetic-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if outcome == "truncated" else "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        def adapter(**kwargs):
            assert kwargs["max_retries"] == 0
            return ChatOpenAI(**kwargs, http_async_client=client)

        monkeypatch.setattr(registry, "ChatOpenAI", adapter)
        provider = BudgetedProvider(store, enabled=True)
        actor = LocalActor(user_id=1, session_id="synthetic-session", expires_at=4_000_000_000)
        if outcome != "success":
            reason = (
                "analysis_outcome_unknown"
                if outcome == "timeout"
                else "model_budget_exhausted"
                if outcome == "budget_denied"
                else "analysis_invalid_response"
            )
            with pytest.raises(DomainError, match=reason):
                await provider.analyze("Synthetic query", chunks, actor, "synthetic-command", "synthetic-analysis")
        else:
            _, telemetry = await provider.analyze(
                "Synthetic query", chunks, actor, "synthetic-command", "synthetic-analysis"
            )
            assert telemetry["output_token_limit"] == ANALYSIS_OUTPUT_TOKEN_LIMIT
    assert len(calls) == (0 if outcome == "budget_denied" else 1)
    store.reserve.assert_awaited_once()
    if calls:
        input_bound = sum(len(m["content"].encode("utf-8")) for m in calls[0]["messages"]) + 64
        expected = (Decimal(input_bound) * 2 + Decimal(ANALYSIS_OUTPUT_TOKEN_LIMIT) * 8) / 1_000_000
        assert store.reserve.await_args.args[3] == expected
    if outcome in {"fenced_json", "invalid_json", "empty", "truncated"}:
        diagnostic = store.usage.await_args.args[3]["response_diagnostic"]
        assert diagnostic["finish_reason"] == ("length" if outcome == "truncated" else "stop")
        assert diagnostic["content_chars"] == 0 if outcome == "empty" else diagnostic["content_chars"] > 0
        assert diagnostic["code_fence"] == (outcome == "fenced_json")
        assert "content" not in diagnostic and "reasoning_content" not in diagnostic
