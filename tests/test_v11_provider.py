"""Actual native SDK wire boundary with zero network and zero paid requests."""

import json
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_openai import ChatOpenAI
from openai import APITimeoutError

from app.core.config import settings
from app.schemas.local_mvp import LocalActor
from app.schemas.v11_backend import ReviewCitation
from app.services.llm import registry
from app.services.llm.contracts import LLMRole, ProviderConfig, ProviderError, ProviderErrorCode
from app.services.llm.service import llm_service
from app.services.llm.registry import LLMRegistry
from app.services.llm.review_wait import BUSINESS_WAIT_POLICY
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_provider import ReviewProvider, build_context
from app.services.v11_storage import V11Store
from app.services.v11_rejection import ReviewProviderFailure, ReviewValidationError


@pytest.fixture
def anyio_backend() -> str:
    """Run SDK async transport in the application's asyncio runtime."""
    return "asyncio"


def citation() -> ReviewCitation:
    """One actual supplied line, not a filename-only description."""
    return ReviewCitation(
        citation_id="diff:fixture",
        source_role="change_observation",
        kind="diff",
        snapshot_id="snapshot:fixture",
        generation_id="generation:fixture",
        file_id="file:fixture",
        path="x.py",
        side="RIGHT",
        sha="a" * 40,
        hunk_id="hunk:fixture",
        start_line=1,
        end_line=1,
        text="limit = 2",
        text_hash=canonical_hash("limit = 2"),
        locator="x.py:RIGHT:1",
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "timeout",
        "connect",
        "rate_limit",
        "rate_no_id",
        "rate_secret_id",
        "rate_unsafe_id",
        "unavailable",
        "budget",
        "fenced",
        "bad_citation",
        "length",
        "missing_usage",
        "inconsistent_risk",
        "excess_citations",
        "private_extra",
        "comparison_missing",
        "behavior_right_only",
        "rule_observation",
        "unsent",
    ],
)
@pytest.mark.parametrize("business", [False, True])
async def test_review_native_wire_one_send(monkeypatch: pytest.MonkeyPatch, outcome: str, business: bool) -> None:
    """No SDK retries or repair sends; actual serialized inputs match reservation."""
    monkeypatch.setattr(settings, "V1_CHAT_MODEL", "deepseek-flash")
    monkeypatch.setattr(settings, "LOCAL_MVP_ONLY", True)
    monkeypatch.setattr(settings, "V1_REAL_MODELS_ENABLED", False)
    monkeypatch.setattr(settings, "V11_REAL_MODELS_ENABLED", True)
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
    store = AsyncMock(spec=V11Store)
    if outcome == "budget":
        store.reserve.side_effect = DomainError("model_budget_exhausted")
    user, citations = build_context("Explain this line", [], [], [citation()], line_roles={"diff:fixture": ["add"]})
    if outcome == "rule_observation":
        rule = citation().model_copy(
            update={"citation_id": "rule:policy", "kind": "rule", "source_role": "application_policy"}
        )
        user, citations = build_context("Explain this line", [rule], [], [citation()])
    if outcome == "unsent":
        citations.insert(0, citation().model_copy(update={"citation_id": "diff:not-sent"}))
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload["max_tokens"] == 8192 and "max_completion_tokens" not in payload
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["thinking"] == {"type": "enabled"} and payload["reasoning_effort"] == "low"
        if outcome == "timeout":
            raise httpx.ReadTimeout("synthetic", request=request)
        if outcome == "connect":
            raise httpx.ConnectError("private-transport-body", request=request)
        if outcome in {"rate_limit", "rate_no_id", "rate_secret_id", "rate_unsafe_id", "unavailable"}:
            identifier = {"rate_secret_id": "synthetic-no-network", "rate_unsafe_id": "Bearer private-token"}.get(
                outcome, "provider-request-123"
            )
            return httpx.Response(
                503 if outcome == "unavailable" else 429,
                headers={}
                if outcome == "rate_no_id"
                else {"x-request-id": identifier, "secret-header": "private-header"},
                json={"error": {"message": "private-error-body"}},
            )
        output = {
            "summary": "The line sets limit to 2.",
            "observations": [
                {
                    "text": "The line sets limit to 2.",
                    "kind": "observation",
                    "citation_ids": ["absent" if outcome == "bad_citation" else citations[0].citation_id],
                }
            ],
            "risks": [],
            "validation_suggestions": [],
            "unknowns": [],
            "candidate_steps": [],
            "model_risk": "LOW",
            "comparison_contract": "v11-behavior-comparison.2",
            "comparisons": [
                {
                    "section": "observations",
                    "index": 0,
                    "classification": "literal_edit",
                    "before": "",
                    "after": "limit = 2",
                }
            ],
        }
        content = json.dumps(output)
        if outcome == "inconsistent_risk":
            output["unknowns"] = ["synthetic-private-value-must-not-be-recorded"]
        if outcome == "excess_citations":
            output["observations"][0]["citation_ids"] = [citations[0].citation_id] * 6
        if outcome == "private_extra":
            output["synthetic-private-field-must-not-be-recorded"] = "synthetic-private-value"
        if outcome == "comparison_missing":
            output["comparisons"] = []
        if outcome == "behavior_right_only":
            output["comparisons"][0]["classification"] = "behavior_added"
        content = json.dumps(output)
        if outcome == "fenced":
            content = "```json\n" + content + "\n```"
        data = {
            "id": "synthetic",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length" if outcome == "length" else "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        }
        if outcome != "missing_usage":
            data["usage"] = {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}
        return httpx.Response(200, json=data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

        def adapter(**kwargs):
            assert kwargs["max_retries"] == 0
            return ChatOpenAI(**kwargs, http_async_client=client)

        monkeypatch.setattr(registry, "ChatOpenAI", adapter)
        with pytest.raises(ProviderError, match="local_mvp_model_operations_disabled"):
            await llm_service.call_once_with_usage([], config, metadata={}, overrides={})
        assert not calls
        provider = ReviewProvider(store, enabled=True)
        actor = LocalActor(user_id=1, session_id="synthetic-session", expires_at=4102444800)
        if outcome in {"success", "missing_usage"}:
            _, telemetry = await provider.review(
                user,
                citations,
                actor,
                "synthetic-command",
                "synthetic-analysis",
                business_wait_policy=BUSINESS_WAIT_POLICY if business else None,
            )
            assert telemetry["output_token_limit"] == 8192
            assert telemetry["comparison_contract"] == "v11-behavior-comparison.2"
        else:
            reason = {
                "timeout": "analysis_outcome_unknown",
                "connect": "analysis_outcome_unknown",
                "rate_limit": "analysis_provider_rate_limited",
                "rate_no_id": "analysis_provider_rate_limited",
                "rate_secret_id": "analysis_provider_rate_limited",
                "rate_unsafe_id": "analysis_provider_rate_limited",
                "unavailable": "analysis_outcome_unknown",
                "budget": "model_budget_exhausted",
                "bad_citation": "analysis_citation_invalid",
                "comparison_missing": "analysis_comparison_coverage_invalid",
                "behavior_right_only": "analysis_behavior_requires_both_sides",
                "rule_observation": "analysis_observation_requires_diff",
                "unsent": "analysis_citation_invalid",
            }.get(outcome, "analysis_invalid_response")
            with pytest.raises(DomainError, match=reason) as caught:
                await provider.review(
                    user,
                    citations,
                    actor,
                    "synthetic-command",
                    "synthetic-analysis",
                    business_wait_policy=BUSINESS_WAIT_POLICY if business else None,
                )
            if outcome in {
                "timeout",
                "connect",
                "rate_limit",
                "rate_no_id",
                "rate_secret_id",
                "rate_unsafe_id",
                "unavailable",
            }:
                assert isinstance(caught.value, ReviewProviderFailure)
                diagnostic = caught.value.diagnostic
                assert diagnostic and diagnostic["stage"] == "invoke" and diagnostic["elapsed_ms"] >= 0
                assert diagnostic["provider_processing"] == "unknown"
                assert diagnostic["http_status"] == (
                    429 if outcome.startswith("rate_") else 503 if outcome == "unavailable" else None
                )
                assert diagnostic["provider_request_id"] == (
                    "provider-request-123" if outcome in {"rate_limit", "unavailable"} else None
                )
                assert "private" not in json.dumps(diagnostic)
                assert store.usage.await_count == 0
            if outcome in {"rule_observation", "unsent", "bad_citation", "behavior_right_only"}:
                assert isinstance(caught.value, ReviewValidationError)
                diagnostic = caught.value.diagnostic
                assert diagnostic and diagnostic["section"] == "observations" and diagnostic["index"] == 0
                assert store.usage.await_count == 1
                assert "claim" not in json.dumps(store.usage.await_args.args[3])
                if outcome == "unsent":
                    assert diagnostic["claim"]["references"][0]["actually_sent"] is False
    assert len(calls) == (0 if outcome == "budget" else 1)
    if calls:
        bound = sum(len(message["content"].encode()) for message in calls[0]["messages"]) + 64
        assert store.reserve.await_args.args[3] == (Decimal(bound) * 2 + Decimal(8192) * 8) / 1000000
    if outcome == "missing_usage":
        assert store.usage.await_args.args[4] is None
    if outcome in {"inconsistent_risk", "excess_citations", "private_extra"}:
        diagnostic = store.usage.await_args.args[3]["schema_errors"]
        assert diagnostic
        assert "synthetic-private" not in json.dumps(diagnostic)
        assert "input" not in diagnostic[0] and "msg" not in diagnostic[0]
        if outcome == "inconsistent_risk":
            assert diagnostic[0]["invariant"] == "analysis_uncertainty_requires_unknown"
        if outcome == "excess_citations":
            assert diagnostic[0]["location"] == ["observations", 0, "citation_ids"]


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["adapter", "deadline", "cancel"])
async def test_pinned_failure_stage_and_outer_deadline(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """The local wait timeout stays UNKNOWN; cancellation still propagates, never retries."""
    monkeypatch.setattr(settings, "V11_REAL_MODELS_ENABLED", True)
    config = ProviderConfig(role=LLMRole.CHAT, provider="openai-compatible", model="synthetic", timeout_seconds=0.001)
    invoke = AsyncMock()

    async def slow(*args, **kwargs):
        if failure == "cancel":
            raise asyncio.CancelledError()
        await asyncio.sleep(1)

    invoke.side_effect = slow

    def model(*args, **kwargs):
        if failure == "adapter":
            raise ProviderError(ProviderErrorCode.CONFIGURATION, "private-setup-value")
        return type("SyntheticModel", (), {"ainvoke": invoke})()

    monkeypatch.setattr(LLMRegistry, "get_pinned_once", model)
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await llm_service.call_once_with_usage([], config, metadata={}, overrides={}, budget_scope="review")
    else:
        with pytest.raises(ProviderError) as caught:
            await llm_service.call_once_with_usage([], config, metadata={}, overrides={}, budget_scope="review")
        d = caught.value.call_diagnostic
        assert d and d["stage"] == ("adapter_setup" if failure == "adapter" else "invoke")
        assert d["http_status"] is None and d["provider_request_id"] is None
        if failure == "deadline":
            assert d["timeout_source"] == "outer_wait"
        assert "private-setup-value" not in json.dumps(d)
        assert caught.value.code == (
            ProviderErrorCode.CONFIGURATION if failure == "adapter" else ProviderErrorCode.TIMEOUT
        )
    assert invoke.await_count == (0 if failure == "adapter" else 1)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "scope,extended,business,seconds",
    [
        ("review", False, False, 30),
        ("legacy", False, False, 30),
        ("review", True, False, 60),
        ("review", False, True, 60),
    ],
)
async def test_native_timeout_boundaries(monkeypatch, scope, extended, business, seconds):
    """Native SDK HTTP phase limits and the outer wait both receive the approved bound."""
    monkeypatch.setattr(settings, "V11_REAL_MODELS_ENABLED", True)
    monkeypatch.setattr(settings, "V1_REAL_MODELS_ENABLED", True)
    config = ProviderConfig(
        role=LLMRole.CHAT,
        provider="openai-compatible",
        model="deepseek-flash",
        endpoint="https://api.deepseek.com",
        api_key="synthetic-offline",
        timeout_seconds=30,
        max_retries=0,
    )
    requests = []
    waits = []
    clients = []

    def transport(request):
        requests.append(request)
        assert all(value == seconds for value in request.extensions["timeout"].values())
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "object": "chat.completion",
                "model": "deepseek-flash",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    def factory(actual, **kwargs):
        assert actual.timeout_seconds == seconds and actual.max_retries == 0
        client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        clients.append(client)
        return LLMRegistry._create_openai_compatible(actual, {"http_async_client": client, **kwargs})

    original = asyncio.wait_for

    async def wait(awaitable, timeout):
        waits.append(timeout)
        return await original(awaitable, timeout)

    monkeypatch.setattr(LLMRegistry, "get_pinned_once", factory)
    monkeypatch.setattr(asyncio, "wait_for", wait)
    try:
        await llm_service.call_once_with_usage(
            [],
            config,
            metadata={},
            overrides={},
            budget_scope=scope,
            evaluation_timeout_seconds=60 if extended else None,
            business_wait_policy=BUSINESS_WAIT_POLICY if business else None,
        )
    finally:
        for client in clients:
            await client.aclose()
    assert waits == [seconds] and len(requests) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("origin", ["sdk", "outer_wait"])
async def test_timeout_origin_is_not_remote_processing_evidence(monkeypatch, origin):
    """Known SDK exception and local deadline are distinguishable without response claims."""
    monkeypatch.setattr(settings, "V11_REAL_MODELS_ENABLED", True)
    config = ProviderConfig(role=LLMRole.CHAT, provider="openai-compatible", model="synthetic", timeout_seconds=30)

    async def fail(*args, **kwargs):
        if origin == "sdk":
            raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
        raise TimeoutError()

    invoke = AsyncMock(side_effect=fail)
    monkeypatch.setattr(
        LLMRegistry, "get_pinned_once", lambda *args, **kwargs: type("Model", (), {"ainvoke": invoke})()
    )
    with pytest.raises(ProviderError) as caught:
        await llm_service.call_once_with_usage(
            [], config, metadata={}, overrides={}, budget_scope="review", evaluation_timeout_seconds=60
        )
    assert invoke.await_count == 1
    assert caught.value.call_diagnostic["timeout_source"] == origin
    assert caught.value.call_diagnostic["provider_processing"] == "unknown"


@pytest.mark.anyio
async def test_legacy_cannot_enable_evaluation_timeout(monkeypatch):
    """A caller cannot extend legacy by supplying the review-only parameter."""
    monkeypatch.setattr(settings, "V1_REAL_MODELS_ENABLED", True)
    factory = AsyncMock()
    monkeypatch.setattr(LLMRegistry, "get_pinned_once", factory)
    config = ProviderConfig(role=LLMRole.CHAT, provider="openai-compatible", model="synthetic")
    with pytest.raises(ProviderError):
        await llm_service.call_once_with_usage(
            [], config, metadata={}, overrides={}, budget_scope="legacy", evaluation_timeout_seconds=60
        )
    factory.assert_not_called()
