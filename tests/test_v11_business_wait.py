"""Virtual monotonic time plus native SDK transport: no paid service or long sleeps."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import settings
from app.schemas.local_mvp import LocalActor
from app.schemas.v11_backend import ReviewAnalyze
from app.services.llm.contracts import LLMRole, ProviderCapabilities, ProviderConfig, ProviderError
from app.services.llm.registry import LLMRegistry
from app.services.llm.review_wait import BUSINESS_WAIT_POLICY
from app.services.llm.service import llm_service
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_backend import V11Backend


@pytest.fixture
def anyio_backend():
    """Use the application's asyncio cancellation semantics."""
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "scope,policy,delay,success",
    [
        ("review", BUSINESS_WAIT_POLICY, 45, True),
        ("review", BUSINESS_WAIT_POLICY, 61, False),
        ("legacy", None, 45, False),
    ],
)
async def test_native_response_after_virtual_delay(monkeypatch, scope, policy, delay, success):
    """Advance the loop clock after the POST, preserving real wait_for cancellation."""
    monkeypatch.setattr(settings, "V11_REAL_MODELS_ENABLED", True)
    monkeypatch.setattr(settings, "V1_REAL_MODELS_ENABLED", True)
    loop = asyncio.get_running_loop()
    real_time = loop.time
    offset = [0]
    monkeypatch.setattr(loop, "time", lambda: real_time() + offset[0])
    sends, cancelled = [], []
    config = ProviderConfig(
        role=LLMRole.CHAT,
        provider="openai-compatible",
        model="deepseek-flash",
        endpoint="https://api.deepseek.com",
        api_key="synthetic-offline",
        timeout_seconds=30,
        max_retries=0,
        capabilities=ProviderCapabilities(supports_chat=True, supports_usage=True),
    )

    async def handler(request):
        sends.append(1)
        assert set(request.extensions["timeout"].values()) == {60 if policy else 30}
        offset[0] += delay
        try:
            await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise
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

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            LLMRegistry,
            "get_pinned_once",
            lambda actual, **kwargs: LLMRegistry._create_openai_compatible(
                actual, {"http_async_client": client, **kwargs}
            ),
        )
        if success:
            response, usage = await llm_service.call_once_with_usage(
                [], config, metadata={}, overrides={}, budget_scope=scope, business_wait_policy=policy
            )
            assert response.content == "{}" and usage.output_tokens == 2
        else:
            with pytest.raises(ProviderError) as error:
                await llm_service.call_once_with_usage(
                    [], config, metadata={}, overrides={}, budget_scope=scope, business_wait_policy=policy
                )
            if scope == "review":
                assert error.value.call_diagnostic["timeout_source"] == "outer_wait"
                assert error.value.call_diagnostic["provider_processing"] == "unknown"
            assert cancelled == [1]
    assert sends == [1]


@pytest.mark.anyio
@pytest.mark.parametrize("scope,evaluation", [("legacy", None), ("review", 60)])
async def test_business_policy_cannot_expand_other_scopes(monkeypatch, scope, evaluation):
    """Business and evaluation authority cannot be combined or applied to legacy."""
    monkeypatch.setattr(settings, "LOCAL_MVP_ONLY", False)
    factory = AsyncMock()
    monkeypatch.setattr(LLMRegistry, "get_pinned_once", factory)
    with pytest.raises(ProviderError, match="business_wait_policy_scope_invalid"):
        await llm_service.call_once_with_usage(
            [],
            ProviderConfig(role=LLMRole.CHAT, provider="openai-compatible", model="synthetic"),
            metadata={},
            overrides={},
            budget_scope=scope,
            evaluation_timeout_seconds=evaluation,
            business_wait_policy=BUSINESS_WAIT_POLICY,
        )
    factory.assert_not_called()


@pytest.mark.anyio
async def test_business_cancellation_propagates_without_second_send(monkeypatch):
    """A caller cancellation remains cancellation rather than a fresh attempt."""
    monkeypatch.setattr(settings, "V11_REAL_MODELS_ENABLED", True)
    started, cleaned = asyncio.Event(), asyncio.Event()
    calls = []

    async def invoke(*args, **kwargs):
        calls.append(1)
        started.set()
        try:
            await asyncio.Future()
        finally:
            cleaned.set()

    monkeypatch.setattr(LLMRegistry, "get_pinned_once", lambda *args, **kwargs: SimpleNamespace(ainvoke=invoke))
    config = ProviderConfig(role=LLMRole.CHAT, provider="openai-compatible", model="synthetic")
    task = asyncio.create_task(
        llm_service.call_once_with_usage(
            [], config, metadata={}, overrides={}, budget_scope="review", business_wait_policy=BUSINESS_WAIT_POLICY
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and calls == [1]


@pytest.mark.anyio
async def test_business_command_total_deadline_remains_120(monkeypatch):
    """Actual analyze command deadline cancels source work; no Provider call follows."""
    loop = asyncio.get_running_loop()
    real_time = loop.time
    offset = [0]
    monkeypatch.setattr(loop, "time", lambda: real_time() + offset[0])
    store = AsyncMock()
    store.object_row.return_value = {
        "owner_user_id": 1,
        "object_version": 1,
        "active_snapshot_id": "snapshot:one",
        "active_generation_id": "generation:one",
    }
    store.claim_review.return_value = None
    provider = AsyncMock()
    runtime = V11Backend(SimpleNamespace(local=SimpleNamespace(store=store)), store=store, provider=provider)
    cancelled = []

    async def delayed(*args):
        offset[0] += 119
        await asyncio.sleep(0.001)
        assert not cancelled
        offset[0] += 2
        try:
            await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    store.snapshot.side_effect = delayed
    with pytest.raises(DomainError, match="analysis_outcome_unknown"):
        await runtime.analyze(
            "one",
            LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800),
            ReviewAnalyze(
                command_id="new-command",
                expected_object_version=1,
                expected_state_version=0,
                snapshot_id="snapshot:one",
                generation_id="generation:one",
                query="Explain",
            ),
        )
    assert cancelled == [1]
    store.claim_review.assert_awaited_once()
    store.complete.assert_not_awaited()
    provider.review.assert_not_awaited()
