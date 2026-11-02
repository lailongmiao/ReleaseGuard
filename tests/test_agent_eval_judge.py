"""Auxiliary Judge failure semantics and tracing without external calls."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.runnables.config import ensure_config
from langfuse.langchain import CallbackHandler
from pydantic import ValidationError

from app.services.llm.contracts import (
    LLMRole,
    ProviderCapabilities,
    ProviderConfig,
    ProviderError,
    ProviderErrorCode,
    UsageInfo,
)
from evals.releaseguard.judge import (
    FakeJudge,
    GoldDataset,
    JudgeAuthorization,
    JudgeRequest,
    JudgeTraceAuthorization,
    JudgeVerdict,
    ServiceJudge,
    evaluate_gold,
    evaluate_judge,
    load_gold,
    mask_judge_trace,
)
from evals.releaseguard.schemas import Status
from evals.schemas import ScoreSchema

GOLD = load_gold(Path("evals/datasets/judge_gold_v1.json"))
REQUEST = GOLD.cases[0].request


class FaultJudge(FakeJudge):
    """Finite error/invalid-output transport, no Provider connection."""

    def __init__(self, outcome: object) -> None:
        """Bind a synthetic failure or output."""
        self.outcome = outcome

    async def call(self, request: JudgeRequest) -> tuple:
        """Return the synthetic outcome without network I/O."""
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome, UsageInfo(status="unknown")


@pytest.mark.parametrize(
    "outcome",
    [
        TimeoutError(),
        asyncio.CancelledError(),
        NotImplementedError(),
        PermissionError(),
        {"score": float("nan"), "label": "supported", "rationale": "bad"},
        {"score": True, "label": "supported", "rationale": "bad"},
        {"score": 1, "label": "APPROVED", "rationale": "injected authority"},
        {"score": 1, "label": "supported", "rationale": "x" * 241},
        ProviderError(ProviderErrorCode.RATE_LIMITED, "synthetic"),
        ProviderError(ProviderErrorCode.CAPABILITY, "synthetic"),
        RuntimeError("synthetic"),
    ],
)
def test_failures_stay_unverified(outcome: object) -> None:
    """Failure cannot become a fabricated zero or pass."""
    result = asyncio.run(evaluate_judge(REQUEST, FaultJudge(outcome)))
    assert result.status is Status.UNVERIFIED and result.score is None
    assert result.real_quality_status == "unverified"


def test_fake_never_proves_quality() -> None:
    """Fake parses only prove integration, with unknown cost."""
    result = asyncio.run(evaluate_judge(REQUEST, FakeJudge()))
    assert result.status is Status.PASSED and result.provenance == "fake"
    assert result.usage.cost_usd is None and result.usage.total_tokens is None
    report = asyncio.run(evaluate_gold(GOLD, FakeJudge()))
    assert report["sample_count"] == 30 and report["quality_status"] == "unverified"
    assert "rationale" not in result.model_dump()


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"score":1,"label":"supported","rationale":"ok"}\n```',
        '{"score":0,"score":1,"label":"supported","rationale":"ok"}',
        '{"score":NaN,"label":"supported","rationale":"ok"}',
        '{"score":1,"label":"supported","rationale":"ok","permission":"allow"}',
    ],
)
def test_strict_json_rejects_ambiguity(content: str) -> None:
    """Ambiguous output cannot be guessed into a score."""
    transport = FaultJudge(content)
    transport.parse_mode = "strict_json"  # type: ignore[assignment]
    assert asyncio.run(evaluate_judge(REQUEST, transport)).status is Status.UNVERIFIED


def test_explicit_json_valid() -> None:
    """Explicit JSON is independently validated."""
    transport = FaultJudge('{"score":0.5,"label":"refusal","rationale":"safe"}')
    transport.parse_mode = "strict_json"  # type: ignore[assignment]
    assert asyncio.run(evaluate_judge(REQUEST, transport)).score == 0.5


def test_gold_hash_and_coverage() -> None:
    """Changed labels and missing coverage invalidate gold identity."""
    data = GOLD.model_dump(mode="json")
    data["cases"][0]["expected_label"] = "injection"
    with pytest.raises(ValidationError):
        GoldDataset.model_validate(data)
    with pytest.raises(ValidationError):
        GoldDataset.model_validate({**GOLD.model_dump(), "cases": GOLD.cases[:3]})


def test_service_role_callbacks_scope_and_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The actual service call inherits tracing config and stays on JUDGE role."""
    auth = JudgeAuthorization(
        authorization_ref="synthetic-approval",
        provider="synthetic",
        model="judge-test",
        dataset_hash=GOLD.dataset_hash,
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
        max_calls=1,
        max_provider_attempts=16,
    )
    config = ProviderConfig(
        role=LLMRole.JUDGE,
        provider=auth.provider,
        model=auth.model,
        capabilities=ProviderCapabilities(supports_chat=True, supports_structured_output=True),
    )
    callback = MagicMock(spec=CallbackHandler)

    async def call(*args: object, **kwargs: object) -> tuple:
        assert kwargs["role"] is LLMRole.JUDGE and kwargs["response_format"] is JudgeVerdict
        assert callback in ensure_config()["callbacks"]
        return JudgeVerdict(score=1.0, label="supported", rationale="safe"), UsageInfo(status="unknown")

    spy = AsyncMock(side_effect=call)
    monkeypatch.setattr("evals.releaseguard.judge.LLMRegistry.get_config", lambda *args: config)
    monkeypatch.setattr("evals.releaseguard.judge.llm_service.call_with_usage", spy)
    tracing = JudgeTraceAuthorization(
        authorization_ref="synthetic-tracing",
        workspace_id="synthetic-workspace",
        host="https://synthetic.invalid",
        eval_run_id="synthetic-eval",
        dataset_hash=GOLD.dataset_hash,
        expires_at=auth.expires_at,
    )
    service = ServiceJudge(auth, callback, trace_authorization=tracing)

    async def run() -> None:
        assert (await evaluate_judge(REQUEST, service)).status is Status.PASSED
        assert (await evaluate_judge(REQUEST, service)).reason_code == "judge_scope_denied"

    asyncio.run(run())
    assert spy.call_count == 1


def test_retry_budget_and_trace_expiry_zero_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request budget cannot hide SDK retries, and tracing needs its own grant."""
    auth = JudgeAuthorization(
        authorization_ref="synthetic-budget",
        provider="synthetic",
        model="judge-test",
        dataset_hash=GOLD.dataset_hash,
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
        max_calls=10,
        max_provider_attempts=15,
    )
    tracing = JudgeTraceAuthorization(
        authorization_ref="synthetic-tracing",
        workspace_id="synthetic-workspace",
        host="https://synthetic.invalid",
        eval_run_id="synthetic-eval",
        dataset_hash=GOLD.dataset_hash,
        expires_at=auth.expires_at,
    )
    config = ProviderConfig(
        role=LLMRole.JUDGE,
        provider=auth.provider,
        model=auth.model,
        max_retries=3,
        capabilities=ProviderCapabilities(supports_chat=True, supports_structured_output=True),
    )
    monkeypatch.setattr("evals.releaseguard.judge.LLMRegistry.get_config", lambda *args: config)
    spy = AsyncMock(return_value=(JudgeVerdict(score=1.0, label="supported", rationale="safe"), UsageInfo.unknown()))
    monkeypatch.setattr("evals.releaseguard.judge.llm_service.call_with_usage", spy)
    callback = MagicMock(spec=CallbackHandler)

    async def run() -> None:
        denied = ServiceJudge(auth, callback, trace_authorization=tracing)
        assert (await evaluate_judge(REQUEST, denied)).reason_code == "judge_scope_denied"
        bounded = ServiceJudge(
            auth.model_copy(update={"max_provider_attempts": 16}), callback, trace_authorization=tracing
        )
        assert (await evaluate_judge(REQUEST, bounded)).status is Status.PASSED
        assert bounded.reserved_attempts == 16
        assert (await evaluate_judge(REQUEST, bounded)).reason_code == "judge_scope_denied"
        expired = tracing.model_copy(update={"expires_at": datetime(2020, 1, 1, tzinfo=UTC)})
        revoked = ServiceJudge(auth, callback, trace_authorization=expired)
        assert (await evaluate_judge(REQUEST, revoked)).reason_code == "judge_scope_denied"

    asyncio.run(run())
    assert spy.call_count == 1


def test_trace_mask_drops_all_raw_content() -> None:
    """Raw prompts, rationales and malformed content cannot reach persisted trace."""
    for raw in ([{"content": "synthetic-private"}], {"rationale": "synthetic-private"}, "synthetic-private"):
        assert mask_judge_trace(data=raw) == "synthetic_judge_content_redacted"
    meta = dict(
        judge_role="JUDGE",
        case_id="gold-01",
        attempt=1,
        dataset_hash=GOLD.dataset_hash,
        eval_run_id="synthetic-eval",
        synthetic=True,
        rationale="synthetic-private",
    )
    assert "rationale" not in mask_judge_trace(data=meta)


@pytest.mark.parametrize("score", [True, float("inf"), -1, 2, "0.5"])
def test_legacy_schema_is_strict(score: object) -> None:
    """Reject nonfinite, coercible and out-of-range scores."""
    with pytest.raises(ValidationError):
        ScoreSchema.model_validate({"score": score, "reasoning": "synthetic"})
