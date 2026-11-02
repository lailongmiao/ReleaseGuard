"""Explicit evaluation-only full structured evidence before schema or citation acceptance."""

import asyncio
import hashlib
import hmac
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from app.services.llm.contracts import ProviderConfig, LLMRole, UsageInfo
from app.schemas.v11_backend import ReviewCitation
from app.services.llm.service import llm_service
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_provider import ReviewProvider
from app.services.v11_rejection import ReviewValidationError
from evals.v11_backend import controlled
from evals.v11_backend.structured_evidence import STRUCTURED_LIMIT, structured_evidence
from tests.test_v11_controlled import fixture


@pytest.mark.parametrize("outcome", ["success", "rejected", "revoked", "write_failed"])
@pytest.mark.parametrize("identity", [controlled.FINAL_EVALUATION_ID, controlled.BEHAVIOR_EVALUATION_ID])
def test_authorized_capture_survives_rejection_and_is_never_adopted(monkeypatch, tmp_path, outcome, identity):
    """Verify bounded evidence and authorization without external calls."""
    payload, permit, signature, key, actor, store, provider = fixture(monkeypatch)
    if identity == controlled.BEHAVIOR_EVALUATION_ID:
        monkeypatch.setattr(controlled, "BEHAVIOR_PAYLOAD_HASH", controlled.canonical_hash(payload))
    permit = controlled.EvaluationPermit.model_validate(
        {
            **permit.model_dump(),
            "evaluation_id": identity,
            "retain_unvalidated": True,
            "source_object": payload["object_id"],
            "payload_hash": controlled.canonical_hash(payload),
            "max_cny": Decimal("0.130804"),
        }
    )
    signature = hmac.new(key, permit.model_dump_json().encode(), hashlib.sha256).hexdigest()
    original_output, _ = provider.review.return_value
    original_output.observations[0].citation_ids = [payload["citations"][0]["citation_id"]]
    returned = {
        **original_output.model_dump(mode="json"),
        "comparison_contract": "v11-behavior-comparison.2",
        "comparisons": [
            {"section": "observations", "index": 0, "classification": "literal_edit", "before": "old", "after": "new"}
        ],
    }
    if outcome == "revoked":
        store.object_row.side_effect = [
            {"owner_user_id": 1},
            DomainError("source_not_found"),
            DomainError("source_not_found"),
        ]
    if outcome == "write_failed":
        original_save = controlled.save_new

        def save(path, report):
            if path.name == "unvalidated.json":
                raise OSError("private-failure")
            original_save(path, report)

        monkeypatch.setattr(controlled, "save_new", save)

    async def review(*args, **kwargs):
        await kwargs["capture_unvalidated"](returned)
        if outcome == "rejected":
            raise ReviewValidationError("analysis_test_rejected", "validate_comparisons", "observations", 0)
        return original_output, {}

    provider.review.side_effect = review

    async def run():
        result = await controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path)
        assert result["status"] == ("completed" if outcome == "success" else "failed")
        assert "answer" not in json.dumps(store.complete.await_args.args[2])
        if outcome in {"success", "rejected"}:
            saved = await controlled.read_evaluation(tmp_path, payload["object_id"], actor, store, "unvalidated")
            assert saved["status"] == "unvalidated" and saved["answer"] == returned
            assert saved["review_complete"]
            store.object_row.side_effect = DomainError("source_not_found")
            with pytest.raises(DomainError):
                await controlled.read_evaluation(tmp_path, payload["object_id"], actor, store, "unvalidated")
        else:
            assert not (tmp_path / "unvalidated.json").exists()
        store.object_row.side_effect = None
        with pytest.raises(DomainError):
            await controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path)
        assert provider.review.await_count == 1
        expected_command = (
            controlled.BEHAVIOR_COMMAND_ID
            if identity == controlled.BEHAVIOR_EVALUATION_ID
            else controlled.FINAL_COMMAND_ID
        )
        assert all(call.args[1] == expected_command for call in store.claim_review.await_args_list)
        assert provider.review.await_args.kwargs["evaluation_timeout_seconds"] is None

    asyncio.run(run())


def test_evidence_redaction_preserves_all_claims_and_discloses_loss():
    """Verify bounded evidence and authorization without external calls."""
    raw = {
        "summary": "token=private-value",
        "observations": [
            {"text": "first", "kind": "observation", "citation_ids": ["diff:known"]},
            {"text": "second", "kind": "inference", "citation_ids": ["sk-secret-value"]},
        ],
        "comparisons": [
            {"section": "observations", "index": 1, "classification": "literal_edit", "before": "x", "after": "y"}
        ],
        "reasoning_content": "never-save-this",
        "headers": {"Authorization": "never-save-this"},
    }
    saved = structured_evidence(raw, {"diff:known"})
    text = json.dumps(saved)
    assert all(value not in text for value in ("private-value", "sk-secret-value", "never-save-this", "Authorization"))
    assert len(saved["answer"]["observations"]) == 2
    assert saved["answer"]["observations"][0]["citation_ids"] == ["diff:known"]
    assert not saved["review_complete"] and saved["transformations"]
    oversized = structured_evidence({"observations": [{"text": "中" * 5000}] * 100}, set())
    assert (
        not oversized["review_complete"] and len(json.dumps(oversized, ensure_ascii=False).encode()) < STRUCTURED_LIMIT
    )


@pytest.mark.parametrize("failure", ["citation", "schema", "known"])
def test_real_provider_calls_sink_before_acceptance(monkeypatch, failure):
    """Verify bounded evidence and authorization without external calls."""
    # Exercise real Provider orchestration with one known synthetic response, not a second model system.
    from_fixture = fixture(monkeypatch)
    payload, _, _, _, actor, _, _ = from_fixture
    store = SimpleNamespace(require_real_budget=AsyncMock(), reserve=AsyncMock(), usage=AsyncMock())
    provider = ReviewProvider(store, enabled=True)
    config = ProviderConfig(role=LLMRole.CHAT, provider="openai-compatible", model="deepseek-flash")
    provider.reserve = AsyncMock(return_value=config)
    raw = {
        "summary": "one",
        "observations": [{"kind": "observation", "text": "one", "citation_ids": ["absent"]}],
        "risks": [],
        "validation_suggestions": [],
        "unknowns": [],
        "candidate_steps": [],
        "model_risk": "LOW",
        "comparison_contract": "v11-behavior-comparison.2",
        "comparisons": [],
    }
    if failure == "schema":
        raw["model_risk"] = "invalid"
    if failure == "known":
        raw["observations"][0]["citation_ids"] = [payload["citations"][0]["citation_id"]]
        raw["comparisons"] = [
            {"section": "observations", "index": 0, "classification": "literal_edit", "before": "", "after": "new"}
        ]
    from_dataclass_usage = UsageInfo(status="known", input_tokens=10, output_tokens=10)
    call = AsyncMock(
        return_value=(
            AIMessage(content=json.dumps(raw), response_metadata={"finish_reason": "stop"}),
            from_dataclass_usage,
        )
    )
    monkeypatch.setattr(llm_service, "call_once_with_usage", call)
    captured = []

    async def sink(parsed):
        captured.append(parsed)

    async def run():
        citations = [ReviewCitation.model_validate(c) for c in payload["citations"]]
        if failure == "known":
            await provider.review(
                payload["user"], citations, actor, "synthetic", "synthetic", capture_unvalidated=sink
            )
        else:
            with pytest.raises(DomainError):
                await provider.review(
                    payload["user"], citations, actor, "synthetic", "synthetic", capture_unvalidated=sink
                )
        assert captured == [raw] and call.await_count == 1

    asyncio.run(run())


def test_maximum_valid_answer_can_be_retained_without_truncation():
    """All schema-bounded prose and comparisons fit the private evidence cap together."""
    claim = {"text": "中" * 1000, "kind": "observation", "citation_ids": ["diff:known"] * 5}
    raw = {
        "summary": "中" * 4000,
        "observations": [claim] * 16,
        "risks": [{**claim, "severity": "HIGH"}] * 12,
        "validation_suggestions": [{"text": "中" * 1000, "performed": False}] * 8,
        "unknowns": ["中" * 1000] * 12,
        "candidate_steps": [],
        "model_risk": "UNKNOWN",
        "comparison_contract": "v11-behavior-comparison.2",
        "comparisons": [
            {
                "section": "observations",
                "index": 0,
                "classification": "literal_edit",
                "before": "中" * 500,
                "after": "中" * 500,
            }
        ]
        * 28,
    }
    saved = structured_evidence(raw, {"diff:known"})
    assert saved["review_complete"] and saved["answer"] == raw
    assert len(json.dumps(saved, ensure_ascii=False).encode()) < STRUCTURED_LIMIT
