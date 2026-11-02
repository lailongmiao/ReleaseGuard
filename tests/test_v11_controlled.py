"""Offline-only signed-permit fixtures, durable one-shot claims and no business adoption."""

import asyncio
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas.local_mvp import LocalActor
from app.schemas.v11_backend import ReviewOutput
from app.schemas.v11_backend import CONTRACT
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_provider import SYSTEM, build_context
from app.services.v11_rejection import ReviewProviderFailure
from app.services.v11_storage import V11Store
from evals.v11_backend import controlled
from evals.v11_backend.behavior_cases import cases, evidence


def fixture(monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Synthetic signer exists only in test memory; it is not an operator authorization."""
    facts, citations, roles = evidence([cases()[1]])
    user, selected = build_context("Compare", [], [], citations, {}, roles)
    payload = {
        "case": "B",
        "object_id": "synthetic-behavior",
        "system": SYSTEM,
        "user": user,
        "facts": facts,
        "citations": [c.model_dump(mode="json") for c in selected],
    }
    permit = controlled.EvaluationPermit(
        evaluation_id=controlled.EVALUATION_ID,
        case="B",
        ledger_id=controlled.LEDGER,
        actor_id=1,
        source_object="synthetic-behavior",
        payload_hash=canonical_hash(payload),
        source_hash="a" * 64,
        approval_reference="synthetic-test-only-not-an-authorization",
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
        max_calls=1,
        max_cny=Decimal("0.131072"),
        preserve_old_unknown=True,
        no_business_adoption=True,
    )
    key = b"synthetic-test-memory-only-key-000000"
    monkeypatch.setenv("V11_EVAL_OPERATOR_APPROVAL_KEY", key.decode())
    signature = hmac.new(key, permit.model_dump_json().encode(), hashlib.sha256).hexdigest()
    monkeypatch.setattr(controlled, "code_state", lambda: {"source_hash": "a" * 64})
    store = SimpleNamespace(
        require_real_budget=AsyncMock(),
        real_budget_identity=AsyncMock(return_value={"ledger_id": controlled.LEDGER}),
        connection=SimpleNamespace(authenticate=AsyncMock()),
        object_row=AsyncMock(return_value={"owner_user_id": 1}),
        claim_review=AsyncMock(side_effect=[None, {"status": "unknown"}]),
        complete=AsyncMock(),
    )
    output = ReviewOutput(
        summary="Synthetic",
        observations=[{"kind": "observation", "text": "Synthetic", "citation_ids": [selected[0].citation_id]}],
        risks=[],
        validation_suggestions=[],
        unknowns=[],
        candidate_steps=[],
        model_risk="LOW",
    )
    provider = SimpleNamespace(
        store=store,
        review=AsyncMock(return_value=(output, {})),
        config=lambda role: SimpleNamespace(model="deepseek-flash"),
    )
    actor = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
    return payload, permit, signature, key, actor, store, provider


@pytest.mark.parametrize(
    "outcome", ["completed", "unknown", "failed", "artifact_failure", "started_failure", "unexpected"]
)
def test_one_shot_never_adopts_or_resends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str) -> None:
    """Every new result or failed write leaves a consumed claim, without touching old records."""
    args = fixture(monkeypatch)
    *_, store, provider = args
    if outcome in {"unknown", "failed"}:
        provider.review.side_effect = ReviewProviderFailure(
            "analysis_outcome_unknown" if outcome == "unknown" else "analysis_provider_rate_limited",
            {"stage": "invoke", "http_status": None, "provider_processing": "unknown"},
        )
    if outcome == "unexpected":
        provider.review.side_effect = RuntimeError("synthetic-internal-failure")
    if outcome in {"artifact_failure", "started_failure"}:
        original = controlled.save_new

        def save(path: Path, report: dict) -> None:
            if path.name == ("result.json" if outcome == "artifact_failure" else "started.json"):
                raise OSError("synthetic-private-write-error")
            original(path, report)

        monkeypatch.setattr(controlled, "save_new", save)

    async def run() -> None:
        if outcome in {"artifact_failure", "started_failure", "unexpected"}:
            with pytest.raises((DomainError, OSError, RuntimeError)):
                await controlled.evaluate(*args, tmp_path)
        else:
            result = await controlled.evaluate(*args, tmp_path)
            assert result["status"] == outcome
            artifact = json.loads((tmp_path / "result.json").read_text("utf-8"))
            assert artifact["not_business_analysis"]
        with pytest.raises(DomainError, match="evaluation_consumed_no_resend"):
            await controlled.evaluate(*args, tmp_path)
        assert provider.review.await_count == (0 if outcome == "started_failure" else 1)
        assert store.complete.await_count == (0 if outcome in {"started_failure", "unexpected"} else 1)
        for call in store.claim_review.await_args_list:
            assert call.args[1] == controlled.COMMAND_ID and call.args[4] == "development"
        # Fake store exposes no save_analysis, claim_analysis, revision, approval or execution method.

    asyncio.run(run())


@pytest.mark.parametrize(
    "damage", ["unsigned", "wrong_key", "payload", "source", "actor", "expired", "cap", "ledger", "owner"]
)
def test_invalid_permit_or_scope_sends_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str) -> None:
    """Authorization and all scope checks run before the durable claim or model send."""
    payload, permit, signature, key, actor, store, provider = fixture(monkeypatch)
    if damage == "unsigned":
        key = b""
    elif damage == "wrong_key":
        signature = "0" * 64
    elif damage == "payload":
        payload["user"] += "changed"
    elif damage == "source":
        monkeypatch.setattr(controlled, "code_state", lambda: {"source_hash": "b" * 64})
    elif damage == "actor":
        actor = actor.model_copy(update={"user_id": 2})
    elif damage in {"expired", "cap"}:
        permit = permit.model_copy(
            update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
            if damage == "expired"
            else {"max_cny": Decimal("0.0001")}
        )
        signature = hmac.new(key, permit.model_dump_json().encode(), hashlib.sha256).hexdigest()
    elif damage == "ledger":
        store.real_budget_identity.return_value = {"ledger_id": "synthetic-wrong-ledger"}
    elif damage == "owner":
        store.object_row.return_value = {"owner_user_id": 2}
    with pytest.raises(DomainError):
        asyncio.run(controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path))
    provider.review.assert_not_awaited()
    store.claim_review.assert_not_awaited()


def test_revoked_evaluation_body_is_withheld_and_reads_reauthorize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A paid result never bypasses current source/owner ACL after revocation."""
    args = fixture(monkeypatch)
    *_, actor, store, provider = args
    store.object_row.side_effect = [
        {"owner_user_id": 1},
        DomainError("source_not_found"),
        DomainError("source_not_found"),
    ]

    async def run() -> None:
        result = await controlled.evaluate(*args, tmp_path)
        assert result["status"] == "completed"
        report = json.loads((tmp_path / "result.json").read_text("utf-8"))
        assert report["body_withheld_authorization"] and "output" not in report
        with pytest.raises(DomainError):
            await controlled.read_evaluation(tmp_path, "synthetic-behavior", actor, store)
        assert provider.review.await_count == 1

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["completed", "timeout", "cancel", "write_failure"])
def test_timeout60_signed_scope_and_no_retry(monkeypatch, tmp_path, outcome):
    """The separate approved identity signs the wait; old commands remain untouched."""
    deadlines = []
    original_timeout = asyncio.timeout

    def timeout(seconds):
        deadlines.append(seconds)
        return original_timeout(seconds)

    monkeypatch.setattr(asyncio, "timeout", timeout)
    payload, permit, signature, key, actor, store, provider = fixture(monkeypatch)
    permit = controlled.EvaluationPermit.model_validate(
        {
            **permit.model_dump(),
            "evaluation_id": controlled.TIMEOUT_EVALUATION_ID,
            "call_timeout_seconds": 60,
            "max_cny": Decimal("0.130936"),
        }
    )
    signature = hmac.new(key, permit.model_dump_json().encode(), hashlib.sha256).hexdigest()
    if outcome == "timeout":
        provider.review.side_effect = TimeoutError()
    if outcome == "cancel":
        provider.review.side_effect = asyncio.CancelledError()
    if outcome == "write_failure":
        original = controlled.save_new

        def save(path, report):
            if path.name == "result.json":
                raise OSError("synthetic")
            original(path, report)

        monkeypatch.setattr(controlled, "save_new", save)

    async def run():
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path)
        elif outcome == "write_failure":
            with pytest.raises(DomainError):
                await controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path)
        else:
            result = await controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path)
            assert result["status"] == ("unknown" if outcome == "timeout" else "completed")
        with pytest.raises(DomainError):
            await controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path)
        assert provider.review.await_count == 1
        assert provider.review.await_args.kwargs["evaluation_timeout_seconds"] == 60
        assert deadlines == [120]
        assert all(call.args[1] == controlled.TIMEOUT_COMMAND_ID for call in store.claim_review.await_args_list)

    asyncio.run(run())
    with pytest.raises(ValueError):
        controlled.EvaluationPermit.model_validate({**permit.model_dump(), "call_timeout_seconds": 30})
    with pytest.raises(ValueError):
        controlled.EvaluationPermit.model_validate({**permit.model_dump(), "evaluation_id": controlled.EVALUATION_ID})


@pytest.mark.parametrize(
    "damage",
    [
        {"case": "A"},
        {"payload_hash": "0" * 64},
        {"call_timeout_seconds": 60},
        {"retain_unvalidated": False},
        {"max_cny": Decimal("0.130805")},
        {"evaluation_id": "arbitrary-new-retry"},
    ],
)
def test_behavior_permit_binds_only_approved_experiment(monkeypatch, damage):
    """The added identity cannot sign A, another body, a longer wait or another retry."""
    _, permit, *_ = fixture(monkeypatch)
    data = {
        **permit.model_dump(),
        "evaluation_id": controlled.BEHAVIOR_EVALUATION_ID,
        "payload_hash": controlled.BEHAVIOR_PAYLOAD_HASH,
        "retain_unvalidated": True,
        "max_cny": Decimal("0.130804"),
    }
    assert controlled.EvaluationPermit.model_validate(data).call_timeout_seconds == 30
    with pytest.raises(ValueError):
        controlled.EvaluationPermit.model_validate({**data, **damage})


@pytest.mark.parametrize("count", [16, 17])
def test_amended_development_cap_keeps_atomic_reservation(monkeypatch, count):
    """The seventeenth reservation is possible under the same lock; the eighteenth is denied."""
    db = AsyncMock()
    db.__aenter__.return_value = db
    transaction = AsyncMock()
    db.transaction = lambda: transaction
    rows = [
        None,
        {"count": 1, "valid": True, "total": Decimal("22"), "smoke": Decimal("22")},
        {"contract_version": CONTRACT, "budget_category": "development", "resource": "synthetic", "status": "started"},
        {"exists": 1},
        {"total": Decimal("2"), "current": Decimal("2"), "smoke": Decimal("1"), "category": Decimal("1")},
        {"count": count},
    ]
    db.execute.return_value.fetchone.side_effect = rows
    connection = SimpleNamespace(connect=AsyncMock(return_value=db))
    store = V11Store(connection)
    monkeypatch.setattr(store, "object_row", AsyncMock())
    actor = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)

    async def run():
        if count == 17:
            with pytest.raises(DomainError, match="development_attempt_limit"):
                await store.reserve(
                    actor, controlled.BEHAVIOR_COMMAND_ID, "analysis-1", Decimal("0.130804"), Decimal("80")
                )
        else:
            await store.reserve(
                actor, controlled.BEHAVIOR_COMMAND_ID, "analysis-1", Decimal("0.130804"), Decimal("80")
            )

    asyncio.run(run())
    sql = [call.args[0] for call in db.execute.await_args_list]
    assert sql[0] == "SELECT pg_advisory_xact_lock(71800801)"
    assert sum(statement.startswith("INSERT INTO v1_budget") for statement in sql) == (1 if count == 16 else 0)
