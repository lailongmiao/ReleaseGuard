"""No-network checks for the sole linked acceptance exception."""

import asyncio
import hashlib
import hmac
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.releaseguard.state_machine import DomainError
from app.services.v11_storage import V11Store
from evals.v11_backend import controlled, supplement
from tests.test_v11_controlled import fixture


@pytest.mark.parametrize("outcome", ["ok", "old_missing", "not_unknown", "consumed", "orphan", "budget_changed"])
def test_linked_claim_is_atomic_and_preserves_history(monkeypatch, outcome):
    """Only new command and audit inserts are allowed, under the original budget lock."""
    _, _, _, _, actor, _, _ = fixture(monkeypatch)
    db = AsyncMock()
    db.__aenter__.return_value = db
    db.transaction = lambda: AsyncMock()
    old = [{"resource": "synthetic", "result": {"status": "unknown"}} for _ in range(2)]
    if outcome == "old_missing":
        old.pop()
    if outcome == "not_unknown":
        old[0]["result"]["status"] = "completed"
    db.execute.return_value.fetchall.return_value = old
    db.execute.return_value.fetchone.side_effect = [
        {"exists": 1} if outcome in {"consumed", "orphan"} else None,
        {"amount": Decimal("24") if outcome != "budget_changed" else Decimal("25")},
    ]
    store = V11Store(SimpleNamespace(connect=AsyncMock(return_value=db)))
    store.object_row = AsyncMock()
    store.event = AsyncMock()
    store.evaluation_ceiling = Decimal("24.1")
    if outcome == "ok":
        binding = asyncio.run(supplement.claim(store, actor, "synthetic", "digest", Decimal("24")))
        assert binding["old_commands"] == list(supplement.OLD_COMMANDS)
        assert binding["no_business_adoption"] and binding["acceptance_total_ceiling"] == "43"
        assert store.evaluation_ceiling == Decimal("24.1")
    else:
        with pytest.raises(DomainError):
            asyncio.run(supplement.claim(store, actor, "synthetic", "digest", Decimal("24")))
        store.event.assert_not_awaited()
    sql = [c.args[0] for c in db.execute.await_args_list]
    assert sql[0] == "SELECT pg_advisory_xact_lock(71800801)"
    assert not any(q.startswith(("UPDATE", "DELETE")) for q in sql)
    assert sum(q.startswith("INSERT") for q in sql) == (outcome == "ok")
    if outcome == "ok":
        assert "'acceptance'" in sql[-1]
        assert db.execute.await_args.args[1][-1].obj["claimed"]


@pytest.mark.parametrize("outcome", ["completed", "timeout", "cancel", "save_failure"])
def test_supplement_60_keeps_link_on_result_and_never_resends(monkeypatch, tmp_path, outcome):
    """Success and all uncertain exits consume one linked identity, without business adoption."""
    payload, permit, _, key, actor, store, provider = fixture(monkeypatch)
    monkeypatch.setattr(controlled, "BEHAVIOR_PAYLOAD_HASH", permit.payload_hash)
    permit = controlled.EvaluationPermit.model_validate(
        {
            **permit.model_dump(),
            "evaluation_id": supplement.EVALUATION_ID,
            "retain_unvalidated": True,
            "call_timeout_seconds": 60,
            "max_cny": Decimal("0.130804"),
            "acceptance_increment_baseline": Decimal("24"),
        }
    )
    signature = hmac.new(key, permit.model_dump_json().encode(), hashlib.sha256).hexdigest()
    binding = {"old_commands": list(supplement.OLD_COMMANDS), "no_business_adoption": True}
    claim = AsyncMock(side_effect=[binding, DomainError("evaluation_consumed_no_resend")])
    monkeypatch.setattr(supplement, "claim", claim)
    output, telemetry = provider.review.return_value

    async def review(*args, **kwargs):
        assert kwargs["evaluation_timeout_seconds"] == 60
        if outcome == "timeout":
            raise TimeoutError()
        if outcome == "cancel":
            raise asyncio.CancelledError()
        await kwargs["capture_unvalidated"](output.model_dump(mode="json"))
        return output, telemetry

    provider.review.side_effect = review
    if outcome == "save_failure":
        original = controlled.save_new

        def save(path, report):
            if path.name == "unvalidated.json":
                raise OSError("synthetic")
            original(path, report)

        monkeypatch.setattr(controlled, "save_new", save)

    async def run():
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path)
        else:
            result = await controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path)
            assert result["old_commands"] == list(supplement.OLD_COMMANDS)
            assert (
                result["status"] == {"completed": "completed", "timeout": "unknown", "save_failure": "failed"}[outcome]
            )
        with pytest.raises(DomainError, match="consumed"):
            await controlled.evaluate(payload, permit, signature, key, actor, store, provider, tmp_path)
        provider.review.assert_awaited_once()
        store.claim_review.assert_not_awaited()

    asyncio.run(run())
    assert json.loads((tmp_path / "started.json").read_text())["old_commands"] == list(supplement.OLD_COMMANDS)


@pytest.mark.parametrize(
    "damage",
    [
        {"call_timeout_seconds": 30},
        {"case": "A"},
        {"payload_hash": "0" * 64},
        {"retain_unvalidated": False},
        {"acceptance_increment_baseline": None},
        {"max_cny": Decimal("0.15")},
        {"evaluation_id": "arbitrary-retry"},
    ],
)
def test_supplement_signature_scope(monkeypatch, damage):
    """Signed scope cannot become A, another input, a shorter wait or an arbitrary identity."""
    _, permit, *_ = fixture(monkeypatch)
    data = {
        **permit.model_dump(),
        "evaluation_id": supplement.EVALUATION_ID,
        "payload_hash": controlled.BEHAVIOR_PAYLOAD_HASH,
        "retain_unvalidated": True,
        "call_timeout_seconds": 60,
        "max_cny": Decimal("0.130804"),
        "acceptance_increment_baseline": Decimal("24"),
    }
    controlled.EvaluationPermit.model_validate(data)
    with pytest.raises(ValueError):
        controlled.EvaluationPermit.model_validate({**data, **damage})
