"""Fixed overlap authorization and permanent business denial, without network I/O."""

import asyncio
from decimal import Decimal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import httpx

from app.schemas.local_mvp import LocalActor
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_backend import V11Backend
from app.services.v11_evaluation_policy import (
    DATASET_HASH,
    FORMAL_BATCH,
    LINK_CONTRACT,
    READONLY_REASON,
    RETIRED_BATCH,
    RETIRED_BATCHES,
    RETIRED_SENDS,
)
from evals.v11_backend.contracts import Dataset
from evals.v11_backend.http_evidence import EvidencePermit, HTTPEvaluationEvidence
from evals.v11_backend.transport import FormalGitHubTransport
from app.services.v11_storage import V11Store
from evals.v11_backend import real
from test_v11_acceptance_engineering import dataset_and_auth


def permit() -> EvidencePermit:
    """Use the untouched real dataset; no signing or real transports are constructed."""
    dataset = Dataset.model_validate_json(Path("evals/datasets/v11_acceptance.json").read_text(encoding="utf-8"))
    assert canonical_hash(dataset) == DATASET_HASH
    positions = {}
    for case in dataset.cases:
        for repeat in (1, 2):
            stamp = canonical_hash([FORMAL_BATCH, DATASET_HASH, case.case_id, repeat])
            positions["analysis:" + stamp] = {
                "case_id": case.case_id,
                "repeat": repeat,
                "query": case.query,
                "binding": case.binding.model_dump(mode="json"),
                "comparison": case.comparison.model_dump(mode="json"),
                "sync_command": "sync:" + stamp,
            }
    return EvidencePermit(
        batch_id=FORMAL_BATCH,
        source_hash="a" * 64,
        dataset_hash=DATASET_HASH,
        owner_id=1,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        mode="real",
        designated_ledger_id="synthetic-never-connected",
        ledger_charge_at_start=Decimal("24.60984678"),
        maximum_batch_charge=19,
        positions=positions,
        overlap_exception=LINK_CONTRACT,
    )


def test_retired_permit_and_all_old_paid_positions_denied():
    """The old signature cannot be remounted, and none of its 240 attempts can reserve."""
    sink = object.__new__(HTTPEvaluationEvidence)
    sink.permit = permit().model_copy(update={"batch_id": RETIRED_BATCH})
    with pytest.raises(DomainError, match="batch_retired"):
        sink.validate_overlap_permit()
    connection = SimpleNamespace(connect=AsyncMock())
    store = V11Store(connection)

    async def run():
        for command in RETIRED_SENDS:
            with pytest.raises(DomainError, match="batch_retired"):
                await store.reserve(SimpleNamespace(), command, "analysis-1", Decimal("0.1"), Decimal("100"))
        assert connection.connect.await_count == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "change", [None, "batch", "query", "comparison", "key", "repeat", "extra", "unsigned", "dataset"]
)
def test_exact_formal_positions_only(change):
    """Every identity and input alteration is rejected without a Provider call."""
    value = permit()
    key = next(k for k, p in value.positions.items() if p["case_id"] == "real-03-zh")
    position = value.positions[key]
    if change == "batch":
        value = value.model_copy(update={"batch_id": "another-batch"})
    elif change == "query":
        position["query"] += " another question"
    elif change == "comparison":
        position["comparison"]["head_sha"] = "e" * 40
    elif change == "key":
        value.positions["new-key"] = value.positions.pop(key)
    elif change == "repeat":
        position["repeat"] = 3
    elif change == "extra":
        position["case_id"] = "unapproved-overlap"
    elif change == "unsigned":
        value = value.model_copy(update={"overlap_exception": None})
    elif change == "dataset":
        value = value.model_copy(update={"dataset_hash": "f" * 64})
    sink = object.__new__(HTTPEvaluationEvidence)
    sink.permit = value
    if change is None:
        sink.validate_overlap_permit()
    else:
        with pytest.raises(DomainError):
            sink.validate_overlap_permit()


@pytest.mark.parametrize("kind", ["prepare", "decide", "execute"])
def test_future_global_recovery_does_not_enable_evaluation(monkeypatch, kind):
    """Remove the global pause in memory; permanent restrictions must still reject first."""
    monkeypatch.setattr("app.services.v11_backend.require_rehearsal", lambda: None)
    store = SimpleNamespace(object_row=AsyncMock(return_value={"evaluation_readonly": True}))
    local = SimpleNamespace(command=AsyncMock())
    runtime = V11Backend(SimpleNamespace(local=local), store, Mock(), Mock())
    actor = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)

    async def run():
        assert await runtime.command("obj", actor, SimpleNamespace(), kind) == {
            "status": "blocked",
            "reason": READONLY_REASON,
        }
        for operation in (
            runtime.guard({"object_id": "obj"}, actor),
            runtime.fresh("obj", actor, {}),
            runtime.check_command(SimpleNamespace(row={"change_id": "obj"}), actor, None, kind),
        ):
            with pytest.raises(DomainError, match=READONLY_REASON):
                await operation
        local.command.assert_not_awaited()
        assert runtime.provider.mock_calls == runtime.github.mock_calls == []

    asyncio.run(run())


def test_marked_object_cannot_use_ordinary_analysis():
    """Removing the explicit evaluation runtime cannot restore model eligibility."""
    store = SimpleNamespace(object_row=AsyncMock(return_value={"evaluation_readonly": True, "owner_user_id": 1}))
    runtime = V11Backend(SimpleNamespace(local=SimpleNamespace()), store, Mock(), Mock())
    actor = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
    with pytest.raises(DomainError, match="evaluation_position_not_authorized"):
        asyncio.run(runtime.analyze("obj", actor, SimpleNamespace()))


def test_provider_attempt_identities_bound_before_database():
    """Unapproved sends cannot even reserve, including extra models and repeat-two indexing."""
    sink = object.__new__(HTTPEvaluationEvidence)
    sink.permit = permit()
    allowed = sink.allowed_attempts()
    assert sum("analysis-1" in a for a in allowed.values()) == 80
    assert sum("query-1-batch-0" in a for a in allowed.values()) == 160
    assert sum(sum(x.startswith("index-batch-") for x in a) for a in allowed.values()) == 120
    store = V11Store(SimpleNamespace(connect=Mock(side_effect=AssertionError("must_not_connect"))))
    store.evaluation_attempts, store.evaluation_owner = allowed, 1
    actor = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
    second = next(p["sync_command"] for p in sink.permit.positions.values() if p["repeat"] == 2)
    for command, attempt in ((second, "index-batch-0"), ("new-command", "analysis-1")):
        with pytest.raises(DomainError, match="evaluation_provider_attempt_not_authorized"):
            asyncio.run(store.reserve(actor, command, attempt, Decimal("0.01"), Decimal("100")))


def test_github_attempts_persist_failures_restart_and_stop(tmp_path):
    """A failed request consumes capacity, and restart or save failure never refunds it."""
    calls = []

    def handler(request):
        calls.append(request.url.path)
        raise httpx.ConnectError("synthetic transport failure")

    async def run():
        transport = FormalGitHubTransport(tmp_path, httpx.MockTransport(handler))
        request = httpx.Request("GET", "https://api.github.com/repos/actions/checkout")
        with pytest.raises(httpx.ConnectError):
            await transport.handle_async_request(request)
        restarted = FormalGitHubTransport(tmp_path, httpx.MockTransport(handler))
        assert restarted.attempts == 1
        (tmp_path / "github-attempt-0002.json").mkdir()
        with pytest.raises(OSError):
            await restarted.handle_async_request(request)
        assert len(calls) == 1
        restarted.attempts = 2000
        with pytest.raises(DomainError, match="evaluation_github_attempt_limit"):
            await restarted.handle_async_request(request)
        with pytest.raises(DomainError, match="evaluation_github_scope_invalid"):
            await restarted.handle_async_request(httpx.Request("POST", "https://api.github.com/anything"))
        assert len(calls) == 1

    asyncio.run(run())


def test_formal_budget_uses_fixed_increment_not_fresh_nineteen(monkeypatch):
    """The supplement is already charged; enough remaining batch capacity is sufficient."""
    monkeypatch.setattr(real, "PRICE_EXPIRY", "2099-12-31T23:59:59+00:00")
    dataset, auth = dataset_and_auth()
    auth = auth.model_copy(
        update={
            "authorization_reference": FORMAL_BATCH,
            "public_github_reads": True,
            "model_calls": True,
            "api_origin": "http://127.0.0.1:8000",
            "budget_remaining_cny": 18.934664,
            "price_evidence_hash": canonical_hash(
                {
                    "expiry": real.PRICE_EXPIRY,
                    "prices": {key: [str(p[0]), str(p[1]), p[2]] for key, p in real.PRICES.items()},
                }
            ),
        }
    )
    monkeypatch.setattr(real, "code_state", lambda: {"source_hash": auth.source_fingerprint})
    auth.validate_preflight(dataset)
    with pytest.raises(ValueError, match="budget_preflight_insufficient"):
        auth.model_copy(update={"budget_remaining_cny": 18.75}).validate_preflight(dataset)
    with pytest.raises(ValueError, match="formal_batch_fixed_authorization_required"):
        auth.model_copy(update={"acceptance_increment_baseline": Decimal("24.67518278")}).validate_preflight(dataset)


@pytest.mark.parametrize("batch", RETIRED_BATCHES)
def test_both_failed_batches_cannot_send_or_mount(batch):
    """Neither prior permit nor any prior position can bypass the new retirement."""
    sink = object.__new__(HTTPEvaluationEvidence)
    sink.permit = permit().model_copy(update={"batch_id": batch})
    with pytest.raises(DomainError, match="evaluation_batch_retired"):
        sink.validate_overlap_permit()
    store = V11Store(SimpleNamespace(connect=Mock(side_effect=AssertionError("must_not_connect"))))
    actor = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
    for prefix in ("analysis:", "sync:", "rag:"):
        command = prefix + canonical_hash([batch, DATASET_HASH, "real-03-zh", 1])
        with pytest.raises(DomainError, match="evaluation_batch_retired"):
            asyncio.run(store.reserve(actor, command, "analysis-1", Decimal("0.01"), Decimal("100")))
