"""Counters consume committed facts; cardinality and unknowns are bounded."""

import asyncio

import pytest
from prometheus_client import CollectorRegistry

from app.core import metrics
from app.core.metrics import ReleaseGuardMetrics
from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.services.llm.contracts import UsageInfo
from releaseguard_fixtures import command_fixture, state_fixture


def test_unknown_usage_and_invalid_labels() -> None:
    """Unknown cost/usage has no fake zero consumption sample."""
    registry = CollectorRegistry()
    telemetry = ReleaseGuardMetrics(registry)
    telemetry.usage("judge", UsageInfo.unknown())
    assert registry.get_sample_value("releaseguard_llm_tokens_total", {"role": "judge"}) is None
    assert (
        registry.get_sample_value(
            "releaseguard_llm_measurements_total", {"role": "judge", "measurement": "tokens", "quality": "unknown"}
        )
        == 1
    )
    with pytest.raises(ValueError, match="metric_label_invalid"):
        telemetry.fact("synthetic-run-id", "passed")


def test_repository_replay_is_not_double_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use actual authority/CAS; replay must return without new metrics."""
    registry = CollectorRegistry()
    monkeypatch.setattr(metrics, "releaseguard_metrics", ReleaseGuardMetrics(registry))

    async def case() -> None:
        repo = InMemoryReleaseGuardRepository()
        state = state_fixture()
        await repo.create_change(state.change_request)
        await repo.create_run(state)
        command = command_fixture(state, "COLLECTING_EVIDENCE")
        await repo.commit_run_update(command)
        value = registry.get_sample_value("releaseguard_facts_total", {"category": "state", "outcome": "recorded"})
        await repo.commit_run_update(command)
        assert (
            registry.get_sample_value("releaseguard_facts_total", {"category": "state", "outcome": "recorded"})
            == value
        )

        assert (
            registry.get_sample_value(
                "releaseguard_states_total", {"status": "COLLECTING_EVIDENCE", "risk": "UNKNOWN"}
            )
            == 1
        )

    asyncio.run(case())


def test_observer_failure_does_not_reject_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A metric exporter failure cannot change domain acceptance."""

    def unavailable(*_: object, **__: object) -> None:
        raise RuntimeError("synthetic exporter unavailable")

    monkeypatch.setattr(metrics.releaseguard_metrics, "fact", unavailable)

    async def case() -> None:
        repo = InMemoryReleaseGuardRepository()
        state = state_fixture()
        await repo.create_change(state.change_request)
        await repo.create_run(state)
        result = await repo.commit_run_update(command_fixture(state, "COLLECTING_EVIDENCE"))
        assert result.status.value == "COLLECTING_EVIDENCE"

    asyncio.run(case())
