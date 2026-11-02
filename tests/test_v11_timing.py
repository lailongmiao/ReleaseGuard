"""Monotonic timing preserves cancellation and does not add nested intervals."""

import asyncio
from unittest.mock import AsyncMock
from types import SimpleNamespace

from evals.v11_backend import http_evidence
from app.services.releaseguard.state_machine import DomainError

import pytest

from app.services import v11_timing as timing
from scripts import v11_probe_app


def test_nested_and_concurrent_intervals_use_union(monkeypatch):
    """Deliberately overlapping spans must not inflate attributed wall time."""
    record = {
        "origin": 0.0,
        "spans": [{"start": 1.0, "end": 4.0}, {"start": 2.0, "end": 3.0}, {"start": 3.5, "end": 6.0}],
        "truncated": False,
    }
    monkeypatch.setattr(timing.time, "perf_counter", lambda: 8.0)
    result = timing.snapshot(record)
    assert result["covered_union_seconds"] == 5.0
    assert result["unattributed_seconds"] == 3.0


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_delay_failure_and_cancel_are_measured_once(outcome):
    """Actual synthetic delay is retained even when the coroutine is cancelled."""

    async def run():
        calls = []

        async def operation():
            calls.append(1)
            await asyncio.sleep(0.025)
            if outcome == "failure":
                raise ValueError("synthetic")
            if outcome == "cancel":
                raise asyncio.CancelledError()
            return 7

        record = {"origin": timing.time.perf_counter(), "spans": [], "truncated": False}
        token = timing.current.set(record)
        try:
            if outcome == "success":
                assert await timing.measured("model_wait", operation()) == 7
            else:
                with pytest.raises(ValueError if outcome == "failure" else asyncio.CancelledError):
                    await timing.measured("model_wait", operation())
        finally:
            timing.current.reset(token)
        assert calls == [1]
        item = record["spans"][0]
        assert item["seconds"] >= 0.02
        assert item["outcome"] == ("completed" if outcome == "success" else "interrupted")
        assert set(item) == {"stage", "start", "end", "seconds", "outcome"}

    asyncio.run(run())


def test_no_capture_on_ordinary_path():
    """Without explicit evaluation context timing keeps no process-global history."""
    with timing.span("model_wait"):
        assert timing.current.get() is None


@pytest.mark.parametrize("bad", ["mode", "provider", "database"])
def test_probe_preflight_fails_before_seed(monkeypatch, bad):
    """Bad configuration cannot reach synthetic writes or any test operation."""
    monkeypatch.setenv("V11_PROBE_PROJECT", "releaseguard-v11-123456789abc")
    monkeypatch.setenv("V11_SYNTHETIC_EVIDENCE_DIR", "synthetic")
    monkeypatch.setattr(v11_probe_app.settings, "POSTGRES_DB", "v11_isolated" if bad != "database" else "wrong")
    monkeypatch.setattr(v11_probe_app.settings, "POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setattr(v11_probe_app.settings, "LOCAL_MVP_ONLY", bad != "mode")
    monkeypatch.setattr(v11_probe_app.settings, "V1_REAL_MODELS_ENABLED", bad == "provider")
    monkeypatch.setattr(v11_probe_app.settings, "V11_REAL_MODELS_ENABLED", False)
    monkeypatch.setattr(v11_probe_app.settings, "V11_REAL_READS_ENABLED", False)
    seed = AsyncMock()
    monkeypatch.setattr(v11_probe_app, "seed", seed)

    async def run():
        with pytest.raises(RuntimeError):
            async with v11_probe_app.lifespan(v11_probe_app.app):
                pytest.fail("must not start")
        seed.assert_not_awaited()

    asyncio.run(run())


def test_timing_save_failure_never_masks_original_failure_or_sends(monkeypatch, tmp_path):
    """An auxiliary disk failure stops success and preserves an existing failure."""
    sink = object.__new__(http_evidence.HTTPEvaluationEvidence)
    sink.permit = SimpleNamespace(positions={"one": {}}, batch_id="synthetic")
    sink.directory = tmp_path
    calls = []

    def fail(path, report):
        calls.append(1)
        raise OSError("synthetic")

    monkeypatch.setattr(http_evidence, "save_new", fail)
    with pytest.raises(DomainError, match="timing_save_failed_no_resend"):
        sink.save_timing("one", "analyze", {}, failed=False)
    sink.save_timing("one", "analyze", {}, failed=True)
    assert calls == [1, 1]
