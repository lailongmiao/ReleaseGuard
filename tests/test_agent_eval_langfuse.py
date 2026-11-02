"""No real workspace calls: scoped adapter, lost writes and legacy regressions."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services.releaseguard.normalization import canonical_hash
from evals.evaluator import Evaluator
from evals import main
from evals.helpers import calculate_avg_scores, get_input_output, initialize_metrics_summary, initialize_report
from evals.releaseguard.langfuse_io import (
    LangfuseIO,
    LangfuseScope,
    SDKLangfuseTransport,
    ScoreIntent,
    StoredScore,
    TraceRef,
)
from evals.releaseguard.schemas import Status

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
HASH = "a" * 64


@pytest.fixture
def anyio_backend():
    """Existing AnyIO plugin runs this asyncio-only adapter."""
    return "asyncio"


def scope(**changes):
    """Separate read and write references, never real credentials."""
    values = dict(
        authorization_ref="synthetic-read",
        write_authorization_ref="synthetic-write",
        allow_write=True,
        workspace_id="synthetic-workspace",
        eval_run_id="synthetic-eval",
        dataset_hash=HASH,
        case_ids=("RG-R01",),
        from_timestamp=NOW - timedelta(hours=1),
        to_timestamp=NOW + timedelta(hours=1),
        expires_at=NOW + timedelta(hours=2),
        page_size=2,
        max_pages=2,
    )
    return LangfuseScope(**(values | changes))


def trace(**changes):
    """Safe refs only; fabricated data is explicitly fake transport."""
    values = dict(
        trace_id="synthetic-trace",
        workspace_id="synthetic-workspace",
        eval_run_id="synthetic-eval",
        case_id="RG-R01",
        attempt=1,
        dataset_hash=HASH,
        timestamp=NOW,
        input_present=True,
        output_present=True,
        judge_observation_ids=("synthetic-judge",),
    )
    return TraceRef(**(values | changes))


def intent(**changes):
    """Numeric version-pinned score intent, not a business action."""
    values = dict(
        trace_id="synthetic-trace",
        eval_run_id="synthetic-eval",
        case_id="RG-R01",
        attempt=1,
        dataset_hash=HASH,
        metric_id="tool_safety",
        metric_version="rules.v1",
        prompt_hash=HASH,
        model="deterministic",
        value=1.0,
        provenance="offline_rules",
    )
    return ScoreIntent(**(values | changes))


class FakeTransport:
    """Count transport calls, support applied/lost and unapplied/lost responses."""

    provenance = "fake"

    def __init__(self):
        """Each test has its own workspace and remote score store."""
        self.traces = (trace(),)
        self.scores = {}
        self.pages = 0
        self.writes = 0
        self.targets = {"synthetic-workspace"}
        self.lost = False
        self.apply = True
        self.failure = None

    async def workspaces(self):
        """Simulate revoked access by changing targets."""
        if self.failure:
            raise self.failure
        return self.targets

    async def page(self, scope, page):
        """Bounded pagination returned from synthetic storage."""
        self.pages += 1
        return self.traces[(page - 1) * scope.page_size : page * scope.page_size]

    async def score(self, key):
        """Safe readback of explicit ids."""
        return self.scores.get(key)

    async def write(self, intent):
        """Potential side effect before lost-response fault."""
        self.writes += 1
        if self.apply:
            self.scores[intent.key] = StoredScore(
                score_id=intent.key,
                intent_hash=canonical_hash(intent),
                trace_id=intent.trace_id,
                observation_id=intent.observation_id,
                name=intent.metric_id,
                value=intent.value,
            )
        if self.lost:
            raise TimeoutError()


@pytest.mark.anyio
async def test_off_and_legacy_no_client(monkeypatch):
    """Configured credentials must not cause client creation or verified status."""
    io = LangfuseIO(None, None)
    assert (await io.read(at=NOW)).reason_code == "langfuse_off"
    assert (await io.publish(intent(), at=NOW)).status is Status.UNVERIFIED
    evaluator = Evaluator()
    result = await evaluator.run(False, dry_run=False)
    assert result["verification_status"] == result["quality_status"] == "unverified"
    assert result["total_traces"] == 0 and result["generate_report_path"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"workspace_id": "other"},
        {"eval_run_id": "other"},
        {"dataset_hash": "b" * 64},
        {"case_id": "RG-R02"},
        {"timestamp": NOW + timedelta(hours=2)},
        {"input_present": False},
        {"output_present": False},
    ],
)
@pytest.mark.anyio
async def test_scope_and_empty_io(change):
    """Server ignoring scope cannot introduce out-of-scope facts or writes."""
    transport = FakeTransport()
    transport.traces = (trace(**change),)
    io = LangfuseIO(transport, scope())
    assert (await io.read(at=NOW)).status is Status.UNVERIFIED
    assert (await io.publish(intent(), at=NOW)).status is Status.UNVERIFIED
    assert transport.writes == 0


@pytest.mark.anyio
async def test_read_grant_write_deny_expiry_and_revocation():
    """Revalidate access after a successful read; expired/revoked writes are zero."""
    transport = FakeTransport()
    io = LangfuseIO(transport, scope(allow_write=False, write_authorization_ref=None))
    assert (await io.read(at=NOW)).status is Status.PASSED
    assert (await io.publish(intent(), at=NOW)).reason_code == "langfuse_write_denied"
    io = LangfuseIO(transport, scope())
    await io.read(at=NOW)
    assert (await io.publish(intent(), at=NOW + timedelta(hours=3))).reason_code == "langfuse_scope_denied"
    transport.targets = set()
    assert (await io.publish(intent(), at=NOW)).reason_code == "langfuse_workspace_denied"
    assert transport.writes == 0


@pytest.mark.anyio
async def test_zero_duplicate_and_page_limit():
    """No traces and truncated/duplicate pages cannot be accepted."""
    transport = FakeTransport()
    io = LangfuseIO(transport, scope())
    transport.traces = ()
    assert (await io.read(at=NOW)).reason_code == "langfuse_no_samples"
    transport.traces = (trace(), trace())
    assert (await io.read(at=NOW)).reason_code == "langfuse_duplicate_trace"
    transport.traces = tuple(trace(trace_id=f"t{i}") for i in range(4))
    assert (await io.read(at=NOW)).reason_code == "langfuse_page_limit"
    transport.traces = tuple(trace(trace_id=f"t{i}") for i in range(3))
    assert len((await io.read(at=NOW)).traces) == 3


@pytest.mark.parametrize("apply", [True, False])
@pytest.mark.anyio
async def test_lost_response_never_blind_resend(apply):
    """Applied writes reconcile; absent result remains unknown with zero resend."""
    transport = FakeTransport()
    transport.lost, transport.apply = True, apply
    io = LangfuseIO(transport, scope())
    await io.read(at=NOW)
    first = await io.publish(intent(), at=NOW)
    second = await io.publish(intent(), at=NOW)
    assert transport.writes == 1
    assert (first.status is Status.PASSED) == apply
    assert (second.status is Status.PASSED) == apply
    assert first.workspace_acceptance == "unverified"


@pytest.mark.anyio
async def test_concurrent_dedup_conflict_and_version_change():
    """Same key is one write; changed payload conflicts, changed version is new."""
    transport = FakeTransport()
    io = LangfuseIO(transport, scope())
    await io.read(at=NOW)
    results = await asyncio.gather(*[io.publish(intent(), at=NOW) for _ in range(5)])
    assert all(r.status is Status.PASSED for r in results) and transport.writes == 1
    assert (await io.publish(intent(value=0.5), at=NOW)).reason_code == "langfuse_score_conflict"
    assert (await io.publish(intent(metric_version="rules.v2"), at=NOW)).status is Status.PASSED
    assert transport.writes == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"attempt": 2},
        {"case_id": "RG-R02"},
        {"provenance": "real_judge"},
        {"provenance": "real_judge", "observation_id": "unresolved"},
    ],
)
@pytest.mark.anyio
async def test_score_binding_zero_write(changes):
    """Real Judge scores require an actual generation association."""
    transport = FakeTransport()
    io = LangfuseIO(transport, scope())
    await io.read(at=NOW)
    assert (await io.publish(intent(**changes), at=NOW)).status is Status.UNVERIFIED
    assert transport.writes == 0


@pytest.mark.parametrize(
    "error,reason",
    [(TimeoutError(), "langfuse_timeout"), (RuntimeError("do_not_persist_raw_body"), "langfuse_unavailable")],
)
@pytest.mark.anyio
async def test_redacted_errors(error, reason):
    """Raw SDK errors never enter results."""
    transport = FakeTransport()
    transport.failure = error
    assert (await LangfuseIO(transport, scope()).read(at=NOW)).reason_code == reason


@pytest.mark.anyio
async def test_current_sdk_aliases_and_async_wrapper():
    """Exercise installed SDK request schema and readback mapping without HTTP."""
    item = intent()
    response = SimpleNamespace(
        id=item.key,
        metadata={"intent_hash": canonical_hash(item)},
        trace_id=item.trace_id,
        observation_id=None,
        name=item.metric_id,
        value=1.0,
    )
    api = SimpleNamespace(
        projects=SimpleNamespace(
            get=Mock(return_value=SimpleNamespace(data=[SimpleNamespace(id="synthetic-workspace")]))
        ),
        score=SimpleNamespace(create=Mock()),
        score_v_2=SimpleNamespace(get_by_id=Mock(return_value=response)),
    )
    adapter = SDKLangfuseTransport(SimpleNamespace(api=api))
    assert await adapter.workspaces() == {"synthetic-workspace"}
    await adapter.write(item)
    request = api.score.create.call_args.kwargs["request"]
    assert api.score.create.call_args.kwargs["request_options"] == {"max_retries": 0}
    assert request.dict()["traceId"] == item.trace_id and request.dict()["dataType"] == "NUMERIC"
    assert (await adapter.score(item.key)).intent_hash == canonical_hash(item)


def test_legacy_empty_messages_and_zero_average():
    """Empty messages typed reject; no samples do not become zero quality score."""
    for messages in ([], None, [{}], ["wrong"]):
        assert get_input_output(SimpleNamespace(output={"messages": messages})) == (None, None)
    report = initialize_report("synthetic")
    initialize_metrics_summary(report, [{"name": "relevance"}])
    calculate_avg_scores(report)
    assert report["metrics_summary"]["relevance"]["avg_score"] is None


@pytest.mark.anyio
async def test_legacy_cli_summary_without_report(monkeypatch):
    """Quick/no-report and direct run have no missing-key exception."""
    report = await main.run_evaluation(False)
    assert report["generate_report_path"] is None
    monkeypatch.setattr("sys.argv", ["evals.main", "--quick", "--no-report"])
    spy = Mock()
    monkeypatch.setattr(main, "run_evaluation", lambda **kw: asyncio.sleep(0))
    monkeypatch.setattr(main.asyncio, "run", spy)
    main.main()
    spy.call_args.args[0].close()


@pytest.mark.anyio
async def test_workspace_new_send_requires_durable_gate():
    """Recreated live adapters may read back facts but must never resend unknown writes."""
    transport = FakeTransport()
    transport.provenance = "workspace"
    for _ in range(2):
        io = LangfuseIO(transport, scope())
        await io.read(at=NOW)
        result = await io.publish(intent(), at=NOW)
        assert result.reason_code == "langfuse_durable_write_gate_missing"
        assert result.status is Status.UNVERIFIED
    assert transport.writes == 0
    transport.scores[intent().key] = StoredScore(
        score_id=intent().key,
        intent_hash=canonical_hash(intent()),
        trace_id=intent().trace_id,
        name=intent().metric_id,
        value=intent().value,
    )
    assert (await io.publish(intent(), at=NOW)).reason_code == "langfuse_readback_confirmed"
    assert transport.writes == 0


class SyntheticHTTPError(Exception):
    """Expose only an HTTP code; raw test body must never reach reports."""

    def __init__(self, status_code):
        """Attach a safe code to a deliberately private synthetic body."""
        super().__init__("synthetic_private_body")
        self.status_code = status_code


@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.anyio
async def test_permission_denied_reads_and_writes(status_code):
    """Authentication/permission failures are explicit and never blindly retried."""
    transport = FakeTransport()
    transport.failure = SyntheticHTTPError(status_code)
    io = LangfuseIO(transport, scope())
    outcome = await io.read(at=NOW)
    assert outcome.reason_code == "langfuse_permission_denied" and transport.writes == 0
    transport.failure = None
    await io.read(at=NOW)

    async def denied(intent):
        transport.writes += 1
        raise SyntheticHTTPError(status_code)

    transport.write = denied
    assert (await io.publish(intent(), at=NOW)).reason_code == "langfuse_permission_denied"
    assert (await io.publish(intent(), at=NOW)).reason_code == "langfuse_write_outcome_unknown"
    assert transport.writes == 1


@pytest.mark.anyio
async def test_cancelled_write_hold_and_bounded_safe_read_retry():
    """Cancellation keeps send uncertainty; only readback is retryable."""
    transport = FakeTransport()
    io = LangfuseIO(transport, scope())
    await io.read(at=NOW)

    async def cancelled(intent):
        transport.writes += 1
        raise asyncio.CancelledError()

    transport.write = cancelled
    assert (await io.publish(intent(), at=NOW)).reason_code == "langfuse_timeout"
    assert (await io.publish(intent(), at=NOW)).reason_code == "langfuse_write_outcome_unknown"
    assert transport.writes == 1
    reads = 0

    async def transient(key):
        nonlocal reads
        reads += 1
        if reads < 3:
            raise TimeoutError()
        return None

    transport.score = transient
    assert await io.read_score(intent().key) is None and reads == 3


@pytest.mark.parametrize("kind", ["valid", "foreign_trace", "not_generation", "missing_id"])
@pytest.mark.anyio
async def test_sdk_resolves_judge_observation(kind):
    """A claimed observation must actually resolve to the same trace/generation."""
    item = SimpleNamespace(
        id="synthetic-trace",
        session_id="synthetic-eval",
        tags=["synthetic-phase7"],
        metadata=dict(
            case_id="RG-R01", attempt=1, dataset_hash=HASH, synthetic=True, judge_observation_ids=["synthetic-judge"]
        ),
        observations=[] if kind == "missing_id" else ["synthetic-judge"],
        timestamp=NOW,
        input="redacted",
        output="redacted",
    )
    observation = SimpleNamespace(
        trace_id="other" if kind == "foreign_trace" else item.id,
        type="SPAN" if kind == "not_generation" else "GENERATION",
    )
    api = SimpleNamespace(
        trace=SimpleNamespace(list=Mock(return_value=SimpleNamespace(data=[item]))),
        observations=SimpleNamespace(get=Mock(return_value=observation)),
    )
    adapter = SDKLangfuseTransport(SimpleNamespace(api=api))
    if kind == "valid":
        assert (await adapter.page(scope(), 1))[0].judge_observation_ids == ("synthetic-judge",)
    else:
        with pytest.raises(ValueError, match="langfuse_observation_unresolved"):
            await adapter.page(scope(), 1)
