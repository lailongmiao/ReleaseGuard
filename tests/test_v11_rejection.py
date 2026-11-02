"""Independent claim roles, precise rejection, private diagnostics and unchanged failures."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from psycopg import DataError, OperationalError

from app.api.v1.releaseguard import actor
from app.api.v1.reviews import review_runtime, router
from app.core.config import settings
from app.core.limiter import limiter
from app.core.telemetry import sanitize_event
from app.schemas.local_mvp import LocalActor
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_comparison import ComparedReviewOutput, validate_comparisons
from app.services.v11_policy import validate_output
from app.services.v11_provider import build_context
from app.services.v11_rejection import (
    DIAGNOSTIC_KEY,
    DIAGNOSTIC_LIMIT,
    ReviewValidationError,
    rejection_diagnostic,
)
from app.services.v11_storage import V11Store
from test_v11_provider import citation
from evals.v11_backend.behavior_cases import evidence
from app.services.releaseguard.normalization import canonical_hash


def example(kind: str = "observation", source_kind: str = "diff", path: str = "README.md") -> tuple:
    """Documentation is an actual changed line; policy and CI remain different evidence roles."""
    c = citation().model_copy(
        update={
            "path": path,
            "text": "duration in seconds",
            "text_hash": canonical_hash("duration in seconds"),
            "kind": source_kind,
            "source_role": {"diff": "change_observation", "ci": "ci_observation", "rule": "application_policy"}[
                source_kind
            ],
        }
    )
    output = ComparedReviewOutput.model_validate(
        {
            "summary": "Scoped statement.",
            "observations": [
                {"text": "The description now specifies seconds.", "kind": kind, "citation_ids": [c.citation_id]}
            ],
            "risks": [],
            "validation_suggestions": [],
            "unknowns": [],
            "candidate_steps": [],
            "model_risk": "MEDIUM",
            "comparisons": [
                {
                    "section": "observations",
                    "index": 0,
                    "classification": "documentation" if source_kind == "diff" else "rule",
                    "before": "",
                    "after": "The unit is explicit.",
                    "before_ids": [],
                    "after_ids": [c.citation_id] if source_kind == "diff" else [],
                }
            ],
        }
    )
    user, _ = build_context("Explain wording", [], [], [citation()], line_roles={citation().citation_id: ["add"]})
    sent = json.loads(user)
    sent["sources"][0].update(kind=c.kind, source_role=c.source_role, path=path, text=c.text)
    return output, [c], sent


@pytest.mark.parametrize("path", ["README.md", "action.yml"])
def test_changed_document_description_is_valid_diff_observation(path: str) -> None:
    """Changed README and YAML descriptions remain valid diff evidence."""
    output, _, _ = example(path=path)
    facts, citations, roles = evidence([{"path": path, "before": "duration", "after": "duration in seconds"}])
    user, citations = build_context(
        "Explain documentation wording",
        [],
        [],
        citations,
        {"coverage": facts["coverage"], "index_complete": True},
        roles,
    )
    output.observations[0].citation_ids = [c.citation_id for c in citations]
    output.comparisons[0].before_ids = [c.citation_id for c in citations if c.side == "LEFT"]
    output.comparisons[0].after_ids = [c.citation_id for c in citations if c.side == "RIGHT"]
    validate_output(output, citations)
    validate_comparisons(output, citations, json.loads(user))


@pytest.mark.parametrize("kind,source_kind", [("observation", "ci"), ("rule", "rule"), ("inference", "diff")])
def test_distinct_claim_and_comparison_roles(kind: str, source_kind: str) -> None:
    """Claim epistemic role and comparison category are separate axes."""
    output, citations, sent = example(kind, source_kind)
    validate_output(output, citations)
    validate_comparisons(output, citations, sent)


@pytest.mark.parametrize(
    "kind,source_kind,code",
    [
        ("observation", "rule", "analysis_observation_requires_diff"),
        ("rule", "diff", "analysis_rule_citation_invalid"),
        ("rule", "ci", "analysis_rule_citation_invalid"),
    ],
)
def test_wrong_role_is_not_auto_reclassified(kind: str, source_kind: str, code: str) -> None:
    """Reject wrong evidence roles without editing the output."""
    output, citations, sent = example(kind, source_kind)
    before = output.model_dump_json()
    with pytest.raises(ReviewValidationError) as caught:
        validate_output(output, citations)
    error = caught.value
    diagnostic = rejection_diagnostic(error, output, citations, sent)
    assert (error.code, error.section, error.index) == (code, "observations", 0)
    assert diagnostic["claim"]["kind"] == kind
    assert diagnostic["claim"]["references"][0]["kind"] == source_kind
    assert diagnostic["claim"]["references"][0]["actually_sent"] is True
    assert output.model_dump_json() == before


def test_one_bad_claim_located_and_other_claims_never_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the failing claim is retained, redacted before serialization."""
    output, citations, sent = example("observation", "rule")
    output.observations.insert(
        0, output.observations[0].model_copy(update={"kind": "rule", "text": "unrelated-private-claim"})
    )
    output.comparisons[0].index = 1
    monkeypatch.setattr(settings, "CHAT_LLM_API_KEY", "test-secret-that-must-be-removed")
    output.observations[
        1
    ].text = "Policy test-secret-that-must-be-removed token=confidential bearer abc.def local-change:other-object"
    with pytest.raises(ReviewValidationError) as caught:
        validate_output(output, citations)
    d = rejection_diagnostic(caught.value, output, citations, sent)
    assert d["index"] == 1 and d["comparisons"][0]["classification"] == "rule"
    encoded = json.dumps(d)
    for private in (
        "test-secret-that-must-be-removed",
        "confidential",
        "abc.def",
        "other-object",
        "unrelated-private-claim",
    ):
        assert private not in encoded
    assert len(encoded.encode()) <= DIAGNOSTIC_LIMIT
    caught.value.diagnostic = d
    assert str(caught.value) == "analysis_observation_requires_diff"
    assert "Policy" not in repr(caught.value)
    assert "Policy" not in json.dumps(sanitize_event(None, None, {"event": "review_failed", "diagnostic": d}))


def test_absent_or_unsent_citation_not_disclosed() -> None:
    """Missing source identities cannot disclose unrelated objects."""
    output, citations, sent = example()
    output.observations[0].citation_ids = ["private-foreign-citation"]
    with pytest.raises(ReviewValidationError) as caught:
        validate_output(output, citations)
    d = rejection_diagnostic(caught.value, output, citations, sent)
    assert "private-foreign-citation" not in json.dumps(d)
    assert d["claim"]["references"][0]["actually_sent"] is False
    sent["sources"] = []
    assert (
        rejection_diagnostic(caught.value, output, citations, sent)["comparisons"][0]["after_references"][0][
            "actually_sent"
        ]
        is False
    )
    output.observations[0].citation_ids = []
    with pytest.raises(ValidationError):
        ComparedReviewOutput.model_validate(output.model_dump())


def test_comparison_rejection_has_exact_location_and_bound() -> None:
    """Comparison errors locate the same claim and fit the byte cap."""
    output, citations, sent = example()
    output.comparisons[0].classification = "behavior_added"
    output.comparisons[0].after = "界" * 500
    with pytest.raises(ReviewValidationError) as caught:
        validate_comparisons(output, citations, sent)
    d = rejection_diagnostic(caught.value, output, citations, sent)
    assert d["validator"] == "validate_comparisons" and d["index"] == 0
    assert d["code"] == "analysis_behavior_requires_both_sides"
    assert len(json.dumps(d, ensure_ascii=False).encode()) <= DIAGNOSTIC_LIMIT


def mock_storage(owner: int = 1, revoked: bool = False, retained: bool = True) -> tuple:
    """Model transaction and ordered rows without accessing a database."""
    db = AsyncMock()
    row = {
        "report": {DIAGNOSTIC_KEY: {"claim": "restricted"}} if retained else {},
        "dependencies": {"sources": {"source": 1}},
        "snapshot_id": "snapshot:one",
        "generation": "generation:one",
    }
    cursor = AsyncMock()
    cursor.fetchone.side_effect = [row, {"acl": [] if revoked else [1], "deleted": False}]
    db.execute.return_value = cursor

    @asynccontextmanager
    async def context():
        yield db

    db.transaction = context
    connection = SimpleNamespace(connect=AsyncMock(return_value=context()), authenticate=AsyncMock())
    store = V11Store(connection)
    store.object_row = AsyncMock(return_value={"owner_user_id": owner})
    return store, db


@pytest.mark.parametrize("case", ["owner", "reviewer", "revoked", "cross_object", "old_failure"])
def test_local_rejection_read_reauthorizes_before_body(case: str) -> None:
    """Membership and every saved source are checked before text is returned."""

    async def run() -> None:
        store, db = mock_storage(
            owner=2 if case == "reviewer" else 1, revoked=case == "revoked", retained=case != "old_failure"
        )
        current = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
        if case == "cross_object":
            store.object_row.side_effect = DomainError("local_object_not_found")
        if case in {"reviewer", "revoked", "cross_object"}:
            with pytest.raises(DomainError):
                await store.rejection("one", "analysis:one", current)
        else:
            result = await store.rejection("one", "analysis:one", current)
            assert result["availability"] == (
                "not_retained_unrecoverable" if case == "old_failure" else "saved_redacted"
            )
            assert db.execute.await_args_list[0].args[1] == ("one", "analysis:one", 1)

    asyncio.run(run())


def test_diagnostic_save_cannot_change_failure_or_budget_and_revocation_drops_body() -> None:
    """Revocation drops diagnostic content but preserves the original failed result."""

    async def run() -> None:
        store, db = mock_storage()
        store.object_row.side_effect = DomainError("source_not_found")
        current = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
        failure = {
            "status": "failed",
            "reason": "analysis_observation_requires_diff",
            DIAGNOSTIC_KEY: {"claim": "private"},
        }
        await store.analysis_failure("one", current, "analysis:one", failure)
        sql, args = db.execute.await_args.args
        assert "status='started'" in sql and "v11_analyses" in sql
        assert "v1_budget" not in sql and "v1_commands" not in sql
        assert args[1].obj == {"status": "failed", "reason": "analysis_observation_requires_diff"}
        assert DIAGNOSTIC_KEY in failure

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["diagnostic_write", "diagnostic_authorization"])
@pytest.mark.parametrize("status", ["failed", "unknown"])
def test_optional_diagnostic_failure_preserves_original_rejection(
    fault: str, status: str, caplog: pytest.LogCaptureFixture
) -> None:
    """An optional diagnostic failure cannot prevent the known rejection from being saved."""

    async def run() -> None:
        store, db = mock_storage()
        current = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
        failure = {
            "status": status,
            "reason": "analysis_outcome_unknown" if status == "unknown" else "analysis_observation_requires_diff",
        }
        persisted = []

        async def execute(query: str, args: tuple) -> SimpleNamespace:
            report = next(arg.obj for arg in args if hasattr(arg, "obj"))
            if DIAGNOSTIC_KEY in report:
                raise DataError("synthetic-private-diagnostic-content")
            persisted.append(report)
            return SimpleNamespace(fetchone=AsyncMock(return_value={"analysis_id": "analysis:one"}))

        db.execute.side_effect = execute
        if fault == "diagnostic_authorization":
            store.connection.authenticate.side_effect = OperationalError("synthetic-private-session-content")
        await store.analysis_failure("one", current, "analysis:one", {**failure, DIAGNOSTIC_KEY: {"claim": "private"}})
        assert persisted == [failure]
        assert all(
            "v1_budget" not in call.args[0] and "v1_commands" not in call.args[0]
            for call in db.execute.await_args_list
        )

    asyncio.run(run())
    assert "synthetic-private-diagnostic-content" not in caplog.text
    assert "synthetic-private-session-content" not in caplog.text


def test_diagnostic_does_not_overwrite_a_previously_finalized_slot() -> None:
    """An empty started-slot CAS must not enter the optional diagnostic transaction."""

    async def run() -> None:
        store, db = mock_storage()
        db.execute.return_value.fetchone.side_effect = [None]
        current = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
        await store.analysis_failure(
            "one",
            current,
            "analysis:one",
            {"status": "failed", "reason": "analysis_observation_requires_diff", DIAGNOSTIC_KEY: {"claim": "private"}},
        )
        assert db.execute.await_count == 1
        store.connection.authenticate.assert_not_awaited()
        store.object_row.assert_not_awaited()

    asyncio.run(run())


def test_core_failure_persistence_error_is_not_reported_as_saved() -> None:
    """Failure of the required state write still propagates and keeps the existing hold."""

    async def run() -> None:
        store, db = mock_storage()
        db.execute.side_effect = OperationalError("synthetic-storage-unavailable")
        current = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
        with pytest.raises(OperationalError):
            await store.analysis_failure(
                "one", current, "analysis:one", {"status": "failed", "reason": "analysis_observation_requires_diff"}
            )
        assert db.execute.await_count == 1

    asyncio.run(run())


def test_rejection_route_is_local_only_and_never_exposes_body_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nonlocal clients and failed reads never receive restricted text."""
    monkeypatch.setattr(settings, "LOCAL_MVP_ONLY", True)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router, prefix="/reviews")
    app.dependency_overrides[actor] = lambda: LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
    store = SimpleNamespace(
        rejection=AsyncMock(
            side_effect=ReviewValidationError(
                "analysis_observation_requires_diff", "validate_output", "observations", 0
            )
        )
    )
    store.rejection.side_effect.diagnostic = {"claim": "private"}
    app.dependency_overrides[review_runtime] = lambda: SimpleNamespace(store=store)

    async def run() -> None:
        for host, status in [("192.0.2.1", 404), ("127.0.0.1", 409)]:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, client=(host, 1234)), base_url="http://local.invalid"
            ) as client:
                r = await client.get("/reviews/one/analyses/one/rejection")
                assert r.status_code == status and "private" not in r.text
        assert store.rejection.await_count == 1

    asyncio.run(run())
