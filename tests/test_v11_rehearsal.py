"""Restored capability must not confer authority on old or ineligible reports."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.schemas.local_mvp import LocalActor
from app.schemas.v11_backend import CIObservation, Coverage
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_backend import V11Backend
from app.services.v11_policy import CAPABILITIES, qualification, require_rehearsal, rehearsal_seal
from test_v11_policy import output


def test_restoration_is_not_quality_acceptance() -> None:
    """The capability is deployed code, not an environment policy bypass."""
    require_rehearsal()
    assert CAPABILITIES.version == "review-capabilities.rehearsal.2"
    assert CAPABILITIES.semantic_acceptance == "deferred"
    assert CAPABILITIES.prepare == CAPABILITIES.approval == CAPABILITIES.run == "available_subject_to_authorization"


@pytest.mark.parametrize(
    "reason", ["readonly", "partial", "ci", "unknown", "expired", "commit", "branch", "repository"]
)
def test_qualification_keeps_each_independent_gate(reason: str) -> None:
    """Complete output is insufficient if any original application constraint fails."""
    coverage = Coverage(
        manifest_completeness="complete", content_completeness="complete", analysis_coverage="complete"
    )
    ci = CIObservation(
        head_sha="a" * 40, status="success", complete=True, checks_passed=True, captured_at=datetime.now(UTC)
    )
    report = output()
    if reason == "partial":
        coverage.analysis_coverage = "partial"
    if reason == "ci":
        ci.checks_passed = False
    if reason == "unknown":
        report.unknowns = ["Unresolved observation"]
    result = qualification(
        "setup-python" if reason == "repository" else "checkout",
        "commit" if reason == "commit" else "branch_compare" if reason == "branch" else "pr",
        coverage,
        ci,
        report,
        ["snapshot_expired"] if reason == "expired" else [],
        evaluation_readonly=reason == "readonly",
    )
    assert not result.eligible and result.reason_codes
    assert "review_rehearsal_paused_this_phase" not in result.reason_codes


@pytest.mark.parametrize("version", [None, "review-capabilities.limited.1", "future-policy", CAPABILITIES.version])
def test_old_approval_capability_is_not_reactivated(version: str | None) -> None:
    """Even otherwise valid report seals cannot restore an obsolete approval epoch."""
    store = SimpleNamespace(object_row=AsyncMock(return_value={"evaluation_readonly": False}))
    backend = V11Backend(SimpleNamespace(local=SimpleNamespace()), store, Mock(), Mock())
    report = {"analysis_id": "analysis:one", "rehearsal_eligibility": {"eligible": True}, "output": "bounded"}
    backend.analysis = AsyncMock(return_value=report)
    seal = rehearsal_seal(report)
    assert seal != canonical_hash({key: value for key, value in report.items() if key != "rehearsal_eligibility"})
    product = {
        "object_id": "one",
        "analysis_id": "analysis:one",
        "binding_hash": seal,
        "rehearsal_capability_version": version,
    }
    actor = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
    if version == CAPABILITIES.version:
        assert asyncio.run(backend.guard(product, actor)) == seal
    else:
        with pytest.raises(DomainError, match="review_approval_capability_changed"):
            asyncio.run(backend.guard(product, actor))
    assert backend.provider.mock_calls == backend.github.mock_calls == []


@pytest.mark.parametrize("status,evaluation", [("failed", False), ("unknown", False), ("completed", True)])
def test_saved_output_does_not_override_adoption_status(status: str, evaluation: bool) -> None:
    """Readable output in failed or explicitly nonbusiness evidence is not actionable."""
    coverage = Coverage(
        manifest_completeness="complete", content_completeness="complete", analysis_coverage="complete"
    )
    at = datetime.now(UTC)
    ci = CIObservation(head_sha="a" * 40, status="success", complete=True, checks_passed=True, captured_at=at)
    store = SimpleNamespace(
        analysis_row=AsyncMock(
            return_value={
                "snapshot_id": "s",
                "generation": "g",
                "revision": 1,
                "dependencies": {},
                "status": status,
                "report": {
                    "output": output().model_dump(mode="json"),
                    "coverage": coverage.model_dump(mode="json"),
                    "evaluation_only": evaluation,
                },
            }
        ),
        snapshot=AsyncMock(return_value={"captured_at": at}),
        observation=AsyncMock(return_value={"captured_at": at, "facts": ci.model_dump(mode="json")}),
        dependencies=AsyncMock(return_value={}),
        object_row=AsyncMock(return_value={"repository_id": "checkout", "kind": "pr"}),
    )
    backend = V11Backend(SimpleNamespace(local=SimpleNamespace()), store, Mock(), Mock())
    backend.checked_dependencies = AsyncMock(return_value={})
    actor = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
    report = asyncio.run(backend.analysis("one", "analysis:one", actor))
    assert not report["rehearsal_eligibility"]["eligible"]
    assert ("evaluation_report_not_business" if evaluation else "analysis_not_completed") in report[
        "rehearsal_eligibility"
    ]["reason_codes"]
