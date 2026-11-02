"""Safety rules require authority and reject deliberate counterexamples."""

import asyncio
from pathlib import Path

import pytest

from evals.releaseguard.dataset import load_dataset
from evals.releaseguard.metrics import evaluate_case
from evals.releaseguard.runner import run_dataset
from evals.releaseguard.schemas import METRICS, CaseResult, RunConfig, Status

DATASET = load_dataset(Path("evals/datasets/releaseguard_v1.json"))


@pytest.fixture(scope="module")
def actual() -> tuple[CaseResult, ...]:
    """Measure actual isolated services, separate from expected labels."""
    return asyncio.run(run_dataset(DATASET, RunConfig(repeat=1)))


@pytest.mark.parametrize("index", range(len(DATASET.cases)))
def test_all_actual_rules(actual: tuple[CaseResult, ...], index: int) -> None:
    """Each required rule passes on a valid measured trajectory."""
    result = evaluate_case(DATASET.cases[index], actual[index])
    assert len(result.metrics) == 8
    assert all(m.status is Status.PASSED for m in result.metrics), [
        (m.metric_id, m.reason_codes) for m in result.metrics
    ]


@pytest.mark.parametrize("metric", METRICS)
def test_bad_authority_never_passes(actual: tuple[CaseResult, ...], metric: str) -> None:
    """Every rule independently refuses broken authority bindings."""
    source = actual[0]
    assert source.facts.domain is not None
    broken = source.facts.domain.model_copy(update={"request_hash": "0" * 64})
    result = evaluate_case(
        DATASET.cases[0], source.model_copy(update={"facts": source.facts.model_copy(update={"domain": broken})})
    )
    assert next(m for m in result.metrics if m.metric_id == metric).status is Status.FAILED


@pytest.mark.parametrize(
    "mutation,metric",
    [
        ({"calls": 2}, "tool_safety"),
        ({"calls": 2}, "idempotency"),
        ({"approvals": ()}, "approval_compliance"),
        ({"citations": ()}, "citation_validity"),
        ({"snapshots": ()}, "citation_validity"),
        ({"gate_citations": ()}, "evidence_completeness"),
        ({"ledger": ()}, "tool_safety"),
        ({"network_calls": 1}, "tool_safety"),
        ({"blocked_write_attempts": 1}, "tool_safety"),
    ],
)
def test_specific_violations(actual: tuple[CaseResult, ...], mutation: dict, metric: str) -> None:
    """A high aggregate score cannot compensate for a violated safety metric."""
    source = actual[0]
    result = evaluate_case(
        DATASET.cases[0], source.model_copy(update={"facts": source.facts.model_copy(update=mutation)})
    )
    assert next(m for m in result.metrics if m.metric_id == metric).status is Status.FAILED


def test_orphan_diagnostic_required(actual: tuple[CaseResult, ...]) -> None:
    """Hold is safe only with zero resend and an explicit unknown diagnosis."""
    index = next(i for i, c in enumerate(DATASET.cases) if c.scenario == "orphan_after_claim")
    source = actual[index]
    result = evaluate_case(
        DATASET.cases[index], source.model_copy(update={"facts": source.facts.model_copy(update={"diagnostics": ()})})
    )
    assert next(m for m in result.metrics if m.metric_id == "recovery_correctness").status is Status.FAILED


def test_missing_facts_unverified(actual: tuple[CaseResult, ...]) -> None:
    """Missing records stay in the denominator without inventing success."""
    result = evaluate_case(
        DATASET.cases[0], actual[0].model_copy(update={"facts": actual[0].facts.model_copy(update={"domain": None})})
    )
    assert all(m.status is Status.UNVERIFIED for m in result.metrics)


def test_duplicate_audit_is_not_extra_valid_evidence(actual: tuple[CaseResult, ...]) -> None:
    """Duplicated audit ids cannot produce a passing authority join."""
    source = actual[0]
    facts = source.facts.model_copy(update={"audits": (*source.facts.audits, source.facts.audits[0])})
    result = evaluate_case(DATASET.cases[0], source.model_copy(update={"facts": facts}))
    assert all(m.status is Status.FAILED for m in result.metrics)
