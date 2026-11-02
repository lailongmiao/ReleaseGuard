"""Actual domain facts, case isolation and deny-before-I/O boundary evidence."""

import asyncio
import socket
import subprocess
from pathlib import Path

import pytest

from evals.releaseguard.boundary import OfflineBoundary
from evals.releaseguard.dataset import load_dataset
from evals.releaseguard.runner import run_case, run_dataset
from evals.releaseguard.schemas import RunConfig

DATASET = load_dataset(Path("evals/datasets/releaseguard_v1.json"))


@pytest.fixture(scope="module")
def observed() -> tuple:
    """Run each isolated case twice; expectations never supply observed facts."""
    return asyncio.run(run_dataset(DATASET, RunConfig(repeat=2)))


@pytest.mark.parametrize("index", range(len(DATASET.cases)))
def test_actual_trajectory_and_repeat(observed: tuple, index: int) -> None:
    """Check reviewed state/effect oracle, identity and deterministic repetition."""
    case = DATASET.cases[index]
    first, second = observed[index], observed[index + len(DATASET.cases)]
    facts = first.facts
    assert facts.completed, facts.diagnostics
    assert facts.domain is not None and facts.gate is not None
    assert facts.domain.status.value in case.oracle.states
    assert facts.calls == case.oracle.calls
    assert facts.target_value == case.oracle.target_value
    assert facts.gate.sufficient == case.oracle.gate_sufficient
    assert (facts.domain.receipt.final_status.value if facts.domain.receipt else None) == case.oracle.receipt
    assert set(case.oracle.diagnostics) <= set(facts.diagnostics)
    assert first.semantic_hash == second.semantic_hash
    assert facts.network_calls == facts.subprocess_calls == 0
    assert facts.blocked_network_attempts == facts.blocked_process_attempts == facts.blocked_write_attempts == 0
    assert all(e.run_id == facts.domain.run_id for e in facts.domain.execution_events)


def test_boundary_zero_external_effects(tmp_path: Path) -> None:
    """Attempts are independently intercepted before network/process/host writes."""
    with OfflineBoundary() as boundary:
        with socket.socket() as sock, pytest.raises(PermissionError):
            sock.connect(("127.0.0.1", 1))
        with pytest.raises(PermissionError):
            subprocess.run(["never-started"])
        with pytest.raises(PermissionError):
            (tmp_path / "never-written").write_text("synthetic")
    assert boundary.network_attempts == boundary.process_attempts == boundary.write_attempts == 1
    assert not (tmp_path / "never-written").exists()


def test_timeout_remains_in_denominator() -> None:
    """A missing authority is incomplete evidence, never an empty success."""
    result = asyncio.run(run_case(DATASET.cases[0], timeout=1e-9))
    assert not result.facts.completed
    assert "runner_timeout_or_cancelled" in result.facts.diagnostics


def test_risk_fixture_does_not_consume_oracle() -> None:
    """An altered expected label cannot alter the actual risk fixture."""
    case = DATASET.cases[0]
    wrong = case.model_copy(update={"oracle": case.oracle.model_copy(update={"risk": "HIGH"})})
    result = asyncio.run(run_case(wrong))
    assert result.facts.domain is not None and result.facts.domain.risk_assessment is not None
    assert result.facts.domain.risk_assessment.level.value == case.input_risk == "LOW"
