"""Offline contract tests; synthetic records do not assert real execution."""

import pytest
from pydantic import ValidationError

from app.schemas.releaseguard import ExecutionPlan, ExecutionReceipt
from releaseguard_fixtures import (
    approval_fixture,
    audit_fixture,
    event_fixture,
    plan_fixture,
    receipt_fixture,
    verification_fixture,
)


def test_all_records_roundtrip() -> None:
    """Revision, hashes and audit fields survive independent JSON restoration."""
    for record in (plan_fixture(), approval_fixture(), event_fixture(), verification_fixture(), receipt_fixture()):
        assert type(record).model_validate_json(record.model_dump_json()) == record
        assert record.revision == 1 and record.audit.input_hash == record.input_hash


def test_plan_hash_covers_executable_meaning_and_excludes_generated_metadata() -> None:
    """Dry-run mode, permissions and steps affect hash; generated ids do not."""
    original = plan_fixture()
    assert plan_fixture(plan_id="other-plan", run_id="other-run").plan_hash == original.plan_hash
    assert plan_fixture(dry_run=True).plan_hash != original.plan_hash
    assert plan_fixture(permissions=["SIMULATED", "READ_ONLY"]).plan_hash != original.plan_hash
    steps = [step.model_dump() for step in original.steps]
    steps[0]["arguments_hash"] = "e" * 64
    assert plan_fixture(steps=steps).plan_hash != original.plan_hash
    second = {**steps[0], "step_id": "second-step", "order": 2, "idempotency_key": "second-key"}
    first_order = plan_fixture(steps=[steps[0], second])
    assert plan_fixture(steps=[{**second, "order": 1}, {**steps[0], "order": 2}]).plan_hash != first_order.plan_hash
    forged = original.model_dump()
    forged["dry_run"] = True
    with pytest.raises(ValidationError, match="hash_mismatch"):
        ExecutionPlan.model_validate(forged)


@pytest.mark.parametrize(
    "missing",
    ["plan_hash", "approval_id", "verification_id", "execution_event_ids", "evidence_assessment", "risk_assessment"],
)
def test_success_receipt_requires_complete_shape(missing: str) -> None:
    """No incomplete synthetic success receipt can pass structural validation."""
    with pytest.raises(ValidationError):
        receipt_fixture(**{missing: [] if missing == "execution_event_ids" else None})


def test_early_failure_receipt_and_audit_consistency() -> None:
    """Early failure does not invent an execution plan or a verification result."""
    receipt = receipt_fixture(
        final_status="FAILED",
        plan_hash=None,
        approval_id=None,
        verification_id=None,
        evidence_assessment=None,
        risk_assessment=None,
        execution_event_ids=[],
        evidence_ids=[],
        audit=audit_fixture(reason="synthetic early failure"),
    )
    assert ExecutionReceipt.model_validate_json(receipt.model_dump_json()) == receipt
    with pytest.raises(ValidationError):
        receipt_fixture(input_hash="c" * 64)
    with pytest.raises(ValidationError):
        event_fixture(actor_id="conflicting-actor")


def test_invalid_approval_and_verification_are_rejected() -> None:
    """Decisions need attribution; passed verification needs independent checks."""
    with pytest.raises(ValidationError):
        approval_fixture(approver_id=None)
    with pytest.raises(ValidationError):
        verification_fixture(checks=[])
    with pytest.raises(ValidationError):
        verification_fixture(independent_source_types=[])
    with pytest.raises(ValidationError):
        verification_fixture(status="FAILED")
    with pytest.raises(ValidationError):
        plan_fixture(rollback_steps=[])
