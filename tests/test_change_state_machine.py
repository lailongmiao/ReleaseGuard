"""State table, rejection, approval invalidation and recovery-shape tests."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.schemas.releaseguard import (
    ApprovalDecision,
    ApprovalInvalidation,
    ChangeRequest,
    ChangeRunState,
    ChangeStatus,
    RunCommand,
    request_input_hash,
    ChangeRequestInput,
)
from app.services.releaseguard.state_machine import DomainError, apply_run_command, legal_targets
from releaseguard_fixtures import (
    NOW,
    advance,
    approval_fixture,
    audit_fixture,
    command_fixture,
    plan_fixture,
    risk_assessment,
    state_fixture,
    receipt_fixture,
    edge_inputs,
    event_fixture,
    verification_fixture,
)


DOCUMENTED_EDGES = {
    "RECEIVED": {"COLLECTING_EVIDENCE"},
    "COLLECTING_EVIDENCE": {"WAITING_FOR_EVIDENCE", "EVIDENCE_SUFFICIENT"},
    "WAITING_FOR_EVIDENCE": {"COLLECTING_EVIDENCE"},
    "EVIDENCE_SUFFICIENT": {"RISK_ASSESSED"},
    "RISK_ASSESSED": {"PLAN_READY"},
    "PLAN_READY": {"WAITING_FOR_APPROVAL"},
    "WAITING_FOR_APPROVAL": {"APPROVED", "REJECTED", "WAITING_FOR_EVIDENCE"},
    "APPROVED": {"DRY_RUN"},
    "DRY_RUN": {"EXECUTING_SANDBOX", "DRY_RUN_FAILED", "DRY_RUN_PASSED"},
    "DRY_RUN_PASSED": {"RECEIPTED"},
    "DRY_RUN_FAILED": {"FAILED"},
    "EXECUTING_SANDBOX": {"VERIFYING", "EXECUTION_FAILED", "EXECUTION_UNKNOWN"},
    "EXECUTION_UNKNOWN": {"VERIFYING", "RECOVERY_REQUIRED", "FAILED"},
    "EXECUTION_FAILED": {"RECOVERY_REQUIRED", "FAILED"},
    "RECOVERY_REQUIRED": {"RECOVERING"},
    "RECOVERING": {"VERIFYING", "FAILED", "EXECUTION_UNKNOWN"},
    "VERIFYING": {"SUCCEEDED", "RECOVERY_REQUIRED", "FAILED"},
    "SUCCEEDED": {"RECEIPTED"},
    "REJECTED": {"RECEIPTED"},
    "FAILED": {"RECEIPTED"},
    "RECEIPTED": set(),
}
EXPECTED_EDGES = {
    ChangeStatus(source): {ChangeStatus(target) for target in targets}
    | ({ChangeStatus.FAILED} if source not in {"SUCCEEDED", "REJECTED", "FAILED", "RECEIPTED"} else set())
    for source, targets in DOCUMENTED_EDGES.items()
}
EDGES = [(source, target) for source, targets in EXPECTED_EDGES.items() for target in targets]
ILLEGAL = [
    (source, target) for source in ChangeStatus for target in ChangeStatus if target not in EXPECTED_EDGES[source]
]


def test_table_matches_independent_documented_specification() -> None:
    """Changing implementation edges cannot silently shrink the test matrix."""
    assert {source: set(legal_targets(source)) for source in ChangeStatus} == EXPECTED_EDGES


@pytest.mark.parametrize("source,target", EDGES)
def test_all_legal_edges(source: ChangeStatus, target: ChangeStatus) -> None:
    """Every documented edge validates synthetic preconditions and records audit."""
    state = state_fixture(source, dry_run=target is ChangeStatus.DRY_RUN_PASSED)
    result = advance(state, target)
    assert result.status is target and result.state_version == state.state_version + 1
    record = result.transition_history[-1]
    assert record.from_status is source and record.to_status is target
    assert record.audit.actor_id == "synthetic-actor" and record.input_hash == record.audit.input_hash
    assert ChangeRunState.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("source,target", ILLEGAL)
def test_unlisted_edges_fail_without_mutation(source: ChangeStatus, target: ChangeStatus) -> None:
    """No arbitrary status string can skip approval, verification or terminality."""
    state = state_fixture(source)
    before = state.model_dump_json()
    with pytest.raises(DomainError, match="invalid_state_transition"):
        apply_run_command(state, command_fixture(state, target))
    assert state.model_dump_json() == before


def test_unknown_risk_cannot_enter_plan_path() -> None:
    """An explicit UNKNOWN outcome is valid but blocks plan-ready progression."""
    state = state_fixture(ChangeStatus.EVIDENCE_SUFFICIENT)
    state = apply_run_command(
        state,
        command_fixture(
            state,
            ChangeStatus.RISK_ASSESSED,
            inputs={"risk_assessment": risk_assessment(level="UNKNOWN", uncertainty=["synthetic missing context"])},
        ),
    )
    with pytest.raises(DomainError, match="transition_precondition_failed"):
        advance(state, ChangeStatus.PLAN_READY)
    assert state.risk_assessment.level.value == "UNKNOWN"


def test_expired_approval_and_dry_run_only_plan_fail_closed() -> None:
    """Approval validity is rechecked at operation time, including after restore."""
    state = state_fixture(ChangeStatus.WAITING_FOR_APPROVAL)
    state = apply_run_command(
        state,
        command_fixture(
            state, ChangeStatus.APPROVED, inputs={"approval": approval_fixture(expires_at=NOW + timedelta(seconds=1))}
        ),
    )
    restored = ChangeRunState.model_validate_json(state.model_dump_json())
    with pytest.raises(DomainError, match="transition_precondition_failed"):
        apply_run_command(
            restored,
            command_fixture(
                restored,
                ChangeStatus.DRY_RUN,
                audit=audit_fixture(occurred_at=NOW + timedelta(seconds=1)).model_dump(exclude={"input_hash"}),
            ),
        )
    dry_plan = plan_fixture(dry_run=True, plan_id="synthetic-dry-plan")
    changed = apply_run_command(state, command_fixture(state, action="replace_plan", replacement_plan=dry_plan))
    assert changed.status is ChangeStatus.PLAN_READY and changed.approval is None


def test_revision_and_plan_replacement_preserve_original_approval() -> None:
    """New hashes invalidate by appending a fact instead of rewriting a decision."""
    state = state_fixture(ChangeStatus.APPROVED)
    updated = state.change_request.model_dump()
    updated.update(revision=2, title="synthetic revised request")
    updated["request_hash"] = request_input_hash(
        ChangeRequestInput.model_validate({key: updated[key] for key in ChangeRequestInput.model_fields})
    )
    revised = apply_run_command(
        state, command_fixture(state, action="revise", replacement_request=ChangeRequest.model_validate(updated))
    )
    assert revised.revision == 2 and revised.status is ChangeStatus.RECEIVED and revised.approval is None
    assert state.approval.decision is ApprovalDecision.APPROVED
    assert any(isinstance(record, ApprovalInvalidation) for record in revised.record_history)
    assert revised.request_history[0] == state.change_request
    replaced = apply_run_command(
        state,
        command_fixture(
            state, action="replace_plan", replacement_plan=plan_fixture(plan_id="replacement-plan", dry_run=True)
        ),
    )
    assert replaced.status is ChangeStatus.PLAN_READY and replaced.approval is None
    for active in (
        ChangeStatus.DRY_RUN,
        ChangeStatus.EXECUTING_SANDBOX,
        ChangeStatus.SUCCEEDED,
        ChangeStatus.RECEIPTED,
    ):
        current = state_fixture(active)
        with pytest.raises(DomainError, match="plan_change_conflict"):
            apply_run_command(
                current,
                command_fixture(
                    current, action="replace_plan", replacement_plan=plan_fixture(plan_id="replacement-plan")
                ),
            )


def test_command_replay_and_conflict() -> None:
    """Repeated commands do not duplicate history; changed payloads are rejected."""
    state = state_fixture()
    command = command_fixture(state, ChangeStatus.COLLECTING_EVIDENCE)
    result = apply_run_command(state, command)
    assert apply_run_command(result, command) == result
    conflict = command_fixture(
        state,
        ChangeStatus.FAILED,
        audit=audit_fixture(reason="synthetic changed command").model_dump(exclude={"input_hash"}),
    )
    with pytest.raises(DomainError, match="command_conflict"):
        apply_run_command(result, conflict)
    forged = command.model_dump()
    forged["target_status"] = "FAILED"
    with pytest.raises(ValidationError, match="hash_mismatch"):
        RunCommand.model_validate(forged)


def test_recovered_receipt_requires_compensation_and_independent_verification() -> None:
    """Synthetic recovery has distinct events and cannot masquerade as success."""
    state = state_fixture(ChangeStatus.RECOVERING)
    state = advance(state, ChangeStatus.VERIFYING)
    state = advance(state, ChangeStatus.SUCCEEDED)
    receipt = advance(state, ChangeStatus.RECEIPTED)
    assert receipt.receipt.final_status.value == "RECOVERED"
    bad = receipt_fixture(approval_id="foreign-approval")
    normal = state_fixture(ChangeStatus.SUCCEEDED)
    with pytest.raises(DomainError, match="transition_precondition_failed"):
        apply_run_command(normal, command_fixture(normal, ChangeStatus.RECEIPTED, inputs={"receipt": bad}))


@pytest.mark.parametrize(
    "source,target",
    [
        (ChangeStatus.COLLECTING_EVIDENCE, ChangeStatus.EVIDENCE_SUFFICIENT),
        (ChangeStatus.COLLECTING_EVIDENCE, ChangeStatus.WAITING_FOR_EVIDENCE),
        (ChangeStatus.EVIDENCE_SUFFICIENT, ChangeStatus.RISK_ASSESSED),
        (ChangeStatus.RISK_ASSESSED, ChangeStatus.PLAN_READY),
        (ChangeStatus.PLAN_READY, ChangeStatus.WAITING_FOR_APPROVAL),
        (ChangeStatus.WAITING_FOR_APPROVAL, ChangeStatus.APPROVED),
        (ChangeStatus.WAITING_FOR_APPROVAL, ChangeStatus.REJECTED),
        (ChangeStatus.WAITING_FOR_APPROVAL, ChangeStatus.WAITING_FOR_EVIDENCE),
        (ChangeStatus.DRY_RUN, ChangeStatus.EXECUTING_SANDBOX),
        (ChangeStatus.DRY_RUN, ChangeStatus.DRY_RUN_FAILED),
        (ChangeStatus.EXECUTING_SANDBOX, ChangeStatus.VERIFYING),
        (ChangeStatus.EXECUTING_SANDBOX, ChangeStatus.EXECUTION_FAILED),
        (ChangeStatus.EXECUTION_FAILED, ChangeStatus.RECOVERY_REQUIRED),
        (ChangeStatus.RECOVERY_REQUIRED, ChangeStatus.RECOVERING),
        (ChangeStatus.RECOVERING, ChangeStatus.VERIFYING),
        (ChangeStatus.VERIFYING, ChangeStatus.SUCCEEDED),
        (ChangeStatus.VERIFYING, ChangeStatus.RECOVERY_REQUIRED),
        (ChangeStatus.SUCCEEDED, ChangeStatus.RECEIPTED),
        (ChangeStatus.FAILED, ChangeStatus.RECEIPTED),
        (ChangeStatus.REJECTED, ChangeStatus.RECEIPTED),
        (ChangeStatus.RECEIVED, ChangeStatus.FAILED),
    ],
)
def test_required_results_and_failure_reasons_cannot_be_omitted(source: ChangeStatus, target: ChangeStatus) -> None:
    """A legal edge alone cannot bypass its documented preconditions."""
    state = state_fixture(source)
    before = state.model_dump_json()
    with pytest.raises(DomainError, match="transition_precondition_failed"):
        apply_run_command(state, command_fixture(state, target))
    assert state.model_dump_json() == before


def test_dry_run_only_plan_cannot_enter_execution() -> None:
    """Fresh approvals and a supplied passing simulation still cannot execute a simulation-only plan."""
    state = state_fixture(ChangeStatus.PLAN_READY)
    plan = plan_fixture(plan_id="synthetic-dry-only", dry_run=True)
    state = apply_run_command(state, command_fixture(state, action="replace_plan", replacement_plan=plan))
    state = apply_run_command(
        state,
        command_fixture(
            state,
            ChangeStatus.WAITING_FOR_APPROVAL,
            inputs={
                "approval": approval_fixture(
                    approval_id="synthetic-dry-pending",
                    decision="PENDING",
                    approver_id=None,
                    decided_at=None,
                    plan_hash=plan.plan_hash,
                )
            },
        ),
    )
    state = apply_run_command(
        state,
        command_fixture(
            state,
            ChangeStatus.APPROVED,
            inputs={"approval": approval_fixture(approval_id="synthetic-dry-approved", plan_hash=plan.plan_hash)},
        ),
    )
    state = advance(state, ChangeStatus.DRY_RUN)
    inputs = edge_inputs(state, ChangeStatus.EXECUTING_SANDBOX)["inputs"].model_dump()
    inputs["simulation_result"]["plan_hash"] = plan.plan_hash
    with pytest.raises(DomainError, match="transition_precondition_failed"):
        apply_run_command(state, command_fixture(state, ChangeStatus.EXECUTING_SANDBOX, inputs=inputs))


def test_expiry_rechecked_at_execution_but_receipt_preserves_start_fact() -> None:
    """Expiry prevents starting execution; later expiry does not invalidate a completed historical start."""
    state = state_fixture(ChangeStatus.WAITING_FOR_APPROVAL)
    state = apply_run_command(
        state,
        command_fixture(
            state, ChangeStatus.APPROVED, inputs={"approval": approval_fixture(expires_at=NOW + timedelta(seconds=1))}
        ),
    )
    state = advance(state, ChangeStatus.DRY_RUN)
    with pytest.raises(DomainError, match="transition_precondition_failed"):
        apply_run_command(
            state,
            command_fixture(
                state,
                ChangeStatus.EXECUTING_SANDBOX,
                inputs=edge_inputs(state, ChangeStatus.EXECUTING_SANDBOX)["inputs"],
                audit=audit_fixture(occurred_at=NOW + timedelta(seconds=1)).model_dump(exclude={"input_hash"}),
            ),
        )
    state = advance(state, ChangeStatus.EXECUTING_SANDBOX)
    state = advance(state, ChangeStatus.VERIFYING)
    state = advance(state, ChangeStatus.SUCCEEDED)
    late = NOW + timedelta(seconds=2)
    result = apply_run_command(
        state,
        command_fixture(
            state,
            ChangeStatus.RECEIPTED,
            inputs={"receipt": receipt_fixture(issued_at=late, audit=audit_fixture(occurred_at=late))},
            audit=audit_fixture(occurred_at=late).model_dump(exclude={"input_hash"}),
        ),
    )
    assert result.receipt is not None and result.receipt.final_status.value == "SUCCEEDED"


@pytest.mark.parametrize(
    "events",
    [
        [event_fixture()],
        [
            event_fixture(event_id="synthetic-start", status="STARTED"),
            event_fixture(),
            event_fixture(event_id="another-success"),
        ],
        [event_fixture(event_id="synthetic-start", status="STARTED", step_id="unknown-step"), event_fixture()],
    ],
)
def test_invalid_event_sequences_fail_without_changing_history(events: list[object]) -> None:
    """Missing starts, repeated terminal observations and foreign steps are rejected."""
    state = state_fixture(ChangeStatus.EXECUTING_SANDBOX)
    before = state.model_dump_json()
    with pytest.raises(DomainError, match="transition_precondition_failed"):
        apply_run_command(state, command_fixture(state, ChangeStatus.VERIFYING, inputs={"execution_events": events}))
    assert state.model_dump_json() == before


@pytest.mark.parametrize("binding", [{"revision": 2}, {"plan_hash": "0" * 64}, {"run_id": "foreign-run"}])
def test_old_or_foreign_approval_and_verification_cannot_be_used(binding: dict[str, object]) -> None:
    """Decisions and verification results must match all current version and run fields."""
    for source, target, field, record in [
        (ChangeStatus.WAITING_FOR_APPROVAL, ChangeStatus.APPROVED, "approval", approval_fixture(**binding)),
        (ChangeStatus.VERIFYING, ChangeStatus.SUCCEEDED, "verification_result", verification_fixture(**binding)),
    ]:
        state = state_fixture(source)
        with pytest.raises(DomainError, match="transition_precondition_failed"):
            apply_run_command(state, command_fixture(state, target, inputs={field: record}))


def test_restore_rejects_a_fabricated_transition_chain() -> None:
    """JSON status edits cannot turn a legal history into an approval bypass."""
    values = state_fixture(ChangeStatus.COLLECTING_EVIDENCE).model_dump()
    values["transition_history"][0]["to_status"] = "APPROVED"
    with pytest.raises(ValidationError, match="invalid_state_transition"):
        ChangeRunState.model_validate(values)
