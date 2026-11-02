"""Synthetic domain fixtures; never contact providers, databases or executors."""

from typing import Any

from app.schemas.evidence import EvidenceAccessScope, EvidenceItem, EvidenceSourceType
from app.schemas.releaseguard import (
    ApprovalRecord,
    AuditFields,
    EvidenceAssessment,
    ExecutionPlan,
    ExecutionPlanInput,
    ExecutionEvent,
    ExecutionReceipt,
    RiskAssessment,
    RunBound,
    VerificationResult,
    build_execution_plan,
    AuditContext,
    ChangeRunState,
    ChangeStatus,
    RecoveryAuthorization,
    RunCommand,
    RunCommandInput,
    SimulationResult,
    StateInputs,
    build_run_command,
    new_change_run,
)
from app.services.releaseguard.state_machine import apply_run_command
from test_change_request_contract import NOW, request_fixture


def binding(**overrides: Any) -> dict[str, Any]:
    """Build a current-run binding to synthetic request inputs."""
    request = request_fixture()
    return {
        "change_id": request.change_id,
        "run_id": "synthetic-run",
        "revision": request.revision,
        "request_hash": request.request_hash,
        **overrides,
    }


def evidence_fixture() -> EvidenceItem:
    """Use the Phase 1 EvidenceItem directly, with an explicit synthetic source."""
    return EvidenceItem(
        evidence_id="synthetic:ci",
        source_type=EvidenceSourceType.CI,
        source_uri="fixture://synthetic/ci",
        captured_at=NOW,
        content_hash="a" * 64,
        access_scope=EvidenceAccessScope(public=True),
        snippet="Synthetic CI observation",
        confidence=1,
    )


def evidence_assessment(**overrides: Any) -> EvidenceAssessment:
    """Build a consistent sufficient assessment for the synthetic evidence."""
    return EvidenceAssessment.model_validate(
        {
            **binding(),
            "sufficient": True,
            "required_items": ["ci"],
            "present_items": ["ci"],
            "missing_items": [],
            "evidence_ids": ["synthetic:ci"],
            "policy_version": "synthetic-policy.v1",
            "evaluated_at": NOW,
            **overrides,
        }
    )


def risk_assessment(**overrides: Any) -> RiskAssessment:
    """Build a supplied risk result without running a risk scoring policy."""
    return RiskAssessment.model_validate(
        {
            **binding(),
            "level": "LOW",
            "factors": ["synthetic CI observed"],
            "blast_radius": "synthetic service",
            "evidence_ids": ["synthetic:ci"],
            "controls": ["synthetic rollback"],
            "policy_version": "synthetic-policy.v1",
            "assessed_at": NOW,
            **overrides,
        }
    )


def run_binding() -> RunBound:
    """Return the typed identity expected by an assessment context."""
    return RunBound.model_validate(binding())


def audit_fixture(**overrides: Any) -> AuditFields:
    """Build an explicit audit context for synthetic supplied records."""
    return AuditFields.model_validate(
        {
            "actor_id": "synthetic-actor",
            "actor_type": "system",
            "reason": None,
            "correlation_id": "synthetic-correlation",
            "occurred_at": NOW,
            "policy_version": "synthetic-policy.v1",
            "input_hash": "b" * 64,
            **overrides,
        }
    )


def plan_fixture(**overrides: Any) -> ExecutionPlan:
    """Describe synthetic steps without running any tool."""
    values = {
        **binding(),
        "plan_id": "synthetic-plan",
        "input_hash": "b" * 64,
        "audit": audit_fixture(),
        "steps": [
            {
                "step_id": "synthetic-step",
                "order": 1,
                "tool_name": "synthetic-tool",
                "permission": "SIMULATED",
                "arguments_hash": "c" * 64,
                "idempotency_key": "synthetic-step-key",
                "expected_effect": "synthetic effect",
                "timeout_seconds": 10,
                "compensation_step_id": "synthetic-rollback",
            }
        ],
        "rollback_steps": [
            {
                "step_id": "synthetic-rollback",
                "order": 1,
                "tool_name": "synthetic-compensation",
                "permission": "SIMULATED",
                "arguments_hash": "d" * 64,
                "idempotency_key": "synthetic-rollback-key",
                "expected_effect": "synthetic compensation",
                "timeout_seconds": 10,
            }
        ],
        "permissions": ["SIMULATED"],
        "rollback_plan": "synthetic rollback",
        "verification_plan": "synthetic independent check",
        "dry_run": False,
        **overrides,
    }
    return build_execution_plan(ExecutionPlanInput.model_validate(values))


def approval_fixture(**overrides: Any) -> ApprovalRecord:
    """Supply a synthetic decision; no authentication or approval workflow runs."""
    return ApprovalRecord.model_validate(
        {
            **binding(),
            "approval_id": "synthetic-approved",
            "plan_hash": plan_fixture().plan_hash,
            "input_hash": "b" * 64,
            "audit": audit_fixture(),
            "decision": "APPROVED",
            "approver_id": "synthetic-approver",
            "requested_at": NOW,
            "decided_at": NOW,
            **overrides,
        }
    )


def event_fixture(**overrides: Any) -> ExecutionEvent:
    """Supply a synthetic event shape instead of an actual execution result."""
    return ExecutionEvent.model_validate(
        {
            **binding(),
            "event_id": "synthetic-complete",
            "plan_hash": plan_fixture().plan_hash,
            "step_id": "synthetic-step",
            "status": "SUCCEEDED",
            "input_hash": "b" * 64,
            "occurred_at": NOW,
            "actor_id": "synthetic-actor",
            "audit": audit_fixture(),
            **overrides,
        }
    )


def verification_fixture(**overrides: Any) -> VerificationResult:
    """Provide a declared independent synthetic observation, not a live probe."""
    return VerificationResult.model_validate(
        {
            **binding(),
            "verification_id": "synthetic-verification",
            "plan_hash": plan_fixture().plan_hash,
            "input_hash": "b" * 64,
            "audit": audit_fixture(),
            "status": "PASSED",
            "verified_at": NOW,
            "checks": [
                {
                    "check_id": "synthetic-check",
                    "description": "synthetic independent observation",
                    "status": "PASSED",
                    "evidence_ids": ["synthetic:ci"],
                    "source_type": "synthetic-independent",
                    "observed_at": NOW,
                }
            ],
            "independent_source_types": ["synthetic-independent"],
            **overrides,
        }
    )


def receipt_fixture(**overrides: Any) -> ExecutionReceipt:
    """Supply a synthetic success shape; run-state tests validate its references."""
    return ExecutionReceipt.model_validate(
        {
            **binding(),
            "receipt_id": "synthetic-receipt",
            "plan_hash": plan_fixture().plan_hash,
            "input_hash": "b" * 64,
            "audit": audit_fixture(),
            "evidence_ids": ["synthetic:ci"],
            "evidence_assessment": evidence_assessment(),
            "risk_assessment": risk_assessment(),
            "approval_id": "synthetic-approved",
            "verification_id": "synthetic-verification",
            "execution_event_ids": ["synthetic-start", "synthetic-complete"],
            "final_status": "SUCCEEDED",
            "issued_at": NOW,
            **overrides,
        }
    )


def command_fixture(state: ChangeRunState, target: ChangeStatus | None = None, **overrides: Any) -> RunCommand:
    """Seal a typed transition or replacement with deterministic ids and times."""
    audit = audit_fixture().model_dump(exclude={"input_hash"})
    return build_run_command(
        RunCommandInput.model_validate(
            {
                "change_id": state.change_id,
                "run_id": state.run_id,
                "command_id": "synthetic-command-" + str(state.state_version + 1),
                "expected_revision": state.revision,
                "expected_state_version": state.state_version,
                "target_status": target,
                "audit": audit,
                **overrides,
            }
        )
    )


def edge_inputs(state: ChangeRunState, target: ChangeStatus) -> dict[str, Any]:
    """Supply explicit synthetic preconditions for a given documented edge."""
    inputs: dict[str, Any] = {}
    reason = None
    plan_hash = state.execution_plan.plan_hash if state.execution_plan else plan_fixture().plan_hash
    if target is ChangeStatus.EVIDENCE_SUFFICIENT:
        inputs.update(evidence_items=[evidence_fixture()], evidence_assessment=evidence_assessment())
    if target is ChangeStatus.WAITING_FOR_EVIDENCE:
        if state.status is ChangeStatus.COLLECTING_EVIDENCE:
            inputs.update(
                evidence_items=[],
                evidence_assessment=evidence_assessment(
                    sufficient=False, present_items=[], missing_items=["ci"], evidence_ids=[]
                ),
            )
        else:
            inputs["approval"] = approval_fixture(approval_id="synthetic-needs-changes", decision="NEEDS_CHANGES")
    if target is ChangeStatus.RISK_ASSESSED:
        inputs["risk_assessment"] = risk_assessment()
    if target is ChangeStatus.PLAN_READY:
        inputs["execution_plan"] = plan_fixture()
    if target is ChangeStatus.WAITING_FOR_APPROVAL:
        inputs["approval"] = approval_fixture(
            approval_id="synthetic-pending", decision="PENDING", approver_id=None, decided_at=None
        )
    if target is ChangeStatus.APPROVED:
        inputs["approval"] = approval_fixture()
    if target is ChangeStatus.REJECTED:
        inputs["approval"] = approval_fixture(approval_id="synthetic-rejected", decision="REJECTED")
    if target in {ChangeStatus.EXECUTING_SANDBOX, ChangeStatus.DRY_RUN_FAILED, ChangeStatus.DRY_RUN_PASSED}:
        passed = target is not ChangeStatus.DRY_RUN_FAILED
        inputs["simulation_result"] = SimulationResult.model_validate(
            {
                **binding(),
                "simulation_id": "synthetic-simulation",
                "plan_hash": plan_hash,
                "passed": passed,
                "input_hash": "b" * 64,
                "audit": audit_fixture(reason=None if passed else "synthetic simulation failed"),
            }
        )
    if target is ChangeStatus.EXECUTION_FAILED:
        inputs["execution_events"] = [
            event_fixture(event_id="synthetic-start", status="STARTED"),
            event_fixture(
                event_id="synthetic-failed",
                status="FAILED",
                reason="synthetic step failed",
                audit=audit_fixture(reason="synthetic step failed"),
            ),
        ]
    if target is ChangeStatus.EXECUTION_UNKNOWN:
        step = "synthetic-rollback" if state.status is ChangeStatus.RECOVERING else "synthetic-step"
        inputs["execution_events"] = [
            event_fixture(event_id="unknown-start", step_id=step, status="STARTED"),
            event_fixture(
                event_id="unknown-result",
                step_id=step,
                status="UNKNOWN",
                reason="synthetic_unknown",
                audit=audit_fixture(reason="synthetic_unknown"),
            ),
        ]
    if target is ChangeStatus.VERIFYING:
        rollback = state.status is ChangeStatus.RECOVERING
        inputs["execution_events"] = [
            event_fixture(
                event_id="synthetic-rollback-start" if rollback else "synthetic-start",
                step_id="synthetic-rollback" if rollback else "synthetic-step",
                status="STARTED",
            ),
            event_fixture(
                event_id="synthetic-rollback-complete" if rollback else "synthetic-complete",
                step_id="synthetic-rollback" if rollback else "synthetic-step",
            ),
        ]
    if target is ChangeStatus.SUCCEEDED:
        inputs["verification_result"] = verification_fixture()
    if target is ChangeStatus.RECOVERY_REQUIRED:
        reason = "synthetic recovery required"
        if state.status is ChangeStatus.VERIFYING:
            inputs["verification_result"] = verification_fixture(
                status="FAILED",
                checks=[
                    {
                        "check_id": "synthetic-check",
                        "description": "synthetic failed check",
                        "status": "FAILED",
                        "evidence_ids": ["synthetic:ci"],
                        "source_type": "synthetic-independent",
                        "observed_at": NOW,
                        "failure_reason": "synthetic failure",
                    }
                ],
            )
    if target is ChangeStatus.RECOVERING:
        inputs["recovery_authorization"] = RecoveryAuthorization.model_validate(
            {
                **binding(),
                "authorization_id": "synthetic-recovery-authorization",
                "plan_hash": plan_fixture().plan_hash,
                "authorized": True,
                "input_hash": "b" * 64,
                "audit": audit_fixture(),
            }
        )
    if target is ChangeStatus.FAILED:
        reason = "synthetic unrecoverable failure"
    if state.status is ChangeStatus.EXECUTION_UNKNOWN:
        unknown = state.execution_events[-1]
        succeeded = target is not ChangeStatus.FAILED
        inputs["execution_events"] = [
            event_fixture(
                event_id="synthetic-reconciled",
                step_id=unknown.step_id,
                status="SUCCEEDED" if succeeded else "FAILED",
                reason=None if succeeded else "synthetic_no_effect",
                reconciles_event_id=unknown.event_id,
                audit=audit_fixture(reason=None if succeeded else "synthetic_no_effect"),
            )
        ]
    if target is ChangeStatus.RECEIPTED:
        recovered = any(record.from_status is ChangeStatus.RECOVERING for record in state.transition_history)
        final = (
            "SIMULATED"
            if state.status is ChangeStatus.DRY_RUN_PASSED
            else "RECOVERED"
            if recovered and state.status is ChangeStatus.SUCCEEDED
            else state.status.value
        )
        inputs["receipt"] = receipt_fixture(
            final_status=final,
            plan_hash=state.execution_plan.plan_hash if state.execution_plan else None,
            approval_id=state.approval.approval_id if state.approval else None,
            verification_id=state.verification_result.verification_id if state.verification_result else None,
            execution_event_ids=[event.event_id for event in state.execution_events],
            evidence_ids=[item.evidence_id for item in state.evidence_items],
            evidence_assessment=state.evidence_assessment,
            risk_assessment=state.risk_assessment,
            audit=audit_fixture(reason=None if final in {"SUCCEEDED", "RECOVERED"} else "synthetic terminal failure"),
            mode="offline_simulation" if final == "SIMULATED" else None,
            simulation_id=state.simulation_result.simulation_id
            if final == "SIMULATED" and state.simulation_result
            else None,
        )
    # A dry-run-only fixture carries its own sealed plan through approvals.
    if "approval" in inputs:
        values = inputs["approval"].model_dump()
        values["plan_hash"] = plan_hash
        inputs["approval"] = ApprovalRecord.model_validate(values)
    return {
        "inputs": StateInputs.model_validate(inputs),
        "audit": AuditContext.model_validate(audit_fixture(reason=reason).model_dump(exclude={"input_hash"})),
    }


def advance(state: ChangeRunState, target: ChangeStatus) -> ChangeRunState:
    """Apply a documented edge using supplied synthetic results."""
    return apply_run_command(state, command_fixture(state, target, **edge_inputs(state, target)))


def state_fixture(status: ChangeStatus = ChangeStatus.RECEIVED, *, dry_run: bool = False) -> ChangeRunState:
    """Reach each source state through actual validated domain transitions."""
    if status is ChangeStatus.DRY_RUN_PASSED or dry_run:
        state = new_change_run(request_fixture(), "synthetic-run")
        for target in (
            ChangeStatus.COLLECTING_EVIDENCE,
            ChangeStatus.EVIDENCE_SUFFICIENT,
            ChangeStatus.RISK_ASSESSED,
            ChangeStatus.PLAN_READY,
            ChangeStatus.WAITING_FOR_APPROVAL,
            ChangeStatus.APPROVED,
            ChangeStatus.DRY_RUN,
            ChangeStatus.DRY_RUN_PASSED,
        ):
            args = edge_inputs(state, target)
            if target is ChangeStatus.PLAN_READY:
                args["inputs"] = StateInputs(execution_plan=plan_fixture(dry_run=True))
            state = apply_run_command(state, command_fixture(state, target, **args))
            if target is status:
                return state
        return state
    paths = {
        ChangeStatus.WAITING_FOR_EVIDENCE: [ChangeStatus.COLLECTING_EVIDENCE, ChangeStatus.WAITING_FOR_EVIDENCE],
        ChangeStatus.DRY_RUN_FAILED: [ChangeStatus.DRY_RUN_FAILED],
        ChangeStatus.EXECUTION_FAILED: [ChangeStatus.EXECUTION_FAILED],
        ChangeStatus.EXECUTION_UNKNOWN: [ChangeStatus.EXECUTION_UNKNOWN],
        ChangeStatus.RECOVERY_REQUIRED: [ChangeStatus.EXECUTION_FAILED, ChangeStatus.RECOVERY_REQUIRED],
        ChangeStatus.RECOVERING: [
            ChangeStatus.EXECUTION_FAILED,
            ChangeStatus.RECOVERY_REQUIRED,
            ChangeStatus.RECOVERING,
        ],
        ChangeStatus.REJECTED: [ChangeStatus.REJECTED],
        ChangeStatus.FAILED: [ChangeStatus.FAILED],
    }
    state = new_change_run(request_fixture(), "synthetic-run")
    if status is ChangeStatus.RECEIVED:
        return state
    main = [
        ChangeStatus.COLLECTING_EVIDENCE,
        ChangeStatus.EVIDENCE_SUFFICIENT,
        ChangeStatus.RISK_ASSESSED,
        ChangeStatus.PLAN_READY,
        ChangeStatus.WAITING_FOR_APPROVAL,
        ChangeStatus.APPROVED,
        ChangeStatus.DRY_RUN,
        ChangeStatus.EXECUTING_SANDBOX,
        ChangeStatus.VERIFYING,
        ChangeStatus.SUCCEEDED,
        ChangeStatus.RECEIPTED,
    ]
    branch_from = {
        ChangeStatus.DRY_RUN_FAILED: ChangeStatus.DRY_RUN,
        ChangeStatus.EXECUTION_FAILED: ChangeStatus.EXECUTING_SANDBOX,
        ChangeStatus.EXECUTION_UNKNOWN: ChangeStatus.EXECUTING_SANDBOX,
        ChangeStatus.RECOVERY_REQUIRED: ChangeStatus.EXECUTING_SANDBOX,
        ChangeStatus.RECOVERING: ChangeStatus.EXECUTING_SANDBOX,
        ChangeStatus.REJECTED: ChangeStatus.WAITING_FOR_APPROVAL,
    }
    if status in paths:
        if status in branch_from:
            for target in main[: main.index(branch_from[status]) + 1]:
                state = advance(state, target)
        for target in paths[status]:
            state = advance(state, target)
        return state
    for target in main[: main.index(status) + 1]:
        state = advance(state, target)
    return state
