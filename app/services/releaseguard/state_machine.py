"""Pure deterministic domain transitions; no Graph, tools, network or clock reads."""

from typing import Any

import structlog
from pydantic import ValidationError

from app.schemas.releaseguard import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    REVISION_STATES,
    PLAN_REPLACEMENT_STATES,
    ApprovalDecision,
    ApprovalInvalidation,
    ChangeRunState,
    ChangeStatus,
    CommandAction,
    DomainRecord,
    EventStatus,
    ExecutionEvent,
    ExecutionPlan,
    RunCommand,
    TransitionRecord,
    VerificationStatus,
    approval_valid_at,
    matches_binding,
    record_key,
)
from app.services.releaseguard.normalization import canonical_hash


logger = structlog.get_logger(__name__)


class DomainError(ValueError):
    """Stable, safe error category that never embeds raw command input."""

    def __init__(self, code: str) -> None:
        """Retain only a machine-readable error code."""
        self.code = code
        super().__init__(code)


def legal_targets(status: ChangeStatus) -> frozenset[ChangeStatus]:
    """Include the explicitly documented unrecoverable-failure edge."""
    return ALLOWED_TRANSITIONS[status] | (
        frozenset({ChangeStatus.FAILED})
        if status not in TERMINAL_STATES and status is not ChangeStatus.EXECUTION_UNKNOWN
        else frozenset()
    )


def apply_run_command(state: ChangeRunState, command: RunCommand) -> ChangeRunState:
    """Return a validated new aggregate, leaving the caller's state untouched."""
    state = ChangeRunState.model_validate_json(state.model_dump_json())
    command = RunCommand.model_validate_json(command.model_dump_json())
    if (command.change_id, command.run_id) != (state.change_id, state.run_id):
        raise DomainError("identity_conflict")
    for record in state.transition_history + state.fact_history:
        if record.command_id == command.command_id:
            if record.input_hash != command.audit.input_hash:
                raise DomainError("command_conflict")
            return state
    if command.expected_revision != state.revision:
        raise DomainError("revision_conflict")
    if command.expected_state_version != state.state_version:
        raise DomainError("version_conflict")
    last_time = max(
        [item.audit.occurred_at for item in state.transition_history + state.fact_history],
        default=state.change_request.updated_at,
    )
    if command.audit.occurred_at < last_time:
        raise DomainError("transition_precondition_failed")
    values = state.model_dump(mode="python")
    if command.action is CommandAction.REVISE:
        _revise(state, command, values)
    elif command.action is CommandAction.REPLACE_PLAN:
        _replace_plan(state, command, values)
    elif command.action is CommandAction.APPEND_EXECUTION_FACTS:
        if state.status not in {
            ChangeStatus.DRY_RUN,
            ChangeStatus.EXECUTING_SANDBOX,
            ChangeStatus.EXECUTION_UNKNOWN,
            ChangeStatus.RECOVERING,
            ChangeStatus.VERIFYING,
            ChangeStatus.RECOVERY_REQUIRED,
        }:
            raise DomainError("execution_fact_status_blocked")
        allowed_facts = {
            ChangeStatus.DRY_RUN: {"simulation_result"},
            ChangeStatus.EXECUTING_SANDBOX: {"execution_events"},
            ChangeStatus.EXECUTION_UNKNOWN: {"execution_events", "receipt"},
            ChangeStatus.RECOVERING: {"execution_events"},
            ChangeStatus.VERIFYING: {"verification_result", "verification_evidence"},
            ChangeStatus.RECOVERY_REQUIRED: {"recovery_approval", "execution_events"},
        }[state.status]
        if any(
            getattr(command.inputs, field) not in (None, []) and field not in allowed_facts
            for field in type(command.inputs).model_fields
        ):
            raise DomainError("execution_fact_scope_denied")
        _merge_inputs(state, command, values)
    else:
        _transition(state, command, values)
    values["state_version"] = state.state_version + 1
    record = TransitionRecord(
        change_id=state.change_id,
        run_id=state.run_id,
        revision=values["revision"],
        request_hash=values["request_hash"],
        transition_id=canonical_hash({"run_id": state.run_id, "command_id": command.command_id}),
        command_id=command.command_id,
        action=command.action,
        from_status=state.status,
        to_status=values["status"],
        from_revision=state.revision,
        plan_hash=values["execution_plan"].plan_hash
        if isinstance(values["execution_plan"], ExecutionPlan)
        else values["execution_plan"]["plan_hash"]
        if values["execution_plan"]
        else None,
        state_version=values["state_version"],
        input_hash=command.audit.input_hash,
        audit=command.audit,
    )
    values["fact_history" if command.action is CommandAction.APPEND_EXECUTION_FACTS else "transition_history"].append(
        record
    )
    try:
        result = ChangeRunState.model_validate(values)
    except ValidationError:
        raise DomainError("transition_precondition_failed") from None
    logger.info(
        "change_state_transitioned",
        from_status=state.status.value,
        to_status=result.status.value,
        revision=result.revision,
        state_version=result.state_version,
        input_hash=command.audit.input_hash,
    )
    return result


def _append_record(values: dict[str, Any], record: DomainRecord) -> None:
    """Append a new fact or accept byte-equivalent replay, never overwrite it."""
    existing = values["record_history"]
    key = record_key(record)
    for item in existing:
        item_key = record_key(item) if not isinstance(item, dict) else _dict_record_key(item)
        if key == item_key:
            if canonical_hash(item) != canonical_hash(record):
                raise DomainError("event_conflict" if record.record_type == "event" else "record_conflict")
            return
    existing.append(record)


def _dict_record_key(value: dict[str, Any]) -> str:
    """Read a serialized record key without copying or altering its payload."""
    kind = value["record_type"]
    ids = {
        "plan": "plan_id",
        "approval": "approval_id",
        "event": "event_id",
        "verification": "verification_id",
        "receipt": "receipt_id",
        "approval_invalidation": "invalidation_id",
        "simulation": "simulation_id",
        "recovery_authorization": "authorization_id",
    }
    return kind + ":" + (value[ids[kind]] if kind in ids else canonical_hash(value))


def _merge_inputs(state: ChangeRunState, command: RunCommand, values: dict[str, Any]) -> None:
    """Merge supplied typed results with immutable record and evidence history."""
    inputs = command.inputs
    if inputs.execution_events and state.receipt is not None and not state.receipt.terminal:
        # Keep the diagnostic in immutable history while publishing newer facts.
        values["receipt"] = None
    for item in inputs.verification_evidence:
        old = next((record for record in state.evidence_history if record.evidence_id == item.evidence_id), None)
        if old is not None and old != item:
            raise DomainError("record_conflict")
        if old is None:
            values["evidence_history"].append(item)
    if inputs.recovery_approval is not None:
        if inputs.recovery_approval.scope != "recovery" or not matches_binding(inputs.recovery_approval, state):
            raise DomainError("recovery_authorization_required")
        _append_record(values, inputs.recovery_approval)
    if inputs.evidence_items is not None:
        values["evidence_items"] = inputs.evidence_items
        evidence = {
            item["evidence_id"] if isinstance(item, dict) else item.evidence_id: item
            for item in values["evidence_history"]
        }
        for item in inputs.evidence_items:
            if item.evidence_id in evidence and canonical_hash(evidence[item.evidence_id]) != canonical_hash(item):
                raise DomainError("record_conflict")
            if item.evidence_id not in evidence:
                values["evidence_history"].append(item)
                evidence[item.evidence_id] = item
    for field in (
        "evidence_assessment",
        "risk_assessment",
        "execution_plan",
        "approval",
        "verification_result",
        "receipt",
        "simulation_result",
        "recovery_authorization",
    ):
        item = getattr(inputs, field)
        if item is None:
            continue
        if not matches_binding(item, state):
            raise DomainError("transition_precondition_failed")
        if field == "execution_plan" and state.execution_plan is not None and item != state.execution_plan:
            raise DomainError("plan_change_conflict")
        at = (
            item.audit.occurred_at
            if hasattr(item, "audit")
            else getattr(item, "evaluated_at", getattr(item, "assessed_at", None))
        )
        if at is not None and at > command.audit.occurred_at:
            raise DomainError("transition_precondition_failed")
        _append_record(values, item)
        values[field] = item
    for event in inputs.execution_events:
        if not matches_binding(event, state) or event.occurred_at > command.audit.occurred_at:
            raise DomainError("transition_precondition_failed")
        _append_record(values, event)
        if not any(
            (item.event_id if not isinstance(item, dict) else item["event_id"]) == event.event_id
            for item in values["execution_events"]
        ):
            values["execution_events"].append(event)


def _transition(state: ChangeRunState, command: RunCommand, values: dict[str, Any]) -> None:
    """Apply an explicit legal edge and enforce its result preconditions."""
    target = command.target_status
    if target is None or target not in legal_targets(state.status):
        raise DomainError("invalid_state_transition")
    allowed_inputs = {
        ChangeStatus.COLLECTING_EVIDENCE: {"evidence_items"},
        ChangeStatus.WAITING_FOR_EVIDENCE: {"evidence_items", "evidence_assessment", "approval"},
        ChangeStatus.EVIDENCE_SUFFICIENT: {"evidence_items", "evidence_assessment"},
        ChangeStatus.RISK_ASSESSED: {"risk_assessment"},
        ChangeStatus.PLAN_READY: {"execution_plan"},
        ChangeStatus.WAITING_FOR_APPROVAL: {"approval"},
        ChangeStatus.APPROVED: {"approval"},
        ChangeStatus.REJECTED: {"approval"},
        ChangeStatus.DRY_RUN: set(),
        ChangeStatus.DRY_RUN_FAILED: {"simulation_result"},
        ChangeStatus.DRY_RUN_PASSED: {"simulation_result"},
        ChangeStatus.EXECUTING_SANDBOX: {"simulation_result"},
        ChangeStatus.EXECUTION_FAILED: {"execution_events"},
        ChangeStatus.EXECUTION_UNKNOWN: {"execution_events"},
        ChangeStatus.VERIFYING: {"execution_events", "verification_result"},
        ChangeStatus.SUCCEEDED: {"verification_result"},
        ChangeStatus.RECOVERY_REQUIRED: {"verification_result", "execution_events"},
        ChangeStatus.RECOVERING: {"recovery_authorization"},
        ChangeStatus.FAILED: {"execution_events", "verification_result"},
        ChangeStatus.RECEIPTED: {"receipt"},
    }[target]
    if any(
        getattr(command.inputs, field) not in (None, []) and field not in allowed_inputs
        for field in type(command.inputs).model_fields
    ):
        raise DomainError("transition_precondition_failed")
    _merge_inputs(state, command, values)
    values["status"] = target
    assessment = command.inputs.evidence_assessment or state.evidence_assessment
    approval = command.inputs.approval or state.approval
    plan = command.inputs.execution_plan or state.execution_plan
    simulation = command.inputs.simulation_result or state.simulation_result
    verification = command.inputs.verification_result or state.verification_result
    if target is ChangeStatus.FAILED:
        if not command.audit.reason:
            raise DomainError("transition_precondition_failed")
        values["errors"].append(command.audit.reason)
    if state.status is ChangeStatus.COLLECTING_EVIDENCE and target is ChangeStatus.WAITING_FOR_EVIDENCE:
        if assessment is None or assessment.sufficient:
            raise DomainError("transition_precondition_failed")
    if target is ChangeStatus.WAITING_FOR_APPROVAL and (
        approval is None or approval.decision is not ApprovalDecision.PENDING
    ):
        raise DomainError("transition_precondition_failed")
    if state.status is ChangeStatus.WAITING_FOR_APPROVAL and target in {
        ChangeStatus.APPROVED,
        ChangeStatus.REJECTED,
        ChangeStatus.WAITING_FOR_EVIDENCE,
    }:
        expected = {
            ChangeStatus.APPROVED: ApprovalDecision.APPROVED,
            ChangeStatus.REJECTED: ApprovalDecision.REJECTED,
            ChangeStatus.WAITING_FOR_EVIDENCE: ApprovalDecision.NEEDS_CHANGES,
        }[target]
        if (
            approval is None
            or approval.decision is not expected
            or approval.decided_at is None
            or approval.decided_at > command.audit.occurred_at
            or approval.expires_at is not None
            and command.audit.occurred_at >= approval.expires_at
        ):
            raise DomainError("transition_precondition_failed")
        if target is ChangeStatus.WAITING_FOR_EVIDENCE:
            for field in (
                "evidence_assessment",
                "risk_assessment",
                "execution_plan",
                "approval",
                "verification_result",
                "simulation_result",
                "recovery_authorization",
            ):
                values[field] = None
    if target in {ChangeStatus.APPROVED, ChangeStatus.DRY_RUN, ChangeStatus.EXECUTING_SANDBOX}:
        if approval is None or not approval_valid_at(approval, command.audit.occurred_at):
            raise DomainError("transition_precondition_failed")
    if state.status is ChangeStatus.DRY_RUN and target in {
        ChangeStatus.EXECUTING_SANDBOX,
        ChangeStatus.DRY_RUN_FAILED,
        ChangeStatus.DRY_RUN_PASSED,
    }:
        if simulation is None or simulation.passed != (target is not ChangeStatus.DRY_RUN_FAILED):
            raise DomainError("transition_precondition_failed")
        if target is ChangeStatus.EXECUTING_SANDBOX:
            if plan is None or plan.dry_run or approval is None:
                raise DomainError("transition_precondition_failed")
            values["execution_started_at"] = command.audit.occurred_at
            values["execution_approval_id"] = approval.approval_id
        if target is ChangeStatus.DRY_RUN_PASSED and (plan is None or not plan.dry_run):
            raise DomainError("transition_precondition_failed")
    events = [
        item if not isinstance(item, dict) else ExecutionEvent.model_validate(item)
        for item in values["execution_events"]
    ]
    if target is ChangeStatus.EXECUTION_FAILED and not any(event.status is EventStatus.FAILED for event in events):
        raise DomainError("transition_precondition_failed")
    if target is ChangeStatus.EXECUTION_UNKNOWN and not any(event.status is EventStatus.UNKNOWN for event in events):
        raise DomainError("transition_precondition_failed")
    if state.status is ChangeStatus.EXECUTION_UNKNOWN:
        latest_events = {event.step_id: event.status for event in events}
        if (
            EventStatus.UNKNOWN in latest_events.values()
            or target is ChangeStatus.FAILED
            and EventStatus.SUCCEEDED in latest_events.values()
        ):
            raise DomainError("execution_outcome_unknown")
        values["receipt"] = None
    if target is ChangeStatus.RECOVERY_REQUIRED:
        if plan is None or not plan.rollback_steps or not command.audit.reason:
            raise DomainError("transition_precondition_failed")
        if state.status is ChangeStatus.VERIFYING and (
            verification is None or verification.status is VerificationStatus.PASSED
        ):
            raise DomainError("transition_precondition_failed")
    if target is ChangeStatus.RECOVERING:
        authorization = command.inputs.recovery_authorization or state.recovery_authorization
        if authorization is None or not authorization.authorized:
            raise DomainError("transition_precondition_failed")
        if (
            plan
            and plan.manifest
            and not (
                authorization.approval_id
                and authorization.authorization_hash
                and authorization.effect_ids
                and authorization.compensation_step_ids
            )
        ):
            raise DomainError("recovery_authorization_required")
    if target is ChangeStatus.VERIFYING:
        selected = (
            plan.rollback_steps if plan and state.recovery_authorization is not None else plan.steps if plan else []
        )
        if plan and plan.manifest and state.recovery_authorization is not None:
            selected_ids = state.recovery_authorization.compensation_step_ids if state.recovery_authorization else ()
            selected = [step for step in selected if step.step_id in selected_ids]
        latest = {event.step_id: event.status for event in events}
        if not selected or any(latest.get(step.step_id) is not EventStatus.SUCCEEDED for step in selected):
            raise DomainError("transition_precondition_failed")
    if target is ChangeStatus.SUCCEEDED and (
        verification is None or verification.status is not VerificationStatus.PASSED
    ):
        raise DomainError("transition_precondition_failed")


def _invalidate_approval(state: ChangeRunState, command: RunCommand, values: dict[str, Any]) -> None:
    """Append invalidation without modifying the original approval decision."""
    approval = state.approval
    if approval is None:
        return
    expired = approval.expires_at is not None and command.audit.occurred_at >= approval.expires_at
    invalidation = ApprovalInvalidation(
        **{
            field: getattr(approval, field)
            for field in ("change_id", "run_id", "revision", "request_hash", "plan_hash")
        },
        invalidation_id=canonical_hash({"command_id": command.command_id, "approval_id": approval.approval_id}),
        approval_id=approval.approval_id,
        decision=ApprovalDecision.EXPIRED if expired else ApprovalDecision.INVALIDATED,
        input_hash=command.audit.input_hash,
        audit=command.audit,
    )
    _append_record(values, invalidation)


def _revise(state: ChangeRunState, command: RunCommand, values: dict[str, Any]) -> None:
    """Reset current results for a new revision while preserving every old fact."""
    request = command.replacement_request
    if (
        state.status not in REVISION_STATES
        or request is None
        or request.change_id != state.change_id
        or request.revision != state.revision + 1
    ):
        raise DomainError("revision_conflict")
    old = state.change_request
    if (
        (request.requester_id, request.idempotency_key, request.created_at)
        != (old.requester_id, old.idempotency_key, old.created_at)
        or request.updated_at < old.updated_at
        or request.updated_at > command.audit.occurred_at
    ):
        raise DomainError("revision_conflict")
    _invalidate_approval(state, command, values)
    values.update(
        change_request=request,
        revision=request.revision,
        request_hash=request.request_hash,
        status=ChangeStatus.RECEIVED,
    )
    values["request_history"].append(request)
    values["evidence_items"] = []
    values["execution_events"] = []
    for field in (
        "evidence_assessment",
        "risk_assessment",
        "execution_plan",
        "approval",
        "verification_result",
        "receipt",
        "simulation_result",
        "recovery_authorization",
        "execution_started_at",
        "execution_approval_id",
    ):
        values[field] = None


def _replace_plan(state: ChangeRunState, command: RunCommand, values: dict[str, Any]) -> None:
    """Require another approval cycle even for an explicit equivalent replacement."""
    plan = command.replacement_plan
    if (
        state.status not in PLAN_REPLACEMENT_STATES
        or plan is None
        or not matches_binding(plan, state)
        or plan.audit.occurred_at > command.audit.occurred_at
    ):
        raise DomainError("plan_change_conflict")
    _invalidate_approval(state, command, values)
    _append_record(values, plan)
    values.update(
        status=ChangeStatus.PLAN_READY,
        execution_plan=plan,
        approval=None,
        simulation_result=None,
        recovery_authorization=None,
        verification_result=None,
    )
