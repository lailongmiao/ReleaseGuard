"""Dry Run predicts on local copies and never calls the executor."""

from datetime import datetime

from pydantic import ValidationError

from app.schemas.execution import Direction, ExecutionMode, ExecutionRequest, SimulationPrediction
from app.schemas.releaseguard import ChangeRunState, ChangeStatus, SimulationResult, StateInputs
from app.services.releaseguard.audit import record_audit
from app.services.releaseguard.execution_commands import audit_fields, execution_command, identity
from app.services.releaseguard.execution_preflight import ExecutionPreflight
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.execution_adapter import ExecutorAdapter, TargetObservation


class DryRunService:
    """Pure prediction after fresh per-step identity, Evidence and approval checks."""

    def __init__(self, preflight: ExecutionPreflight, adapter: ExecutorAdapter) -> None:
        """Reuse the protected preflight and synthetic inspection boundary."""
        self.preflight, self.adapter = preflight, adapter

    async def run(self, request: ExecutionRequest, *, at: datetime) -> ChangeRunState:
        """Save simulation facts; target write calls remain zero on all paths."""
        grant = await self.preflight.check(request, at=at, stage="dry_run")
        run, context = grant.run, grant.context
        if run.status is ChangeStatus.APPROVED:
            run = await self.preflight.service.domain.commit_run_update(
                execution_command(
                    run,
                    context,
                    command_id=request.command_id + ":dry-start",
                    at=at,
                    code="dry_run_started",
                    target=ChangeStatus.DRY_RUN,
                )
            )
        plan = run.execution_plan
        if plan is None or plan.manifest is None:
            raise DomainError("manifest_conflict")
        predictions: dict[str, int] = {}
        observations, audits, passed, results = [], [], True, []
        for pin in sorted((p for p in plan.manifest.steps if p.direction is Direction.FORWARD), key=lambda p: p.order):
            current = request.model_copy(update={"state_version": run.state_version, "step_id": pin.step_id})
            checked = await self.preflight.check(current, at=at, stage="dry_run")
            audits.extend(checked.authorization.audit_ids)
            if pin.resource_id not in predictions:
                try:
                    observation = TargetObservation.model_validate(
                        (await self.adapter.inspect_target(pin, at=at)).model_dump(mode="python")
                    )
                except (ValidationError, AttributeError):
                    raise DomainError("simulation_observation_invalid") from None
                if observation.resource_id != pin.resource_id or observation.observed_at != at:
                    raise DomainError("simulation_observation_invalid")
                observations.append(observation)
                predictions[pin.resource_id] = observation.value
            results.append(
                SimulationPrediction(
                    step_id=pin.step_id,
                    resource_id=pin.resource_id,
                    observed_value=predictions[pin.resource_id],
                    expected_value=checked.value.expected_value,
                    desired_value=checked.value.desired_value,
                    passed=predictions[pin.resource_id] == checked.value.expected_value,
                )
            )
            if predictions[pin.resource_id] != checked.value.expected_value:
                passed = False
                break
            predictions[pin.resource_id] = checked.value.desired_value
        event = await record_audit(
            self.preflight.service.approvals,
            context.invocation,
            "simulation_completed",
            "simulation_passed" if passed else "simulation_failed",
            at=at,
        )
        code = "simulation_passed" if passed else "simulation_failed"
        result = SimulationResult.model_validate(
            {
                **identity(run),
                "simulation_id": request.command_id + ":simulation",
                "plan_hash": plan.plan_hash,
                "passed": passed,
                "mode": ExecutionMode.SIMULATION,
                "manifest_hash": plan.manifest.manifest_hash,
                "authorization_hash": grant.authorization.authorization_hash,
                "prediction_hash": canonical_hash({"observations": observations, "predictions": predictions}),
                "predictions": results,
                "audit_ids": (*audits, event.event_id),
                "input_hash": plan.input_hash,
                "audit": audit_fields(context, at=at, code=code, input_hash=plan.input_hash),
            }
        )
        target = (
            (ChangeStatus.DRY_RUN_PASSED if plan.dry_run else ChangeStatus.EXECUTING_SANDBOX)
            if passed
            else ChangeStatus.DRY_RUN_FAILED
        )
        return await self.preflight.service.domain.commit_run_update(
            execution_command(
                run,
                context,
                command_id=request.command_id + ":dry-result",
                at=at,
                code=code,
                target=target,
                inputs=StateInputs(simulation_result=result),
            )
        )
