"""Service-only domain commands with explicit time, identity and CAS."""

from datetime import datetime

from app.schemas.authorization import RequestContext
from app.schemas.releaseguard import (
    ActorType,
    AuditContext,
    AuditFields,
    ChangeRunState,
    ChangeStatus,
    CommandAction,
    RunBound,
    RunCommand,
    RunCommandInput,
    StateInputs,
    build_run_command,
)


def identity(run: ChangeRunState) -> dict[str, str | int]:
    """Extract only domain identity fields."""
    return run.model_dump(include=set(RunBound.model_fields))


def audit_fields(context: RequestContext, *, at: datetime, code: str, input_hash: str) -> AuditFields:
    """Attribute a safe immutable domain fact to a freshly resolved actor."""
    return AuditFields(
        actor_id=context.principal.principal_id,
        actor_type=ActorType.USER,
        reason=code,
        correlation_id=context.invocation.correlation_id,
        occurred_at=at,
        policy_version=context.invocation.policy_version,
        input_hash=input_hash,
    )


def execution_command(
    run: ChangeRunState,
    context: RequestContext,
    *,
    command_id: str,
    at: datetime,
    code: str,
    target: ChangeStatus | None = None,
    inputs: StateInputs | None = None,
) -> RunCommand:
    """Build a hashed append or real state edge; never fake a transition self-loop."""
    return build_run_command(
        RunCommandInput(
            change_id=run.change_id,
            run_id=run.run_id,
            expected_revision=run.revision,
            expected_state_version=run.state_version,
            command_id=command_id,
            action=CommandAction.TRANSITION if target is not None else CommandAction.APPEND_EXECUTION_FACTS,
            target_status=target,
            inputs=inputs or StateInputs(),
            audit=AuditContext.model_validate(
                audit_fields(context, at=at, code=code, input_hash=context.invocation.input_hash).model_dump(
                    exclude={"input_hash"}
                )
            ),
        )
    )
