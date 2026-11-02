"""Explicitly synthetic, genuinely approved execution integration fixtures."""

from datetime import datetime, timedelta

from app.repositories.releaseguard import RunLookup
from app.schemas.approval import ApprovalRequest
from app.schemas.authorization import RequestContext
from app.schemas.execution import (
    Direction,
    ExecutionManifest,
    ExecutionMode,
    ExecutionRequest,
    ManifestStep,
    SandboxInput,
)
from app.schemas.releaseguard import CommandAction
from app.schemas.tooling import ToolDescriptor, ToolInvocation
from app.services.releaseguard.approval import ApprovalService, approval_domain_command
from app.services.releaseguard.authorization_context import load_authorization_binding
from app.services.releaseguard.normalization import canonical_hash
from tests.releaseguard_fixtures import command_fixture, plan_fixture
from tests.test_approval_contract import approval_setup, approver_context, create_waiting, decision_command
from tests.test_authorization import policy_fixture, rule_fixture
from tests.test_change_request_contract import NOW


class TrustedResolver:
    """A test-only authenticated session, deliberately outside Graph state."""

    def __init__(self, context: RequestContext) -> None:
        """Retain server identity and permit tests to revoke that identity."""
        self.context = context

    async def resolve(self, call: ToolInvocation, *, at: datetime) -> RequestContext:
        """Never take identity or tenant from the execution request."""
        return self.context.model_copy(update={"invocation": call})


async def request_for(
    service: ApprovalService,
    *,
    step: str = "step-0",
    direction: Direction = Direction.FORWARD,
    approval: str = "synthetic-approval",
    command: str = "execute",
) -> ExecutionRequest:
    """Read current CAS bindings to submit one exact intent."""
    run = await service.domain.get_run(RunLookup(run_id="synthetic-run"))
    assert run.execution_plan is not None and run.execution_plan.manifest is not None
    entry = await service.approvals.get(approval)
    return ExecutionRequest(
        command_id=command,
        change_id=run.change_id,
        run_id=run.run_id,
        revision=run.revision,
        request_hash=run.request_hash,
        thread_id=entry.request.binding.invocation.thread_id,
        state_version=run.state_version,
        plan_hash=run.execution_plan.plan_hash,
        manifest_hash=run.execution_plan.manifest.manifest_hash,
        mode=run.execution_plan.manifest.mode,
        approval_request_id=approval,
        step_id=step,
        direction=direction,
    )


def sandbox_descriptor() -> ToolDescriptor:
    """A pinned finite integer operation, not shell or an external tool."""
    return ToolDescriptor(
        tool_name="sandbox-register",
        version="v1",
        capabilities={"mutating"},
        resource_types={"synthetic_register"},
        action_types={"compare_set"},
        input_schema="sandbox-register.v1",
        output_schema="sandbox-result.v1",
        side_effect_level="mutating",
        timeout_ms=1000,
        idempotency_required=True,
        adapter_kind="internal",
    )


def manifest_fixture(*, simulation: bool = False, count: int = 2) -> ExecutionManifest:
    """Freeze forward and inverse steps over a single synthetic register."""
    descriptor = sandbox_descriptor()
    steps = []
    for i in range(count):
        steps.append(
            ManifestStep(
                step_id=f"step-{i}",
                order=i + 1,
                tool_name=descriptor.tool_name,
                version=descriptor.version,
                descriptor_hash=descriptor.descriptor_hash,
                input_hash=canonical_hash(SandboxInput(expected_value=i, desired_value=i + 1)),
                input_ref=f"input-{i}",
                resource_id="synthetic-resource",
                environment="staging",
                expected_value=i,
                desired_value=i + 1,
                dependencies=(f"step-{i - 1}",) if i else (),
            )
        )
    for index, i in enumerate(reversed(range(count))):
        steps.append(
            ManifestStep(
                step_id=f"undo-{i}",
                order=index + 1,
                direction=Direction.COMPENSATION,
                tool_name=descriptor.tool_name,
                version=descriptor.version,
                descriptor_hash=descriptor.descriptor_hash,
                input_hash=canonical_hash(SandboxInput(expected_value=i + 1, desired_value=i)),
                input_ref=f"undo-input-{i}",
                resource_id="synthetic-resource",
                environment="staging",
                expected_value=i + 1,
                desired_value=i,
                compensates=f"step-{i}",
                dependencies=(f"undo-{i + 1}",) if i < count - 1 else (),
            )
        )
    return ExecutionManifest(
        mode=ExecutionMode.SIMULATION if simulation else ExecutionMode.SANDBOX,
        steps=tuple(steps),
        policy_version="synthetic-policy.v1",
    )


async def execution_setup(
    *, simulation: bool = False, count: int = 2
) -> tuple[ApprovalService, RequestContext, ToolDescriptor]:
    """Reuse Phase 5 Gate/Citation fixtures, replace the plan, then approve it."""
    service, _, old_context, _ = await approval_setup()
    descriptor, manifest = sandbox_descriptor(), manifest_fixture(simulation=simulation, count=count)
    run = await service.domain.get_run(RunLookup(run_id="synthetic-run"))
    forward, rollback = [], []
    for pin in manifest.steps:
        step = dict(
            step_id=pin.step_id,
            order=pin.order,
            tool_name=pin.tool_name,
            permission="MUTATING",
            arguments_hash=pin.input_hash,
            idempotency_key=f"key-{pin.step_id}",
            expected_effect="synthetic register transition",
            timeout_seconds=1,
        )
        if pin.direction is Direction.FORWARD:
            step["compensation_step_id"] = f"undo-{pin.order - 1}"
            forward.append(step)
        else:
            rollback.append(step)
    plan = plan_fixture(
        plan_id="execution-plan",
        steps=forward,
        rollback_steps=rollback,
        permissions=["MUTATING"],
        dry_run=simulation,
        manifest=manifest,
    )
    run = await service.domain.commit_run_update(
        command_fixture(run, action=CommandAction.REPLACE_PLAN, replacement_plan=plan)
    )
    pin = manifest.steps[0]
    call = old_context.invocation.model_copy(
        update=dict(
            tool_name=pin.tool_name,
            version=pin.version,
            descriptor_hash=pin.descriptor_hash,
            input_schema=pin.input_schema,
            input_hash=pin.input_hash,
            resource_type=pin.resource_type,
            resource_id=pin.resource_id,
            action=pin.action,
        )
    )
    context = old_context.model_copy(update={"invocation": call})
    service.permission_policy = policy_fixture(
        rules=(
            rule_fixture(
                resource_type="synthetic_register",
                resource_ids={"synthetic-resource"},
                actions={"compare_set"},
                environments={"staging"},
                side_effects={"mutating"},
                effect="approval_required",
            ),
            rule_fixture(
                rule_id="inspect-rule",
                resource_type="synthetic_register",
                resource_ids={"synthetic-resource"},
                actions={"inspect"},
                environments={"staging"},
                side_effects={"read_only"},
                effect="allow",
            ),
        )
    )
    _, binding = await load_authorization_binding(
        service.domain,
        service.evidence,
        context,
        descriptor,
        service.permission_policy,
        service.evidence_policy,
        at=NOW,
        scope="plan",
    )
    request = ApprovalRequest(
        approval_request_id="synthetic-approval",
        idempotency_key="plan-key",
        binding=binding,
        authorization_hash=canonical_hash(binding),
        expected_state_version=run.state_version,
        requested_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
        reason_code="plan_review",
    )
    await create_waiting(service, request, context, descriptor)
    entry = await service.decide(decision_command(request), approver_context(context), descriptor, at=NOW)
    assert entry.decision is not None
    run = await service.domain.get_run(RunLookup(run_id=run.run_id))
    await service.domain.commit_run_update(approval_domain_command(run, entry.decision, command_id="approved-plan"))
    return service, context, descriptor
