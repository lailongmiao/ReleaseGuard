"""Eval-owned synthetic fixtures composed from public domain services."""

from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.core.langgraph.graph import build_agent_graph
from app.core.langgraph.releaseguard_execution import ExecutionRuntime
from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config, releaseguard_thread_id
from app.repositories.in_memory_approval import InMemoryApprovalRepository
from app.repositories.in_memory_evidence import InMemoryEvidenceRepository
from app.repositories.in_memory_execution import InMemoryExecutionRepository
from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.repositories.releaseguard import RunLookup
from app.schemas.approval import ApprovalCommand, ApprovalRequest
from app.schemas.authorization import (
    PermissionEffect,
    PermissionPolicy,
    PermissionRule,
    RequestContext,
    VerifiedPrincipal,
)
from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceCitation,
    EvidenceItem,
    EvidencePolicy,
    EvidenceRequirement,
    EvidenceSnapshot,
    EvidenceSourceType,
)
from app.schemas.execution import (
    Direction,
    ExecutionIntent,
    ExecutionManifest,
    ExecutionMode,
    ExecutionRequest,
    ManifestStep,
    SandboxInput,
)
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import (
    AuditFields,
    AuditContext,
    ActorType,
    Environment,
    ChangeWindow,
    ChangeRunState,
    ChangeIdentity,
    ChangeRequestInput,
    ChangeStatus,
    CreateChangeRequest,
    EvidenceAssessment,
    ExecutionPlanInput,
    RiskAssessment,
    RunCommandInput,
    StateInputs,
    build_change_request,
    build_execution_plan,
    build_run_command,
    new_change_run,
)
from app.schemas.tooling import SideEffect, ToolDescriptor, ToolInvocation
from app.services.releaseguard.approval import ApprovalService, approval_domain_command
from app.services.releaseguard.authorization_context import load_authorization_binding
from app.services.releaseguard.execution import ExecutionService
from app.services.releaseguard.execution_preflight import ExecutionPreflight
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.verification import VerificationService
from app.services.tools.sandbox_executor import SandboxExecutor, SyntheticTargetStore
from app.services.tools.verification_adapter import IndependentRegisterReader

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class TrustedResolver:
    """Synthetic identity lives outside checkpoint; it is never user-supplied."""

    def __init__(self, context: RequestContext) -> None:
        """Retain only isolated identity state."""
        self.context = context

    async def resolve(self, call: ToolInvocation, *, at: datetime) -> RequestContext:
        """Resolve the invocation from the controlled identity."""
        return self.context.model_copy(update={"invocation": call})


class Fault:
    """One shot, deterministic crash-window injection."""

    def __init__(self, stage: str) -> None:
        """Arm a single known boundary."""
        self.stage = stage

    async def hit(self, stage: str) -> None:
        """Crash once, without timing races or sensitive exceptions."""
        if self.stage == stage:
            self.stage = "done"
            raise RuntimeError("synthetic_fault")


def principal(*, approver: bool = False) -> VerifiedPrincipal:
    """Create a synthetic, verified, tenant-bound identity."""
    return VerifiedPrincipal(
        principal_id="synthetic-approver" if approver else "synthetic-requester",
        session_id="approver-session" if approver else "synthetic-session",
        tenant_id="synthetic-tenant",
        roles=frozenset({"requester", "approver"}),
        authenticated_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )


async def passthrough(state: GraphState) -> dict[str, Any]:
    """Keep Chat offline and independent from execution runtime."""
    return {"messages": state.messages}


class Fixture:
    """Fresh stores and a finite target per case and attempt."""

    def __init__(self) -> None:
        """Initialize private, single-process synthetic stores."""
        self.domain = InMemoryReleaseGuardRepository()
        self.evidence = InMemoryEvidenceRepository()
        self.approvals = InMemoryApprovalRepository()
        self.ledger = InMemoryExecutionRepository()
        self.target = SyntheticTargetStore({"synthetic-resource": 0})
        self.at = NOW

    async def run(self) -> ChangeRunState:
        """Read authority instead of graph summary."""
        return await self.domain.get_run(RunLookup(run_id="synthetic-run"))

    async def request(
        self,
        *,
        command: str = "execute",
        step: str = "step-0",
        approval: str = "synthetic-approval",
        direction: Direction = Direction.FORWARD,
    ) -> ExecutionRequest:
        """Bind each command to the current CAS and exact approved intent."""
        run = await self.run()
        entry = await self.approvals.get(approval)
        plan = run.execution_plan
        assert plan is not None and plan.manifest is not None
        return ExecutionRequest(
            command_id=command,
            change_id=run.change_id,
            run_id=run.run_id,
            revision=run.revision,
            request_hash=run.request_hash,
            thread_id=entry.request.binding.invocation.thread_id,
            state_version=run.state_version,
            plan_hash=plan.plan_hash,
            manifest_hash=plan.manifest.manifest_hash,
            mode=plan.manifest.mode,
            approval_request_id=approval,
            step_id=step,
            direction=direction,
        )

    async def decide(
        self, decision: str = "APPROVED", *, own: bool = False, review: ApprovalRequest | None = None
    ) -> None:
        """Decide via real approval service and domain CAS."""
        request = review or self.review
        actor = principal(approver=not own)
        context = self.context.model_copy(
            update={
                "principal": actor,
                "invocation": request.binding.invocation.model_copy(
                    update={"principal_id": actor.principal_id, "session_id": actor.session_id}
                ),
            }
        )
        entry = await self.service.decide(
            ApprovalCommand.model_validate(
                dict(
                    command_id=request.approval_request_id + ":decision",
                    approval_request_id=request.approval_request_id,
                    authorization_hash=request.authorization_hash,
                    expected_state_version=request.expected_state_version + (0 if review else 1),
                    decision=decision,
                    reason_code="synthetic_review",
                )
            ),
            context,
            self.descriptor,
            at=self.at,
        )
        if review is None:
            assert entry.decision is not None
            await self.domain.commit_run_update(
                approval_domain_command(await self.run(), entry.decision, command_id="approved-plan")
            )

    async def initialize(
        self, *, level: str = "LOW", simulation: bool = False, count: int = 1, decide: bool = True
    ) -> "Fixture":
        """Build genuine Evidence, risk, plan, pending and approval transitions."""
        request = build_change_request(
            CreateChangeRequest(
                identity=ChangeIdentity(change_id="synthetic-change", revision=1),
                body=ChangeRequestInput(
                    requester_id="synthetic-requester",
                    service="SyntheticService",
                    environment=Environment.STAGING,
                    title="synthetic change",
                    description="isolated evaluation",
                    change_window=ChangeWindow(start=NOW, end=NOW + timedelta(hours=1), timezone="UTC"),
                    rollback_plan="restore register",
                    verification_plan="independent register read",
                ),
                idempotency_key="synthetic-key",
                occurred_at=NOW,
            )
        )
        await self.domain.create_change(request)
        run = await self.domain.create_run(new_change_run(request, "synthetic-run"))
        self.bound: dict[str, Any] = dict(
            change_id=run.change_id, run_id=run.run_id, revision=run.revision, request_hash=run.request_hash
        )
        item = EvidenceItem(
            evidence_id="synthetic:ci",
            source_type=EvidenceSourceType.CI,
            source_uri="fixture://synthetic/ci",
            captured_at=NOW,
            content_hash="a" * 64,
            access_scope=EvidenceAccessScope(public=True),
            snippet="synthetic CI observation",
            confidence=1,
        )
        self.snapshot = EvidenceSnapshot(
            snapshot_id="sha256:"
            + canonical_hash(item.model_dump(include={"evidence_id", "source_uri", "source_version", "content_hash"})),
            item=item,
            persisted_at=NOW,
        )
        await self.evidence.append_snapshot(self.snapshot)
        self.thread = releaseguard_thread_id(run.change_id, run.run_id)
        self.citation = EvidenceCitation(
            **self.bound,
            citation_id="synthetic-citation",
            snapshot_id=self.snapshot.snapshot_id,
            evidence_id=item.evidence_id,
            content_hash=item.content_hash,
            captured_at=NOW,
            required_item="ci",
            thread_id=self.thread,
            policy_version="synthetic-policy.v1",
            cited_at=NOW,
        )
        await self.evidence.append_citation(self.citation)
        self.evidence_policy = EvidencePolicy(
            policy_version="synthetic-policy.v1",
            requirements=[
                EvidenceRequirement(
                    requirement_id="ci", source_types=frozenset({EvidenceSourceType.CI}), min_confidence=0.8
                )
            ],
        )
        self.descriptor = ToolDescriptor(
            tool_name="sandbox-register",
            version="v1",
            capabilities=frozenset({SideEffect.MUTATING}),
            resource_types=frozenset({"synthetic_register"}),
            action_types=frozenset({"compare_set"}),
            input_schema="sandbox-register.v1",
            output_schema="sandbox-result.v1",
            side_effect_level=SideEffect.MUTATING,
            timeout_ms=1000,
            idempotency_required=True,
            adapter_kind="internal",
        )
        pins = []
        for direction in (Direction.FORWARD, Direction.COMPENSATION):
            for order, i in enumerate(range(count) if direction is Direction.FORWARD else reversed(range(count)), 1):
                forward = direction is Direction.FORWARD
                pins.append(
                    ManifestStep(
                        step_id=("step-" if forward else "undo-") + str(i),
                        order=order,
                        direction=direction,
                        tool_name=self.descriptor.tool_name,
                        version="v1",
                        descriptor_hash=self.descriptor.descriptor_hash,
                        input_hash=canonical_hash(
                            SandboxInput(expected_value=i if forward else i + 1, desired_value=i + 1 if forward else i)
                        ),
                        input_ref="input-" + str(i) + direction.value,
                        resource_id="synthetic-resource",
                        environment="staging",
                        expected_value=i if forward else i + 1,
                        desired_value=i + 1 if forward else i,
                        compensates=None if forward else "step-" + str(i),
                        dependencies=(("step-" + str(i - 1),) if i else ())
                        if forward
                        else (("undo-" + str(i + 1),) if i < count - 1 else ()),
                    )
                )
        self.manifest = ExecutionManifest(
            mode=ExecutionMode.SIMULATION if simulation else ExecutionMode.SANDBOX,
            steps=tuple(pins),
            policy_version="synthetic-policy.v1",
        )
        steps, rollback = [], []
        for pin in pins:
            value = dict(
                step_id=pin.step_id,
                order=pin.order,
                tool_name=pin.tool_name,
                permission="MUTATING",
                arguments_hash=pin.input_hash,
                idempotency_key="key-" + pin.step_id,
                expected_effect="synthetic register",
                timeout_seconds=1,
            )
            if pin.direction is Direction.FORWARD:
                value["compensation_step_id"] = "undo-" + str(pin.order - 1)
                steps.append(value)
            else:
                rollback.append(value)
        audit = AuditFields(
            actor_id="synthetic-system",
            actor_type=ActorType.SYSTEM,
            correlation_id="synthetic-correlation",
            occurred_at=NOW,
            policy_version="synthetic-policy.v1",
            input_hash="b" * 64,
        )
        self.plan = build_execution_plan(
            ExecutionPlanInput.model_validate(
                dict(
                    **self.bound,
                    plan_id="synthetic-plan",
                    input_hash="b" * 64,
                    audit=audit,
                    steps=steps,
                    rollback_steps=rollback,
                    permissions=["MUTATING"],
                    rollback_plan="restore register",
                    verification_plan="independent register",
                    dry_run=simulation,
                    manifest=self.manifest,
                )
            )
        )
        self.risk = RiskAssessment.model_validate(
            dict(
                **self.bound,
                level=level,
                factors=["supplied synthetic risk"],
                blast_radius="isolated register",
                evidence_ids=[item.evidence_id],
                controls=["rollback"],
                uncertainty=["unknown synthetic risk"] if level == "UNKNOWN" else [],
                policy_version="synthetic-policy.v1",
                assessed_at=NOW,
            )
        )
        assessment = EvidenceAssessment(
            **self.bound,
            sufficient=True,
            required_items=["ci"],
            present_items=["ci"],
            missing_items=[],
            evidence_ids=[item.evidence_id],
            policy_version="synthetic-policy.v1",
            evaluated_at=NOW,
        )
        for status, inputs in (
            (ChangeStatus.COLLECTING_EVIDENCE, StateInputs()),
            (ChangeStatus.EVIDENCE_SUFFICIENT, StateInputs(evidence_items=[item], evidence_assessment=assessment)),
            (ChangeStatus.RISK_ASSESSED, StateInputs(risk_assessment=self.risk)),
            (ChangeStatus.PLAN_READY, StateInputs(execution_plan=self.plan)),
        ):
            run = await self.domain.commit_run_update(
                build_run_command(
                    RunCommandInput(
                        change_id=run.change_id,
                        run_id=run.run_id,
                        command_id="setup-" + status.value,
                        expected_revision=run.revision,
                        expected_state_version=run.state_version,
                        target_status=status,
                        audit=AuditContext.model_validate(audit.model_dump(exclude={"input_hash"})),
                        inputs=inputs,
                    )
                )
            )
        pin = pins[0]
        call = ToolInvocation.model_validate(
            dict(
                **self.bound,
                invocation_id="synthetic-call",
                thread_id=self.thread,
                tool_name=pin.tool_name,
                version=pin.version,
                descriptor_hash=pin.descriptor_hash,
                input_schema=pin.input_schema,
                input_hash=pin.input_hash,
                principal_id="synthetic-requester",
                session_id="synthetic-session",
                tenant_id="synthetic-tenant",
                resource_type=pin.resource_type,
                resource_id=pin.resource_id,
                action=pin.action,
                environment="staging",
                policy_version="synthetic-policy.v1",
                correlation_id="synthetic-correlation",
                approval_id="synthetic-approval:decision",
                evidence_ids=(item.evidence_id,),
                citation_ids=(self.citation.citation_id,),
                occurred_at=NOW,
            )
        )
        self.context = RequestContext(principal=principal(), invocation=call, resource_tenant_id="synthetic-tenant")
        rules = tuple(
            PermissionRule(
                rule_id=action,
                tenant_id="synthetic-tenant",
                roles=frozenset({"requester", "approver"}),
                resource_type="synthetic_register",
                resource_ids=frozenset({"synthetic-resource"}),
                actions=frozenset({action}),
                environments=frozenset({"staging"}),
                side_effects=frozenset({SideEffect(effect)}),
                effect=PermissionEffect(grant),
            )
            for action, effect, grant in (
                ("compare_set", "mutating", "approval_required"),
                ("inspect", "read_only", "allow"),
            )
        )
        self.service = ApprovalService(
            self.domain,
            self.evidence,
            self.approvals,
            PermissionPolicy(policy_version="synthetic-policy.v1", rules=rules),
            self.evidence_policy,
        )
        _, binding = await load_authorization_binding(
            self.domain,
            self.evidence,
            self.context,
            self.descriptor,
            self.service.permission_policy,
            self.evidence_policy,
            at=NOW,
            scope="plan",
        )
        self.review = ApprovalRequest(
            approval_request_id="synthetic-approval",
            idempotency_key="synthetic-approval-key",
            binding=binding,
            authorization_hash=canonical_hash(binding),
            expected_state_version=run.state_version,
            requested_at=NOW,
            expires_at=NOW + timedelta(minutes=10),
            reason_code="synthetic_review",
        )
        entry = await self.service.create(self.review, self.context, self.descriptor, at=NOW)
        await self.domain.commit_run_update(approval_domain_command(run, entry.pending, command_id="pending-plan"))
        if decide:
            await self.decide()
        self.resolver = TrustedResolver(self.context)
        self.execution = ExecutionService(
            ExecutionPreflight(self.service, self.resolver, self.descriptor), self.ledger, SandboxExecutor(self.target)
        )
        self.verification = VerificationService(self.execution, IndependentRegisterReader(self.target))
        self.runtime = ExecutionRuntime(self.execution, self.verification, lambda: self.at)
        self.graph = build_agent_graph(
            passthrough, passthrough, InMemorySaver(), self.domain, self.evidence, execution_runtime=self.runtime
        )
        self.config = releaseguard_runnable_config(run.change_id, run.run_id, run.revision)
        return self

    async def state(self) -> GraphState:
        """Build only the intent; business facts remain Repository authority."""
        run = await self.run()
        return GraphState(
            workflow_kind="releaseguard",
            releaseguard=ReleaseGuardGraphContext(
                change_request=run.change_request,
                run_id=run.run_id,
                expected_revision=run.revision,
                execution_intent=ExecutionIntent(request=await self.request()),
            ),
        )
