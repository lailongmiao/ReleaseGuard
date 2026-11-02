"""Local product orchestration over the existing services and single Graph.

Only synthetic sources and the bounded integer sandbox are wired. PostgreSQL
holds authoritative facts and a claim-before-command journal. A interrupted
command is a diagnostic hold, never permission to resend an uncertain effect.
"""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast, override

from langgraph.graph.state import CompiledStateGraph

from app.core.langgraph.releaseguard_approval import ApprovalRuntime, resume_approval
from app.core.langgraph.releaseguard_execution import ExecutionRuntime
from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config, releaseguard_thread_id
from app.core.logging import logger
from app.repositories.approval import ApprovalRepository
from app.repositories.evidence import EvidenceRepository
from app.repositories.execution import ExecutionRepository
from app.repositories.releaseguard import ReleaseGuardRepository, RunLookup
from app.schemas.approval import ApprovalCommand, ApprovalRequest
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
from app.schemas.local_mvp import LocalActor, LocalCommand, LocalCreate, LocalPrepare, LocalRevision, LocalView
from app.schemas.releaseguard import (
    ActorType,
    ApprovalDecision,
    AuditContext,
    AuditFields,
    ChangeIdentity,
    ChangeRunState,
    ChangeStatus,
    CommandAction,
    CreateChangeRequest,
    ExecutionPlanInput,
    RiskAssessment,
    RiskLevel,
    RunCommandInput,
    StateInputs,
    build_change_request,
    build_execution_plan,
    build_run_command,
)
from app.schemas.tooling import ToolInvocation
from app.services.local_mvp_policy import (
    DESCRIPTOR,
    EVIDENCE_POLICY,
    PERMISSION_POLICY,
    POLICY_VERSION,
    PRODUCT_EVIDENCE_POLICY,
    RESOURCE_ID,
    LocalIndependentReader,
    LocalExecutionResolver,
    LocalRegistry,
    LocalResolver,
    LocalSandboxAdapter,
    LocalSourceAdapter,
    evidence_access,
    principal_id,
)
from app.services.local_mvp_storage import LocalPostgresStore, ScopedRepository, unit, review_scope
from app.services.releaseguard.approval import ApprovalService
from app.services.releaseguard.authorization_context import load_authorization_binding
from app.services.releaseguard.execution import ExecutionService
from app.services.releaseguard.execution_preflight import ExecutionPreflight
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard.verification import VerificationService


class ScopedExecutionRuntime(ExecutionRuntime):
    """Per-command coordinator; durable root journal owns crash/replay policy."""

    @override
    async def advance(self, intent: ExecutionIntent) -> ChangeRunState:
        """No runtime authority is recovered from a client or old checkpoint."""
        runtime = ExecutionRuntime(self.execution, self.verification, self.clock)
        return await runtime.advance(intent)


class LocalMVPService:
    """Trusted HTTP commands, object locks and server-pinned dependencies."""

    def __init__(
        self, graph: Callable[[], Awaitable[CompiledStateGraph]], store: LocalPostgresStore | None = None
    ) -> None:
        """Use the application's graph factory, not another business graph."""
        self.graph = graph
        self.store = store or LocalPostgresStore()
        self.domain = cast(ReleaseGuardRepository, ScopedRepository("domain"))
        self.evidence = cast(EvidenceRepository, ScopedRepository("evidence"))
        self.approvals = cast(ApprovalRepository, ScopedRepository("approvals"))
        self.ledger = cast(ExecutionRepository, ScopedRepository("ledger"))
        self.resolver = LocalResolver()
        self.product_runtime: Any = None
        self.approval_service = ApprovalService(
            self.domain,
            self.evidence,
            self.approvals,
            PERMISSION_POLICY,
            EVIDENCE_POLICY,
            evidence_policy_resolver=self.evidence_policy,
        )
        self.approval_runtime = ApprovalRuntime(
            self.approval_service, LocalRegistry(), self.resolver, lambda: datetime.now(UTC)
        )
        self.execution = ExecutionService(
            ExecutionPreflight(self.approval_service, LocalExecutionResolver(self.resolver), DESCRIPTOR),
            self.ledger,
            LocalSandboxAdapter(),
        )
        self.verification = VerificationService(self.execution, LocalIndependentReader())
        self.execution_runtime = ScopedExecutionRuntime(self.execution, self.verification, lambda: datetime.now(UTC))

    def dependencies(self) -> dict[str, Any]:
        """Inject object-scoped services into the same Chat/ReleaseGuard graph."""
        return {
            "evidence_repository": self.evidence,
            "evidence_adapters": {key: LocalSourceAdapter(key) for key in ("ci", "history")},
            "evidence_access": evidence_access,
            "approval_runtime": self.approval_runtime,
            "execution_runtime": self.execution_runtime,
            "analysis_runtime": self.product_runtime.graph_analyze if self.product_runtime else None,
        }

    def evidence_policy(self) -> Any:
        """Only server-created object mode selects the evidence policy."""
        return PRODUCT_EVIDENCE_POLICY if unit().facts.product else EVIDENCE_POLICY

    async def run(self) -> ChangeRunState:
        """Read the only run belonging to the authenticated locked aggregate."""
        return await self.domain.get_run(RunLookup(run_id=unit().facts.creation["run_id"]))

    async def view(self) -> LocalView:
        """A success message cannot substitute for saved state or a receipt."""
        current = unit()
        if current.facts.product:
            if self.product_runtime is None:
                raise DomainError("product_runtime_unavailable")
            if current.facts.version == "local-mvp.v3":
                await self.product_runtime.reviews.store.object_row(current.row["change_id"], current.actor)
            else:
                for source_id in ("github-checkout", "markdown-rules"):
                    await self.product_runtime.store.source(source_id, current.actor)
        state = await self.run()
        approval_id = current.facts.creation.get("approval_request_id")
        approval = await self.approvals.get(approval_id) if approval_id else None
        held = state.status in {ChangeStatus.EXECUTION_UNKNOWN, ChangeStatus.RECOVERY_REQUIRED} or any(
            value.get("status") == "started" for value in current.facts.commands.values()
        )
        return LocalView(
            state=state,
            approval=approval,
            target_write_calls=current.target.calls,
            recovery="safe_hold" if held else "saved_facts",
            source_provenance="readonly_github_markdown" if current.facts.product else "synthetic",
            analysis_provenance=current.facts.product.get("analysis_provenance", "unverified")
            if current.facts.product
            else "deterministic",
            execution_mode=state.execution_plan.manifest.mode.value
            if state.execution_plan and state.execution_plan.manifest
            else None,
        )

    def owner(self) -> None:
        """Approvers can inspect/decide, not create plans or execute as owners."""
        if unit().actor.user_id != unit().row["owner_user_id"]:
            raise DomainError("local_owner_required")

    async def create(self, body: LocalCreate, actor: LocalActor) -> LocalView:
        """Derive scoped identifiers; equal key/body preserves original result."""
        if body.change.requester_id != principal_id(actor.user_id):
            raise DomainError("local_requester_mismatch")
        if body.change.environment.value not in {"development", "staging"}:
            raise DomainError("local_environment_denied")
        if body.product:
            if self.product_runtime is None:
                raise DomainError("product_runtime_unavailable")
            for source_id in ("github-checkout", "markdown-rules"):
                await self.product_runtime.store.source(source_id, actor)
        change_id = "local-change:" + canonical_hash({"actor": actor.user_id, "key": body.idempotency_key})
        scope = review_scope.get()
        creation_digest = (
            canonical_hash({"body": body, "review": scope["binding"]})
            if scope and "binding" in scope
            else canonical_hash(body)
        )
        async with self.store.open(change_id, actor, (body.approver_user_id, creation_digest)):
            current = unit()
            if current.facts.creation.get("result"):
                return LocalView.model_validate(current.facts.creation["result"])
            if not current.facts.creation:
                run_id = "local-run:" + canonical_hash({"change": change_id})
                request = build_change_request(
                    CreateChangeRequest(
                        identity=ChangeIdentity(change_id=change_id, revision=1),
                        body=body.change,
                        idempotency_key=body.idempotency_key,
                        occurred_at=current.at,
                    )
                )
                current.facts.creation = {
                    "run_id": run_id,
                    "thread_id": releaseguard_thread_id(change_id, run_id),
                    "request": request.model_dump(mode="json"),
                    "evidence_scenario": body.evidence_scenario,
                }
                if body.product:
                    current.facts.version = "local-mvp.v2"
                    current.facts.product = {
                        "binding": body.product.model_dump(mode="json"),
                        "analysis_status": "not_started",
                    }
                if scope and "binding" in scope:
                    current.facts.version = "local-mvp.v3"
                    current.facts.product = {
                        "contract_version": "releaseguard.review.v1.1",
                        "binding": scope["binding"],
                        "analysis_status": "not_started",
                    }
                    await self.product_runtime.reviews.register(current, actor)
                await current.save()
            await self.invoke()
            result = await self.view()
            current.facts.creation["result"] = result.model_dump(mode="json")
            await current.save()
            return result

    async def invoke(self, **updates: Any) -> dict:
        """Run the application's one Graph with server-built typed context."""
        current = unit()
        context = ReleaseGuardGraphContext.model_validate(
            {
                "change_request": current.facts.creation["request"],
                "run_id": current.facts.creation["run_id"],
                "expected_revision": current.facts.creation["request"]["revision"],
                **updates,
            }
        )
        graph = await self.graph()
        config = releaseguard_runnable_config(
            context.change_request.change_id, context.run_id, context.expected_revision
        )
        state = GraphState(workflow_kind="releaseguard", releaseguard=context)
        result = await graph.ainvoke(state.model_dump(mode="python"), config)
        snapshot = await graph.aget_state(config)
        logger.info(
            "local_graph_command_saved",
            change_id=context.change_request.change_id,
            status="waiting" if snapshot.next else "saved",
        )
        return result

    async def edge(self, target: ChangeStatus, inputs: StateInputs, command_id: str) -> ChangeRunState:
        """Existing deterministic domain service owns every preparatory edge."""
        run = await self.run()
        return await self.domain.commit_run_update(
            build_run_command(
                RunCommandInput(
                    change_id=run.change_id,
                    run_id=run.run_id,
                    command_id=command_id,
                    expected_revision=run.revision,
                    expected_state_version=run.state_version,
                    target_status=target,
                    inputs=inputs,
                    audit=AuditContext(
                        actor_id=principal_id(unit().actor.user_id),
                        actor_type=ActorType.USER,
                        reason="local_product_preparation",
                        correlation_id=run.run_id,
                        occurred_at=unit().at,
                        policy_version=POLICY_VERSION,
                    ),
                )
            )
        )

    async def command(
        self, change_id: str, actor: LocalActor, body: LocalCommand | ApprovalCommand, kind: str
    ) -> LocalView:
        """Serialize command id/hash/version across processes; uncertain never retries."""
        if self.product_runtime and kind in {"prepare", "decide", "execute"}:
            await self.product_runtime.fresh(change_id, actor)
        async with self.store.open(change_id, actor):
            current = unit()
            if current.facts.version == "local-mvp.v3":
                await self.product_runtime.reviews.check_command(current, actor, body, kind)
            if kind == "decide":
                if actor.user_id != current.row["approver_user_id"]:
                    raise DomainError("local_approver_required")
            else:
                self.owner()
            digest = canonical_hash({"kind": kind, "body": body, "actor": actor.user_id})
            prior = current.facts.commands.get(body.command_id)
            if prior:
                if prior["hash"] != digest:
                    raise DomainError("idempotency_conflict")
                if prior["status"] == "completed":
                    return LocalView.model_validate(prior["result"])
                if prior["status"] == "failed":
                    raise DomainError(prior["error"])
                raise DomainError("local_command_outcome_unknown")
            if any(value["status"] == "started" for value in current.facts.commands.values()):
                raise DomainError("local_command_reconciliation_required")
            run = await self.run()
            if body.expected_state_version != run.state_version:
                raise DomainError("version_conflict")
            current.facts.commands[body.command_id] = {"hash": digest, "status": "started"}
            await current.save()
            try:
                if kind == "prepare" and isinstance(body, LocalPrepare):
                    await self.prepare(body)
                elif kind == "revise" and isinstance(body, LocalRevision):
                    await self.revise(body)
                elif kind == "decide" and isinstance(body, ApprovalCommand):
                    graph = await self.graph()
                    await resume_approval(
                        graph, body, releaseguard_runnable_config(run.change_id, run.run_id, run.revision)
                    )
                elif kind == "revoke":
                    approval_id = current.facts.creation.get("approval_request_id")
                    if not approval_id:
                        raise DomainError("local_approval_missing")
                    entry = await self.approvals.get(approval_id)
                    context = await self.resolver.resolve(
                        entry.request.binding.invocation, invalidation_only=current.facts.version == "local-mvp.v3"
                    )
                    await self.approval_service.invalidate(
                        approval_id, context, ApprovalDecision.REVOKED, "local_owner_revoked", at=current.at
                    )
                elif kind == "execute" and isinstance(body, LocalCommand):
                    await self.execute(body)
                else:
                    raise DomainError("local_command_invalid")
            except DomainError as error:
                current.facts.commands[body.command_id].update(status="failed", error=error.code)
                await current.save()
                raise
            current.facts.commands[body.command_id]["status"] = "completed"
            result = await self.view()
            current.facts.commands[body.command_id]["result"] = result.model_dump(mode="json")
            await current.save()
            return result

    async def prepare(self, body: LocalPrepare) -> None:
        """Two sources → Graph Gate → deterministic risk/plan → approval interrupt."""
        current = unit()
        command_id = "local-prepare:" + canonical_hash(
            {"change": current.row["change_id"], "command": body.command_id}
        )
        run = await self.run()
        if current.facts.product:
            product = current.facts.product
            if body.mode != "offline_simulation" or body.analysis_id != product.get("analysis_id"):
                raise DomainError("product_analysis_or_mode_denied")
            if current.facts.version == "local-mvp.v3" and run.status in {
                ChangeStatus.RECEIVED,
                ChangeStatus.WAITING_FOR_EVIDENCE,
            }:
                await self.edge(ChangeStatus.COLLECTING_EVIDENCE, StateInputs(), command_id + ":collect")
                await self.invoke(evidence_policy=self.evidence_policy(), evidence_evaluated_at=current.at)
                run = await self.run()
            if run.status is not ChangeStatus.EVIDENCE_SUFFICIENT:
                raise DomainError("local_prepare_status_conflict")
            if (
                product["analysis"]["risk"] == "UNKNOWN"
                or product["analysis"]["unknowns"]
                or not product["analysis"]["candidate_steps"]
            ):
                raise DomainError("product_analysis_unknown")
        else:
            if run.status is not ChangeStatus.RECEIVED:
                raise DomainError("local_prepare_status_conflict")
            await self.edge(ChangeStatus.COLLECTING_EVIDENCE, StateInputs(), command_id + ":collect")
            await self.invoke(evidence_policy=EVIDENCE_POLICY, evidence_evaluated_at=current.at)
        run = await self.run()
        if run.status is not ChangeStatus.EVIDENCE_SUFFICIENT:
            return
        bound = {key: getattr(run, key) for key in ("change_id", "run_id", "revision", "request_hash")}
        floor = RiskLevel.HIGH if run.change_request.environment.value == "staging" else RiskLevel.MEDIUM
        level = RiskLevel(current.facts.product["analysis"]["risk"]) if current.facts.product else floor
        if level in {RiskLevel.LOW, RiskLevel.MEDIUM} and floor is RiskLevel.HIGH or level is RiskLevel.LOW:
            level = floor
        risk = RiskAssessment(
            **bound,
            level=level,
            factors=[
                "readonly simulation environment rule"
                if current.facts.product
                else "bounded synthetic environment rule"
            ],
            blast_radius="one isolated local integer",
            evidence_ids=[item.evidence_id for item in run.evidence_items],
            controls=["designated approver", "independent observation"],
            policy_version=self.evidence_policy().policy_version,
            assessed_at=current.at,
        )
        await self.edge(ChangeStatus.RISK_ASSESSED, StateInputs(risk_assessment=risk), command_id + ":risk")
        pins = tuple(
            ManifestStep(
                step_id="step-0" if forward else "undo-0",
                order=1,
                direction=Direction.FORWARD if forward else Direction.COMPENSATION,
                tool_name=DESCRIPTOR.tool_name,
                version=DESCRIPTOR.version,
                descriptor_hash=DESCRIPTOR.descriptor_hash,
                input_hash=canonical_hash(
                    SandboxInput(expected_value=0 if forward else 1, desired_value=1 if forward else 0)
                ),
                input_ref="local-forward" if forward else "local-compensation",
                resource_id=RESOURCE_ID,
                environment=cast(Literal["development", "staging"], run.change_request.environment.value),
                expected_value=0 if forward else 1,
                desired_value=1 if forward else 0,
                compensates=None if forward else "step-0",
            )
            for forward in (True, False)
        )
        manifest = ExecutionManifest(mode=ExecutionMode(body.mode), steps=pins, policy_version=POLICY_VERSION)
        plan_input_hash = canonical_hash({"request": run.request_hash, "manifest": manifest.manifest_hash})
        if current.facts.product:
            plan_input_hash = canonical_hash(
                {
                    "request": run.request_hash,
                    "manifest": manifest.manifest_hash,
                    "product": current.facts.product["binding_hash"],
                }
            )
        audit = AuditFields(
            actor_id=principal_id(current.actor.user_id),
            actor_type=ActorType.USER,
            reason="local_plan_pinned",
            correlation_id=run.run_id,
            occurred_at=current.at,
            policy_version=POLICY_VERSION,
            input_hash=plan_input_hash,
        )
        step_values = [
            dict(
                step_id=p.step_id,
                order=p.order,
                tool_name=p.tool_name,
                permission="MUTATING",
                arguments_hash=p.input_hash,
                idempotency_key=run.run_id + ":" + p.step_id,
                expected_effect="synthetic integer compare set",
                timeout_seconds=1,
                **({"compensation_step_id": "undo-0"} if p.direction is Direction.FORWARD else {}),
            )
            for p in pins
        ]
        plan = build_execution_plan(
            ExecutionPlanInput.model_validate(
                dict(
                    **bound,
                    plan_id="local-plan:" + canonical_hash(bound),
                    input_hash=plan_input_hash,
                    audit=audit,
                    steps=[step_values[0]],
                    rollback_steps=[step_values[1]],
                    permissions=["MUTATING"],
                    rollback_plan=run.change_request.rollback_plan,
                    verification_plan=run.change_request.verification_plan,
                    dry_run=body.mode == "offline_simulation",
                    manifest=manifest,
                )
            )
        )
        run = await self.edge(ChangeStatus.PLAN_READY, StateInputs(execution_plan=plan), command_id + ":plan")
        citation_ids = tuple(
            c.citation_id
            for c in current.evidence._citations.values()
            if c.revision == run.revision and c.required_item in {"ci", "history"}
        )
        approval_id = "local-approval:" + canonical_hash({"run": run.run_id, "plan": plan.plan_hash})
        call = ToolInvocation.model_validate(
            dict(
                **bound,
                invocation_id="local-call:" + canonical_hash(bound),
                thread_id=current.facts.creation["thread_id"],
                tool_name=DESCRIPTOR.tool_name,
                version=DESCRIPTOR.version,
                descriptor_hash=DESCRIPTOR.descriptor_hash,
                input_schema=pins[0].input_schema,
                input_hash=pins[0].input_hash,
                principal_id=principal_id(current.actor.user_id),
                session_id=current.actor.session_id,
                tenant_id="local-mvp",
                resource_type="synthetic_register",
                resource_id=RESOURCE_ID,
                action="compare_set",
                environment=run.change_request.environment.value,
                policy_version=POLICY_VERSION,
                correlation_id=run.run_id,
                approval_id=approval_id + ":decision",
                evidence_ids=tuple(item.evidence_id for item in run.evidence_items),
                citation_ids=citation_ids,
                occurred_at=current.at,
            )
        )
        context = await self.resolver.resolve(call)
        _, binding = await load_authorization_binding(
            self.domain,
            self.evidence,
            context,
            DESCRIPTOR,
            PERMISSION_POLICY,
            self.evidence_policy(),
            at=current.at,
            scope="plan",
        )
        request = ApprovalRequest(
            approval_request_id=approval_id,
            idempotency_key=approval_id,
            binding=binding,
            authorization_hash=canonical_hash(binding),
            expected_state_version=run.state_version,
            requested_at=current.at,
            expires_at=current.at + timedelta(seconds=body.approval_ttl_seconds),
            reason_code="local_designated_approval_required",
        )
        current.facts.creation["approval_request_id"] = approval_id
        await current.save()
        await self.invoke(approval_request=request)

    async def revise(self, body: LocalRevision) -> None:
        """Append a revision and invalidate old request authority before re-planning."""
        current, run = unit(), await self.run()
        if body.change.requester_id != principal_id(current.actor.user_id):
            raise DomainError("local_requester_mismatch")
        if body.change.environment.value not in {"development", "staging"}:
            raise DomainError("local_environment_denied")
        draft = build_change_request(
            CreateChangeRequest(
                identity=ChangeIdentity(change_id=run.change_id, revision=run.revision + 1),
                body=body.change,
                idempotency_key=run.change_request.idempotency_key,
                occurred_at=current.at,
            )
        )
        replacement = type(draft).model_validate({**draft.model_dump(), "created_at": run.change_request.created_at})
        # Domain validates allowed status/CAS before any authority is changed.
        command = build_run_command(
            RunCommandInput(
                change_id=run.change_id,
                run_id=run.run_id,
                command_id="local-revise:" + canonical_hash({"change": run.change_id, "command": body.command_id}),
                expected_revision=run.revision,
                expected_state_version=run.state_version,
                action=CommandAction.REVISE,
                replacement_request=replacement,
                audit=AuditContext(
                    actor_id=principal_id(current.actor.user_id),
                    actor_type=ActorType.USER,
                    reason="local_revision_changed",
                    correlation_id=run.run_id,
                    occurred_at=current.at,
                    policy_version=POLICY_VERSION,
                ),
            )
        )
        revised = await self.domain.commit_run_update(command)
        approval_id = current.facts.creation.get("approval_request_id")
        if approval_id:
            entry = await self.approvals.get(approval_id)
            if not entry.invalidations:
                context = await self.resolver.resolve(
                    entry.request.binding.invocation, invalidation_only=current.facts.version == "local-mvp.v3"
                )
                await self.approval_service.invalidate(
                    approval_id, context, ApprovalDecision.INVALIDATED, "local_revision_changed", at=current.at
                )
        current.facts.creation["request"] = revised.change_request.model_dump(mode="json")
        current.facts.creation["evidence_scenario"] = body.evidence_scenario
        current.facts.creation.pop("approval_request_id", None)
        if current.facts.product:
            binding = current.facts.product["binding"]
            current.facts.product = {"binding": binding, "analysis_status": "not_started"}
            if current.facts.version == "local-mvp.v3":
                current.facts.product.update(contract_version="releaseguard.review.v1.1", object_id=run.change_id)
        await current.save()

    async def execute(self, body: LocalCommand) -> None:
        """No execution before valid approval; no unknown-result resume endpoint."""
        current = unit()
        run = await self.run()
        if (
            run.status is not ChangeStatus.APPROVED
            or run.execution_plan is None
            or run.execution_plan.manifest is None
        ):
            raise DomainError("local_execution_status_denied")
        plan, manifest = run.execution_plan, run.execution_plan.manifest
        request = ExecutionRequest(
            command_id="local-exec:" + canonical_hash({"change": run.change_id, "command": body.command_id}),
            change_id=run.change_id,
            run_id=run.run_id,
            revision=run.revision,
            request_hash=run.request_hash,
            thread_id=current.facts.creation["thread_id"],
            state_version=run.state_version,
            plan_hash=plan.plan_hash,
            manifest_hash=manifest.manifest_hash,
            mode=manifest.mode,
            approval_request_id=current.facts.creation["approval_request_id"],
            step_id="step-0",
        )
        # Fresh Gate/permission/approval before even entering the Graph node.
        await self.execution.preflight.check(request, at=current.at, stage="dry_run")
        await self.invoke(execution_intent=ExecutionIntent(request=request))

    async def get(self, change_id: str, actor: LocalActor) -> LocalView:
        """Queries hydrate only authorized objects from PostgreSQL."""
        async with self.store.open(change_id, actor):
            return await self.view()

    async def audit(self, change_id: str, actor: LocalActor) -> dict[str, Any]:
        """Saved facts with bounded references; no raw provider/tool responses."""
        async with self.store.open(change_id, actor):
            await self.view()
            run = await self.run()
            snapshot = await self.approvals.export_snapshot()
            return {
                "change_id": run.change_id,
                "run_id": run.run_id,
                "thread_id": unit().facts.creation["thread_id"],
                "transitions": [v.model_dump(mode="json") for v in run.transition_history],
                "events": [v.model_dump(mode="json") for v in snapshot.audit_events],
                "product_events": await self.product_runtime.store.audit(change_id) if unit().facts.product else [],
                "production_execution": "disabled",
            }

    async def listing(self, actor: LocalActor, *, inbox: bool, cursor: str, limit: int) -> dict[str, Any]:
        """Expose only bounded object metadata, never evidence or private totals."""
        identifiers = await self.store.list_records(actor, inbox=inbox, cursor=cursor, limit=limit + 1)
        items = []
        product_seen = False
        for identifier in identifiers[:limit]:
            async with self.store.open(identifier, actor):
                if unit().facts.product:
                    if self.product_runtime is None:
                        raise DomainError("product_runtime_unavailable")
                    product_seen = True
                    for source_id in ("github-checkout", "markdown-rules"):
                        await self.product_runtime.store.source(source_id, actor)
                run = await self.run()
                if not inbox or run.status is ChangeStatus.WAITING_FOR_APPROVAL:
                    items.append({"change_id": identifier, "status": run.status.value, "revision": run.revision})
        # Recheck access before exposing both rows and the scanned-position cursor.
        await self.store.authenticate(actor.session_id, actor.expires_at)
        if product_seen and self.product_runtime is not None:
            for source_id in ("github-checkout", "markdown-rules"):
                await self.product_runtime.store.source(source_id, actor)
        return {"items": items, "next_cursor": identifiers[limit - 1] if len(identifiers) > limit else None}

    async def evidence_view(self, change_id: str, actor: LocalActor) -> dict[str, Any]:
        """The scoped aggregate is the authority, not checkpoint text."""
        async with self.store.open(change_id, actor):
            await self.view()
            return {
                "snapshots": [value.model_dump(mode="json") for value in unit().evidence._snapshots.values()],
                "citations": [value.model_dump(mode="json") for value in unit().evidence._citations.values()],
            }
