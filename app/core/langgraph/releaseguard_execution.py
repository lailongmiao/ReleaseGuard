"""Protected offline services in the existing Graph; no production wiring."""

from collections.abc import Callable
from datetime import datetime

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from app.core.langgraph.releaseguard_nodes import _context, _validate_run_identity, _with_context
from app.repositories.releaseguard import RunLookup
from app.schemas.execution import Direction, ExecutionIntent, ExecutionRequest
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import ChangeRunState, ChangeStatus, EventStatus
from app.services.releaseguard.dry_run import DryRunService
from app.services.releaseguard.execution import ExecutionService
from app.services.releaseguard.execution_commands import execution_command
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.receipt import ReceiptService
from app.services.releaseguard.recovery import RecoveryService
from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard.verification import VerificationService


class ExecutionRuntime:
    """Synthetic command coordinator, outside checkpoint and not durable."""

    def __init__(
        self, execution: ExecutionService, verification: VerificationService, clock: Callable[[], datetime]
    ) -> None:
        """Inject one authority, sender, independent reader and trusted clock."""
        if verification.execution is not execution:
            raise DomainError("execution_runtime_dependency_mismatch")
        self.execution, self.verification, self.clock = execution, verification, clock
        self.dry_run = DryRunService(execution.preflight, execution.adapter)
        self.recovery, self.receipt = RecoveryService(execution), ReceiptService(execution)
        self._intents: dict[str, str] = {}
        self._versions: dict[str, int] = {}
        self._continuations: dict[str, ExecutionIntent] = {}
        self._consumed_interrupts: dict[str, int] = {}

    async def load(self, intent: ExecutionIntent) -> ChangeRunState:
        """Accept replay only for a server-registered exact intent and own progress."""
        request = intent.request
        run = await self.execution.preflight.service.domain.get_run(RunLookup(run_id=request.run_id))
        plan = run.execution_plan
        if (
            plan is None
            or plan.manifest is None
            or request.plan_hash != plan.plan_hash
            or request.manifest_hash != plan.manifest.manifest_hash
            or request.mode is not plan.manifest.mode
            or any(getattr(request, k) != getattr(run, k) for k in ("change_id", "run_id", "revision", "request_hash"))
        ):
            raise DomainError("execution_graph_binding_conflict")
        digest = canonical_hash(intent)
        prior = self._intents.get(request.command_id)
        if prior is not None and prior != digest:
            raise DomainError("execution_intent_conflict")
        if prior is None and run.state_version != request.state_version:
            raise DomainError("execution_graph_version_conflict")
        if prior is not None and run.state_version != self._versions[request.command_id]:
            raise DomainError("execution_graph_store_fork")
        self._intents[request.command_id] = digest
        self._versions[request.command_id] = run.state_version
        return run

    def request(
        self,
        intent: ExecutionIntent,
        run: ChangeRunState,
        *,
        step: str | None = None,
        direction: Direction | None = None,
    ) -> ExecutionRequest:
        """Derive CAS and selected step from authoritative state, not resume facts."""
        return intent.request.model_copy(
            update={
                "state_version": run.state_version,
                "step_id": step or intent.request.step_id,
                "direction": direction or intent.request.direction,
            }
        )

    async def edge(self, intent: ExecutionIntent, run: ChangeRunState, target: ChangeStatus) -> ChangeRunState:
        """Service-only state edges follow verified saved outcomes."""
        entry = await self.execution.preflight.service.approvals.get(intent.request.approval_request_id)
        actor = await self.execution.preflight.resolver.resolve(entry.request.binding.invocation, at=self.clock())
        return await self.execution.preflight.service.domain.commit_run_update(
            execution_command(
                run,
                actor,
                command_id=intent.request.command_id + ":" + target.value,
                at=self.clock(),
                code="execution_graph_progress",
                target=target,
            )
        )

    async def advance(self, intent: ExecutionIntent) -> ChangeRunState:
        """A finite structured progression, with isolated UNKNOWN/recovery waits."""
        replay = intent.request.command_id in self._intents
        run = await self.load(intent)
        initial_version = run.state_version
        service = self.execution.preflight.service
        try:
            if intent.action == "reconcile" and run.status is ChangeStatus.EXECUTION_UNKNOWN:
                run = await self.recovery.reconcile(self.request(intent, run), at=self.clock())
            elif intent.action == "recover" and run.status is ChangeStatus.RECOVERY_REQUIRED:
                run = await self.recovery.start(self.request(intent, run), at=self.clock())
            elif (
                intent.action != "continue"
                and not replay
                and run.status
                not in {
                    ChangeStatus.RECOVERING,
                    ChangeStatus.VERIFYING,
                    ChangeStatus.SUCCEEDED,
                    ChangeStatus.FAILED,
                    ChangeStatus.RECEIPTED,
                }
            ):
                raise DomainError("execution_intent_status_conflict")
            for _ in range(140):
                plan = run.execution_plan
                assert plan is not None and plan.manifest is not None
                if run.status in {ChangeStatus.APPROVED, ChangeStatus.DRY_RUN}:
                    run = await self.dry_run.run(self.request(intent, run), at=self.clock())
                elif run.status in {ChangeStatus.EXECUTING_SANDBOX, ChangeStatus.RECOVERING}:
                    recovering = run.status is ChangeStatus.RECOVERING
                    selected = (
                        set(run.recovery_authorization.compensation_step_ids)
                        if recovering and run.recovery_authorization
                        else {s.step_id for s in plan.steps}
                    )
                    direction = Direction.COMPENSATION if recovering else Direction.FORWARD
                    pins = sorted(
                        (p for p in plan.manifest.steps if p.direction is direction and p.step_id in selected),
                        key=lambda p: p.order,
                    )
                    latest = {event.step_id: event.status for event in run.execution_events}
                    pending = next((p for p in pins if latest.get(p.step_id) is not EventStatus.SUCCEEDED), None)
                    if pending is None:
                        run = await self.edge(intent, run, ChangeStatus.VERIFYING)
                    else:
                        request = self.request(intent, run, step=pending.step_id, direction=direction)
                        run = await (
                            self.recovery.compensate(request, at=self.clock())
                            if recovering
                            else self.execution.step(request, at=self.clock())
                        )
                elif run.status is ChangeStatus.EXECUTION_FAILED:
                    effects = await self.recovery.effects(run)
                    run = await self.edge(
                        intent, run, ChangeStatus.RECOVERY_REQUIRED if effects else ChangeStatus.FAILED
                    )
                elif run.status is ChangeStatus.DRY_RUN_FAILED:
                    run = await self.edge(intent, run, ChangeStatus.FAILED)
                elif run.status is ChangeStatus.VERIFYING:
                    if run.verification_result is not None:
                        passed = run.verification_result.status.value == "PASSED"
                        run = await self.edge(
                            intent,
                            run,
                            ChangeStatus.SUCCEEDED
                            if passed
                            else (
                                ChangeStatus.FAILED if run.recovery_authorization else ChangeStatus.RECOVERY_REQUIRED
                            ),
                        )
                    else:
                        # The reader uses historical start authority; compensation
                        # still requires its separately scoped live authority.
                        request = self.request(intent, run)
                        if run.recovery_authorization:
                            snapshot = await service.approvals.export_snapshot()
                            forward = next(
                                e
                                for e in snapshot.entries
                                if e.decision and e.decision.approval_id == run.execution_approval_id
                            )
                            request = request.model_copy(
                                update={
                                    "approval_request_id": forward.request.approval_request_id,
                                    "direction": Direction.FORWARD,
                                    "step_id": plan.steps[0].step_id,
                                }
                            )
                        run = await self.verification.verify(request, at=self.clock())
                elif run.status in {
                    ChangeStatus.DRY_RUN_PASSED,
                    ChangeStatus.SUCCEEDED,
                    ChangeStatus.FAILED,
                    ChangeStatus.REJECTED,
                }:
                    return await self.receipt.issue(self.request(intent, run), at=self.clock())
                elif run.status is ChangeStatus.EXECUTION_UNKNOWN:
                    if run.receipt is None:
                        run = await self.receipt.issue(self.request(intent, run), at=self.clock())
                    return run
                elif run.status in {ChangeStatus.RECOVERY_REQUIRED, ChangeStatus.RECEIPTED}:
                    return run
                else:
                    raise DomainError("execution_status_blocked")
            raise DomainError("execution_progress_bound_exceeded")
        finally:
            # This readback permits same-process replay after a domain commit
            # before Graph checkpoint. It is explicitly not a durable coordinator.
            current = await service.domain.get_run(RunLookup(run_id=run.run_id))
            entries = await self.execution.ledger.entries(run.run_id)
            allowed = {
                "publish:" + event.event_id
                for claim, events in entries
                if claim.owner == intent.request.command_id
                or (
                    intent.action == "reconcile"
                    and claim.step_id == intent.request.step_id
                    and claim.direction is intent.request.direction
                )
                for event in events
            }
            allowed.update(
                prefix + claim.claim_id
                for claim, _ in entries
                if claim.owner == intent.request.command_id
                for prefix in ("outcome:", "unknown:")
            )
            progress = [
                record
                for record in current.transition_history + current.fact_history
                if record.state_version > initial_version
            ]
            if any(
                record.command_id not in allowed and not record.command_id.startswith(intent.request.command_id + ":")
                for record in progress
            ):
                raise DomainError("execution_graph_store_fork")
            self._versions[intent.request.command_id] = current.state_version


def execution_state(state: GraphState, run: ChangeRunState, intent: ExecutionIntent) -> GraphState:
    """Publish only authoritative domain references to the checkpoint."""
    context = state.releaseguard
    if context is None:
        raise DomainError("releaseguard_state_required")
    _validate_run_identity(context, run)
    return _with_context(
        state,
        ReleaseGuardGraphContext.model_validate(
            {
                **context.model_dump(),
                "domain_state": run,
                "state_version": run.state_version,
                "execution_intent": intent,
                "routing_outcome": "execution_wait"
                if run.status in {ChangeStatus.EXECUTION_UNKNOWN, ChangeStatus.RECOVERY_REQUIRED}
                else "terminal",
            }
        ),
    )


async def execute_releaseguard(state: GraphState, config: RunnableConfig, runtime: ExecutionRuntime) -> GraphState:
    """One existing Graph node drives services and waits for typed safe intent."""
    context = _context(state, config)
    intent = context.execution_intent
    if intent is None or intent.request.thread_id != config.get("configurable", {}).get("thread_id"):
        raise DomainError("execution_graph_binding_conflict")
    root_command = intent.request.command_id
    for _ in range(runtime._consumed_interrupts.get(root_command, 0)):
        # LangGraph restarts a node and indexes its interrupt calls. Consume
        # already accepted values without reapplying their domain commands.
        interrupt({"kind": "releaseguard.execution.v1", "replay": True})
    resumed = runtime._continuations.get(root_command)
    if resumed is not None:
        intent = resumed
    run = await runtime.advance(intent)
    updated = execution_state(state, run, intent)
    while run.status in {ChangeStatus.EXECUTION_UNKNOWN, ChangeStatus.RECOVERY_REQUIRED}:
        review = None
        if run.status is ChangeStatus.RECOVERY_REQUIRED:
            review = await runtime.recovery.prepare(
                runtime.request(intent, run),
                at=runtime.clock(),
                request_id=intent.request.command_id + ":recovery-review",
            )
        supplied = interrupt(
            {
                "kind": "releaseguard.execution.v1",
                "status": run.status.value,
                "state_version": run.state_version,
                "request": runtime.request(intent, run).model_dump(mode="json"),
                "recovery_request": review.model_dump(mode="json") if review else None,
            }
        )
        candidate = ExecutionIntent.model_validate(supplied)
        if config.get("metadata", {}).get("execution_intent_hash") != canonical_hash(candidate):
            raise DomainError("execution_resume_intent_conflict")
        expected_action = "reconcile" if run.status is ChangeStatus.EXECUTION_UNKNOWN else "recover"
        if candidate.action != expected_action or candidate.request.state_version != run.state_version:
            raise DomainError("execution_resume_intent_conflict")
        intent = candidate
        runtime._continuations[root_command] = candidate
        runtime._consumed_interrupts[root_command] = runtime._consumed_interrupts.get(root_command, 0) + 1
        run = await runtime.advance(intent)
        updated = execution_state(updated, run, intent)
    await runtime.execution.hit("before_node_return")
    return updated


async def resume_execution(graph: CompiledStateGraph, intent: ExecutionIntent, config: RunnableConfig) -> dict:
    """Only the latest root waiting task may consume a typed execution intent."""
    intent = ExecutionIntent.model_validate_json(intent.model_dump_json())
    if config.get("configurable", {}).get("checkpoint_id") or config.get("configurable", {}).get("checkpoint_ns"):
        raise DomainError("execution_checkpoint_fork")
    snapshot = await graph.aget_state(config)
    waiting = [task for task in snapshot.tasks if task.name == "releaseguard_execution" and task.interrupts]
    if len(waiting) != 1 or not snapshot.values:
        raise DomainError("execution_checkpoint_missing_or_completed")
    state = GraphState.model_validate(snapshot.values)
    context = _context(state, config)
    prior = context.execution_intent
    if prior is None:
        raise DomainError("execution_graph_binding_conflict")
    old, new = prior.request, intent.request
    payload = waiting[0].interrupts[0].value
    if (
        not isinstance(payload, dict)
        or new.state_version != payload.get("state_version")
        or intent.action != ("reconcile" if payload.get("status") == "EXECUTION_UNKNOWN" else "recover")
    ):
        raise DomainError("execution_resume_intent_conflict")
    if any(
        getattr(old, key) != getattr(new, key)
        for key in (
            "thread_id",
            "change_id",
            "run_id",
            "revision",
            "request_hash",
            "plan_hash",
            "manifest_hash",
            "mode",
        )
    ):
        raise DomainError("execution_resume_binding_conflict")
    resume_config: RunnableConfig = {
        **config,
        "metadata": {**config.get("metadata", {}), "execution_intent_hash": canonical_hash(intent)},
    }
    return await graph.ainvoke(Command(resume=intent.model_dump(mode="json")), resume_config)
