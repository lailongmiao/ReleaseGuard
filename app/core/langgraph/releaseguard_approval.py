"""Approval wait/resume in the existing graph, without execution capabilities."""

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from app.core.langgraph.releaseguard_nodes import _context, _with_context, _validate_run_identity
from app.repositories.releaseguard import RunLookup
from app.schemas.approval import ApprovalCommand, ApprovalRequest
from app.schemas.authorization import RequestContext
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import ApprovalDecision, ChangeRunState
from app.schemas.tooling import ToolInvocation
from app.services.releaseguard.approval import ApprovalService, approval_domain_command
from app.services.releaseguard.audit import record_audit
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.registry import ToolRegistry


class ApprovalContextResolver(Protocol):
    """Trusted authentication/object owner resolution; never interpret messages."""

    async def resolve(self, invocation: ToolInvocation, config: RunnableConfig) -> RequestContext:
        """Resolve the currently authenticated actor/session, including after resume."""
        ...


class ApprovalRuntime:
    """Local dependencies remain outside checkpoint and caller-supplied commands."""

    def __init__(
        self,
        service: ApprovalService,
        registry: ToolRegistry,
        resolver: ApprovalContextResolver,
        clock: Callable[[], datetime],
    ) -> None:
        """Inject authoritative services and a trusted clock, not a production fallback."""
        self.service, self.registry, self.resolver, self.clock = service, registry, resolver, clock


def _request(state: GraphState, config: RunnableConfig) -> ApprovalRequest:
    """Verify the full Graph identity before loading approval facts."""
    context = _context(state, config)
    request = context.approval_request
    if request is None or request.binding.invocation.thread_id != config.get("configurable", {}).get("thread_id"):
        raise DomainError("approval_graph_binding_mismatch")
    return request


def _copy_state(state: GraphState, run: ChangeRunState, *, outcome: str | None = None) -> GraphState:
    """Use only authoritative domain output, never a resume-supplied snapshot."""
    if state.releaseguard is None:
        raise DomainError("releaseguard_state_required")
    _validate_run_identity(state.releaseguard, run)
    context = ReleaseGuardGraphContext.model_validate(
        {
            **state.releaseguard.model_dump(),
            "domain_state": run,
            "state_version": run.state_version,
            "approval_outcome": outcome,
        }
    )
    return _with_context(state, context)


async def prepare_approval_wait(state: GraphState, config: RunnableConfig, runtime: ApprovalRuntime) -> GraphState:
    """Persist request/audit then one idempotent pending domain transition."""
    request = _request(state, config)
    call = request.binding.invocation
    context = await runtime.resolver.resolve(call, config)
    descriptor = runtime.registry.descriptor(call.tool_name, call.version)
    entry = await runtime.service.create(request, context, descriptor, at=runtime.clock())
    run = await runtime.service.domain.get_run(RunLookup(run_id=call.run_id))
    command = approval_domain_command(
        run,
        entry.pending,
        command_id=request.approval_request_id + ":pending_commit",
        expected_state_version=request.expected_state_version,
    )
    await runtime.service.domain.commit_run_update(command)
    current = await runtime.service.domain.get_run(RunLookup(run_id=call.run_id))
    if current.state_version != request.expected_state_version + 1 or current.approval != entry.pending:
        raise DomainError("approval_domain_fork")
    return _copy_state(state, current)


async def await_approval_decision(state: GraphState, config: RunnableConfig, runtime: ApprovalRuntime) -> GraphState:
    """Treat resume as intent and re-resolve identity, policy, Gate and history."""
    request = _request(state, config)
    graph_context = _context(state, config)
    call = request.binding.invocation
    actor = await runtime.resolver.resolve(call, config)
    entry = await runtime.service.approvals.get(request.approval_request_id)
    if entry.request != request:
        raise DomainError("approval_checkpoint_fork")
    # This check is independent of the decision payload and also protects cancellation.
    runtime.service._check_actor(entry, actor, at=runtime.clock())
    if not entry.invalidations and runtime.clock() >= request.expires_at:
        entry = await runtime.service.invalidate(
            request.approval_request_id, actor, ApprovalDecision.EXPIRED, "approval_expired", at=runtime.clock()
        )
    if entry.invalidations:
        outcome = "expired" if entry.invalidations[0].decision is ApprovalDecision.EXPIRED else "revoked"
        return _with_context(state, graph_context.model_copy(update={"approval_outcome": outcome}))
    supplied = interrupt(
        {
            "kind": "releaseguard.approval.v1",
            "request": request.model_dump(mode="json"),
            "expected_state_version": request.expected_state_version + 1,
        }
    )
    try:
        command = ApprovalCommand.model_validate(supplied)
    except ValidationError:
        raise DomainError("approval_resume_payload_invalid") from None
    if command.approval_request_id != request.approval_request_id:
        raise DomainError("approval_resume_binding_mismatch")
    if config.get("metadata", {}).get("approval_command_hash") != canonical_hash(command):
        raise DomainError("approval_resume_intent_conflict")
    try:
        descriptor = runtime.registry.descriptor(call.tool_name, call.version)
        decided = await runtime.service.decide(command, actor, descriptor, at=runtime.clock())
        if decided.decision is None:
            raise DomainError("approval_decision_missing")
        run = await runtime.service.domain.get_run(RunLookup(run_id=call.run_id))
        domain_command = approval_domain_command(
            run,
            decided.decision,
            command_id=command.command_id,
            expected_state_version=command.expected_state_version,
        )
        await runtime.service.domain.commit_run_update(domain_command)
        current = await runtime.service.domain.get_run(RunLookup(run_id=call.run_id))
        transition = next((item for item in current.transition_history if item.command_id == command.command_id), None)
        if (
            current.state_version != command.expected_state_version + 1
            or transition is None
            or transition.audit.input_hash != domain_command.audit.input_hash
        ):
            raise DomainError("approval_domain_fork")
        return _copy_state(state, current, outcome="decided")
    except DomainError as error:
        await record_audit(
            runtime.service.approvals,
            actor.invocation,
            "consistency_stopped",
            error.code,
            at=runtime.clock(),
            discriminator=command.command_id,
            authorization_hash=request.authorization_hash,
        )
        raise


async def resume_approval(graph: CompiledStateGraph, command: ApprovalCommand, config: RunnableConfig) -> dict:
    """Supported entry rejects absent/completed/wrong checkpoint before invoking Graph."""
    command = ApprovalCommand.model_validate_json(command.model_dump_json())
    configurable = config.get("configurable", {})
    if configurable.get("checkpoint_id") or configurable.get("checkpoint_ns"):
        raise DomainError("approval_checkpoint_fork")
    snapshot = await graph.aget_state(config)
    waiting = [task for task in snapshot.tasks if task.name == "releaseguard_approval_wait"]
    # LangGraph's latest-state view applies pending resume writes: a failed
    # interrupted task can have next=() while retaining its error/interrupt.
    failed_wait = (
        snapshot.next == ()
        and len(snapshot.tasks) == 1
        and len(waiting) == 1
        and waiting[0].error
        and waiting[0].interrupts
    )
    if (snapshot.next != ("releaseguard_approval_wait",) and not failed_wait) or not snapshot.values:
        raise DomainError("approval_checkpoint_missing_or_completed")
    state = GraphState.model_validate(snapshot.values)
    request = _request(state, config)
    if (
        command.approval_request_id != request.approval_request_id
        or command.authorization_hash != request.authorization_hash
        or command.expected_state_version != request.expected_state_version + 1
    ):
        raise DomainError("approval_resume_binding_mismatch")
    if not any(
        task.name == "releaseguard_approval_wait" and (task.interrupts or task.error) for task in snapshot.tasks
    ):
        raise DomainError("approval_checkpoint_missing_or_completed")
    resume_config: RunnableConfig = {
        **config,
        "metadata": {**config.get("metadata", {}), "approval_command_hash": canonical_hash(command)},
    }
    return await graph.ainvoke(Command(resume=command.model_dump(mode="json")), resume_config)
