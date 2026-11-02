"""ReleaseGuard Graph node boundaries backed by the Phase 2 repository."""

from collections.abc import Mapping
from typing import Any

import structlog
from langchain_core.runnables.config import RunnableConfig

from app.core.langgraph.releaseguard_identity import validate_releaseguard_thread_id
from app.repositories.evidence import EvidenceRepository
from app.repositories.releaseguard import (
    ChangeLookup,
    ReleaseGuardRepository,
    RunLookup,
)
from app.schemas.evidence import (
    EvidenceCitation,
    EvidenceCollectionStatus,
    EvidenceSourceResult,
    EvidenceSourceType,
    EvidenceSnapshot,
)
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import (
    ActorType,
    AuditContext,
    ChangeRunState,
    ChangeStatus,
    RunCommandInput,
    StateInputs,
    build_run_command,
    new_change_run,
)
from app.services.evidence_sources import (
    EvidenceAccessContext,
    EvidenceSourceAdapter,
    EvidenceSourceRequest,
    collect_source,
)
from app.services.releaseguard.evidence_policy import evaluate_evidence_sufficiency
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError


logger = structlog.get_logger(__name__)


def _context(state: GraphState, config: RunnableConfig) -> ReleaseGuardGraphContext:
    """Validate workflow, checkpoint thread, and non-sensitive trace binding."""
    if state.workflow_kind != "releaseguard" or state.releaseguard is None:
        raise DomainError("releaseguard_workflow_required")
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id")
    if not isinstance(thread_id, str):
        raise DomainError("releaseguard_thread_binding_mismatch")
    try:
        validate_releaseguard_thread_id(state, thread_id)
    except ValueError:
        raise DomainError("releaseguard_thread_binding_mismatch") from None

    metadata = config.get("metadata", {})
    request = state.releaseguard.change_request
    expected_metadata: dict[str, Any] = {
        "workflow_kind": "releaseguard",
        "change_id": request.change_id,
        "run_id": state.releaseguard.run_id,
        "revision": state.releaseguard.expected_revision,
    }
    checked_metadata = {**metadata, "run_id": metadata.get("domain_run_id", metadata.get("run_id"))}
    if any(checked_metadata.get(key) != value for key, value in expected_metadata.items()):
        raise DomainError("releaseguard_metadata_binding_mismatch")
    logger.info("releaseguard_context_validated", **expected_metadata, thread_id=thread_id)
    return state.releaseguard


def _with_context(state: GraphState, context: ReleaseGuardGraphContext) -> GraphState:
    """Revalidate a complete GraphState after applying typed repository output."""
    values = state.model_dump(mode="python")
    values["releaseguard"] = context.model_dump(mode="python")
    return GraphState.model_validate(values)


def _validate_run_identity(context: ReleaseGuardGraphContext, run: ChangeRunState) -> None:
    """Require a repository result to remain bound to the requested run/revision."""
    request = context.change_request
    if (run.change_id, run.run_id) != (request.change_id, context.run_id):
        raise DomainError("identity_conflict")
    if run.revision != context.expected_revision:
        raise DomainError("revision_conflict")
    if run.request_hash != request.request_hash or run.change_request != request:
        raise DomainError("identity_conflict")


def _snapshot_for_item(item: Any, persisted_at: Any) -> EvidenceSnapshot:
    """Build the immutable snapshot identity without retaining source payloads."""
    snapshot_id = "sha256:" + canonical_hash(
        {
            "evidence_id": item.evidence_id,
            "source_uri": item.source_uri,
            "source_version": item.source_version,
            "content_hash": item.content_hash,
        }
    )
    return EvidenceSnapshot(snapshot_id=snapshot_id, item=item, persisted_at=persisted_at)


def _citation_for_snapshot(
    snapshot: EvidenceSnapshot,
    *,
    requirement_id: str,
    context: ReleaseGuardGraphContext,
    thread_id: str,
    policy_version: str,
    cited_at: Any,
) -> EvidenceCitation:
    """Create a deterministic citation bound to the current graph identity."""
    item = snapshot.item
    identity = {
        "snapshot_id": snapshot.snapshot_id,
        "requirement_id": requirement_id,
        "change_id": context.change_request.change_id,
        "run_id": context.run_id,
        "revision": context.expected_revision,
        "thread_id": thread_id,
        "policy_version": policy_version,
    }
    return EvidenceCitation(
        citation_id="evidence-citation:" + canonical_hash(identity),
        snapshot_id=snapshot.snapshot_id,
        evidence_id=item.evidence_id,
        source_version=item.source_version,
        content_hash=item.content_hash,
        captured_at=item.captured_at,
        required_item=requirement_id,
        change_id=context.change_request.change_id,
        run_id=context.run_id,
        revision=context.expected_revision,
        request_hash=context.change_request.request_hash,
        thread_id=thread_id,
        policy_version=policy_version,
        cited_at=cited_at,
    )


async def collect_and_assess_evidence(
    state: GraphState,
    repository: EvidenceRepository | None,
    adapters: Mapping[str, EvidenceSourceAdapter],
    access: EvidenceAccessContext | None,
    config: RunnableConfig,
) -> GraphState:
    """Collect read-only evidence, persist snapshots, and prepare one domain command."""
    context = _context(state, config)
    if context.domain_state is None or context.evidence_policy is None:
        raise DomainError("evidence_precondition_failed")
    if context.domain_state.status is not ChangeStatus.COLLECTING_EVIDENCE:
        raise DomainError("evidence_status_precondition_failed")
    evaluated_at = context.evidence_evaluated_at
    if evaluated_at is None or evaluated_at.tzinfo is None:
        raise DomainError("evidence_evaluation_time_required")
    if repository is None:
        raise DomainError("evidence_repository_unavailable")
    thread_id = config.get("configurable", {}).get("thread_id")
    if not isinstance(thread_id, str):
        raise DomainError("releaseguard_thread_binding_mismatch")
    access_context = access or EvidenceAccessContext(principal_id=context.change_request.requester_id)
    collected_results: list[EvidenceSourceResult] = []
    persisted: dict[str, EvidenceSnapshot] = {
        snapshot.snapshot_id: snapshot for snapshot in context.evidence_snapshots
    }
    citations: dict[str, EvidenceCitation] = {
        citation.citation_id: citation for citation in context.evidence_citations
    }
    for requirement in context.evidence_policy.requirements:
        adapter = adapters.get(requirement.requirement_id)
        if adapter is None:
            source_type = next(iter(requirement.source_types), EvidenceSourceType.DOCUMENT)
            collected_results.append(
                EvidenceSourceResult(
                    source_id="unconfigured-source",
                    source_type=source_type,
                    requirement_id=requirement.requirement_id,
                    collection_status=EvidenceCollectionStatus.UNAVAILABLE,
                    captured_at=evaluated_at,
                    audit_reference="source-unconfigured",
                    degradation_reason="source_adapter_unavailable",
                )
            )
            continue
        request = EvidenceSourceRequest(
            principal_id=access_context.principal_id,
            tenant_id=access_context.tenant_id,
            roles=access_context.roles,
            change_id=context.change_request.change_id,
            run_id=context.run_id,
            revision=context.expected_revision,
            requirement_id=requirement.requirement_id,
            query_hash=context.change_request.request_hash,
        )
        result = await collect_source(adapter, request, captured_at=evaluated_at)
        collected_results.append(result)
        for item in result.items:
            snapshot = _snapshot_for_item(item, result.captured_at)
            persisted[snapshot.snapshot_id] = await repository.append_snapshot(snapshot)
            citation = _citation_for_snapshot(
                persisted[snapshot.snapshot_id],
                requirement_id=requirement.requirement_id,
                context=context,
                thread_id=thread_id,
                policy_version=context.evidence_policy.policy_version,
                cited_at=evaluated_at,
            )
            citations[citation.citation_id] = await repository.append_citation(citation)
    snapshots = list(persisted.values())
    citation_values = list(citations.values())
    assessment = evaluate_evidence_sufficiency(
        context.domain_state,
        snapshots=snapshots,
        citations=citation_values,
        policy=context.evidence_policy,
        evaluated_at=evaluated_at,
        thread_id=thread_id,
        access=access_context,
        source_results=collected_results,
    )
    command_id = "evidence-assessment:" + canonical_hash(
        {
            "run_id": context.run_id,
            "state_version": context.domain_state.state_version,
            "assessment": assessment.model_dump(mode="json"),
        }
    )
    command = build_run_command(
        RunCommandInput(
            change_id=context.domain_state.change_id,
            run_id=context.domain_state.run_id,
            command_id=command_id,
            expected_revision=context.domain_state.revision,
            expected_state_version=context.domain_state.state_version,
            target_status=(
                ChangeStatus.EVIDENCE_SUFFICIENT if assessment.sufficient else ChangeStatus.WAITING_FOR_EVIDENCE
            ),
            inputs=StateInputs(
                evidence_items=[snapshot.item for snapshot in snapshots],
                evidence_assessment=assessment,
            ),
            audit=AuditContext(
                actor_id="evidence-gate",
                actor_type=ActorType.SYSTEM,
                reason="evidence_sufficiency_evaluated",
                correlation_id="evidence-gate:" + context.run_id,
                occurred_at=evaluated_at,
                policy_version=context.evidence_policy.policy_version,
            ),
        )
    )
    updated = ReleaseGuardGraphContext.model_validate(
        {
            **context.model_dump(mode="python"),
            "evidence_snapshots": snapshots,
            "evidence_citations": citation_values,
            "evidence_source_results": collected_results,
            "evidence_assessment": assessment,
            "pending_command": command,
        }
    )
    return _with_context(state, updated)


async def register_or_load_change(
    state: GraphState,
    repository: ReleaseGuardRepository,
    config: RunnableConfig,
) -> GraphState:
    """Create by stable requester/key input and reject any silent identity rebind."""
    context = _context(state, config)
    request = context.change_request
    # Creation is idempotent only for revision 1. Later revisions must already
    # exist in the authority; registering them must never create a new request.
    stored = (
        await repository.create_change(request)
        if request.revision == 1
        else await repository.get_change(ChangeLookup(change_id=request.change_id, revision=request.revision))
    )
    if (
        stored.change_id,
        stored.revision,
        stored.request_hash,
        stored.requester_id,
        stored.idempotency_key,
    ) != (
        request.change_id,
        request.revision,
        request.request_hash,
        request.requester_id,
        request.idempotency_key,
    ):
        raise DomainError("identity_conflict")
    updated = ReleaseGuardGraphContext.model_validate({**context.model_dump(mode="python"), "change_request": stored})
    logger.info("releaseguard_change_registered", change_id=stored.change_id, revision=stored.revision)
    return _with_context(state, updated)


async def register_or_load_run(
    state: GraphState,
    repository: ReleaseGuardRepository,
    config: RunnableConfig,
) -> GraphState:
    """Create an empty run or read back and validate an existing run identity."""
    context = _context(state, config)
    initial = new_change_run(context.change_request, context.run_id)
    try:
        run = await repository.create_run(initial)
    except DomainError as error:
        if error.code != "identity_conflict":
            raise
        run = await repository.get_run(RunLookup(run_id=context.run_id))
        _validate_run_identity(context, run)

    latest = await repository.get_change(ChangeLookup(change_id=context.change_request.change_id))
    if latest != context.change_request:
        raise DomainError("revision_conflict")
    _validate_run_identity(context, run)
    updated = ReleaseGuardGraphContext.model_validate(
        {
            **context.model_dump(mode="python"),
            "domain_state": run,
            "state_version": run.state_version,
            "pending_command": context.pending_command,
        }
    )
    logger.info(
        "releaseguard_run_registered",
        change_id=run.change_id,
        run_id=run.run_id,
        revision=run.revision,
        state_version=run.state_version,
    )
    return _with_context(state, updated)


async def load_domain_state(
    state: GraphState,
    repository: ReleaseGuardRepository,
    config: RunnableConfig,
) -> GraphState:
    """Refresh checkpoint cache from the repository's current revision and run."""
    context = _context(state, config)
    latest = await repository.get_change(ChangeLookup(change_id=context.change_request.change_id))
    if latest != context.change_request:
        raise DomainError("revision_conflict")
    run = await repository.get_run(RunLookup(run_id=context.run_id))
    _validate_run_identity(context, run)
    updated = ReleaseGuardGraphContext.model_validate(
        {
            **context.model_dump(mode="python"),
            "domain_state": run,
            "state_version": run.state_version,
        }
    )
    logger.info(
        "releaseguard_domain_state_loaded",
        change_id=run.change_id,
        run_id=run.run_id,
        revision=run.revision,
        state_version=run.state_version,
        status=run.status.value,
    )
    return _with_context(state, updated)


async def commit_domain_command(
    state: GraphState,
    repository: ReleaseGuardRepository,
    config: RunnableConfig,
) -> GraphState:
    """Apply one prepared typed command and copy only the repository result."""
    context = _context(state, config)
    domain_state = context.domain_state
    command = context.pending_command
    if domain_state is None or command is None:
        raise DomainError("command_precondition_failed")
    _validate_run_identity(context, domain_state)
    if (command.change_id, command.run_id, command.expected_revision) != (
        domain_state.change_id,
        domain_state.run_id,
        domain_state.revision,
    ):
        raise DomainError("version_conflict")

    result = await repository.commit_run_update(command)
    _validate_run_identity(context, result)
    if result.state_version != command.expected_state_version + 1:
        raise DomainError("repository_result_mismatch")
    matching_transition = next(
        (record for record in result.transition_history if record.command_id == command.command_id),
        None,
    )
    if matching_transition is None or matching_transition.audit.input_hash != command.audit.input_hash:
        raise DomainError("repository_result_mismatch")

    current = await repository.get_run(RunLookup(run_id=command.run_id))
    _validate_run_identity(context, current)
    current_transition = next(
        (record for record in current.transition_history if record.command_id == command.command_id),
        None,
    )
    if (
        current.state_version < result.state_version
        or current_transition is None
        or current_transition.audit.input_hash != command.audit.input_hash
    ):
        raise DomainError("repository_result_mismatch")
    updated = ReleaseGuardGraphContext.model_validate(
        {
            **context.model_dump(mode="python"),
            "domain_state": current,
            "state_version": current.state_version,
            "pending_command": None,
        }
    )
    logger.info(
        "releaseguard_command_committed",
        change_id=current.change_id,
        run_id=current.run_id,
        revision=current.revision,
        state_version=current.state_version,
        status=current.status.value,
        command_id=command.command_id,
        input_hash=command.audit.input_hash,
    )
    return _with_context(state, updated)
