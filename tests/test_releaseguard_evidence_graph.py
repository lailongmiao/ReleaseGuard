"""Offline integration tests for the Phase 4 Evidence path in one Graph."""

import asyncio
from datetime import UTC, datetime

import pytest
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from app.core.langgraph.graph import build_agent_graph
from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config
from app.repositories.in_memory_evidence import InMemoryEvidenceRepository
from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.repositories.releaseguard import RunLookup
from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceItem,
    EvidencePolicy,
    EvidenceRequirement,
    EvidenceSourceResult,
    EvidenceSourceType,
)
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import ChangeStatus, RunCommand, new_change_run
from app.services.evidence_sources import InMemoryEvidenceSourceAdapter
from tests.releaseguard_fixtures import command_fixture
from tests.test_change_request_contract import request_fixture


CAPTURED_AT = datetime(2026, 1, 1, tzinfo=UTC)


class FailBeforeEvidenceCommitRepository(InMemoryReleaseGuardRepository):
    """Fail one domain commit after the Evidence write has already completed."""

    def __init__(self) -> None:
        """Enable one deterministic pre-commit failure."""
        super().__init__()
        self.fail_once = True

    async def commit_run_update(self, command: RunCommand):
        """Leave domain state untouched once, then accept the exact retry."""
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("synthetic_evidence_domain_commit_failure")
        return await super().commit_run_update(command)


def item_fixture() -> EvidenceItem:
    """Build the only source item used by this synthetic integration."""
    return EvidenceItem(
        evidence_id="synthetic:ci",
        source_type=EvidenceSourceType.CI,
        source_uri="fixture://ci/build-1",
        source_version="build-1",
        captured_at=CAPTURED_AT,
        content_hash="a" * 64,
        access_scope=EvidenceAccessScope(public=True),
        snippet="synthetic build passed",
        confidence=1.0,
    )


def policy_fixture() -> EvidencePolicy:
    """Require one CI observation for the graph's deterministic Gate."""
    return EvidencePolicy(
        policy_version="synthetic-policy.v1",
        requirements=[
            EvidenceRequirement(
                requirement_id="ci",
                source_types=frozenset({EvidenceSourceType.CI}),
                min_confidence=0.8,
            )
        ],
    )


def source_fixture(*, items: list[EvidenceItem]) -> EvidenceSourceResult:
    """Build a verified source response without network or provider access."""
    return EvidenceSourceResult(
        source_id="synthetic-ci-source",
        source_type=EvidenceSourceType.CI,
        requirement_id="ci",
        collection_status="verified",
        items=items,
        source_version="build-1",
        captured_at=CAPTURED_AT,
        audit_reference="synthetic-source-audit",
    )


async def chat_passthrough(state: GraphState, config: RunnableConfig) -> dict[str, str]:
    """Avoid Provider access in the offline Graph fixture."""
    del config
    return {"long_term_memory": state.long_term_memory}


async def tool_passthrough(state: GraphState) -> dict[str, list[object]]:
    """Avoid tool execution in the offline Graph fixture."""
    return {"messages": state.messages}


async def collecting_state(repository: InMemoryReleaseGuardRepository, run_id: str):
    """Create and persist the Phase 2 collecting-evidence state."""
    request = request_fixture()
    await repository.create_change(request)
    run = new_change_run(request, run_id)
    await repository.create_run(run)
    return await repository.commit_run_update(command_fixture(run, ChangeStatus.COLLECTING_EVIDENCE))


def test_graph_collects_persists_assesses_and_commits_evidence() -> None:
    """One ReleaseGuard Graph writes Evidence first and then the domain assessment."""

    async def scenario() -> None:
        domain_repository = InMemoryReleaseGuardRepository()
        run = await collecting_state(domain_repository, "evidence-run")
        request = run.change_request
        source = InMemoryEvidenceSourceAdapter(source_fixture(items=[item_fixture()]))
        evidence_repository = InMemoryEvidenceRepository()
        state = GraphState(
            workflow_kind="releaseguard",
            releaseguard=ReleaseGuardGraphContext(
                change_request=request,
                run_id=run.run_id,
                expected_revision=request.revision,
                state_version=run.state_version,
                domain_state=run,
                evidence_policy=policy_fixture(),
                evidence_evaluated_at=CAPTURED_AT,
            ),
        )
        graph = build_agent_graph(
            chat_passthrough,
            tool_passthrough,
            InMemorySaver(),
            domain_repository,
            evidence_repository,
            {"ci": source},
        )

        result = await graph.ainvoke(
            state,
            releaseguard_runnable_config(request.change_id, run.run_id, request.revision),
        )

        context = result["releaseguard"]
        assert context.domain_state.status is ChangeStatus.EVIDENCE_SUFFICIENT
        assert context.evidence_assessment is not None
        assert context.evidence_assessment.sufficient is True
        assert len(context.evidence_snapshots) == 1
        assert len(context.evidence_citations) == 1
        stored = await domain_repository.get_run(RunLookup(run_id=run.run_id))
        assert stored.status is ChangeStatus.EVIDENCE_SUFFICIENT
        assert len(stored.transition_history) == 2
        assert await evidence_repository.get_snapshot(context.evidence_snapshots[0].snapshot_id)

    asyncio.run(scenario())


def test_graph_missing_adapter_commits_explicit_waiting_evidence() -> None:
    """An unconfigured required source remains a deterministic waiting outcome."""

    async def scenario() -> None:
        domain_repository = InMemoryReleaseGuardRepository()
        run = await collecting_state(domain_repository, "missing-source-run")
        request = run.change_request
        state = GraphState(
            workflow_kind="releaseguard",
            releaseguard=ReleaseGuardGraphContext(
                change_request=request,
                run_id=run.run_id,
                expected_revision=request.revision,
                state_version=run.state_version,
                domain_state=run,
                evidence_policy=policy_fixture(),
                evidence_evaluated_at=CAPTURED_AT,
            ),
        )
        graph = build_agent_graph(
            chat_passthrough,
            tool_passthrough,
            InMemorySaver(),
            domain_repository,
            InMemoryEvidenceRepository(),
        )

        result = await graph.ainvoke(
            state,
            releaseguard_runnable_config(request.change_id, run.run_id, request.revision),
        )

        context = result["releaseguard"]
        assert context.domain_state.status is ChangeStatus.WAITING_FOR_EVIDENCE
        assert context.evidence_assessment is not None
        assert context.evidence_assessment.sufficient is False
        assert context.routing_outcome == "awaiting_evidence"

    asyncio.run(scenario())


def test_evidence_snapshot_survives_domain_commit_failure_and_replays_safely() -> None:
    """A domain failure leaves an auditable snapshot and a retryable checkpoint."""

    async def scenario() -> None:
        domain_repository = FailBeforeEvidenceCommitRepository()
        domain_repository.fail_once = False
        run = await collecting_state(domain_repository, "evidence-retry-run")
        domain_repository.fail_once = True
        request = run.change_request
        evidence_repository = InMemoryEvidenceRepository()
        source = InMemoryEvidenceSourceAdapter(source_fixture(items=[item_fixture()]))
        state = GraphState(
            workflow_kind="releaseguard",
            releaseguard=ReleaseGuardGraphContext(
                change_request=request,
                run_id=run.run_id,
                expected_revision=request.revision,
                state_version=run.state_version,
                domain_state=run,
                evidence_policy=policy_fixture(),
                evidence_evaluated_at=CAPTURED_AT,
            ),
        )
        config = releaseguard_runnable_config(request.change_id, run.run_id, request.revision)
        graph = build_agent_graph(
            chat_passthrough,
            tool_passthrough,
            InMemorySaver(),
            domain_repository,
            evidence_repository,
            {"ci": source},
        )

        with pytest.raises(RuntimeError, match="synthetic_evidence_domain_commit_failure"):
            await graph.ainvoke(state, config)

        snapshots = await evidence_repository.find_snapshots("synthetic:ci")
        assert len(snapshots) == 1
        stored = await domain_repository.get_run(RunLookup(run_id=run.run_id))
        assert stored.status is ChangeStatus.COLLECTING_EVIDENCE

        resumed = await graph.ainvoke(None, config)

        context = resumed["releaseguard"]
        assert context.domain_state.status is ChangeStatus.EVIDENCE_SUFFICIENT
        assert len((await domain_repository.get_run_history(RunLookup(run_id=run.run_id))).transitions) == 2

    asyncio.run(scenario())
