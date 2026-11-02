"""Offline async repository contracts using synthetic identities and observations."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.repositories.releaseguard import (
    ChangeLookup,
    IdempotencyLookup,
    RecordLookup,
    ReleaseGuardRepository,
    RevisionAppend,
    RunLookup,
    SnapshotPayload,
)
from app.schemas.releaseguard import (
    ChangeRequest,
    ChangeRequestInput,
    ChangeRunState,
    ChangeStatus,
    EvidenceAssessment,
    new_change_run,
    record_key,
    request_input_hash,
)
from app.services.releaseguard.state_machine import DomainError
from releaseguard_fixtures import audit_fixture, command_fixture, edge_inputs, event_fixture, request_fixture


PATH = [
    ChangeStatus.COLLECTING_EVIDENCE,
    ChangeStatus.EVIDENCE_SUFFICIENT,
    ChangeStatus.RISK_ASSESSED,
    ChangeStatus.PLAN_READY,
    ChangeStatus.WAITING_FOR_APPROVAL,
    ChangeStatus.APPROVED,
    ChangeStatus.DRY_RUN,
    ChangeStatus.EXECUTING_SANDBOX,
    ChangeStatus.VERIFYING,
    ChangeStatus.SUCCEEDED,
    ChangeStatus.RECEIPTED,
]


def variant(**overrides: Any) -> ChangeRequest:
    """Recompute business hashes for alternate synthetic identities or revisions."""
    values = request_fixture().model_dump()
    values.update(overrides)
    body = ChangeRequestInput.model_validate({key: values[key] for key in ChangeRequestInput.model_fields})
    values["request_hash"] = request_input_hash(body)
    return ChangeRequest.model_validate(values)


async def setup_run(repo: ReleaseGuardRepository) -> ChangeRunState:
    """Use only the protocol to create a new current-revision run."""
    request = await repo.create_change(request_fixture())
    return await repo.create_run(new_change_run(request, "synthetic-run"))


async def reach(repo: ReleaseGuardRepository, target: ChangeStatus) -> ChangeRunState:
    """Persist a legal synthetic path, including each original command result."""
    state = await setup_run(repo)
    for status in PATH[: PATH.index(target) + 1]:
        state = await repo.commit_run_update(command_fixture(state, status, **edge_inputs(state, status)))
    return state


def test_protocol_creation_idempotency_and_requester_scope() -> None:
    """Same key/body returns the original object, while requester scope is isolated."""

    async def scenario() -> None:
        repo: ReleaseGuardRepository = InMemoryReleaseGuardRepository()
        original = await repo.create_change(request_fixture())
        replay = await repo.create_change(variant(change_id="different-generated-id"))
        assert replay == original
        with pytest.raises(DomainError, match="idempotency_conflict"):
            await repo.create_change(variant(title="synthetic conflicting body"))
        other = await repo.create_change(variant(change_id="other-change", requester_id="other-requester"))
        assert other.change_id != original.change_id
        assert (
            await repo.find_by_idempotency(
                IdempotencyLookup(requester_id=original.requester_id, idempotency_key=original.idempotency_key)
            )
            == original
        )
        assert await repo.get_change(ChangeLookup(change_id=original.change_id, revision=1)) == original

    asyncio.run(scenario())


def test_concurrent_same_key_creates_one_logical_object() -> None:
    """Concurrent equivalent creates never produce duplicate identity entries."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        results = await asyncio.gather(
            *(repo.create_change(variant(change_id="generated-" + str(i))) for i in range(12))
        )
        assert all(result == results[0] for result in results)
        assert len(json.loads((await repo.export_snapshot()).data)["changes"]) == 1

    asyncio.run(scenario())


def test_concurrent_conflicting_creates_have_one_winner() -> None:
    """A different business input cannot overwrite the first creation."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        results = await asyncio.gather(
            repo.create_change(request_fixture()),
            repo.create_change(variant(title="synthetic alternate")),
            return_exceptions=True,
        )
        assert sum(isinstance(result, ChangeRequest) for result in results) == 1
        assert [result.code for result in results if isinstance(result, DomainError)] == ["idempotency_conflict"]
        assert len(json.loads((await repo.export_snapshot()).data)["changes"]) == 1

    asyncio.run(scenario())


def test_identity_conflicts_and_missing_reads_are_stable() -> None:
    """Missing lookups never create entries, and explicit identities cannot collide."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        state = await setup_run(repo)
        with pytest.raises(DomainError, match="identity_conflict"):
            await repo.create_change(variant(idempotency_key="different-key"))
        with pytest.raises(DomainError, match="identity_conflict"):
            await repo.create_run(state)
        missing: list[Callable[[], Awaitable[object]]] = [
            lambda: repo.get_change(ChangeLookup(change_id="missing")),
            lambda: repo.get_change(ChangeLookup(change_id=state.change_id, revision=2)),
            lambda: repo.get_run(RunLookup(run_id="missing")),
            lambda: repo.get_record(RecordLookup(run_id=state.run_id, record_key="event:missing")),
            lambda: repo.find_by_idempotency(IdempotencyLookup(requester_id="missing", idempotency_key="missing")),
        ]
        before = await repo.export_snapshot()
        for operation in missing:
            with pytest.raises(DomainError, match="not_found"):
                await operation()
        assert await repo.export_snapshot() == before

    asyncio.run(scenario())


def test_concurrent_state_updates_use_compare_and_swap() -> None:
    """Only one command against the old state_version can append a transition."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        state = await setup_run(repo)
        results = await asyncio.gather(
            *(
                repo.commit_run_update(
                    command_fixture(state, ChangeStatus.COLLECTING_EVIDENCE, command_id="concurrent-" + str(i))
                )
                for i in range(8)
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(result, ChangeRunState) for result in results) == 1
        assert [result.code for result in results if isinstance(result, DomainError)] == ["version_conflict"] * 7
        history = await repo.get_run_history(RunLookup(run_id=state.run_id))
        assert len(history.transitions) == 1

    asyncio.run(scenario())


def test_failed_command_has_no_partial_writes() -> None:
    """Rejected commands append neither records nor replay results nor revisions."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        state = await reach(repo, ChangeStatus.COLLECTING_EVIDENCE)
        before = await repo.export_snapshot()
        with pytest.raises(DomainError, match="transition_precondition_failed"):
            await repo.commit_run_update(command_fixture(state, ChangeStatus.EVIDENCE_SUFFICIENT))
        assert await repo.export_snapshot() == before

    asyncio.run(scenario())


def test_reads_and_original_replay_results_are_isolated() -> None:
    """Old replay returns the original result without reverting the current store."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        initial = await setup_run(repo)
        first = command_fixture(initial, ChangeStatus.COLLECTING_EVIDENCE)
        first_result = await repo.commit_run_update(first)
        second = command_fixture(
            first_result,
            ChangeStatus.EVIDENCE_SUFFICIENT,
            **edge_inputs(first_result, ChangeStatus.EVIDENCE_SUFFICIENT),
        )
        latest = await repo.commit_run_update(second)
        replay = await repo.commit_run_update(first)
        assert replay == first_result and replay != latest
        replay.errors.append("synthetic caller mutation")
        latest.evidence_items.clear()
        history = await repo.get_run_history(RunLookup(run_id=initial.run_id))
        key = record_key(history.records[0])
        history.records.clear()
        history.transitions.clear()
        record = await repo.get_record(RecordLookup(run_id=initial.run_id, record_key=key))
        assert isinstance(record, EvidenceAssessment)
        record.evidence_ids.clear()
        stored = await repo.get_run(RunLookup(run_id=initial.run_id))
        assert stored.state_version == 2 and stored.evidence_items and not stored.errors
        assert stored.evidence_assessment is not None and stored.evidence_assessment.evidence_ids
        assert len((await repo.get_run_history(RunLookup(run_id=initial.run_id))).transitions) == 2
        with pytest.raises(DomainError, match="command_conflict"):
            await repo.commit_run_update(
                command_fixture(
                    initial,
                    ChangeStatus.COLLECTING_EVIDENCE,
                    audit=audit_fixture(reason="synthetic different payload").model_dump(exclude={"input_hash"}),
                )
            )

    asyncio.run(scenario())


def test_event_replay_is_append_only_and_conflict_is_atomic() -> None:
    """Same event payload replays, while changing an existing event id fails."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        state = await reach(repo, ChangeStatus.VERIFYING)
        query = RecordLookup(run_id=state.run_id, record_key="event:synthetic-complete")
        assert await repo.get_record(query) == event_fixture()
        before = await repo.export_snapshot()
        changed = event_fixture(
            reason="synthetic changed observation", audit=audit_fixture(reason="synthetic changed observation")
        )
        with pytest.raises(DomainError, match="event_conflict"):
            await repo.commit_run_update(
                command_fixture(
                    state,
                    ChangeStatus.FAILED,
                    inputs={"execution_events": [changed]},
                    audit=audit_fixture(reason="synthetic terminal failure").model_dump(exclude={"input_hash"}),
                )
            )
        assert await repo.export_snapshot() == before
        result = await repo.commit_run_update(
            command_fixture(
                state,
                ChangeStatus.FAILED,
                inputs={"execution_events": state.execution_events},
                audit=audit_fixture(reason="synthetic terminal failure").model_dump(exclude={"input_hash"}),
            )
        )
        assert result.execution_events == state.execution_events
        assert await repo.get_record(query) == event_fixture()

    asyncio.run(scenario())


def test_revision_append_preserves_creation_and_blocks_stale_run_writes() -> None:
    """Global revision CAS preserves the original idempotent creation result."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        state = await setup_run(repo)
        revised = variant(revision=2, title="synthetic revised input")
        await repo.append_revision(RevisionAppend(request=revised, expected_revision=1))
        assert await repo.get_change(ChangeLookup(change_id=state.change_id)) == revised
        assert await repo.create_change(request_fixture()) == state.change_request
        before = await repo.export_snapshot()
        with pytest.raises(DomainError, match="revision_conflict"):
            await repo.append_revision(RevisionAppend(request=revised, expected_revision=1))
        with pytest.raises(DomainError, match="revision_conflict"):
            await repo.commit_run_update(command_fixture(state, ChangeStatus.COLLECTING_EVIDENCE))
        with pytest.raises(DomainError, match="revision_conflict"):
            await repo.create_run(new_change_run(state.change_request, "stale-run"))
        assert await repo.export_snapshot() == before
        await repo.create_run(new_change_run(revised, "revised-run"))

    asyncio.run(scenario())


def test_revise_command_commits_request_and_run_together() -> None:
    """A domain revision and its invalidation history share one storage transaction."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        state = await reach(repo, ChangeStatus.APPROVED)
        revised = variant(revision=2, title="synthetic revised input")
        result = await repo.commit_run_update(command_fixture(state, action="revise", replacement_request=revised))
        assert result.status is ChangeStatus.RECEIVED and result.approval is None
        assert result.revision == 2
        assert await repo.get_change(ChangeLookup(change_id=state.change_id)) == result.change_request
        assert await repo.get_change(ChangeLookup(change_id=state.change_id, revision=1)) == state.change_request
        restored = InMemoryReleaseGuardRepository()
        await restored.restore_snapshot(await repo.export_snapshot())
        assert await restored.get_run(RunLookup(run_id=state.run_id)) == result

    asyncio.run(scenario())


def test_active_run_blocks_global_revision_append_without_partial_writes() -> None:
    """A second run cannot revise the request under an active synthetic execution."""

    async def scenario() -> None:
        repo = InMemoryReleaseGuardRepository()
        state = await reach(repo, ChangeStatus.DRY_RUN)
        before = await repo.export_snapshot()
        with pytest.raises(DomainError, match="revision_conflict"):
            await repo.append_revision(RevisionAppend(request=variant(revision=2), expected_revision=1))
        assert await repo.get_run(RunLookup(run_id=state.run_id)) == state
        assert await repo.export_snapshot() == before

    asyncio.run(scenario())


def test_snapshot_roundtrip_continuation_and_replay() -> None:
    """Restored objects retain hashes, history, exact command results and CAS."""

    async def scenario() -> None:
        source = InMemoryReleaseGuardRepository()
        state = await reach(source, ChangeStatus.SUCCEEDED)
        payload = await source.export_snapshot()
        restored: ReleaseGuardRepository = InMemoryReleaseGuardRepository()
        summary = await restored.restore_snapshot(payload)
        assert (summary.changes, summary.runs, summary.commands) == (1, 1, state.state_version)
        assert await restored.export_snapshot() == payload
        original = json.loads(payload.data)["command_results"][0]
        replay = command_fixture(new_change_run(request_fixture(), state.run_id), ChangeStatus.COLLECTING_EVIDENCE)
        assert (await restored.commit_run_update(replay)).model_dump(mode="json") == original["state"]
        assert await restored.get_run(RunLookup(run_id=state.run_id)) == state
        command = command_fixture(state, ChangeStatus.RECEIPTED, **edge_inputs(state, ChangeStatus.RECEIPTED))
        receipt = await restored.commit_run_update(command)
        assert receipt.receipt is not None and receipt.receipt.final_status.value == "SUCCEEDED"
        assert await restored.commit_run_update(command) == receipt
        assert await restored.restore_snapshot(await restored.export_snapshot()) == summary.model_copy(
            update={"commands": summary.commands + 1}
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "tamper,code",
    [
        ("version", "unsupported_contract_version"),
        ("nested_version", "unsupported_contract_version"),
        ("request_hash", "hash_mismatch"),
        ("plan_hash", "hash_mismatch"),
        ("foreign_run", "invalid_snapshot"),
        ("transition", "invalid_snapshot"),
        ("result", "invalid_snapshot"),
        ("missing_command", "invalid_snapshot"),
        ("orphan_command", "invalid_snapshot"),
        ("duplicate_request", "invalid_snapshot"),
        ("duplicate_command", "command_conflict"),
    ],
)
def test_snapshot_tampering_is_rejected_atomically(tamper: str, code: str) -> None:
    """Invalid versions, hashes, bindings or replay chains never install any data."""

    async def scenario() -> None:
        source = InMemoryReleaseGuardRepository()
        await reach(source, ChangeStatus.PLAN_READY)
        values = json.loads((await source.export_snapshot()).data)
        if tamper == "version":
            values["contract_version"] = "releaseguard.v999"
        elif tamper == "nested_version":
            values["runs"][0]["execution_plan"]["audit"]["contract_version"] = "releaseguard.v999"
        elif tamper == "request_hash":
            values["changes"][0]["request_hash"] = "0" * 64
        elif tamper == "plan_hash":
            values["runs"][0]["execution_plan"]["plan_hash"] = "0" * 64
        elif tamper == "foreign_run":
            values["runs"][0]["record_history"][0]["run_id"] = "foreign-run"
        elif tamper == "transition":
            values["runs"][0]["transition_history"][0]["to_status"] = "SUCCEEDED"
        elif tamper == "result":
            values["command_results"][0]["state"]["errors"] = ["synthetic forged result"]
        elif tamper == "missing_command":
            values["command_results"].pop()
        elif tamper == "orphan_command":
            values["runs"] = []
        elif tamper == "duplicate_request":
            values["changes"].append(values["changes"][0])
        elif tamper == "duplicate_command":
            values["command_results"].append(values["command_results"][0])
        target = InMemoryReleaseGuardRepository()
        empty = await target.export_snapshot()
        with pytest.raises(DomainError, match=code):
            await target.restore_snapshot(SnapshotPayload(data=json.dumps(values)))
        assert await target.export_snapshot() == empty

    asyncio.run(scenario())


def test_restore_never_overwrites_an_existing_repository() -> None:
    """Different snapshots conflict instead of replacing local work or audit history."""

    async def scenario() -> None:
        source = InMemoryReleaseGuardRepository()
        await setup_run(source)
        target = InMemoryReleaseGuardRepository()
        await target.create_change(variant(change_id="other-change", idempotency_key="other-key"))
        before = await target.export_snapshot()
        with pytest.raises(DomainError, match="snapshot_conflict"):
            await target.restore_snapshot(await source.export_snapshot())
        assert await target.export_snapshot() == before

    asyncio.run(scenario())
