"""Single-process offline repository; no persistence, credentials or external I/O."""

import asyncio
from collections import defaultdict

from pydantic import ValidationError

from app.core.metrics import observe_fact, observe_run, observe_state
from app.core.logging import logger

from app.repositories.releaseguard import (
    ChangeLookup,
    IdempotencyLookup,
    RecordLookup,
    RepositorySnapshot,
    RepositorySummary,
    RevisionAppend,
    RunHistory,
    RunLookup,
    SnapshotPayload,
    StoredCommandResult,
)
from app.schemas.releaseguard import (
    ChangeRequest,
    ChangeRunState,
    ChangeStatus,
    DomainRecord,
    RunCommand,
    new_change_run,
    record_key,
)
from app.services.releaseguard.state_machine import DomainError, apply_run_command


class InMemoryReleaseGuardRepository:
    """Atomic asynchronous fixture store with isolated reads and exact replay."""

    def __init__(self) -> None:
        """Initialize empty storage and a single-process transaction lock."""
        self._lock = asyncio.Lock()
        self._changes: dict[tuple[str, int], ChangeRequest] = {}
        self._latest: dict[str, int] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._runs: dict[str, ChangeRunState] = {}
        self._results: dict[tuple[str, str], StoredCommandResult] = {}

    async def create_change(self, request: ChangeRequest) -> ChangeRequest:
        """Return the original creation object for equal requester/key/body."""
        request = ChangeRequest.model_validate_json(request.model_dump_json())
        if request.revision != 1:
            raise DomainError("revision_conflict")
        async with self._lock:
            key = (request.requester_id, request.idempotency_key)
            if key in self._keys:
                original = self._changes[(self._keys[key], 1)]
                if original.request_hash != request.request_hash:
                    raise DomainError("idempotency_conflict")
                return original.model_copy(deep=True)
            if request.change_id in self._latest:
                raise DomainError("identity_conflict")
            self._changes[(request.change_id, 1)] = request
            self._latest[request.change_id] = 1
            self._keys[key] = request.change_id
            return request.model_copy(deep=True)

    async def get_change(self, query: ChangeLookup) -> ChangeRequest:
        """Read a specific or latest revision without exposing stored collections."""
        async with self._lock:
            revision = query.revision if query.revision is not None else self._latest.get(query.change_id)
            result = self._changes.get((query.change_id, revision)) if revision is not None else None
            if result is None:
                raise DomainError("not_found")
            return result.model_copy(deep=True)

    async def find_by_idempotency(self, query: IdempotencyLookup) -> ChangeRequest:
        """Read the first creation result even if later revisions exist."""
        async with self._lock:
            change_id = self._keys.get((query.requester_id, query.idempotency_key))
            if change_id is None:
                raise DomainError("not_found")
            return self._changes[(change_id, 1)].model_copy(deep=True)

    def _check_revision(self, command: RevisionAppend) -> None:
        """Validate revision CAS before changing any storage index."""
        request = command.request
        latest = self._latest.get(request.change_id)
        if latest is None:
            raise DomainError("not_found")
        if latest != command.expected_revision or request.revision != latest + 1:
            raise DomainError("revision_conflict")
        old = self._changes[(request.change_id, latest)]
        if (request.requester_id, request.idempotency_key, request.created_at) != (
            old.requester_id,
            old.idempotency_key,
            old.created_at,
        ) or request.updated_at < old.updated_at:
            raise DomainError("revision_conflict")
        active = {
            ChangeStatus.DRY_RUN,
            ChangeStatus.DRY_RUN_PASSED,
            ChangeStatus.EXECUTION_UNKNOWN,
            ChangeStatus.EXECUTING_SANDBOX,
            ChangeStatus.EXECUTION_FAILED,
            ChangeStatus.RECOVERY_REQUIRED,
            ChangeStatus.RECOVERING,
            ChangeStatus.VERIFYING,
        }
        if any(run.change_id == request.change_id and run.status in active for run in self._runs.values()):
            raise DomainError("revision_conflict")

    def _store_revision(self, request: ChangeRequest) -> None:
        """Append a checked immutable revision inside a transaction."""
        self._changes[(request.change_id, request.revision)] = request
        self._latest[request.change_id] = request.revision

    async def append_revision(self, command: RevisionAppend) -> ChangeRequest:
        """Append a new business version; stale run writes then fail closed."""
        command = RevisionAppend.model_validate_json(command.model_dump_json())
        async with self._lock:
            self._check_revision(command)
            self._store_revision(command.request)
            return command.request.model_copy(deep=True)

    async def create_run(self, state: ChangeRunState) -> ChangeRunState:
        """Register only empty runs for an existing current request revision."""
        state = ChangeRunState.model_validate_json(state.model_dump_json())
        async with self._lock:
            if state.run_id in self._runs:
                raise DomainError("identity_conflict")
            request = self._changes.get((state.change_id, state.revision))
            if request is None:
                raise DomainError("not_found")
            if request != state.change_request or self._latest[state.change_id] != state.revision:
                raise DomainError("revision_conflict")
            if state != new_change_run(request, state.run_id):
                raise DomainError("transition_precondition_failed")
            self._runs[state.run_id] = state
            observe_run(change_id=state.change_id, run_id=state.run_id, revision=state.revision)
            observe_state(state)
            return state.model_copy(deep=True)

    async def get_run(self, query: RunLookup) -> ChangeRunState:
        """Return a deep copy so callers cannot mutate the stored aggregate."""
        async with self._lock:
            state = self._runs.get(query.run_id)
            if state is None:
                raise DomainError("not_found")
            return state.model_copy(deep=True)

    async def commit_run_update(self, command: RunCommand) -> ChangeRunState:
        """Commit state, audit facts, revision and original replay result together."""
        command = RunCommand.model_validate_json(command.model_dump_json())
        async with self._lock:
            state = self._runs.get(command.run_id)
            if state is None:
                raise DomainError("not_found")
            key = (command.run_id, command.command_id)
            if key in self._results:
                result = self._results[key]
                if result.command != command:
                    raise DomainError("command_conflict")
                return result.state.model_copy(deep=True)
            if self._latest[state.change_id] != state.revision:
                raise DomainError("revision_conflict")
            result = apply_run_command(state, command)
            if result.revision != state.revision:
                self._check_revision(RevisionAppend(request=result.change_request, expected_revision=state.revision))
            # Every possible rejection precedes this transaction's mutations.
            self._runs[command.run_id] = result
            if result.revision != state.revision:
                self._store_revision(result.change_request)
            self._results[key] = StoredCommandResult(command=command, state=result)
            references = dict(
                change_id=result.change_id,
                run_id=result.run_id,
                revision=result.revision,
                state_version=result.state_version,
                correlation_id=command.audit.correlation_id,
            )
            observe_fact("state", **references)
            observe_state(result)
            old_events = {event.event_id for event in state.execution_events}
            for event in result.execution_events:
                if event.event_id not in old_events:
                    logger.info(
                        "execution_fact_observed",
                        step_id=event.step_id,
                        execution_id=event.event_id,
                        status=event.status.value,
                        approval_id=result.execution_approval_id,
                        **references,
                    )
            if result.evidence_assessment != state.evidence_assessment and result.evidence_assessment:
                observe_fact("gate", "passed" if result.evidence_assessment.sufficient else "denied", **references)
            if result.risk_assessment != state.risk_assessment and result.risk_assessment:
                observe_fact(
                    "gate", "unknown" if result.risk_assessment.level.value == "UNKNOWN" else "recorded", **references
                )
            if result.verification_result != state.verification_result and result.verification_result:
                observe_fact(
                    "verification",
                    "passed" if result.verification_result.status.value == "PASSED" else "failed",
                    **references,
                )
            if result.receipt != state.receipt and result.receipt:
                observe_fact("receipt", receipt_id=result.receipt.receipt_id, **references)
            return result.model_copy(deep=True)

    async def get_run_history(self, query: RunLookup) -> RunHistory:
        """Read isolated copies of historical records and transitions."""
        state = await self.get_run(query)
        return RunHistory(records=state.record_history, transitions=state.transition_history)

    async def get_record(self, query: RecordLookup) -> DomainRecord:
        """Read one append-only record, or report a stable not_found error."""
        state = await self.get_run(RunLookup(run_id=query.run_id))
        for record in state.record_history:
            if record_key(record) == query.record_key:
                return record
        raise DomainError("not_found")

    def _snapshot(self) -> RepositorySnapshot:
        """Build a deterministic export while the transaction lock is held."""
        return RepositorySnapshot(
            changes=[self._changes[key] for key in sorted(self._changes)],
            runs=[self._runs[key] for key in sorted(self._runs)],
            command_results=[self._results[key] for key in sorted(self._results)],
        )

    async def export_snapshot(self) -> SnapshotPayload:
        """Serialize domain-only state, without filesystem or database writes."""
        async with self._lock:
            return SnapshotPayload(data=self._snapshot().model_dump_json())

    async def restore_snapshot(self, payload: SnapshotPayload) -> RepositorySummary:
        """Validate complete replay before installing data; never overwrite a store."""
        try:
            snapshot = RepositorySnapshot.model_validate_json(payload.data)
        except ValidationError as exc:
            errors = exc.errors(include_input=False)
            if any(error["loc"] and error["loc"][-1] == "contract_version" for error in errors):
                raise DomainError("unsupported_contract_version") from None
            if any(error["type"] == "hash_mismatch" for error in errors):
                raise DomainError("hash_mismatch") from None
            raise DomainError("invalid_snapshot") from None
        changes, latest, keys, runs, results = _validate_snapshot(snapshot)
        async with self._lock:
            if self._changes or self._runs:
                if self._snapshot() != snapshot:
                    raise DomainError("snapshot_conflict")
            else:
                self._changes, self._latest, self._keys, self._runs, self._results = (
                    changes,
                    latest,
                    keys,
                    runs,
                    results,
                )
            return RepositorySummary(changes=len(changes), runs=len(runs), commands=len(results))


def _validate_snapshot(
    snapshot: RepositorySnapshot,
) -> tuple[
    dict[tuple[str, int], ChangeRequest],
    dict[str, int],
    dict[tuple[str, str], str],
    dict[str, ChangeRunState],
    dict[tuple[str, str], StoredCommandResult],
]:
    """Check identities, indexes and deterministic replay for every restored run."""
    changes: dict[tuple[str, int], ChangeRequest] = {}
    latest: dict[str, int] = {}
    keys: dict[tuple[str, str], str] = {}
    for request in sorted(snapshot.changes, key=lambda item: (item.change_id, item.revision)):
        key = (request.change_id, request.revision)
        if key in changes or request.revision != latest.get(request.change_id, 0) + 1:
            raise DomainError("invalid_snapshot")
        creation_key = (request.requester_id, request.idempotency_key)
        if request.revision == 1:
            if creation_key in keys:
                raise DomainError("idempotency_conflict")
            keys[creation_key] = request.change_id
        else:
            previous = changes[(request.change_id, request.revision - 1)]
            if (request.requester_id, request.idempotency_key, request.created_at) != (
                previous.requester_id,
                previous.idempotency_key,
                previous.created_at,
            ) or request.updated_at < previous.updated_at:
                raise DomainError("invalid_snapshot")
        changes[key], latest[request.change_id] = request, request.revision
    runs: dict[str, ChangeRunState] = {}
    results: dict[tuple[str, str], StoredCommandResult] = {}
    grouped: dict[str, list[StoredCommandResult]] = defaultdict(list)
    for result in snapshot.command_results:
        key = (result.command.run_id, result.command.command_id)
        if key in results:
            raise DomainError("command_conflict")
        grouped[result.command.run_id].append(result)
        results[key] = result
    for run in snapshot.runs:
        if run.run_id in runs or any(
            changes.get((request.change_id, request.revision)) != request for request in run.request_history
        ):
            raise DomainError("invalid_snapshot")
        current = new_change_run(run.request_history[0], run.run_id)
        for result in sorted(grouped.pop(run.run_id, []), key=lambda item: item.state.state_version):
            try:
                current = apply_run_command(current, result.command)
            except (DomainError, ValidationError, ValueError):
                raise DomainError("invalid_snapshot") from None
            if current != result.state:
                raise DomainError("invalid_snapshot")
        if current != run:
            raise DomainError("invalid_snapshot")
        runs[run.run_id] = run
    if grouped:
        raise DomainError("invalid_snapshot")
    return changes, latest, keys, runs, results
