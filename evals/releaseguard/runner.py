"""Actual isolated service and single Graph trajectories with explicit injection."""

import asyncio
from datetime import timedelta
from time import perf_counter
from typing import Any, override

from pydantic import ValidationError

from app.core.langgraph.releaseguard_execution import resume_execution
from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceCollectionStatus,
    EvidenceSourceResult,
    EvidenceSnapshot,
    EvidenceSourceType,
)
from app.schemas.execution import Direction, ExecutionIntent
from app.schemas.graph import GraphState
from app.schemas.releaseguard import ApprovalDecision, ChangeRunState, ChangeStatus
from app.services.evidence_sources import EvidenceAccessContext
from app.services.releaseguard.dry_run import DryRunService
from app.services.releaseguard.evidence_policy import evaluate_evidence_sufficiency
from app.services.releaseguard.execution_commands import execution_command
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.receipt import ReceiptService
from app.services.releaseguard.recovery import RecoveryService
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.execution_adapter import TargetObservation
from app.services.tools.sandbox_executor import SandboxExecutor
from app.services.tools.verification_adapter import IndependentRegisterReader
from evals.releaseguard.fixtures import Fault, Fixture, NOW
from evals.releaseguard.boundary import OfflineBoundary
from evals.releaseguard.schemas import CaseFacts, CaseResult, CaseSpec, Dataset, LedgerFacts, RunConfig


class ResponseFault(SandboxExecutor):
    """Inject a lost response or cancellation after a genuine finite effect."""

    def __init__(self, fixture: Fixture, code: str) -> None:
        """No network or host side effects are supported."""
        super().__init__(fixture.target)
        self.code = code

    @override
    async def execute_step(self, step: Any, value: Any, *, claim_id: str) -> bool:
        """Preserve the send fact even when the response cannot be observed."""
        applied = await super().execute_step(step, value, claim_id=claim_id)
        if self.code == "timeout":
            raise TimeoutError("synthetic_timeout")
        if self.code == "cancel":
            raise asyncio.CancelledError()
        return applied


class FalseSuccess(SandboxExecutor):
    """A fault injected adapter claiming success without changing the target."""

    @override
    async def execute_step(self, step: Any, value: Any, *, claim_id: str) -> bool:
        """Count actual attempted sends but leave the independent target unchanged."""
        self.targets.calls += 1
        return True


class InvalidReader(IndependentRegisterReader):
    """Inject invalid observation data; never pretend this is a live source."""

    @override
    async def observe(self, resource_id: str, *, at: Any) -> TargetObservation:
        """A bool masquerading as integer must be rejected by actual verification."""
        return TargetObservation.model_construct(resource_id=resource_id, value=True, version=1, observed_at=at)


async def capture_error(call: Any, diagnostics: list[str]) -> Any:
    """Store only stable safe classifications, never exception bodies."""
    try:
        return await call
    except DomainError as error:
        diagnostics.append(error.code)
    except ValidationError:
        diagnostics.append("schema_rejected")
    except RuntimeError as error:
        if str(error) != "synthetic_fault":
            raise
        diagnostics.append("synthetic_fault")
    return None


async def partial_recovery(f: Fixture, *, fail: bool) -> None:
    """Use actual service scopes and independently approved inverse effects."""
    await DryRunService(f.execution.preflight, f.execution.adapter).run(await f.request(), at=f.at)
    await f.execution.step(await f.request(), at=f.at)
    f.target._values["synthetic-resource"] = 99
    await f.execution.step(await f.request(step="step-1"), at=f.at)
    await f.domain.commit_run_update(
        execution_command(
            await f.run(),
            f.context,
            command_id="require-recovery",
            at=f.at,
            code="partial_failure",
            target=ChangeStatus.RECOVERY_REQUIRED,
        )
    )
    f.target._values["synthetic-resource"] = 1
    recovery = RecoveryService(f.execution)
    review = await recovery.prepare(await f.request(), at=f.at, request_id="recovery-review")
    await f.decide(review=review)
    await recovery.start(
        await f.request(step="undo-0", approval=review.approval_request_id, direction=Direction.COMPENSATION), at=f.at
    )
    if fail:
        f.target._values["synthetic-resource"] = 99
    await recovery.compensate(
        await f.request(step="undo-0", approval=review.approval_request_id, direction=Direction.COMPENSATION), at=f.at
    )
    if not fail:
        await f.domain.commit_run_update(
            execution_command(
                await f.run(),
                f.context,
                command_id="verify-recovery",
                at=f.at,
                code="compensations_confirmed",
                target=ChangeStatus.VERIFYING,
            )
        )
        await f.verification.verify(
            await f.request(approval=review.approval_request_id, step="undo-0", direction=Direction.COMPENSATION),
            at=f.at,
        )
    await ReceiptService(f.execution).issue(
        await f.request(
            approval=review.approval_request_id, step="undo-0", direction=Direction.COMPENSATION, command="receipt"
        ),
        at=f.at,
    )


async def trajectory(f: Fixture, case: CaseSpec, diagnostics: list[str]) -> None:
    """Dispatch only reviewed scenarios, through existing service contracts."""
    s = case.scenario
    if s.startswith("gate_"):
        return
    if s == "risk_unknown":
        return
    if s.startswith("approval_"):
        request = await f.request()
        action = s.removeprefix("approval_")
        if action in {"rejected", "needs_changes"}:
            await f.decide("REJECTED" if action == "rejected" else "NEEDS_CHANGES")
        elif action == "self":
            await capture_error(f.decide(own=True), diagnostics)
        elif action == "expired":
            f.at += timedelta(minutes=11)
        elif action == "revoked":
            await f.service.invalidate(
                "synthetic-approval", f.context, ApprovalDecision.REVOKED, "synthetic_revocation", at=f.at
            )
        await capture_error(
            DryRunService(f.execution.preflight, f.execution.adapter).run(request, at=f.at), diagnostics
        )
        return
    if s.startswith("deny_"):
        selector = s.removeprefix("deny_")
        if selector == "default":
            f.service.permission_policy = f.service.permission_policy.model_copy(update={"rules": ()})
        elif selector == "tenant":
            f.resolver.context = f.context.model_copy(update={"resource_tenant_id": "other-tenant"})
        else:
            key, value = {
                "resource": ("resource_ids", frozenset({"other-resource"})),
                "action": ("actions", frozenset({"other-action"})),
                "environment": ("environments", frozenset({"production"})),
            }[selector]
            f.service.permission_policy = f.service.permission_policy.model_copy(
                update={
                    "rules": tuple(rule.model_copy(update={key: value}) for rule in f.service.permission_policy.rules)
                }
            )
        await capture_error(
            DryRunService(f.execution.preflight, f.execution.adapter).run(await f.request(), at=f.at), diagnostics
        )
        return
    if s.startswith("drift_"):
        selector = s.removeprefix("drift_")
        request = await f.request()
        if selector in {"plan", "manifest", "input"}:
            field = {"plan": "plan_hash", "manifest": "manifest_hash", "input": "request_hash"}[selector]
            request = request.model_copy(update={field: "0" * 64})
        elif selector == "descriptor":
            payload = f.descriptor.model_dump(exclude={"descriptor_hash"})
            payload["version"] = "v2"
            f.execution.preflight.descriptor = type(f.descriptor).model_validate(payload)
        elif selector == "policy":
            f.service.permission_policy = f.service.permission_policy.model_copy(
                update={"policy_version": "synthetic-policy.v2"}
            )
        elif selector == "evidence":
            f.service.evidence_policy = f.evidence_policy.model_copy(update={"requirements": []})
        await capture_error(
            DryRunService(f.execution.preflight, f.execution.adapter).run(request, at=f.at), diagnostics
        )
        return
    if s in {"partial_recovery", "compensation_failed"}:
        await partial_recovery(f, fail=s == "compensation_failed")
        return
    if s == "chat_isolation":
        await f.graph.ainvoke(
            GraphState(long_term_memory="synthetic"), {"configurable": {"thread_id": "synthetic-chat"}}
        )
        return
    if s in {"graph_success", "risk_low", "risk_medium", "risk_high", "simulation"}:
        await f.graph.ainvoke(await f.state(), f.config)
        return
    if s in {"typed_resume", "old_root", "two_waits", "unknown_reconcile"}:
        f.execution.faults = Fault("after_send")
        state = await f.state()
        await f.graph.ainvoke(state, f.config)
        if s == "old_root":
            assert state.releaseguard is not None and state.releaseguard.execution_intent is not None
            snapshot = await f.graph.aget_state(f.config)
            await capture_error(
                resume_execution(f.graph, state.releaseguard.execution_intent, snapshot.config), diagnostics
            )
            return
        if s == "two_waits":
            request = await f.request(command="bad-resume")
            await capture_error(
                resume_execution(
                    f.graph,
                    ExecutionIntent(request=request.model_copy(update={"plan_hash": "0" * 64}), action="reconcile"),
                    f.config,
                ),
                diagnostics,
            )
        await resume_execution(
            f.graph, ExecutionIntent(request=await f.request(command="reconcile"), action="reconcile"), f.config
        )
        return
    if s.startswith("graph_fault_"):
        f.execution.faults = Fault(s.removeprefix("graph_fault_"))
        await capture_error(f.graph.ainvoke(await f.state(), f.config), diagnostics)
        await capture_error(f.graph.ainvoke(None, f.config), diagnostics)
        return
    await DryRunService(f.execution.preflight, f.execution.adapter).run(await f.request(command="dry"), at=f.at)
    if s.startswith("orphan_"):
        f.execution.faults = Fault(s.removeprefix("orphan_"))
        await capture_error(f.execution.step(await f.request(), at=f.at), diagnostics)
        await capture_error(f.execution.step(await f.request(command="new-owner"), at=f.at), diagnostics)
        return
    if s in {"timeout", "cancel"}:
        f.execution.adapter = ResponseFault(f, s)
        await f.execution.step(await f.request(), at=f.at)
        await ReceiptService(f.execution).issue(await f.request(command="unknown"), at=f.at)
        return
    if s in {"repeat", "concurrent", "intent_conflict", "opposite"}:
        request = await f.request()
        if s in {"concurrent", "opposite"}:
            barrier = asyncio.Event()

            async def contender(conflict: bool) -> None:
                await barrier.wait()
                other = request.model_copy(
                    update={"command_id": "second", **({"plan_hash": "0" * 64} if conflict else {})}
                )
                await capture_error(f.execution.step(other, at=f.at), diagnostics)

            tasks = [
                asyncio.create_task(contender(s == "opposite")),
                asyncio.create_task(capture_error(f.execution.step(request, at=f.at), diagnostics)),
            ]
            barrier.set()
            await asyncio.gather(*tasks)
        else:
            await f.execution.step(request, at=f.at)
            repeat = await f.request(command="repeat")
            if s == "intent_conflict":
                repeat = repeat.model_copy(update={"manifest_hash": "0" * 64})
            await capture_error(f.execution.step(repeat, at=f.at), diagnostics)
        return
    if s in {"false_success", "invalid_observation"}:
        if s == "false_success":
            f.execution.adapter = FalseSuccess(f.target)
        await f.execution.step(await f.request(), at=f.at)
        await f.domain.commit_run_update(
            execution_command(
                await f.run(),
                f.context,
                command_id="verify",
                at=f.at,
                code="effects_confirmed",
                target=ChangeStatus.VERIFYING,
            )
        )
        if s == "invalid_observation":
            f.verification.reader = InvalidReader(f.target)
        await capture_error(f.verification.verify(await f.request(), at=f.at), diagnostics)
        if s == "false_success":
            await capture_error(
                ReceiptService(f.execution).issue(await f.request(command="failed"), at=f.at), diagnostics
            )
        return
    if s == "receipt_conflict":
        await f.graph.ainvoke(await f.state(), f.config)
        run = await f.run()
        assert run.receipt is not None
        request = await f.request(command=run.receipt.receipt_id.removesuffix(":receipt"))
        await capture_error(
            ReceiptService(f.execution).issue(request.model_copy(update={"step_id": "impostor"}), at=f.at), diagnostics
        )
        data = run.model_dump(mode="json")
        data["receipt"]["claim_ids"] = []
        try:
            ChangeRunState.model_validate(data)
        except ValidationError:
            diagnostics.append("schema_rejected")
        return
    raise ValueError("scenario_not_registered")


def gate_inputs(f: Fixture, case: CaseSpec) -> tuple[list[Any], list[Any], list[Any]]:
    """Inject only typed finite Gate inputs, separate from persisted authority."""
    snapshots, citations, sources = [f.snapshot], [f.citation], []
    scenario = case.scenario.removeprefix("gate_")
    if scenario == "missing":
        snapshots, citations = [], []
    elif scenario == "conflicting":
        item = f.snapshot.item.model_copy(update={"evidence_id": "synthetic:other", "content_hash": "b" * 64})
        snapshot = EvidenceSnapshot(
            snapshot_id="sha256:"
            + canonical_hash(item.model_dump(include={"evidence_id", "source_uri", "source_version", "content_hash"})),
            item=item,
            persisted_at=NOW,
        )
        snapshots.append(snapshot)
        citations.append(
            f.citation.model_copy(
                update={
                    "citation_id": "other-citation",
                    "snapshot_id": snapshot.snapshot_id,
                    "evidence_id": item.evidence_id,
                    "content_hash": item.content_hash,
                }
            )
        )
    elif scenario == "expired":
        snapshots = [
            f.snapshot.model_copy(update={"item": f.snapshot.item.model_copy(update={"freshness_until": NOW})})
        ]
    elif scenario in {"unavailable", "permission_denied"}:
        sources = [
            EvidenceSourceResult(
                source_id="synthetic-source",
                source_type=EvidenceSourceType.CI,
                requirement_id="ci",
                collection_status=EvidenceCollectionStatus(scenario),
                captured_at=NOW,
                audit_reference="synthetic-source-audit",
                degradation_reason="synthetic_unavailable",
            )
        ]
    elif scenario in {"hash_drift", "version", "dangling", "access"}:
        if scenario == "access":
            snapshots = [
                f.snapshot.model_copy(
                    update={
                        "item": f.snapshot.item.model_copy(
                            update={"access_scope": EvidenceAccessScope(principals=frozenset({"other-principal"}))}
                        )
                    }
                )
            ]
        else:
            key, value = {
                "hash_drift": ("content_hash", "0" * 64),
                "version": ("source_version", "v9"),
                "dangling": ("snapshot_id", "0" * 64),
            }[scenario]
            citations = [f.citation.model_copy(update={key: value})]
    return snapshots, citations, sources


async def _run_case(case: CaseSpec, *, attempt: int = 1, timeout: float = 20) -> CaseResult:
    """Keep failures in the denominator and release cancelled case resources."""
    started, f, diagnostics = perf_counter(), Fixture(), []
    completed = True
    try:
        async with asyncio.timeout(timeout):
            await f.initialize(
                level=case.input_risk,
                simulation=case.scenario == "simulation",
                count=2 if case.scenario in {"partial_recovery", "compensation_failed"} else 1,
                decide=case.scenario
                not in {"approval_pending", "approval_rejected", "approval_needs_changes", "approval_self"},
            )
            await trajectory(f, case, diagnostics)
    except DomainError as error:
        diagnostics.append(error.code)
    except (TimeoutError, asyncio.CancelledError):
        completed = False
        diagnostics.append("runner_timeout_or_cancelled")
    except Exception:
        completed = False
        diagnostics.append("runner_internal_error")
    try:
        run = await f.run()
    except DomainError:
        run = None
    if run is None or not hasattr(f, "snapshot"):
        facts = CaseFacts(
            case_id=case.case_id,
            provenance="actual_offline_services",
            calls=f.target.calls,
            reads=f.target.reads,
            target_value=0,
            target_version=0,
            completed=False,
            diagnostics=tuple(sorted(set(diagnostics))),
        )
        return CaseResult(
            case_id=case.case_id,
            attempt=attempt,
            facts=facts,
            semantic_hash=canonical_hash(facts),
            wall_seconds=perf_counter() - started,
        )
    snapshots, citations, sources = gate_inputs(f, case)
    gate = evaluate_evidence_sufficiency(
        run,
        snapshots=snapshots,
        citations=citations,
        source_results=sources,
        policy=f.evidence_policy,
        evaluated_at=f.at,
        thread_id=f.thread,
        access=EvidenceAccessContext(
            principal_id="synthetic-requester", tenant_id="synthetic-tenant", roles=frozenset({"requester"})
        ),
    )
    persisted_citations = [f.citation]
    if run.verification_result:
        persisted_citations += [await f.evidence.get_citation(cid) for cid in run.verification_result.citation_ids]
    persisted_snapshots = [await f.evidence.get_snapshot(cid.snapshot_id) for cid in persisted_citations]
    store = await f.approvals.export_snapshot()
    observation = await f.target.read("synthetic-resource", at=f.at)
    facts = CaseFacts(
        case_id=case.case_id,
        provenance="fault_injected_offline"
        if "fault" in case.scenario
        or (case.scenario.startswith("gate_") and case.scenario != "gate_sufficient")
        or case.scenario.startswith("drift_")
        or case.scenario.startswith("orphan")
        or case.scenario
        in {"partial_recovery", "compensation_failed", "timeout", "cancel", "false_success", "invalid_observation"}
        else "actual_offline_services",
        domain=run,
        gate=gate,
        gate_policy=f.evidence_policy,
        gate_access=EvidenceAccessContext(
            principal_id="synthetic-requester", tenant_id="synthetic-tenant", roles=frozenset({"requester"})
        ),
        thread_id=f.thread,
        gate_snapshots=tuple(snapshots),
        gate_citations=tuple(citations),
        source_results=tuple(sources),
        snapshots=tuple(persisted_snapshots),
        citations=tuple(persisted_citations),
        approvals=store.entries,
        audits=store.audit_events,
        ledger=tuple(LedgerFacts(claim=c, events=tuple(events)) for c, events in await f.ledger.entries(run.run_id)),
        calls=f.target.calls,
        reads=f.target.reads,
        target_value=observation.value,
        target_version=observation.version,
        diagnostics=tuple(sorted(set(diagnostics))),
        completed=completed,
    )
    return CaseResult(
        case_id=case.case_id,
        attempt=attempt,
        facts=facts,
        semantic_hash=canonical_hash(facts),
        wall_seconds=perf_counter() - started,
    )


async def run_dataset(dataset: Dataset, config: RunConfig) -> tuple[CaseResult, ...]:
    """Each attempt creates fresh stores; business replay happens inside a case."""
    return tuple(
        [
            await run_case(case, attempt=attempt, timeout=config.timeout_seconds)
            for attempt in range(1, config.repeat + 1)
            for case in dataset.cases
        ]
    )


_case_lock = asyncio.Lock()


async def run_case(case: CaseSpec, *, attempt: int = 1, timeout: float = 20) -> CaseResult:
    """Serialize trusted Python I/O guards; internal contenders still race via barrier."""
    async with _case_lock:
        with OfflineBoundary() as boundary:
            result = await _run_case(case, attempt=attempt, timeout=timeout)
        facts = result.facts.model_copy(
            update={
                "blocked_network_attempts": boundary.network_attempts,
                "blocked_process_attempts": boundary.process_attempts,
                "blocked_write_attempts": boundary.write_attempts,
            }
        )
        return result.model_copy(update={"facts": facts, "semantic_hash": canonical_hash(facts)})
