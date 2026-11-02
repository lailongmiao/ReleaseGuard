"""Receipts reflect saved facts and references; they never authorize effects."""

from datetime import datetime

from app.repositories.releaseguard import RunLookup
from app.schemas.authorization import PermissionEffect
from app.schemas.execution import Direction, ExecutionMode, ExecutionRequest
from app.schemas.releaseguard import ChangeRunState, ChangeStatus, ExecutionReceipt, ReceiptStatus, StateInputs
from app.services.authorization import evaluate_permission
from app.services.releaseguard.audit import record_audit
from app.services.releaseguard.execution import ExecutionService
from app.services.releaseguard.execution_commands import audit_fields, execution_command, identity
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError


class ReceiptService:
    """Offline joins are verified across stores before issuing a receipt."""

    def __init__(self, execution: ExecutionService) -> None:
        """Keep the authoritative stores outside Graph state."""
        self.execution = execution

    async def issue(self, request: ExecutionRequest, *, at: datetime) -> ChangeRunState:
        """Keep UNKNOWN nonterminal and link subsequent receipts to prior diagnostics."""
        service = self.execution.preflight.service
        entry = await service.approvals.get(request.approval_request_id)
        context = await self.execution.preflight.resolver.resolve(entry.request.binding.invocation, at=at)
        permission = evaluate_permission(
            context, self.execution.preflight.descriptor, service.permission_policy, at=at
        )
        if permission.effect not in {PermissionEffect.ALLOW, PermissionEffect.APPROVAL_REQUIRED}:
            raise DomainError(permission.reason_code)
        run = await service.domain.get_run(RunLookup(run_id=request.run_id))
        plan = run.execution_plan
        if (
            plan is None
            or plan.manifest is None
            or request.plan_hash != plan.plan_hash
            or request.manifest_hash != plan.manifest.manifest_hash
            or request.thread_id != entry.request.binding.invocation.thread_id
            or request.mode is not plan.manifest.mode
            or any(
                getattr(request, key) != getattr(run, key)
                for key in ("change_id", "run_id", "revision", "request_hash")
            )
        ):
            raise DomainError("receipt_binding_conflict")
        if entry.request.binding.invocation.run_id != run.run_id or entry.request.binding.plan_hash != plan.plan_hash:
            raise DomainError("receipt_binding_conflict")
        intent_hash = canonical_hash(request.model_dump(exclude={"state_version"}))
        if run.receipt and run.receipt.receipt_id == request.command_id + ":receipt":
            if run.receipt.intent_hash != intent_hash:
                raise DomainError("receipt_intent_conflict")
            return run
        if request.state_version != run.state_version:
            raise DomainError("receipt_binding_conflict")
        outcome = {
            ChangeStatus.DRY_RUN_PASSED: ReceiptStatus.SIMULATED,
            ChangeStatus.EXECUTION_UNKNOWN: ReceiptStatus.UNKNOWN,
            ChangeStatus.FAILED: ReceiptStatus.FAILED,
            ChangeStatus.REJECTED: ReceiptStatus.REJECTED,
            ChangeStatus.SUCCEEDED: ReceiptStatus.RECOVERED if run.recovery_authorization else ReceiptStatus.SUCCEEDED,
        }.get(run.status)
        if outcome is None:
            raise DomainError("receipt_status_blocked")
        snapshot = await service.approvals.export_snapshot()
        history = [e for e in snapshot.entries if e.request.binding.invocation.run_id == run.run_id]
        audit = await record_audit(
            service.approvals,
            context.invocation,
            "receipt_issued",
            "receipt_issued",
            at=at,
            discriminator=request.command_id,
        )
        audits = {a.event_id for a in snapshot.audit_events if a.run_id == run.run_id} | {audit.event_id}
        ledger = await self.execution.ledger.entries(run.run_id)
        event_ids = {event.event_id for _, events in ledger for event in events}
        if event_ids != {event.event_id for event in run.execution_events}:
            raise DomainError("receipt_ledger_fork")
        if any(not set(event.audit_ids) <= audits for event in run.execution_events):
            raise DomainError("receipt_audit_missing")
        citation_ids = set(context.invocation.citation_ids)
        if run.verification_result:
            citation_ids.update(run.verification_result.citation_ids)
        for identifier in citation_ids:
            citation = await service.evidence.get_citation(identifier)
            stored = await service.evidence.get_snapshot(citation.snapshot_id)
            if (
                stored.item not in run.evidence_history
                or citation.run_id != run.run_id
                or citation.revision != run.revision
            ):
                raise DomainError("receipt_citation_fork")
        receipt = ExecutionReceipt.model_validate(
            {
                **identity(run),
                "receipt_id": request.command_id + ":receipt",
                "intent_hash": intent_hash,
                "plan_hash": plan.plan_hash,
                "manifest_hash": plan.manifest.manifest_hash,
                "mode": ExecutionMode.SIMULATION if outcome is ReceiptStatus.SIMULATED else ExecutionMode.SANDBOX,
                "simulation_id": run.simulation_result.simulation_id if run.simulation_result else None,
                "evidence_ids": sorted({item.evidence_id for item in run.evidence_history}),
                "evidence_assessment": run.evidence_assessment,
                "risk_assessment": run.risk_assessment,
                "approval_id": run.execution_approval_id or (run.approval.approval_id if run.approval else None),
                "verification_id": run.verification_result.verification_id if run.verification_result else None,
                "final_status": outcome,
                "execution_event_ids": sorted(event_ids),
                "issued_at": at,
                "citation_ids": sorted(citation_ids),
                "authorization_ids": sorted({e.request.authorization_hash for e in history}),
                "claim_ids": sorted({claim.claim_id for claim, _ in ledger}),
                "compensation_ids": [
                    event.event_id for event in run.execution_events if event.direction is Direction.COMPENSATION
                ],
                "audit_ids": sorted(audits),
                "prior_receipt_ids": [
                    record.receipt_id for record in run.record_history if isinstance(record, ExecutionReceipt)
                ],
                "terminal": outcome is not ReceiptStatus.UNKNOWN,
                "input_hash": plan.input_hash,
                "audit": audit_fields(context, at=at, code="receipt_issued", input_hash=plan.input_hash),
            }
        )
        return await service.domain.commit_run_update(
            execution_command(
                run,
                context,
                command_id=request.command_id + ":issued",
                at=at,
                code="receipt_issued",
                target=None if outcome is ReceiptStatus.UNKNOWN else ChangeStatus.RECEIPTED,
                inputs=StateInputs(receipt=receipt),
            )
        )
