"""Independent target observations bound to immutable Evidence and Citations."""

from datetime import datetime, timedelta

from pydantic import ValidationError

from app.schemas.authorization import PermissionEffect
from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceCitation,
    EvidenceItem,
    EvidenceSnapshot,
    EvidenceSourceType,
)
from app.schemas.execution import Direction, ExecutionMode, ExecutionRequest
from app.schemas.releaseguard import (
    ChangeRunState,
    ChangeStatus,
    StateInputs,
    VerificationCheck,
    VerificationResult,
    VerificationStatus,
)
from app.schemas.tooling import ToolInvocation
from app.services.authorization import evaluate_permission
from app.services.releaseguard.audit import record_audit
from app.services.releaseguard.execution import ExecutionService
from app.services.releaseguard.execution_commands import audit_fields, execution_command, identity
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.verification_adapter import VerificationAdapter, observation_descriptor
from app.services.tools.execution_adapter import TargetObservation


class VerificationService:
    """No executor response or free text is accepted as a passed observation."""

    def __init__(self, execution: ExecutionService, reader: VerificationAdapter) -> None:
        """Pin a separate read adapter outside checkpoint state."""
        self.execution, self.reader = execution, reader

    async def verify(self, request: ExecutionRequest, *, at: datetime) -> ChangeRunState:
        """Read independently, then persist snapshots before committing their refs."""
        grant = await self.execution.preflight.check(request, at=at, stage="verify")
        run, service = grant.run, self.execution.preflight.service
        plan = run.execution_plan
        assert plan is not None and plan.manifest is not None
        if self.reader.source_pin != plan.manifest.verification_adapter or self.reader is self.execution.adapter:
            raise DomainError("verification_source_untrusted")
        direction = Direction.COMPENSATION if run.recovery_authorization else Direction.FORWARD
        selected = (
            set(run.recovery_authorization.compensation_step_ids)
            if run.recovery_authorization
            else {s.step_id for s in plan.steps}
        )
        pins = sorted(
            (p for p in plan.manifest.steps if p.direction is direction and p.step_id in selected),
            key=lambda p: p.order,
        )
        expected = {p.resource_id: p.desired_value for p in pins}
        descriptor, checks, items, citations, observations = observation_descriptor(), [], [], [], []
        if not expected:
            raise DomainError("verification_checks_missing")
        for resource_id, desired in sorted(expected.items()):
            call = ToolInvocation.model_validate(
                {
                    **grant.context.invocation.model_dump(),
                    "resource_id": resource_id,
                    "tool_name": descriptor.tool_name,
                    "version": descriptor.version,
                    "descriptor_hash": descriptor.descriptor_hash,
                    "input_schema": descriptor.input_schema,
                    "action": "inspect",
                    "occurred_at": at,
                }
            )
            context = await self.execution.preflight.resolver.resolve(call, at=at)
            decision = evaluate_permission(context, descriptor, service.permission_policy, at=at)
            if decision.effect is not PermissionEffect.ALLOW:
                raise DomainError(decision.reason_code)
            try:
                observation = TargetObservation.model_validate(
                    (await self.reader.observe(resource_id, at=at)).model_dump(mode="python")
                )
            except (ValidationError, AttributeError):
                raise DomainError("verification_observation_invalid") from None
            if (
                observation.resource_id != resource_id
                or observation.version < 0
                or run.execution_started_at is None
                or not run.execution_started_at <= observation.observed_at <= at
            ):
                raise DomainError("verification_observation_invalid")
            observations.append(observation)
            digest = canonical_hash(observation)
            reference = canonical_hash(
                {
                    **identity(run),
                    "thread_id": request.thread_id,
                    "policy_version": plan.manifest.policy_version,
                    "manifest_hash": plan.manifest.manifest_hash,
                    "observation_hash": digest,
                }
            )
            item = EvidenceItem(
                evidence_id="verification:" + reference,
                source_type=EvidenceSourceType.RUNTIME_READ,
                source_uri="synthetic://independent-register/" + resource_id,
                source_version=str(observation.version),
                captured_at=observation.observed_at,
                content_hash=digest,
                access_scope=EvidenceAccessScope(tenant_id=context.principal.tenant_id),
                snippet="synthetic integer observation",
                confidence=1,
                freshness_until=at + timedelta(minutes=5),
                metadata={
                    "mode": ExecutionMode.SANDBOX.value,
                    "value": observation.value,
                    "resource_id": resource_id,
                    "manifest_hash": plan.manifest.manifest_hash,
                },
            )
            snapshot_id = canonical_hash(
                item.model_dump(include={"evidence_id", "source_uri", "source_version", "content_hash"})
            )
            snapshot = EvidenceSnapshot(snapshot_id="sha256:" + snapshot_id, item=item, persisted_at=at)
            await service.evidence.append_snapshot(snapshot)
            citation = EvidenceCitation.model_validate(
                {
                    **identity(run),
                    "citation_id": "verification-cite:" + reference,
                    "snapshot_id": snapshot.snapshot_id,
                    "evidence_id": item.evidence_id,
                    "source_version": item.source_version,
                    "content_hash": item.content_hash,
                    "captured_at": item.captured_at,
                    "required_item": "verification",
                    "thread_id": request.thread_id,
                    "policy_version": plan.manifest.policy_version,
                    "cited_at": at,
                }
            )
            await service.evidence.append_citation(citation)
            items.append(item)
            citations.append(citation.citation_id)
            passed = observation.value == desired and observation.version > 0
            checks.append(
                VerificationCheck(
                    check_id=resource_id,
                    description="independent synthetic target comparison",
                    status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                    evidence_ids=[item.evidence_id],
                    source_type="synthetic_independent_register",
                    observed_at=observation.observed_at,
                    failure_reason=None if passed else "target_state_mismatch",
                )
            )
        audit = await record_audit(
            service.approvals,
            grant.context.invocation,
            "verification_completed",
            "independent_verification_completed",
            at=at,
            discriminator=request.command_id,
        )
        passed = all(check.status is VerificationStatus.PASSED for check in checks)
        result = VerificationResult.model_validate(
            {
                **identity(run),
                "verification_id": request.command_id + ":verification",
                "plan_hash": plan.plan_hash,
                "mode": ExecutionMode.SANDBOX,
                "status": "PASSED" if passed else "FAILED",
                "checks": checks,
                "independent_source_types": ["synthetic_independent_register"],
                "verified_at": at,
                "citation_ids": citations,
                "audit_ids": [audit.event_id],
                "observation_hash": canonical_hash(observations),
                "input_hash": plan.input_hash,
                "audit": audit_fields(
                    grant.context, at=at, code="independent_verification_completed", input_hash=plan.input_hash
                ),
            }
        )
        run = await service.domain.commit_run_update(
            execution_command(
                run,
                grant.context,
                command_id=request.command_id + ":facts",
                at=at,
                code="verification_facts_saved",
                inputs=StateInputs(verification_result=result, verification_evidence=items),
            )
        )
        return await service.domain.commit_run_update(
            execution_command(
                run,
                grant.context,
                command_id=request.command_id + ":result",
                at=at,
                code="verification_passed" if passed else "verification_failed",
                target=ChangeStatus.SUCCEEDED
                if passed
                else (ChangeStatus.FAILED if run.recovery_authorization else ChangeStatus.RECOVERY_REQUIRED),
            )
        )
