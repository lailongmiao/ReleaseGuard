"""Pinned local policy and read-only synthetic CI/history sources."""

from datetime import UTC, datetime, timedelta
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain_core.runnables import RunnableConfig
from app.repositories.releaseguard import RunLookup

from app.schemas.authorization import (
    PermissionPolicy,
    PermissionRule,
    PermissionEffect,
    RequestContext,
    VerifiedPrincipal,
)
from app.schemas.evidence import (
    EvidencePolicy,
    EvidenceRequirement,
    EvidenceSourceType,
    EvidenceItem,
    EvidenceAccessScope,
)
from app.schemas.evidence import EvidenceSourceResult, EvidenceCollectionStatus, EvidenceSnapshot, EvidenceCitation
from app.schemas.execution import ManifestStep, SandboxInput
from app.schemas.tooling import ToolDescriptor, ToolInvocation, SideEffect
from app.schemas.local_mvp import LocalActor
from app.schemas.v1_backend import Collection
from app.services.evidence_sources import EvidenceAccessContext, EvidenceSourceRequest, InMemoryEvidenceSourceAdapter
from app.services.local_mvp_storage import unit
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.registry import ToolRegistry
from app.services.tools.execution_adapter import TargetObservation


POLICY_VERSION = "local-mvp.v1"
RESOURCE_ID = "local-synthetic-register"
DESCRIPTOR = ToolDescriptor(
    tool_name="sandbox-register",
    version="v1",
    capabilities=frozenset({SideEffect.MUTATING}),
    resource_types=frozenset({"synthetic_register"}),
    action_types=frozenset({"compare_set"}),
    input_schema="sandbox-register.v1",
    output_schema="sandbox-result.v1",
    side_effect_level=SideEffect.MUTATING,
    timeout_ms=1000,
    idempotency_required=True,
    adapter_kind="internal",
)
EVIDENCE_POLICY = EvidencePolicy(
    policy_version=POLICY_VERSION,
    requirements=[
        EvidenceRequirement(requirement_id="ci", source_types=frozenset({EvidenceSourceType.CI}), min_confidence=0.8),
        EvidenceRequirement(
            requirement_id="history", source_types=frozenset({EvidenceSourceType.HISTORY}), min_confidence=0.8
        ),
    ],
)
PRODUCT_EVIDENCE_POLICY = EvidencePolicy(
    policy_version="product-evidence.v1",
    requirements=[
        EvidenceRequirement(requirement_id="ci", source_types=frozenset({EvidenceSourceType.CI}), min_confidence=0.8),
        EvidenceRequirement(
            requirement_id="history", source_types=frozenset({EvidenceSourceType.DOCUMENT}), min_confidence=0.8
        ),
    ],
)
PERMISSION_POLICY = PermissionPolicy(
    policy_version=POLICY_VERSION,
    rules=tuple(
        PermissionRule(
            rule_id=action,
            tenant_id="local-mvp",
            roles=frozenset({"requester", "approver"}),
            resource_type="synthetic_register",
            resource_ids=frozenset({RESOURCE_ID}),
            actions=frozenset({action}),
            environments=frozenset({"development", "staging"}),
            side_effects=frozenset({effect}),
            effect=decision,
        )
        for action, effect, decision in (
            ("compare_set", SideEffect.MUTATING, PermissionEffect.APPROVAL_REQUIRED),
            ("inspect", SideEffect.READ_ONLY, PermissionEffect.ALLOW),
        )
    ),
)


def principal_id(user_id: int) -> str:
    """Use opaque database identity, never email/name or caller claims."""
    return "local-user:" + str(user_id)


def evidence_access() -> EvidenceAccessContext:
    """Graph gets current authenticated object-bound access from the server."""
    current = unit()
    return EvidenceAccessContext(principal_id=principal_id(current.actor.user_id))


class LocalResolver:
    """Roles are per object and re-derived after each authenticated request."""

    def __init__(self) -> None:
        """Product authority requires an injected live source guard."""
        self.product_guard: Callable[[dict, LocalActor], Awaitable[str]] | None = None

    async def resolve(
        self,
        invocation: ToolInvocation,
        config: RunnableConfig | None = None,
        *,
        at: datetime | None = None,
        invalidation_only: bool = False,
    ) -> RequestContext:
        """Resolve current actor without trusting checkpoint/session identifiers."""
        current = unit()
        actor = current.actor
        product_hash = None
        if current.facts.product and not (invalidation_only and current.facts.version == "local-mvp.v3"):
            if self.product_guard is None:
                raise DomainError("product_guard_unavailable")
            product_hash = await self.product_guard(current.facts.product, actor)
        roles = {"requester"} if actor.user_id == current.row["owner_user_id"] else {"approver"}
        principal = VerifiedPrincipal(
            principal_id=principal_id(actor.user_id),
            session_id=actor.session_id,
            tenant_id="local-mvp",
            roles=frozenset(roles),
            authenticated_at=current.at,
            expires_at=datetime.fromtimestamp(actor.expires_at, UTC),
        )
        call = invocation.model_copy(
            update={
                "principal_id": principal.principal_id,
                "session_id": actor.session_id,
                "occurred_at": at or current.at,
            }
        )
        return RequestContext(
            principal=principal,
            invocation=call,
            resource_tenant_id="local-mvp",
            product_hash=product_hash,
            product_binding_version="v3" if current.facts.version == "local-mvp.v3" else "v2",
        )


class LocalExecutionResolver:
    """Adapt the existing execution protocol without weakening identity checks."""

    def __init__(self, resolver: LocalResolver | None = None) -> None:
        """Reuse the same guarded resolver for product and legacy requests."""
        self.resolver = resolver or LocalResolver()

    async def resolve(self, call: ToolInvocation, *, at: datetime) -> RequestContext:
        """Both approval and execution derive identity from the same server unit."""
        return await self.resolver.resolve(call, at=at)


class LocalRegistry(ToolRegistry):
    """Only a pinned finite sandbox descriptor; no transport dispatch."""

    @override
    def descriptor(self, name: str, version: str) -> ToolDescriptor:
        """Unknown versions fail closed; discovery never widens capabilities."""
        if (name, version) != (DESCRIPTOR.tool_name, DESCRIPTOR.version):
            raise DomainError("tool_unknown_version")
        return DESCRIPTOR.model_copy(deep=True)


class LocalSandboxAdapter:
    """Persist synthetic effects before returning; no host/network target exists."""

    async def inspect_target(self, step: ManifestStep, *, at: datetime) -> TargetObservation:
        """A separate finite target read; simulation does not call compare_set."""
        return await unit().target.read(step.resource_id, at=at)

    async def execute_step(self, step: ManifestStep, value: SandboxInput, *, claim_id: str) -> bool:
        """The previous durable claim is required by ExecutionService."""
        result = await unit().target.compare_set(step, value, claim_id=claim_id)
        await unit().save()
        return result

    async def query_outcome(self, claim_id: str) -> bool | None:
        """Absence stays unknown, never permission for another send."""
        return await unit().target.outcome(claim_id)


class LocalIndependentReader:
    """Independent target observation; cannot read executor return values."""

    source_pin = "independent-register.v1"

    async def observe(self, resource_id: str, *, at: datetime) -> TargetObservation:
        """Read saved finite target state, not a success message."""
        return await unit().target.read(resource_id, at=at)


async def collect_local_sources(
    run: Any,
) -> tuple[list[EvidenceSnapshot], list[EvidenceCitation], list[EvidenceSourceResult]]:
    """Two server-pinned read-only synthetic adapters, explicitly not production."""
    current, snapshots, citations, results = unit(), [], [], []
    scenario = current.facts.creation["evidence_scenario"]
    for required, source_type in (("ci", EvidenceSourceType.CI), ("history", EvidenceSourceType.HISTORY)):
        items = []
        unavailable = scenario == "unavailable" and required == "ci"
        missing = scenario == "missing" and required == "history"
        if not missing and not unavailable:
            captured = current.at - timedelta(hours=2) if scenario == "expired" else current.at
            fresh = current.at - timedelta(hours=1) if scenario == "expired" else current.at + timedelta(hours=1)
            for variant in range(2 if scenario == "conflicting" and required == "ci" else 1):
                content = {"source": required, "version": "local-source.v1", "synthetic": True, "variant": variant}
                item = EvidenceItem(
                    evidence_id=run.change_id + ":r" + str(run.revision) + ":" + required + ":" + str(variant),
                    source_type=source_type,
                    source_uri="fixture://local-mvp/" + required,
                    source_version="local-source.v1",
                    captured_at=captured,
                    freshness_until=fresh,
                    content_hash=canonical_hash(content),
                    access_scope=EvidenceAccessScope(
                        principals=frozenset(
                            {principal_id(current.row["owner_user_id"]), principal_id(current.row["approver_user_id"])}
                        )
                    ),
                    snippet="Synthetic CI result" if required == "ci" else "Synthetic change and incident history",
                    confidence=1,
                    metadata={"synthetic": True},
                )
                items.append(item)
        result = EvidenceSourceResult(
            source_id="local-" + required,
            source_type=source_type,
            requirement_id=required,
            collection_status=EvidenceCollectionStatus.UNAVAILABLE
            if unavailable
            else EvidenceCollectionStatus.VERIFIED,
            items=items,
            captured_at=current.at,
            audit_reference=run.change_id + ":source:" + required,
            degradation_reason="synthetic_source_unavailable" if unavailable else None,
        )
        result = await InMemoryEvidenceSourceAdapter(result).collect(
            EvidenceSourceRequest(
                principal_id=principal_id(current.actor.user_id),
                change_id=run.change_id,
                run_id=run.run_id,
                revision=run.revision,
                requirement_id=required,
                query_hash=run.request_hash,
            )
        )
        results.append(result)
        for item in result.items:
            snapshot = EvidenceSnapshot(
                snapshot_id=canonical_hash(
                    item.model_dump(include={"evidence_id", "source_uri", "source_version", "content_hash"})
                ),
                item=item,
                persisted_at=current.at,
            )
            citation = EvidenceCitation(
                citation_id=item.evidence_id + ":citation",
                snapshot_id=snapshot.snapshot_id,
                evidence_id=item.evidence_id,
                source_version=item.source_version,
                content_hash=item.content_hash,
                captured_at=item.captured_at,
                required_item=required,
                change_id=run.change_id,
                run_id=run.run_id,
                revision=run.revision,
                request_hash=run.request_hash,
                thread_id=current.facts.creation["thread_id"],
                policy_version=POLICY_VERSION,
                cited_at=current.at,
            )
            snapshots.append(snapshot)
            citations.append(citation)
    return snapshots, citations, results


class LocalSourceAdapter:
    """Read-only server source with request-specific object scope."""

    def __init__(self, requirement: str) -> None:
        """Pin CI/history rather than accept client URLs or source payloads."""
        self.requirement = requirement

    @property
    def source_id(self) -> str:
        """The actual source identity follows the trusted locked object mode."""
        if unit().facts.version == "local-mvp.v3":
            return (
                "review-github-" + unit().facts.product["binding"]["repository_id"]
                if self.requirement == "ci"
                else "markdown-rules"
            )
        return (
            ("github-checkout" if self.requirement == "ci" else "markdown-rules")
            if unit().facts.product
            else "local-" + self.requirement
        )

    async def collect(self, request: EvidenceSourceRequest) -> EvidenceSourceResult:
        """Return deterministic local synthetic facts for the locked object."""
        from_run = await unit().domain.get_run(RunLookup(run_id=request.run_id))
        if unit().facts.product:
            current = unit()
            product = current.facts.product
            collection = Collection.model_validate(product["collections"][self.requirement])
            source_type = EvidenceSourceType.CI if self.requirement == "ci" else EvidenceSourceType.DOCUMENT
            if collection.status != "verified":
                return EvidenceSourceResult(
                    source_id=self.source_id,
                    source_type=source_type,
                    requirement_id=self.requirement,
                    collection_status=EvidenceCollectionStatus(collection.status),
                    captured_at=current.at,
                    audit_reference=product["analysis_id"],
                    degradation_reason=collection.reason or "source_unavailable",
                )
            payload = {document.document_id: document.content_hash for document in collection.documents}
            digest = canonical_hash(payload)
            principals = set.intersection(*(set(value["acl"]) for value in product["sources"].values()))
            item = EvidenceItem(
                evidence_id="product:"
                + canonical_hash(
                    {
                        "run": from_run.run_id,
                        "revision": from_run.revision,
                        "required": self.requirement,
                        "hash": digest,
                    }
                ),
                source_type=source_type,
                source_uri=collection.documents[0].locator
                if self.requirement == "ci"
                else "https://local.invalid/knowledge/rules",
                source_version=collection.head_sha or digest,
                captured_at=collection.documents[0].captured_at,
                content_hash=digest,
                access_scope=EvidenceAccessScope(principals=frozenset(principal_id(i) for i in principals)),
                snippet="\n".join(d.content for d in collection.documents)[:2000],
                confidence=1.0,
                freshness_until=collection.documents[0].captured_at + timedelta(minutes=15),
                metadata={
                    "source_provenance": "readonly",
                    "source_id": self.source_id,
                    "aggregate": True,
                    "checks_passed": collection.checks_passed if self.requirement == "ci" else None,
                },
            )
            return EvidenceSourceResult(
                source_id=self.source_id,
                source_type=source_type,
                requirement_id=self.requirement,
                collection_status=EvidenceCollectionStatus.VERIFIED,
                items=[item],
                captured_at=current.at,
                audit_reference=product["analysis_id"],
                source_version=item.source_version,
            )
        _, _, results = await collect_local_sources(from_run)
        return next(result for result in results if result.requirement_id == self.requirement)
