"""Readonly product orchestration inside the existing single business Graph."""

import asyncio
from contextvars import ContextVar
from typing import Any

from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.local_mvp import LocalActor
from app.schemas.releaseguard import ChangeStatus, StateInputs
from app.schemas.v1_backend import AnalyzeInput, Collection, ProductBinding, SearchInput, SourceCommand
from app.services.llm.contracts import LLMRole
from app.services.local_mvp_storage import unit
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.evidence_policy import evaluate_evidence_sufficiency
from app.services.local_mvp_policy import evidence_access
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_provider import BudgetedProvider, PROMPT_VERSION
from app.services.v1_rag import STRATEGY, chunk_documents, fuse
from app.services.v1_sources import GitHubSource, MarkdownSource
from app.services.v1_storage import SOURCE_IDS, V1Store

active_analysis: ContextVar[dict | None] = ContextVar("v1_authenticated_analysis", default=None)


class V1Backend:
    """Reuses domain, approval, execution and Graph; adds only readonly analysis."""

    def __init__(
        self,
        local: Any,
        store: V1Store | None = None,
        provider: Any = None,
        github: GitHubSource | None = None,
        markdown: MarkdownSource | None = None,
    ) -> None:
        """Inject explicit test doubles without treating them as real evidence."""
        self.local = local
        self.store = store or V1Store(local.store)
        self.provider = provider or BudgetedProvider(self.store)
        self.github = github or GitHubSource()
        self.markdown = markdown or MarkdownSource()
        self.reviews: Any = None
        local.product_runtime = self
        local.resolver.product_guard = self.guard

    async def sources(self, actor: LocalActor) -> dict:
        """Only authorized source configuration, no credential references."""
        return {
            "items": [
                {key: value for key, value in row.items() if key != "authz_version"}
                for row in await self.store.sources(actor)
                if row["source_id"] in SOURCE_IDS
            ]
        }

    async def sync(self, source_id: str, actor: LocalActor, body: SourceCommand) -> dict:
        """Persist claim, release all locks, collect/embed, then CAS publish."""
        if source_id not in SOURCE_IDS:
            raise DomainError("source_not_found")
        source = await self.store.source(source_id, actor, owner=True)
        prior = await self.store.claim(
            actor, body.command_id, canonical_hash({"kind": "sync", "source": source_id, "body": body}), source_id
        )
        if prior is not None:
            return prior
        if source["version"] != body.expected_version:
            result = {"status": "failed", "reason": "source_version_conflict"}
            await self.store.complete(actor, body.command_id, result)
            return result
        try:
            manifest_digest = await self.markdown.manifest_digest() if source_id == SOURCE_IDS[1] else None
            if source_id == SOURCE_IDS[0]:
                if body.pr_number is None:
                    raise DomainError("source_pr_required")
                collection = await self.github.collect(
                    body.pr_number, required_checks=tuple(source["required_checks"])
                )
            else:
                collection = await self.markdown.collect()
            if collection.status != "verified":
                raise DomainError(collection.reason or "source_unavailable")
            config = self.provider.config(LLMRole.EMBEDDING)
            fingerprint = {
                "documents": {d.document_id: d.content_hash for d in collection.documents},
                "model": config.model,
                "provider": config.provider,
                "dimensions": config.capabilities.embedding_dimensions,
                "strategy": STRATEGY,
                "required_checks": source["required_checks"],
            }
            if source_id == SOURCE_IDS[1]:
                if await self.markdown.manifest_digest() != manifest_digest:
                    raise DomainError("markdown_deleted_or_changed")
                fingerprint["corpus_manifest"] = manifest_digest
            if source["generation"]:
                previous = await self.store.generation(source["generation"])
                if previous["fingerprint"] == fingerprint:
                    await self.local.store.authenticate(actor.session_id, actor.expires_at)
                    if await self.store.source(source_id, actor, owner=True) != source:
                        raise DomainError("source_changed_during_sync")
                    result = {"status": "completed", "generation": source["generation"], "unchanged": True}
                    await self.store.complete(actor, body.command_id, result)
                    return result
            generation = "generation:" + canonical_hash(
                {
                    "source": source_id,
                    "fingerprint": fingerprint,
                    "command": body.command_id,
                    "version": source["version"],
                }
            )
            chunks = chunk_documents(collection.documents, generation)
            vectors = await self.provider.embed([c.text for c in chunks], actor, body.command_id, "index-1")
            chunks = [c.model_copy(update={"vector": vector}) for c, vector in zip(chunks, vectors, strict=True)]
            if source_id == SOURCE_IDS[1]:
                after = await self.markdown.collect()
                if (
                    after.status != "verified"
                    or {d.document_id: d.content_hash for d in after.documents} != fingerprint["documents"]
                    or await self.markdown.manifest_digest() != manifest_digest
                ):
                    raise DomainError("markdown_deleted_or_changed")
            await self.store.publish(
                actor,
                source,
                generation,
                {"fingerprint": fingerprint, "collection": collection.model_dump(mode="json")},
                chunks,
            )
            result = {
                "status": "completed",
                "generation": generation,
                "chunks": len(chunks),
                "checks_passed": collection.checks_passed,
            }
        except DomainError as error:
            result = {"status": "failed", "reason": error.code}
        except (TimeoutError, asyncio.CancelledError):
            # Do not complete or clear an uncertain external attempt.
            raise DomainError("provider_command_outcome_unknown") from None
        await self.store.complete(actor, body.command_id, result)
        return result

    async def bindings(self, actor: LocalActor, binding: ProductBinding) -> dict:
        """Current source rights and active generations, before any ranking."""
        result = {}
        for identifier in SOURCE_IDS:
            source = await self.store.source(identifier, actor)
            if not source["generation"]:
                raise DomainError("index_not_ready")
            config = await self.store.generation(source["generation"])
            model = self.provider.config(LLMRole.EMBEDDING)
            if (
                config["fingerprint"]["model"],
                config["fingerprint"]["provider"],
                config["fingerprint"]["dimensions"],
            ) != (model.model, model.provider, model.capabilities.embedding_dimensions):
                raise DomainError("index_model_mismatch")
            result[identifier] = {
                "version": source["version"],
                "generation": source["generation"],
                "acl": source["acl"],
                "fingerprint": config["fingerprint"],
            }
            if identifier == SOURCE_IDS[0]:
                collection = Collection.model_validate(config["collection"])
                if not collection.documents or collection.documents[0].resource != f"pr:{binding.pr_number}":
                    raise DomainError("index_resource_mismatch")
        current = await self.markdown.collect()
        if (
            current.status != "verified"
            or result[SOURCE_IDS[1]]["fingerprint"].get("corpus_manifest") != await self.markdown.manifest_digest()
            or {d.document_id: d.content_hash for d in current.documents}
            != result[SOURCE_IDS[1]]["fingerprint"]["documents"]
        ):
            raise DomainError("markdown_deleted_or_changed")
        return result

    async def guard(self, product: dict, actor: LocalActor) -> str:
        """Resolver recomputes product seals from live DB source permissions."""
        if product.get("contract_version") == "releaseguard.review.v1.1":
            if self.reviews is None:
                raise DomainError("review_runtime_unavailable")
            return await self.reviews.guard(product, actor)
        binding = ProductBinding.model_validate(product["binding"])
        current = await self.bindings(actor, binding)
        if current != product.get("sources") or product.get("analysis_status") != "completed":
            raise DomainError("product_analysis_stale_or_missing")
        if not product.get("checks_passed"):
            raise DomainError("product_ci_blocked")
        expected = canonical_hash(
            {
                "analysis_id": product["analysis_id"],
                "analysis": product["analysis"],
                "sources": current,
                "binding": product["binding"],
                "evidence_hash": product["evidence_hash"],
                "citations": product["retrieved"],
                "prompt": PROMPT_VERSION,
            }
        )
        if expected != product.get("binding_hash"):
            raise DomainError("product_binding_integrity_failure")
        return expected

    async def fresh(self, change_id: str, actor: LocalActor) -> None:
        """Remote reads occur outside object locks; inner guards recheck DB seals."""
        async with self.local.store.open(change_id, actor):
            product = dict(unit().facts.product)
        if not product:
            return
        if product.get("contract_version") == "releaseguard.review.v1.1":
            if self.reviews is None:
                raise DomainError("review_runtime_unavailable")
            await self.reviews.fresh(change_id, actor, product)
            return
        await self.guard(product, actor)
        binding = ProductBinding.model_validate(product["binding"])
        source = await self.store.source(SOURCE_IDS[0], actor)
        collection = await self.github.collect(binding.pr_number, required_checks=tuple(source["required_checks"]))
        if (
            not collection.checks_passed
            or not collection.documents
            or collection.documents[0].content_hash != product["github_hash"]
        ):
            raise DomainError("product_source_changed_or_unavailable")

    async def analyze(self, change_id: str, actor: LocalActor, body: AnalyzeInput) -> dict:
        """No object lock spans source or Provider latency; stale work is diagnostic."""
        async with self.local.store.open(change_id, actor):
            self.local.owner()
            current, run = unit(), await self.local.run()
            if not current.facts.product:
                raise DomainError("product_mode_required")
            binding = ProductBinding.model_validate(current.facts.product["binding"])
            initial_version, revision = run.state_version, run.revision
        sources = await self.bindings(actor, binding)
        prior = await self.store.claim(
            actor, body.command_id, canonical_hash({"kind": "analyze", "change": change_id, "body": body}), change_id
        )
        if prior is not None:
            return prior
        analysis_id = "analysis:" + canonical_hash({"change": change_id, "command": body.command_id})
        try:
            if initial_version != body.expected_state_version:
                raise DomainError("version_conflict")
            source = await self.store.source(SOURCE_IDS[0], actor)
            github, markdown = await asyncio.gather(
                self.github.collect(binding.pr_number, required_checks=tuple(source["required_checks"])),
                self.markdown.collect(),
            )
            await self.local.store.authenticate(actor.session_id, actor.expires_at)
            async with self.local.store.open(change_id, actor):
                run = await self.local.run()
                if (
                    run.state_version != initial_version
                    or run.revision != revision
                    or await self.bindings(actor, binding) != sources
                ):
                    raise DomainError("analysis_inputs_changed")
                if run.status not in {
                    ChangeStatus.RECEIVED,
                    ChangeStatus.WAITING_FOR_EVIDENCE,
                    ChangeStatus.EVIDENCE_SUFFICIENT,
                }:
                    raise DomainError("analysis_revision_required")
                product = unit().facts.product
                if product.get("analysis_status") in {"started", "completed", "unknown"}:
                    raise DomainError("analysis_revision_required")
                if (
                    github.documents
                    and {d.document_id: d.content_hash for d in github.documents}
                    != sources[SOURCE_IDS[0]]["fingerprint"]["documents"]
                ):
                    raise DomainError("index_source_mismatch")
                if run.status is ChangeStatus.EVIDENCE_SUFFICIENT:
                    assessment = evaluate_evidence_sufficiency(
                        run,
                        snapshots=list(unit().evidence._snapshots.values()),
                        citations=[c for c in unit().evidence._citations.values() if c.revision == run.revision],
                        policy=self.local.evidence_policy(),
                        evaluated_at=unit().at,
                        thread_id=unit().facts.creation["thread_id"],
                        access=evidence_access(),
                    )
                    if not assessment.sufficient:
                        raise DomainError("analysis_revision_required")
                product.update(
                    sources=sources,
                    collections={"ci": github.model_dump(mode="json"), "history": markdown.model_dump(mode="json")},
                    checks_passed=github.checks_passed,
                    analysis_id=analysis_id,
                    analysis_status="started",
                )
                if run.status is not ChangeStatus.EVIDENCE_SUFFICIENT:
                    await self.local.edge(ChangeStatus.COLLECTING_EVIDENCE, StateInputs(), analysis_id + ":collect")
                    await self.local.invoke(
                        evidence_policy=self.local.evidence_policy(), evidence_evaluated_at=unit().at
                    )
                run = await self.local.run()
                if run.status is not ChangeStatus.EVIDENCE_SUFFICIENT or not github.checks_passed:
                    product.update(analysis_status="blocked", reason="evidence_or_ci_blocked")
                    await unit().save()
                    result = {"status": "blocked", "reason": "evidence_or_ci_blocked", "analysis_id": analysis_id}
                    await self.store.complete(actor, body.command_id, result)
                    return result
                version = run.state_version
                product["github_hash"] = github.documents[0].content_hash
                product["evidence_hash"] = canonical_hash(
                    [
                        c.model_dump(mode="json")
                        for c in unit().evidence._citations.values()
                        if c.revision == run.revision
                    ]
                )
                await unit().save()
                request = run.change_request
            token = active_analysis.set(
                {
                    "change_id": change_id,
                    "actor": actor,
                    "body": body,
                    "revision": revision,
                    "version": version,
                    "sources": sources,
                    "binding": binding,
                    "analysis_id": analysis_id,
                }
            )
            try:
                graph = await self.local.graph()
                context = ReleaseGuardGraphContext(
                    change_request=request,
                    run_id=run.run_id,
                    expected_revision=revision,
                    analysis_command_id=body.command_id,
                )
                await graph.ainvoke(
                    GraphState(workflow_kind="releaseguard", releaseguard=context).model_dump(mode="python"),
                    releaseguard_runnable_config(change_id, run.run_id, revision),
                )
            finally:
                active_analysis.reset(token)
            result = {"status": "completed", "analysis_id": analysis_id}
        except DomainError as error:
            result = {"status": "failed", "analysis_id": analysis_id, "reason": error.code}
            async with self.local.store.open(change_id, actor):
                if unit().facts.product.get("analysis_id") == analysis_id:
                    status = "unknown" if "outcome_unknown" in error.code else "failed"
                    unit().facts.product.update(analysis_status=status, reason=error.code)
                    await unit().save()
        except (TimeoutError, asyncio.CancelledError):
            raise DomainError("provider_command_outcome_unknown") from None
        await self.store.complete(actor, body.command_id, result)
        return result

    async def graph_analyze(self, state: GraphState) -> GraphState:
        """The sole analysis node requires ephemeral authenticated command context."""
        if self.reviews is not None and self.reviews.active_work():
            return await self.reviews.graph_analyze(state)
        work = active_analysis.get()
        if (
            work is None
            or state.releaseguard is None
            or state.releaseguard.analysis_command_id != work["body"].command_id
        ):
            raise DomainError("analysis_authenticated_command_required")
        actor, body = work["actor"], work["body"]
        vector = (await self.provider.embed([body.query], actor, body.command_id, "query-1"))[0]
        lexical, semantic = await self.store.candidates(actor, work["binding"].pr_number, body.query, vector)
        chunks = fuse(lexical, semantic)
        if not chunks:
            raise DomainError("analysis_retrieval_empty")
        await self.local.store.authenticate(actor.session_id, actor.expires_at)
        if await self.bindings(actor, work["binding"]) != work["sources"]:
            raise DomainError("analysis_inputs_changed")
        output, telemetry = await self.provider.analyze(
            body.query, chunks, actor, body.command_id, work["analysis_id"]
        )
        await self.local.store.authenticate(actor.session_id, actor.expires_at)
        async with self.local.store.open(work["change_id"], actor):
            async with unit().connection.transaction():
                # Lock only the short commit window, never source/Provider I/O.
                await self.store.lock_sources(unit().connection, work["sources"], actor)
                run = await self.local.run()
                if (
                    run.state_version != work["version"]
                    or run.revision != work["revision"]
                    or unit().facts.product.get("analysis_id") != work["analysis_id"]
                    or await self.bindings(actor, work["binding"]) != work["sources"]
                ):
                    raise DomainError("analysis_inputs_changed")
                product = unit().facts.product
                product.update(
                    analysis_status="completed",
                    analysis=output.model_dump(mode="json"),
                    telemetry=telemetry,
                    analysis_provenance=self.provider.provenance,
                    retrieved=[c.model_dump(mode="json", exclude={"vector"}) for c in chunks],
                )
                product["binding_hash"] = canonical_hash(
                    {
                        "analysis_id": product["analysis_id"],
                        "analysis": product["analysis"],
                        "sources": product["sources"],
                        "binding": product["binding"],
                        "evidence_hash": product["evidence_hash"],
                        "citations": product["retrieved"],
                        "prompt": PROMPT_VERSION,
                    }
                )
                await unit().save()
        return state.model_copy(
            update={
                "releaseguard": state.releaseguard.model_copy(
                    update={"analysis_command_id": None, "analysis_result_id": work["analysis_id"]}
                )
            }
        )

    async def analysis(self, change_id: str, actor: LocalActor) -> dict:
        """Content is reauthorized at read time; stale analysis is explicitly marked."""
        async with self.local.store.open(change_id, actor):
            product = dict(unit().facts.product)
        if not product:
            return {"status": "not_applicable"}
        current = await self.bindings(actor, ProductBinding.model_validate(product["binding"]))
        return {
            "status": product.get("analysis_status", "not_started"),
            "stale": current != product.get("sources"),
            "reason": product.get("reason"),
            "analysis_id": product.get("analysis_id"),
            "analysis": product.get("analysis"),
            "citations": product.get("retrieved", []),
            "telemetry": product.get("telemetry"),
            "binding_hash": product.get("binding_hash"),
            "execution_mode": "offline_simulation",
        }

    async def search(self, actor: LocalActor, body: SearchInput) -> dict:
        """One explicitly budgeted idempotent query, with current scoped vectors."""
        async with self.local.store.open(body.change_id, actor):
            product = unit().facts.product
            if not product:
                raise DomainError("product_mode_required")
            binding = ProductBinding.model_validate(product["binding"])
        sources = await self.bindings(actor, binding)
        prior = await self.store.claim(
            actor, body.command_id, canonical_hash({"kind": "search", "body": body}), body.change_id
        )
        if prior is not None:
            if prior.get("sources") != sources:
                raise DomainError("search_result_stale")
            return prior
        try:
            vector = (await self.provider.embed([body.query], actor, body.command_id, "query-1"))[0]
            lexical, semantic = await self.store.candidates(actor, binding.pr_number, body.query, vector)
            await self.local.store.authenticate(actor.session_id, actor.expires_at)
            if await self.bindings(actor, binding) != sources:
                raise DomainError("search_inputs_changed")
            result = {
                "status": "completed" if semantic else "lexical_degraded",
                "items": [c.model_dump(mode="json", exclude={"vector"}) for c in fuse(lexical, semantic, body.limit)],
                "sources": sources,
            }
        except DomainError as error:
            result = {"status": "failed", "reason": error.code, "sources": sources, "items": []}
        await self.store.complete(actor, body.command_id, result)
        return result
