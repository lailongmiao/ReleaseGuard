"""Readonly review orchestration through the existing single business Graph."""

import asyncio
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg.types.json import Jsonb

from app.core.config import settings
from app.core.logging import logger
from app.services.llm.review_wait import BUSINESS_WAIT_POLICY, REVIEW_COMMAND_SECONDS
from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.approval import ApprovalCommand
from app.schemas.local_mvp import LocalActor, LocalCreate
from app.schemas.v1_backend import Collection, SourceDocument
from app.schemas.v11_backend import (
    CONTRACT,
    CIObservation,
    Coverage,
    ReviewAnalyze,
    ReviewBinding,
    ReviewCitation,
    ReviewCreate,
    ReviewOutput,
    ReviewSearch,
    ReviewView,
    SyncInput,
)
from app.services.local_mvp_storage import LocalUnit, review_scope, unit
from app.services.releaseguard.normalization import canonical_hash, canonical_json
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_rejection import DIAGNOSTIC_KEY, ReviewProviderFailure, ReviewValidationError
from app.services.v1_rag import chunk_documents
from app.services.v11_policy import (
    CAPABILITIES,
    PAUSED_COMMANDS,
    INDEX_POLICY,
    PROMPT_VERSION,
    citation_line_roles,
    diff_chunks,
    fuse_review,
    load_rules,
    qualification,
    require_rehearsal,
    rehearsal_seal,
    stale_reasons,
)
from app.services.v11_provider import ReviewProvider, build_context
from app.services.v11_sources import ReviewGitHubSource
from app.services.v11_storage import V11Store
from app.services.v11_timing import evaluation_command, measured
from app.services.v11_evaluation_policy import require_business_object

review_work: ContextVar[dict | None] = ContextVar("v11_graph_authenticated_work", default=None)


class V11Backend:
    """No second Graph, authority source, autonomous retry or real target adapter."""

    def __init__(
        self,
        product: Any,
        store: V11Store | None = None,
        provider: Any = None,
        github: Any = None,
        evaluation: Any = None,
    ) -> None:
        """Test dependencies must be explicitly injected; real calls default disabled."""
        self.product, self.local = product, product.local
        self.store = store or V11Store(self.local.store)
        self.provider = provider or ReviewProvider(self.store, enabled=settings.V11_REAL_MODELS_ENABLED)
        self.github = github or ReviewGitHubSource()
        self.injected_source = github is not None
        self.evaluation = evaluation
        if evaluation is not None:
            self.store.evaluation_recovery_hash = getattr(evaluation.permit, "recovery_manifest_hash", None)
            self.store.evaluation_recovery_boot_id = getattr(evaluation, "recovery_boot_id", None)
            self.store.evaluation_ceiling = (
                evaluation.permit.ledger_charge_at_start + evaluation.permit.maximum_batch_charge
            )
            if evaluation.permit.mode == "real":
                self.store.evaluation_attempts = evaluation.allowed_attempts()
                self.store.evaluation_owner = evaluation.permit.owner_id
                self.store.evaluation_expires_at = evaluation.permit.expires_at
                if github is None:
                    self.github = ReviewGitHubSource(evaluation.github_transport)
        product.reviews = self

    def active_work(self) -> bool:
        """Ephemeral context cannot be reconstructed from checkpoint/model fields."""
        return review_work.get() is not None

    def reads_enabled(self) -> None:
        """Enabling V1 or possessing a token does not authorize V1.1 network reads."""
        if not self.injected_source and not settings.V11_REAL_READS_ENABLED:
            raise DomainError("review_real_reads_disabled")

    async def register(self, current: LocalUnit, actor: LocalActor) -> None:
        """Create object metadata alongside the new local aggregate, retaining its ID."""
        async with current.connection.transaction():
            await self.store.add_object(
                current.row["change_id"],
                actor,
                ReviewBinding.model_validate(current.facts.product["binding"]),
                current.connection,
            )
        current.facts.product["object_id"] = current.row["change_id"]

    async def commit_guard(self, current: LocalUnit) -> None:
        """Linearize v3 domain writes with ACL, rule and object pointer changes."""
        expected = current.facts.product.get("review_dependencies")
        actual = await self.store.dependencies(current.row["change_id"], current.actor, current.connection)
        scope = review_scope.get()
        if scope is not None and scope.get("invalidation_only"):
            if actual["object_version"] != scope["object_version"]:
                raise DomainError("object_version_conflict")
            return
        if expected is None or actual != expected:
            raise DomainError("review_dependencies_changed")

    async def bootstrap(self, owner: int) -> None:
        """Register approved local rule sets without performing embedding or network I/O."""
        await self.store.bootstrap_reviews(owner)
        _, entries = await asyncio.to_thread(load_rules, "checkout")
        async with await self.store.connection.connect() as db, db.transaction():
            for name in sorted({entry["rule_set_id"] for entry, _ in entries}):
                documents = [entry for entry, _ in entries if entry["rule_set_id"] == name]
                await db.execute(
                    "INSERT INTO v11_rule_sets(rule_set_id,source_id,applicability,content_hash,manifest) VALUES(%s,'markdown-rules',%s,%s,%s) ON CONFLICT DO NOTHING",
                    (
                        name,
                        Jsonb({"repositories": documents[0]["repositories"]}),
                        canonical_hash(documents),
                        Jsonb(documents),
                    ),
                )

    async def sync_rules(self, name: str, actor: LocalActor, body: SyncInput) -> dict:
        """Rule owner explicitly indexes a set; an object sync cannot update global rules."""
        await self.store.source("markdown-rules", actor, owner=True)
        if name not in {"application-policy", "checkout-reference"}:
            raise DomainError("source_not_found")
        prior = await self.store.claim_review(
            actor,
            body.command_id,
            canonical_hash([CONTRACT, "rules", name, actor.user_id, body]),
            name,
            settings.V11_BUDGET_CATEGORY,
        )
        if prior is not None:
            return prior
        _, entries = await asyncio.to_thread(load_rules, "checkout")
        chosen = [(entry, text) for entry, text in entries if entry["rule_set_id"] == name]
        manifest = [entry for entry, _ in chosen]
        digest = canonical_hash(manifest)
        generation = "rules:" + canonical_hash([name, digest, INDEX_POLICY])
        # Durable rule-set claim uses the same namespace as other provider commands.
        async with await self.store.connection.connect() as db, db.transaction():
            row = await (
                await db.execute("SELECT * FROM v11_rule_sets WHERE rule_set_id=%s FOR UPDATE", (name,))
            ).fetchone()
            if not row or row["version"] != body.expected_object_version:
                raise DomainError("rule_version_conflict")
            unresolved = await (
                await db.execute(
                    "SELECT 1 FROM v1_commands WHERE resource=%s AND status='started' AND (owner_id,command_id)<>(%s,%s) LIMIT 1",
                    (name, actor.user_id, body.command_id),
                )
            ).fetchone()
            if unresolved:
                raise DomainError("rule_index_unknown_hold")
            if row["generation"] == generation:
                return await self.finish(
                    actor, body.command_id, {"status": "completed", "generation_id": generation, "unchanged": True}
                )
        at = datetime.now(UTC)
        documents = [
            SourceDocument(
                document_id="rule-doc:" + canonical_hash(entry["filename"]),
                source_id="markdown-rules",
                resource=entry["filename"],
                version=entry["sha256"],
                content=text,
                content_hash=canonical_hash(text),
                locator="knowledge:" + entry["filename"],
                captured_at=at,
            )
            for entry, text in chosen
        ]
        chunks = await asyncio.to_thread(chunk_documents, documents, generation)
        vectors = await self.provider.embed([c.text for c in chunks], actor, body.command_id, "index")
        await self.local.store.authenticate(actor.session_id, actor.expires_at)
        async with await self.store.connection.connect() as db, db.transaction():
            source = await (
                await db.execute(
                    "SELECT acl,owner_id,deleted FROM v1_sources WHERE source_id='markdown-rules' FOR SHARE"
                )
            ).fetchone()
            row = await (
                await db.execute("SELECT version FROM v11_rule_sets WHERE rule_set_id=%s FOR UPDATE", (name,))
            ).fetchone()
            if (
                not source
                or source["deleted"]
                or source["owner_id"] != actor.user_id
                or actor.user_id not in source["acl"]
                or not row
                or row["version"] != body.expected_object_version
            ):
                raise DomainError("rule_index_not_adopted")
            await db.execute(
                "INSERT INTO v1_generations(generation,source_id,config) VALUES(%s,'markdown-rules',%s)",
                (generation, Jsonb({"rule_set": name, "content_hash": digest})),
            )
            for chunk, vector in zip(chunks, vectors, strict=True):
                await db.execute(
                    "INSERT INTO v1_chunks(generation,chunk_id,document_id,resource,content,embedding,dimensions,metadata) VALUES(%s,%s,%s,%s,%s,%s::vector,1536,%s)",
                    (
                        generation,
                        chunk.chunk_id,
                        chunk.document.document_id,
                        chunk.document.resource,
                        chunk.text,
                        str(vector),
                        Jsonb(chunk.model_dump(mode="json", exclude={"vector"})),
                    ),
                )
            await db.execute(
                "UPDATE v11_rule_sets SET generation=%s,version=version+1,content_hash=%s,manifest=%s WHERE rule_set_id=%s",
                (generation, digest, Jsonb(manifest), name),
            )
        return await self.finish(actor, body.command_id, {"status": "completed", "generation_id": generation})

    async def checked_dependencies(self, identifier: str, actor: LocalActor) -> dict:
        """Changed local rules require explicit rule synchronization before adoption."""
        expected = await self.store.dependencies(identifier, actor)
        obj = await self.store.object_row(identifier, actor)
        _, entries = await asyncio.to_thread(load_rules, obj["repository_id"])
        sets = {
            name: canonical_hash([entry for entry, _ in entries if entry["rule_set_id"] == name])
            for name in {entry["rule_set_id"] for entry, _ in entries}
        }
        saved = {row["rule_set_id"]: row for row in expected["rules"]}
        if sets.keys() != saved.keys() or any(
            not saved[name]["generation"] or saved[name]["content_hash"] != digest for name, digest in sets.items()
        ):
            raise DomainError("rule_sync_required")
        return expected

    async def create(self, body: ReviewCreate, actor: LocalActor) -> ReviewView:
        """Only this explicit entry point supplies the trusted v3 creation context."""
        await self.store.repository(body.binding.repository_id, actor)
        await self.store.source("markdown-rules", actor)
        token = review_scope.set({"binding": body.binding.model_dump(mode="json")})
        try:
            result = await self.local.create(
                LocalCreate(
                    change=body.change, approver_user_id=body.approver_user_id, idempotency_key=body.idempotency_key
                ),
                actor,
            )
            return await self.view(result.state.change_id, actor)
        finally:
            review_scope.reset(token)

    async def view(self, identifier: str, actor: LocalActor) -> ReviewView:
        """Show independent domain, analysis and qualification states."""
        row = await self.store.object_row(identifier, actor)
        async with self.local.store.open(identifier, actor):
            state = await self.local.run()
            product = dict(unit().facts.product)
            local_view = await self.local.view()
        coverage, ci, output, stale = None, None, None, []
        analysis = product.get("analysis_id")
        saved: dict = {}
        report: dict = {}
        if analysis:
            saved = await self.store.analysis_row(identifier, analysis, actor)
            report = saved["report"]
            if report.get("output"):
                output = ReviewOutput.model_validate(report["output"])
                coverage = Coverage.model_validate(report["coverage"])
            if row["active_snapshot_id"]:
                snapshot = await self.store.snapshot(identifier, row["active_snapshot_id"], actor)
                observed = None
                try:
                    observed = await self.store.observation(identifier, row["active_snapshot_id"], actor)
                except DomainError as error:
                    if error.code != "observation_missing":
                        raise
                stale = stale_reasons(
                    {**snapshot, "captured_at": observed["captured_at"] if observed else snapshot["captured_at"]},
                    saved["dependencies"],
                    await self.store.dependencies(identifier, actor),
                )
        if row["active_snapshot_id"]:
            snapshot = await self.store.snapshot(identifier, row["active_snapshot_id"], actor)
            coverage = coverage or Coverage.model_validate(snapshot["facts"]["coverage"])
            try:
                ci = CIObservation.model_validate(
                    (await self.store.observation(identifier, row["active_snapshot_id"], actor))["facts"]
                )
            except DomainError as error:
                if error.code != "observation_missing":
                    raise
                stale.append("observation_missing")
            if ci is not None and ci.captured_at + timedelta(minutes=15) <= datetime.now(UTC):
                stale = sorted(set([*stale, "snapshot_expired"]))
        if analysis:
            if saved["status"] != "completed":
                stale.append("analysis_not_completed")
            if report.get("evaluation_only"):
                stale.append("evaluation_report_not_business")
        await self.local.store.authenticate(actor.session_id, actor.expires_at)
        await self.store.object_row(identifier, actor)
        return ReviewView(
            object_id=identifier,
            object_version=row["object_version"],
            owner_user_id=row["owner_user_id"],
            approver_user_id=row["approver_user_id"],
            approval_request=local_view.approval.model_dump(mode="json") if local_view.approval else None,
            binding=ReviewBinding(repository_id=row["repository_id"], selector=row["selector"]),
            state=state.model_dump(mode="json"),
            active_analysis_id=analysis,
            active_snapshot_id=row["active_snapshot_id"],
            active_generation_id=row["active_generation_id"],
            collection_status=row["collection_status"],
            analysis_status=product.get("analysis_status", "not_started"),
            freshness="stale" if stale else "current_as_of" if coverage else "unchecked",
            stale_reasons=stale,
            coverage=coverage,
            rehearsal_eligibility=qualification(
                row["repository_id"],
                row["kind"],
                coverage,
                ci,
                output,
                stale,
                evaluation_readonly=row.get("evaluation_readonly", False),
            ),
            capabilities=CAPABILITIES,
        )

    async def verify_repository(self, repository: str, actor: LocalActor) -> dict:
        """Explicit verification never happens as a startup side effect."""
        row = await self.store.repository(repository, actor)
        await self.store.source(row["source_id"], actor, owner=True)
        self.reads_enabled()
        identity = await self.github.identity(repository, row if row["github_id"] else None)
        await self.store.pin_repository(repository, actor, identity)
        return await self.store.repository(repository, actor)

    @evaluation_command
    async def sync(self, identifier: str, actor: LocalActor, body: SyncInput) -> dict:
        """Persist collection claim, perform bounded reads, then publish via CAS."""
        obj = await self.store.object_row(identifier, actor)
        if obj["owner_user_id"] != actor.user_id:
            raise DomainError("local_owner_required")
        self.reads_enabled()
        if obj["verification"] != "verified":
            raise DomainError("repository_unverified")
        evaluation_position = None
        if self.evaluation is not None:
            evaluation_position = self.evaluation.sync_position(body.command_id, obj)
            await self.evaluation.check_overlap(self.store, actor, identifier, evaluation_position)
        prior = await self.store.claim_review(
            actor,
            body.command_id,
            canonical_hash([CONTRACT, "sync", actor.user_id, identifier, body]),
            identifier,
            settings.V11_BUDGET_CATEGORY,
        )
        if prior is not None:
            return prior
        claimed = False
        try:
            async with asyncio.timeout(120):
                obj = await self.store.claim_sync(identifier, actor, body.command_id, body.expected_object_version)
                claimed = True
                binding = ReviewBinding(repository_id=obj["repository_id"], selector=obj["selector"])
                collection = await measured("source_collection", self.github.collect(binding, obj))
                if (
                    evaluation_position
                    and collection.comparison.model_dump(mode="json") != evaluation_position["comparison"]
                ):
                    raise DomainError("evaluation_frozen_source_mismatch")
                await self.local.store.authenticate(actor.session_id, actor.expires_at)
                result = await self.store.publish_snapshot(
                    identifier, actor, body.command_id, body.expected_object_version, collection
                )
                snapshot = await self.store.snapshot(identifier, result["snapshot_id"], actor)
                current = await self.store.object_row(identifier, actor)
                if current["active_generation_id"]:
                    await self.store.release_sync(identifier, actor, body.command_id)
                    return await self.finish(
                        actor,
                        body.command_id,
                        {"status": "completed", **result, "generation_id": current["active_generation_id"]},
                    )
                if evaluation_position and evaluation_position["repeat"] == 2:
                    raise DomainError("evaluation_second_index_forbidden")
                generation = "generation:" + canonical_hash([identifier, result["snapshot_id"], INDEX_POLICY])
                chunks, citations, complete = await measured(
                    "index_chunking",
                    asyncio.to_thread(
                        diff_chunks,
                        identifier,
                        result["snapshot_id"],
                        generation,
                        obj["source_id"],
                        snapshot["facts"],
                        snapshot["captured_at"],
                    ),
                )
                if not chunks:
                    await self.store.release_sync(identifier, actor, body.command_id)
                    return await self.finish(
                        actor,
                        body.command_id,
                        {
                            "status": "completed",
                            **result,
                            "reason": "no_changes" if not collection.files else "unsupported_content",
                        },
                    )
                vectors = await measured(
                    "index_embedding",
                    self.provider.embed([chunk.text for chunk in chunks], actor, body.command_id, "index"),
                )
                chunks = [
                    chunk.model_copy(update={"vector": vector}) for chunk, vector in zip(chunks, vectors, strict=True)
                ]
                await self.local.store.authenticate(actor.session_id, actor.expires_at)
                await measured(
                    "index_storage",
                    self.store.publish_index(
                        identifier,
                        actor,
                        result["snapshot_id"],
                        result["object_version"],
                        generation,
                        chunks,
                        {
                            "policy": INDEX_POLICY,
                            "complete": complete,
                            "citations": [c.model_dump(mode="json") for c in citations],
                        },
                    ),
                )
                result.update(
                    status="completed", generation_id=generation, object_version=result["object_version"] + 1
                )
                await self.store.release_sync(identifier, actor, body.command_id)
        except DomainError as error:
            unknown = "unknown" in error.code or "timeout" in error.code or "unavailable" in error.code and claimed
            if claimed:
                await self.store.finish_sync(identifier, actor, body.command_id, unknown=unknown)
            result = {"status": "unknown" if unknown else "failed", "reason": error.code}
        except (TimeoutError, asyncio.CancelledError):
            raise DomainError("sync_outcome_unknown") from None
        return await self.finish(actor, body.command_id, result)

    async def finish(self, actor: LocalActor, command: str, result: dict) -> dict:
        """Only explicit known completion is a replayable command result."""
        await self.store.complete(actor, command, result)
        return result

    async def rule_citations(
        self, repository: str, snapshot: str, actor: LocalActor
    ) -> tuple[list[ReviewCitation], list[ReviewCitation]]:
        """Policies are verified from the pinned local corpus, independent of model text."""
        await self.store.source("markdown-rules", actor)
        _, entries = await asyncio.to_thread(load_rules, repository)
        mandatory, references = [], []
        for entry, text in entries:
            documents = [e for e, _ in entries if e["rule_set_id"] == entry["rule_set_id"]]
            generation = "rules:" + canonical_hash([entry["rule_set_id"], canonical_hash(documents), INDEX_POLICY])
            citation = ReviewCitation.model_validate(
                {
                    "citation_id": "rule:" + canonical_hash([entry["filename"], entry["sha256"]]),
                    "source_role": entry["role"],
                    "kind": "rule",
                    "snapshot_id": snapshot,
                    "generation_id": generation,
                    "text": text,
                    "text_hash": canonical_hash(text),
                    "locator": "knowledge:" + entry["filename"] + "#" + entry["version_condition"],
                }
            )
            (mandatory if entry["role"] == "application_policy" else references).append(citation)
        return mandatory, references

    @evaluation_command
    async def analyze(self, identifier: str, actor: LocalActor, body: ReviewAnalyze) -> dict:
        """CI is orthogonal to readonly qualification; analysis never advances the Gate."""
        obj = await self.store.object_row(identifier, actor)
        if obj["owner_user_id"] != actor.user_id:
            raise DomainError("local_owner_required")
        if self.evaluation is not None:
            await self.evaluation.authorize(self.store, actor, identifier, body)
        elif obj.get("evaluation_readonly"):
            raise DomainError("evaluation_position_not_authorized")
        prior = await self.store.claim_review(
            actor,
            body.command_id,
            canonical_hash([CONTRACT, "analyze", actor.user_id, identifier, body]),
            identifier,
            settings.V11_BUDGET_CATEGORY,
        )
        if prior is not None:
            return prior
        analysis = "analysis:" + canonical_hash([identifier, body.command_id])
        claimed = False
        try:
            async with asyncio.timeout(REVIEW_COMMAND_SECONDS):
                if (obj["object_version"], obj["active_snapshot_id"], obj["active_generation_id"]) != (
                    body.expected_object_version,
                    body.snapshot_id,
                    body.generation_id,
                ):
                    raise DomainError("review_version_conflict")
                snapshot = await self.store.snapshot(identifier, body.snapshot_id, actor)
                observation = await self.store.observation(identifier, body.snapshot_id, actor)
                if observation["captured_at"] + timedelta(minutes=15) <= datetime.now(UTC):
                    raise DomainError("snapshot_expired")
                expected = await self.checked_dependencies(identifier, actor)
                async with self.local.store.open(identifier, actor):
                    run = await self.local.run()
                    if self.evaluation is not None:
                        self.evaluation.validate_revision(body.command_id, run.revision)
                    if run.state_version != body.expected_state_version:
                        raise DomainError("version_conflict")
                    if run.status.value not in {"RECEIVED", "WAITING_FOR_EVIDENCE", "EVIDENCE_SUFFICIENT"}:
                        raise DomainError("revision_required")
                    input_hash = canonical_hash(
                        [
                            snapshot["content_hash"],
                            body.query,
                            PROMPT_VERSION,
                            [
                                {"rule_set_id": row["rule_set_id"], "content_hash": row["content_hash"]}
                                for row in expected["rules"]
                            ],
                        ]
                    )
                    await self.store.claim_analysis(
                        identifier,
                        actor,
                        body.command_id,
                        analysis,
                        run.revision,
                        body.snapshot_id,
                        body.generation_id,
                        observation["observation_id"],
                        input_hash,
                        expected,
                    )
                    claimed = True
                    unit().facts.product.update(analysis_id=analysis, analysis_status="started")
                    await unit().save()
                token = review_work.set(
                    {
                        "object_id": identifier,
                        "actor": actor,
                        "body": body,
                        "analysis_id": analysis,
                        "expected": expected,
                        "snapshot": snapshot,
                        "observation": observation,
                        "revision": run.revision,
                        "version": run.state_version,
                        "repository": obj["repository_id"],
                        "environment": run.change_request.environment,
                    }
                )
                try:
                    graph = await self.local.graph()
                    context = ReleaseGuardGraphContext(
                        change_request=run.change_request,
                        run_id=run.run_id,
                        expected_revision=run.revision,
                        analysis_command_id=body.command_id,
                    )
                    await graph.ainvoke(
                        GraphState(workflow_kind="releaseguard", releaseguard=context).model_dump(mode="python"),
                        releaseguard_runnable_config(identifier, run.run_id, run.revision),
                    )
                finally:
                    review_work.reset(token)
                result = {"status": "completed", "analysis_id": analysis}
        except DomainError as error:
            status = (
                "unknown" if any(word in error.code for word in ("unknown", "timeout", "unavailable")) else "failed"
            )
            result = {"status": status, "analysis_id": analysis, "reason": error.code}
            if claimed:
                failure: dict = dict(result)
                if isinstance(error, (ReviewValidationError, ReviewProviderFailure)) and error.diagnostic is not None:
                    failure[DIAGNOSTIC_KEY] = error.diagnostic
                await self.store.analysis_failure(identifier, actor, analysis, failure)
                try:
                    async with self.local.store.open(identifier, actor):
                        if unit().facts.product.get("analysis_id") == analysis:
                            unit().facts.product.update(analysis_status=status, reason=error.code)
                            await unit().save()
                except DomainError:
                    # Revocation or a newer authority binding must not be overwritten by diagnostics.
                    pass
        except (TimeoutError, asyncio.CancelledError):
            raise DomainError("analysis_outcome_unknown") from None
        return await self.finish(actor, body.command_id, result)

    async def graph_analyze(self, state: GraphState) -> GraphState:
        """Sole Graph analysis node; every Provider operation runs outside object locks."""
        work = review_work.get()
        if (
            work is None
            or state.releaseguard is None
            or state.releaseguard.analysis_command_id != work["body"].command_id
        ):
            raise DomainError("analysis_authenticated_command_required")
        actor, body, identifier = work["actor"], work["body"], work["object_id"]
        vector = (
            await measured("query_embedding", self.provider.embed([body.query], actor, body.command_id, "query-1"))
        )[0]
        lexical, semantic = await self.store.review_candidates(
            identifier, actor, body.generation_id, body.query, vector
        )
        ranked = fuse_review(lexical, semantic)
        config = await self.store.generation(body.generation_id)
        diffs = [ReviewCitation.model_validate(c) for c in config["citations"]]
        if any(c.snapshot_id != body.snapshot_id or c.generation_id != body.generation_id for c in diffs):
            raise DomainError("diff_citation_identity_invalid")
        mandatory, available = await self.rule_citations(work["repository"], body.snapshot_id, actor)
        references = []
        for chunk in ranked:
            original = next(
                (c for c in available if c.locator.startswith("knowledge:" + chunk.document.resource + "#")), None
            )
            if original is not None:
                references.append(
                    original.model_copy(
                        update={
                            "citation_id": chunk.chunk_id,
                            "text": chunk.text,
                            "text_hash": chunk.text_hash,
                            "generation_id": chunk.generation,
                            "locator": chunk.locator + "#" + original.locator.split("#", 1)[1],
                        }
                    )
                )
        observed_ci = work["observation"]["facts"]
        server_observations = {
            "repository": work["repository"],
            "checkout_usage_references_applicable": work["repository"] == "checkout",
            "environment": work["environment"],
            "comparison": work["snapshot"]["facts"]["comparison"],
            "coverage": work["snapshot"]["facts"]["coverage"],
            "index_complete": config["complete"],
            "ci": {
                key: observed_ci.get(key) for key in ("head_sha", "status", "complete", "checks_passed", "reasons")
            },
        }
        ci_text = canonical_json(server_observations["ci"])
        mandatory.append(
            ReviewCitation(
                citation_id="ci:" + work["observation"]["observation_id"],
                source_role="ci_observation",
                kind="ci",
                snapshot_id=body.snapshot_id,
                generation_id=body.generation_id,
                sha=observed_ci["head_sha"],
                text=ci_text,
                text_hash=canonical_hash(ci_text),
                locator="github-checks:" + observed_ci["head_sha"],
            )
        )
        user, citations = await asyncio.to_thread(
            build_context,
            body.query,
            mandatory,
            references,
            diffs,
            server_observations,
            citation_line_roles(diffs, work["snapshot"]["facts"], identifier),
        )
        await self.local.store.authenticate(actor.session_id, actor.expires_at)
        if await self.checked_dependencies(identifier, actor) != work["expected"]:
            raise DomainError("analysis_inputs_changed")
        if self.evaluation is None:
            output, telemetry = await self.provider.review(
                user,
                citations,
                actor,
                body.command_id,
                work["analysis_id"],
                business_wait_policy=BUSINESS_WAIT_POLICY,
            )
        else:
            capture = await self.evaluation.begin(
                self.store, actor, identifier, body.command_id, user, citations, server_observations
            )
            try:
                output, telemetry = await self.provider.review(
                    user, citations, actor, body.command_id, work["analysis_id"], capture_unvalidated=capture
                )
            except DomainError as error:
                try:
                    await self.evaluation.finish(
                        self.store,
                        actor,
                        body.command_id,
                        "rejected_or_unknown",
                        {"code": error.code, "detail": getattr(error, "diagnostic", None)},
                    )
                except Exception:
                    logger.warning("evaluation_validation_save_failed", original_code=error.code)
                raise
            try:
                await self.evaluation.finish(self.store, actor, body.command_id, "contract_passed")
            except Exception:
                raise DomainError("evaluation_validation_save_failed_no_resend") from None
        telemetry["server_observations"] = server_observations
        coverage = Coverage.model_validate(work["snapshot"]["facts"]["coverage"])
        selected = {c.citation_id for c in citations if c.kind in {"diff", "file_metadata"}}
        complete = config["complete"] and selected == {c.citation_id for c in diffs}
        coverage.analysis_coverage = "complete" if complete else "partial"
        coverage.model_file_ids = sorted(
            {c.file_id for c in citations if c.kind in {"diff", "file_metadata"} and c.file_id}
        )
        coverage.model_changed_lines = self.covered_lines(work["snapshot"]["facts"], citations)
        if not complete:
            coverage.reasons.append("model_context_or_index_limit")
        report = {
            "analysis_id": work["analysis_id"],
            "object_id": identifier,
            "revision": work["revision"],
            "snapshot_id": body.snapshot_id,
            "generation_id": body.generation_id,
            "status": "completed",
            "resolved_comparison": work["snapshot"]["facts"]["comparison"],
            "output": output.model_dump(mode="json"),
            "coverage": coverage.model_dump(mode="json"),
            "citations": [c.model_dump(mode="json") for c in citations],
            "dependencies": work["expected"],
            "ci_observation_id": work["observation"]["observation_id"],
            "telemetry": telemetry,
            "provenance": self.provider.provenance,
            "evaluation_only": self.evaluation is not None,
        }
        await self.local.store.authenticate(actor.session_id, actor.expires_at)
        async with self.local.store.open(identifier, actor):
            async with unit().connection.transaction():
                run = await self.local.run()
                if (run.revision, run.state_version) != (work["revision"], work["version"]):
                    raise DomainError("analysis_not_adopted")
                await self.store.save_analysis(
                    identifier, actor, work["analysis_id"], "completed", report, work["expected"], unit().connection
                )
                unit().facts.product.update(
                    analysis_status="completed",
                    analysis_id=work["analysis_id"],
                    analysis={
                        "risk": output.model_risk.value,
                        "unknowns": output.unknowns,
                        "candidate_steps": output.candidate_steps,
                    },
                    analysis_provenance=self.provider.provenance,
                )
                await unit().save()
        return state.model_copy(
            update={
                "releaseguard": state.releaseguard.model_copy(
                    update={"analysis_command_id": None, "analysis_result_id": work["analysis_id"]}
                )
            }
        )

    def covered_lines(self, facts: dict, citations: list[ReviewCitation]) -> int:
        """Count only actual changed coordinates that were supplied to the model."""
        count = 0
        for file in facts["files"]:
            for hunk in file["hunks"]:
                for line in hunk["lines"]:
                    if line["kind"] == "context":
                        continue
                    side = "LEFT" if line["kind"] == "delete" else "RIGHT"
                    position = line["old_line"] if side == "LEFT" else line["new_line"]
                    count += any(
                        c.kind == "diff"
                        and c.file_id == file["file_id"]
                        and c.hunk_id == hunk["hunk_id"]
                        and c.side == side
                        and c.start_line is not None
                        and c.end_line is not None
                        and c.start_line <= position <= c.end_line
                        for c in citations
                    )
        return count

    async def search(self, identifier: str, actor: LocalActor, body: ReviewSearch) -> dict:
        """Independent query command fixes object, snapshot and generation scope."""
        expected = await self.store.dependencies(identifier, actor)
        snapshot = await self.store.snapshot(identifier, body.snapshot_id, actor)
        prior = await self.store.claim_review(
            actor,
            body.command_id,
            canonical_hash([CONTRACT, "search", actor.user_id, identifier, body]),
            identifier,
            settings.V11_BUDGET_CATEGORY,
            intent_hash=canonical_hash(["search", snapshot["content_hash"], body.query]),
        )
        if prior is not None:
            return prior
        try:
            async with asyncio.timeout(120):
                if (expected["object_version"], expected["snapshot_id"], expected["generation_id"]) != (
                    body.expected_object_version,
                    body.snapshot_id,
                    body.generation_id,
                ):
                    raise DomainError("review_version_conflict")
                vector = (
                    await measured(
                        "query_embedding", self.provider.embed([body.query], actor, body.command_id, "query-1")
                    )
                )[0]
                lexical, semantic = await self.store.review_candidates(
                    identifier, actor, body.generation_id, body.query, vector
                )
                await self.local.store.authenticate(actor.session_id, actor.expires_at)
                if await self.store.dependencies(identifier, actor) != expected:
                    raise DomainError("search_not_adopted")
                config = await self.store.generation(body.generation_id)
                result = {
                    "status": "completed",
                    "citations": config["citations"],
                    "items": [
                        c.model_dump(mode="json", exclude={"vector"})
                        for c in fuse_review(lexical, semantic, body.limit)
                    ],
                }
        except DomainError as error:
            result = {
                "status": "unknown"
                if any(word in error.code for word in ("unknown", "timeout", "unavailable"))
                else "failed",
                "reason": error.code,
            }
        except (TimeoutError, asyncio.CancelledError):
            raise DomainError("search_outcome_unknown") from None
        return await self.finish(actor, body.command_id, result)

    async def command(self, identifier: str, actor: LocalActor, body: Any, kind: str) -> dict:
        """Domain commands share the owner/command ledger with sync, search and analysis."""
        obj = await self.store.object_row(identifier, actor)
        if kind in {"prepare", "decide", "execute"} and obj.get("evaluation_readonly"):
            return {"status": "blocked", "reason": "evaluation_permanently_readonly"}
        if kind in PAUSED_COMMANDS and CAPABILITIES.pause_reason is not None:
            # Apply current policy before even replaying a previously successful command.
            # No claim, budget reservation, remote read or historical result mutation.
            return {"status": "blocked", "reason": CAPABILITIES.pause_reason}
        prior = await self.store.claim_review(
            actor,
            body.command_id,
            canonical_hash([CONTRACT, kind, identifier, actor.user_id, body]),
            identifier,
            settings.V11_BUDGET_CATEGORY,
        )
        if prior is not None:
            return prior
        scope = review_scope.get()
        if scope is None:
            raise DomainError("review_context_required")
        scope["object_version"] = body.expected_object_version
        scope["invalidation_only"] = kind in {"revise", "revoke"}
        try:
            domain_body = (
                ApprovalCommand.model_validate(body.model_dump(exclude={"expected_object_version"}))
                if kind == "decide"
                else body
            )
            result = await self.local.command(identifier, actor, domain_body, kind)
            response = {"status": "completed", "result": result.model_dump(mode="json")}
        except DomainError as error:
            response = {"status": "unknown" if "unknown" in error.code else "blocked", "reason": error.code}
        return await self.finish(actor, body.command_id, response)

    async def analysis(self, identifier: str, analysis: str, actor: LocalActor) -> dict:
        """Append live staleness and qualification without rewriting saved report bytes."""
        saved = await self.store.analysis_row(identifier, analysis, actor)
        snapshot = await self.store.snapshot(identifier, saved["snapshot_id"], actor)
        observed = None
        try:
            observed = await self.store.observation(identifier, saved["snapshot_id"], actor)
        except DomainError as error:
            if error.code != "observation_missing":
                raise
        stale = stale_reasons(
            {**snapshot, "captured_at": observed["captured_at"] if observed else snapshot["captured_at"]},
            saved["dependencies"],
            await self.store.dependencies(identifier, actor),
        )
        obj = await self.store.object_row(identifier, actor)
        ci = CIObservation.model_validate(observed["facts"]) if observed else None
        if observed is None:
            stale.append("observation_missing")
        try:
            await self.checked_dependencies(identifier, actor)
        except DomainError as error:
            if error.code != "rule_sync_required":
                raise
            stale.append("rule_sync_required")
        report = {key: value for key, value in saved["report"].items() if key != DIAGNOSTIC_KEY}
        if saved["status"] != "completed":
            stale.append("analysis_not_completed")
        if report.get("evaluation_only"):
            stale.append("evaluation_report_not_business")
        output = ReviewOutput.model_validate(report["output"]) if report.get("output") else None
        coverage = Coverage.model_validate(report["coverage"]) if report.get("coverage") else None
        return {
            **report,
            "analysis_id": analysis,
            "object_id": identifier,
            "revision": saved["revision"],
            "snapshot_id": saved["snapshot_id"],
            "generation_id": saved["generation"],
            "status": saved["status"],
            "stale_reasons": stale,
            "capabilities": CAPABILITIES.model_dump(mode="json"),
            "rehearsal_eligibility": qualification(
                obj["repository_id"],
                obj["kind"],
                coverage,
                ci,
                output,
                stale,
                evaluation_readonly=obj.get("evaluation_readonly", False),
            ).model_dump(mode="json"),
        }

    async def guard(self, product: dict, actor: LocalActor) -> str:
        """Recompute ReviewSeal from immutable report plus live dependencies."""
        require_business_object(await self.store.object_row(product["object_id"], actor))
        require_rehearsal()
        report = await self.analysis(product["object_id"], product["analysis_id"], actor)
        if not report["rehearsal_eligibility"]["eligible"]:
            raise DomainError("review_rehearsal_ineligible")
        seal = rehearsal_seal(report)
        if product.get("rehearsal_capability_version") != CAPABILITIES.version:
            raise DomainError("review_approval_capability_changed")
        if product.get("binding_hash") != seal:
            raise DomainError("review_binding_integrity_failure")
        return seal

    async def fresh(self, identifier: str, actor: LocalActor, product: dict) -> None:
        """Recheck public identity and frozen SHA outside local object locks."""
        require_business_object(await self.store.object_row(identifier, actor))
        require_rehearsal()
        self.reads_enabled()
        obj = await self.store.object_row(identifier, actor)
        report = await self.analysis(identifier, product.get("analysis_id", ""), actor)
        if not report["rehearsal_eligibility"]["eligible"]:
            raise DomainError("review_rehearsal_ineligible")
        collection = await self.github.collect(ReviewBinding.model_validate(product["binding"]), obj)
        snapshot = await self.store.snapshot(identifier, report["snapshot_id"], actor)
        code = collection.model_dump(mode="json", exclude={"ci", "captured_at", "repository_name"})
        prior_ci = CIObservation.model_validate(
            (await self.store.observation(identifier, report["snapshot_id"], actor))["facts"]
        )
        if (
            canonical_hash(code) != snapshot["content_hash"]
            or not collection.ci.checks_passed
            or not collection.ci.complete
            or canonical_hash(collection.ci.model_dump(exclude={"captured_at"}))
            != canonical_hash(prior_ci.model_dump(exclude={"captured_at"}))
        ):
            raise DomainError("review_source_changed_or_ci_blocked")
        scope = review_scope.get()
        if scope is None:
            raise DomainError("review_context_required")
        scope["fresh"] = {"object_id": identifier, "report": report, "collection": collection}

    async def check_command(self, current: LocalUnit, actor: LocalActor, body: Any, kind: str) -> None:
        """Local command dispatch checks v3 object CAS before any domain mutation."""
        obj = await self.store.object_row(current.row["change_id"], actor)
        if kind in PAUSED_COMMANDS:
            require_business_object(obj)
            require_rehearsal()
        scope = review_scope.get()
        if scope is None or scope.get("object_version") != obj["object_version"]:
            raise DomainError("object_version_conflict")
        if kind == "revise":
            if (getattr(body, "snapshot_id", None), getattr(body, "generation_id", None)) != (
                obj["active_snapshot_id"],
                obj["active_generation_id"],
            ):
                raise DomainError("review_version_conflict")
            if (await self.local.run()).revision >= 20:
                raise DomainError("capacity_exceeded")
            return
        if kind not in {"prepare", "decide", "execute"}:
            return
        fresh = scope.get("fresh")
        if not fresh or fresh["object_id"] != obj["object_id"]:
            raise DomainError("review_remote_check_required")
        report, collection = fresh["report"], fresh["collection"]
        if await self.store.dependencies(obj["object_id"], actor) != report["dependencies"]:
            raise DomainError("review_dependencies_changed")
        if kind == "prepare":
            product = current.facts.product
            product["rehearsal_capability_version"] = CAPABILITIES.version
            product["binding_hash"] = rehearsal_seal(report)
            product["review_dependencies"] = report["dependencies"]
            product["sources"] = {
                source: await self.store.source(source, actor) for source in report["dependencies"]["sources"]
            }
            ci_text = canonical_json(collection.ci)
            ci_document = SourceDocument(
                document_id="ci:" + canonical_hash(collection.ci),
                source_id=obj["source_id"],
                resource=obj["object_id"],
                version=collection.comparison.head_sha,
                content=ci_text,
                content_hash=canonical_hash(ci_text),
                locator="https://github.com/"
                + collection.repository_name
                + "/commit/"
                + collection.comparison.head_sha,
                captured_at=collection.captured_at,
            )
            mandatory, _ = await self.rule_citations(obj["repository_id"], report["snapshot_id"], actor)
            documents = [
                SourceDocument(
                    document_id=c.citation_id,
                    source_id="markdown-rules",
                    resource=c.locator,
                    version=c.generation_id,
                    content=c.text,
                    content_hash=c.text_hash,
                    locator=c.locator,
                    captured_at=collection.captured_at,
                )
                for c in mandatory
            ]
            product["collections"] = {
                "ci": Collection(
                    status="verified",
                    documents=[ci_document],
                    head_sha=collection.comparison.head_sha,
                    checks_passed=True,
                ).model_dump(mode="json"),
                "history": Collection(status="verified", documents=documents, checks_passed=True).model_dump(
                    mode="json"
                ),
            }
