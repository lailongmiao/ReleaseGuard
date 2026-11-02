"""Object-scoped PostgreSQL facts, claims and budget accounting for reviews."""

from decimal import Decimal
from datetime import UTC, datetime
from math import isfinite
from typing import Any, override

from psycopg.types.json import Jsonb
from app.models.v11_backend import ReviewSnapshotRecord
from app.core.config import settings
from app.core.logging import logger

from app.schemas.local_mvp import LocalActor
from app.schemas.v1_backend import Chunk
from app.schemas.v11_backend import CONTRACT, CollectedReview, ReviewBinding
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_storage import V1Store
from app.services.v1_rag import chinese_terms
from app.services.v11_rejection import DIAGNOSTIC_KEY
from app.services.v11_recovery_guard import guard as recovery_guard
from app.services.v11_evaluation_policy import (
    FORMAL_BATCH,
    LINK_CONTRACT,
    OLD_COMMANDS,
    OVERLAP_CASES,
    DATASET_HASH,
    RETIRED_BATCH,
    RETIRED_BATCHES,
    RETIRED_SENDS,
    RETIREMENT_MARKER,
)


class V11Store(V1Store):
    """Every adoption rechecks membership, source ACL and expected versions."""

    evaluation_ceiling: Decimal | None = None
    evaluation_attempts: dict[str, set[str]] | None = None
    evaluation_owner: int | None = None
    evaluation_expires_at: datetime | None = None
    evaluation_recovery_hash: str | None = None
    evaluation_recovery_boot_id: str | None = None

    async def retire_formal_batch(self, actor: LocalActor) -> None:
        """Append fixed revocation evidence only if all four old overlap slots are unused."""
        await self.connection.authenticate(actor.session_id, actor.expires_at)
        await self.source("markdown-rules", actor, owner=True)
        commands = [
            prefix + canonical_hash([batch, DATASET_HASH, case, repeat])
            for batch in RETIRED_BATCHES
            for case in sorted(OVERLAP_CASES)
            for repeat in (1, 2)
            for prefix in ("analysis:", "sync:", "rag:")
        ]
        async with await self.connection.connect() as db, db.transaction():
            await db.execute("SELECT pg_advisory_xact_lock(71801109)")
            await db.execute("SELECT pg_advisory_xact_lock(71800801)")
            for table in ("v1_commands", "v1_budget", "v11_analyses"):
                row = await (
                    await db.execute(f"SELECT 1 FROM {table} WHERE command_id=ANY(%s) LIMIT 1", (commands,))
                ).fetchone()
                if row:
                    raise DomainError("evaluation_retired_overlap_consumed")
            row = await (
                await db.execute(
                    "SELECT 1 FROM v1_commands WHERE contract_version=%s AND result->>'batch_id'=ANY(%s) LIMIT 1",
                    (LINK_CONTRACT, list(RETIRED_BATCHES)),
                )
            ).fetchone()
            if row:
                raise DomainError("evaluation_retired_overlap_consumed")
            saved = {
                "retired_batch": RETIRED_BATCH,
                "retired_batches": list(RETIRED_BATCHES),
                "replacement_batch": FORMAL_BATCH,
                "old_overlap_unconsumed": True,
                "historical_records_unchanged": True,
            }
            await db.execute(
                "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,contract_version,budget_category,result) "
                "VALUES(%s,%s,%s,%s,'completed','v11-formal-retirement.1','acceptance',%s) "
                "ON CONFLICT DO NOTHING",
                (actor.user_id, RETIREMENT_MARKER, canonical_hash(saved), RETIRED_BATCH, Jsonb(saved)),
            )

    async def check_evaluation_overlap(self, identifier: str, actor: LocalActor, *, permitted: bool) -> None:
        """Fail closed on any other uncertain operation for this same source selector."""
        async with await self.connection.connect() as db, db.transaction():
            obj = await self.object_row(identifier, actor, db, lock=True)
            rows = await (
                await db.execute(
                    "SELECT c.owner_id,c.command_id,c.resource FROM v1_commands c JOIN v11_objects o "
                    "ON o.object_id=c.resource WHERE o.repository_id=%s AND o.selector=%s "
                    "AND (c.status='started' OR c.result->>'status'='unknown') ORDER BY c.command_id",
                    (obj["repository_id"], Jsonb(obj["selector"])),
                )
            ).fetchall()
            if any(
                not permitted or row["owner_id"] != actor.user_id or row["command_id"] not in OLD_COMMANDS
                for row in rows
            ):
                raise DomainError("evaluation_unapproved_unknown_overlap")

    async def claim_evaluation_position(self, identifier: str, actor: LocalActor, binding: dict) -> None:
        """Consume one fixed position and permanently mark its object in one transaction.

        Existing command storage supplies durable uniqueness without a migration.
        These records are never finished or rewritten by a business command.
        """
        case, repeat = binding["case_id"], binding["repeat"]
        if binding["batch_id"] != FORMAL_BATCH or case not in OVERLAP_CASES or repeat not in (1, 2):
            raise DomainError("evaluation_overlap_not_authorized")
        marker = f"formal-link:{FORMAL_BATCH}:{case}:{repeat}"
        async with await self.connection.connect() as db, db.transaction():
            await db.execute("SELECT pg_advisory_xact_lock(71801109)")
            retired = await (
                await db.execute(
                    "SELECT result FROM v1_commands WHERE owner_id=%s AND command_id=%s",
                    (actor.user_id, RETIREMENT_MARKER),
                )
            ).fetchone()
            if not retired or retired["result"].get("replacement_batch") != FORMAL_BATCH:
                raise DomainError("evaluation_retirement_required")
            obj = await self.object_row(identifier, actor, db, lock=True)
            if obj["owner_user_id"] != actor.user_id:
                raise DomainError("local_owner_required")
            old = await (
                await db.execute(
                    "SELECT command_id,resource,result FROM v1_commands WHERE owner_id=%s "
                    "AND command_id=ANY(%s) ORDER BY command_id FOR SHARE",
                    (actor.user_id, list(OLD_COMMANDS)),
                )
            ).fetchall()
            if len(old) != len(OLD_COMMANDS) or any(r["result"].get("status") != "unknown" for r in old):
                raise DomainError("evaluation_old_unknown_link_mismatch")
            resources = {r["resource"] for r in old}
            if len(resources) != 1 or identifier in resources:
                raise DomainError("evaluation_old_unknown_link_mismatch")
            old_object = await self.object_row(next(iter(resources)), actor, db, lock=True)
            if any(obj[k] != old_object[k] for k in ("repository_id", "selector")):
                raise DomainError("evaluation_old_unknown_link_mismatch")
            previous = await (
                await db.execute(
                    "SELECT command_id,resource,result FROM v1_commands WHERE contract_version=%s",
                    (LINK_CONTRACT,),
                )
            ).fetchall()
            for row in previous:
                prior = row["result"]
                if row["command_id"] == marker:
                    raise DomainError("evaluation_position_consumed_no_resend")
                if (prior["case_id"] == case) != (row["resource"] == identifier):
                    raise DomainError("evaluation_case_object_mismatch")
            if repeat == 2 and not any(r["result"]["case_id"] == case for r in previous):
                raise DomainError("evaluation_repeat_order_invalid")
            analyses = await (
                await db.execute("SELECT command_id,status FROM v11_analyses WHERE object_id=%s", (identifier,))
            ).fetchall()
            if (repeat == 1 and analyses) or (
                repeat == 2 and (len(analyses) != 1 or analyses[0]["status"] != "completed")
            ):
                raise DomainError("evaluation_prior_analysis_not_eligible")
            saved = {
                **binding,
                "object_id": identifier,
                "old_object_id": old_object["object_id"],
                "old_commands": list(OLD_COMMANDS),
                "no_business_adoption": True,
                "permanently_readonly": True,
            }
            await db.execute(
                "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,contract_version,"
                "budget_category,result) VALUES(%s,%s,%s,%s,'completed',%s,'acceptance',%s)",
                (actor.user_id, marker, canonical_hash(saved), identifier, LINK_CONTRACT, Jsonb(saved)),
            )
            await self.event(db, actor, marker, identifier, "formal_overlap_position_claimed", saved)

    async def save_evaluation_input(self, identifier: str, actor: LocalActor, command: str, digest: str) -> None:
        """Bind the actual compiled input once, before the model reservation/send."""
        async with await self.connection.connect() as db, db.transaction():
            await self.object_row(identifier, actor, db, lock=True)
            row = await (
                await db.execute(
                    "UPDATE v1_commands SET result=result || %s WHERE resource=%s AND owner_id=%s "
                    "AND contract_version=%s AND result->>'analysis_command'=%s "
                    "AND NOT result ? 'actual_input_hash' RETURNING command_id",
                    (Jsonb({"actual_input_hash": digest}), identifier, actor.user_id, LINK_CONTRACT, command),
                )
            ).fetchone()
            if not row:
                raise DomainError("evaluation_input_consumed_or_unbound")

    async def require_real_budget(self) -> None:
        """Synthetic probe reconciliation can never authorize real Provider sends."""
        if not settings.V11_BUDGET_LEDGER_ID:
            raise DomainError("budget_designated_ledger_required")
        async with await self.connection.connect() as db:
            rows = await (
                await db.execute("SELECT evidence_id,ledger_id,reconciled FROM v11_budget_carryovers")
            ).fetchall()
        if not rows or any(
            not row["reconciled"]
            or row["ledger_id"] != settings.V11_BUDGET_LEDGER_ID
            or row["evidence_id"].startswith("synthetic")
            for row in rows
        ):
            raise DomainError("budget_carryover_unreconciled")

    async def dependencies(self, identifier: str, actor: LocalActor, db: Any = None) -> dict:
        """ACL changes are distinct from legacy source generation changes."""
        if db is None:
            async with await self.connection.connect() as connection, connection.transaction():
                return await self.dependencies(identifier, actor, connection)
        obj = await self.object_row(identifier, actor, db, lock=True)
        sources = await (
            await db.execute(
                "SELECT source_id,acl,authz_version,deleted FROM v1_sources WHERE source_id IN (%s,'markdown-rules') ORDER BY source_id FOR SHARE",
                (obj["source_id"],),
            )
        ).fetchall()
        if len(sources) != 2 or any(s["deleted"] or actor.user_id not in s["acl"] for s in sources):
            raise DomainError("source_not_found")
        rules = await (
            await db.execute(
                "SELECT rule_set_id,version,content_hash,generation FROM v11_rule_sets WHERE applicability->'repositories' @> %s OR applicability->'repositories' @> '[\"*\"]'::jsonb ORDER BY rule_set_id FOR SHARE",
                (Jsonb([obj["repository_id"]]),),
            )
        ).fetchall()
        return {
            "sources": {s["source_id"]: s["authz_version"] for s in sources},
            "rules": rules,
            "object_version": obj["object_version"],
            "snapshot_id": obj["active_snapshot_id"],
            "generation_id": obj["active_generation_id"],
            "policy_version": obj["policy_version"],
        }

    async def real_budget_identity(self, actor: LocalActor) -> dict:
        """Only the policy source owner may inspect the reconciled billing identity."""
        await self.connection.authenticate(actor.session_id, actor.expires_at)
        await self.source("markdown-rules", actor, owner=True)
        await self.require_real_budget()
        async with await self.connection.connect() as db:
            rows = await (
                await db.execute(
                    "SELECT evidence_id,evidence_hash,ledger_id,total_charge,smoke_charge,unknown_charge,reconciled FROM v11_budget_carryovers ORDER BY evidence_id"
                )
            ).fetchall()
        await self.source("markdown-rules", actor, owner=True)
        normalized = [
            {key: str(value) if isinstance(value, Decimal) else value for key, value in row.items()} for row in rows
        ]
        return {"ledger_id": settings.V11_BUDGET_LEDGER_ID, "budget_evidence_hash": canonical_hash(normalized)}

    async def claim_review(
        self,
        actor: LocalActor,
        command: str,
        digest: str,
        resource: str,
        category: str,
        intent_hash: str | None = None,
    ) -> dict | None:
        """Use the existing cross-object command namespace without reclassification."""
        async with await self.connection.connect() as db, db.transaction():
            await recovery_guard(
                db, command, actor.user_id, self.evaluation_recovery_hash, self.evaluation_recovery_boot_id
            )
            if intent_hash is not None:
                lock = int(canonical_hash([resource, intent_hash])[:15], 16)
                await db.execute("SELECT pg_advisory_xact_lock(%s)", (lock,))
                held = await (
                    await db.execute(
                        "SELECT 1 FROM v1_commands WHERE resource=%s AND intent_hash=%s AND (status='started' OR result->>'status'='unknown') AND (owner_id,command_id)<>(%s,%s) LIMIT 1",
                        (resource, intent_hash, actor.user_id, command),
                    )
                ).fetchone()
                if held:
                    raise DomainError("provider_intent_unknown_hold")
            await db.execute(
                "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,contract_version,budget_category,intent_hash) VALUES(%s,%s,%s,%s,'started',%s,%s,%s) ON CONFLICT DO NOTHING",
                (actor.user_id, command, digest, resource, CONTRACT, category, intent_hash),
            )
            row = await (
                await db.execute(
                    "SELECT * FROM v1_commands WHERE owner_id=%s AND command_id=%s FOR UPDATE",
                    (actor.user_id, command),
                )
            ).fetchone()
            if not row or row["digest"] != digest:
                raise DomainError("idempotency_conflict")
            if row["status"] == "completed":
                return row["result"]
            if row["result"]:
                raise DomainError("provider_command_outcome_unknown")
            await db.execute(
                "UPDATE v1_commands SET result=%s WHERE owner_id=%s AND command_id=%s",
                (Jsonb({"claimed": True}), actor.user_id, command),
            )
            await self.event(db, actor, command, resource, "command_claimed", {"digest": digest})
            return None

    async def finish_sync(self, identifier: str, actor: LocalActor, command: str, *, unknown: bool) -> None:
        """Known failure ends only its own slot; uncertain provider sends keep the hold."""
        async with await self.connection.connect() as db, db.transaction():
            await self.object_row(identifier, actor, db, lock=True)
            await db.execute(
                "UPDATE v11_objects SET collection_status='failed',sync_claim=CASE WHEN %s THEN sync_claim ELSE NULL END WHERE object_id=%s AND sync_claim=%s",
                (unknown, identifier, command),
            )

    async def release_sync(self, identifier: str, actor: LocalActor, command: str) -> None:
        """Internal known-success completion; no public claim-release endpoint exists."""
        async with await self.connection.connect() as db, db.transaction():
            await self.object_row(identifier, actor, db, lock=True)
            await db.execute(
                "UPDATE v11_objects SET sync_claim=NULL WHERE object_id=%s AND sync_claim=%s", (identifier, command)
            )

    async def snapshot(self, identifier: str, snapshot: str, actor: LocalActor) -> dict:
        """Reauthorize both identity and snapshot ownership before returning content."""
        async with await self.connection.connect() as db, db.transaction():
            await self.object_row(identifier, actor, db, lock=True)
            row = await (
                await db.execute(
                    "SELECT * FROM v11_snapshots WHERE object_id=%s AND snapshot_id=%s", (identifier, snapshot)
                )
            ).fetchone()
            if not row:
                raise DomainError("local_object_not_found")
            validated = ReviewSnapshotRecord.model_validate(row)
            if canonical_hash(validated.facts) != validated.content_hash:
                raise DomainError("snapshot_integrity_failure")
            return row

    async def observation(self, identifier: str, snapshot: str, actor: LocalActor) -> dict:
        """Latest observation remains a separate immutable fact at the same code SHA."""
        async with await self.connection.connect() as db, db.transaction():
            await self.object_row(identifier, actor, db, lock=True)
            row = await (
                await db.execute(
                    "SELECT * FROM v11_observations WHERE object_id=%s AND snapshot_id=%s ORDER BY captured_at DESC,observation_id DESC LIMIT 1",
                    (identifier, snapshot),
                )
            ).fetchone()
            if not row:
                raise DomainError("observation_missing")
            return row

    async def publish_index(
        self,
        identifier: str,
        actor: LocalActor,
        snapshot: str,
        version: int,
        generation: str,
        chunks: list[Chunk],
        config: dict,
    ) -> None:
        """No source-global pointer; generation ownership is enforced by composite FKs."""
        if not chunks or len(chunks) > 64 or sum(len(c.text.encode()) for c in chunks) > 196608:
            raise DomainError("index_capacity_exceeded")
        if any(
            not c.vector or len(c.vector) != 1536 or not any(c.vector) or any(not isfinite(n) for n in c.vector)
            for c in chunks
        ):
            raise DomainError("embedding_invalid_or_dimension_mismatch")
        async with await self.connection.connect() as db, db.transaction():
            obj = await self.object_row(identifier, actor, db, lock=True)
            if obj["object_version"] != version or obj["active_snapshot_id"] != snapshot:
                raise DomainError("index_not_adopted")
            await db.execute(
                "INSERT INTO v1_generations(generation,source_id,object_id,snapshot_id,config) VALUES(%s,%s,%s,%s,%s)",
                (generation, obj["source_id"], identifier, snapshot, Jsonb(config)),
            )
            for chunk in chunks:
                if chunk.generation != generation or chunk.document.resource != identifier:
                    raise DomainError("index_scope_conflict")
                await db.execute(
                    "INSERT INTO v1_chunks(generation,chunk_id,document_id,resource,content,embedding,dimensions,metadata) VALUES(%s,%s,%s,%s,%s,%s::vector,%s,%s)",
                    (
                        generation,
                        chunk.chunk_id,
                        chunk.document.document_id,
                        identifier,
                        chunk.text,
                        str(chunk.vector),
                        1536,
                        Jsonb(chunk.model_dump(mode="json", exclude={"vector"})),
                    ),
                )
            await db.execute(
                "UPDATE v11_objects SET active_generation_id=%s,object_version=object_version+1 WHERE object_id=%s",
                (generation, identifier),
            )

    async def claim_analysis(
        self,
        identifier: str,
        actor: LocalActor,
        command: str,
        analysis: str,
        revision: int,
        snapshot: str,
        generation: str,
        observation: str,
        input_hash: str,
        expected: dict,
    ) -> None:
        """Persistent revision slot and input hold stop new-key and restart resends."""
        async with await self.connection.connect() as db, db.transaction():
            obj = await self.object_row(identifier, actor, db, lock=True)
            if obj["owner_user_id"] != actor.user_id:
                raise DomainError("local_owner_required")
            if obj.get("evaluation_readonly"):
                linked = await (
                    await db.execute(
                        "SELECT 1 FROM v1_commands WHERE resource=%s AND owner_id=%s AND contract_version=%s "
                        "AND result->>'analysis_command'=%s AND (result->>'repeat')::int=%s",
                        (identifier, actor.user_id, LINK_CONTRACT, command, revision),
                    )
                ).fetchone()
                if not linked:
                    raise DomainError("evaluation_position_not_authorized")
            if await self.dependencies(identifier, actor, db) != expected:
                raise DomainError("analysis_inputs_changed")
            prior = await (
                await db.execute(
                    "SELECT status FROM v11_analyses WHERE object_id=%s AND (revision=%s OR (input_hash=%s AND status IN ('started','unknown'))) LIMIT 1",
                    (identifier, revision, input_hash),
                )
            ).fetchone()
            if prior:
                raise DomainError(
                    "analysis_unknown_hold" if prior["status"] in {"started", "unknown"} else "revision_required"
                )
            if not 1 <= revision <= 20:
                raise DomainError("capacity_exceeded")
            await db.execute(
                "INSERT INTO v11_analyses(analysis_id,object_id,revision,snapshot_id,generation,observation_id,owner_id,command_id,input_hash,dependencies) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    analysis,
                    identifier,
                    revision,
                    snapshot,
                    generation,
                    observation,
                    actor.user_id,
                    command,
                    input_hash,
                    Jsonb(expected),
                ),
            )

    async def save_analysis(
        self,
        identifier: str,
        actor: LocalActor,
        analysis: str,
        status: str,
        report: dict,
        expected: dict | None = None,
        db: Any = None,
    ) -> None:
        """Only the original started slot can be adopted, never overwritten."""
        if db is None:
            async with await self.connection.connect() as connection, connection.transaction():
                return await self.save_analysis(identifier, actor, analysis, status, report, expected, connection)
        await self.object_row(identifier, actor, db, lock=True)
        if expected is not None and await self.dependencies(identifier, actor, db) != expected:
            raise DomainError("analysis_not_adopted")
        result = await (
            await db.execute(
                "UPDATE v11_analyses SET status=%s,report=%s WHERE object_id=%s AND analysis_id=%s AND owner_id=%s AND status='started' RETURNING analysis_id",
                (status, Jsonb(report), identifier, analysis, actor.user_id),
            )
        ).fetchone()
        if not result:
            raise DomainError("analysis_slot_conflict")

    async def analysis_row(self, identifier: str, analysis: str, actor: LocalActor) -> dict:
        """Never return saved reports after source revocation."""
        async with await self.connection.connect() as db, db.transaction():
            await self.object_row(identifier, actor, db, lock=True)
            row = await (
                await db.execute(
                    "SELECT * FROM v11_analyses WHERE object_id=%s AND analysis_id=%s", (identifier, analysis)
                )
            ).fetchone()
            if not row:
                raise DomainError("local_object_not_found")
            return row

    async def analysis_failure(self, identifier: str, actor: LocalActor, analysis: str, result: dict) -> None:
        """Finalize only our claimed slot with safe diagnostics, even if access was revoked.

        A bounded redacted rejection can be stored only while source access remains valid.
        It cannot overwrite an adopted report or release an unknown attempt's charge.
        """
        async with await self.connection.connect() as db:
            result = dict(result)
            diagnostic = result.pop(DIAGNOSTIC_KEY, None)
            async with db.transaction():
                saved = await (
                    await db.execute(
                        "UPDATE v11_analyses SET status=%s,report=%s WHERE object_id=%s AND analysis_id=%s AND owner_id=%s AND status='started' RETURNING analysis_id",
                        (result["status"], Jsonb(result), identifier, analysis, actor.user_id),
                    )
                ).fetchone()
            if diagnostic is not None and saved:
                try:
                    # Commit the fact first; retain object-before-analysis lock order here.
                    async with db.transaction():
                        await self.connection.authenticate(actor.session_id, actor.expires_at)
                        await self.object_row(identifier, actor, db, lock=True)
                        await db.execute(
                            "UPDATE v11_analyses SET report=%s WHERE object_id=%s AND analysis_id=%s AND owner_id=%s AND status=%s AND report=%s",
                            (
                                Jsonb({**result, DIAGNOSTIC_KEY: diagnostic}),
                                identifier,
                                analysis,
                                actor.user_id,
                                result["status"],
                                Jsonb(result),
                            ),
                        )
                except DomainError:
                    # Revocation still saves the original failure without its body.
                    pass
                except Exception:
                    logger.exception("review_rejection_diagnostic_not_saved", analysis_id=analysis)

    async def rejection(self, identifier: str, analysis: str, actor: LocalActor) -> dict:
        """Owner-only local diagnostic; recheck saved source dependencies before disclosing text."""
        await self.connection.authenticate(actor.session_id, actor.expires_at)
        async with await self.connection.connect() as db, db.transaction():
            obj = await self.object_row(identifier, actor, db, lock=True)
            if obj["owner_user_id"] != actor.user_id:
                raise DomainError("local_object_not_found")
            row = await (
                await db.execute(
                    "SELECT report,dependencies,snapshot_id,generation FROM v11_analyses "
                    "WHERE object_id=%s AND analysis_id=%s AND owner_id=%s "
                    "AND (status='failed' OR (status='unknown' AND report ? '_rejection_diagnostic'))",
                    (identifier, analysis, actor.user_id),
                )
            ).fetchone()
            if not row:
                raise DomainError("local_object_not_found")
            for source in sorted(row["dependencies"]["sources"]):
                permitted = await (
                    await db.execute("SELECT acl,deleted FROM v1_sources WHERE source_id=%s FOR SHARE", (source,))
                ).fetchone()
                if not permitted or permitted["deleted"] or actor.user_id not in permitted["acl"]:
                    raise DomainError("local_object_not_found")
            await self.connection.authenticate(actor.session_id, actor.expires_at)
            diagnostic = row["report"].get(DIAGNOSTIC_KEY)
            return {
                "object_id": identifier,
                "analysis_id": analysis,
                "snapshot_id": row["snapshot_id"],
                "generation_id": row["generation"],
                "availability": "saved_redacted" if diagnostic is not None else "not_retained_unrecoverable",
                "diagnostic": diagnostic,
            }

    async def review_candidates(
        self, identifier: str, actor: LocalActor, generation: str, query: str, vector: list[float]
    ) -> tuple[list[Chunk], list[Chunk]]:
        """Exact vector and positive lexical rankings are scoped before LIMIT."""
        if len(vector) != 1536 or not any(vector) or any(not isfinite(v) for v in vector):
            raise DomainError("query_vector_invalid")
        async with await self.connection.connect() as db, db.transaction():
            obj = await self.object_row(identifier, actor, db, lock=True)
            if obj["active_generation_id"] != generation:
                raise DomainError("generation_stale")
            generations = [generation]
            rows = await (
                await db.execute(
                    "SELECT generation FROM v11_rule_sets WHERE (applicability->'repositories' @> %s OR applicability->'repositories' @> '[\"*\"]'::jsonb) AND generation IS NOT NULL",
                    (Jsonb([obj["repository_id"]]),),
                )
            ).fetchall()
            generations.extend(row["generation"] for row in rows)
            scope = "generation=ANY(%s) AND (resource=%s OR generation<>%s)"
            semantic = await (
                await db.execute(
                    "SELECT metadata FROM v1_chunks WHERE "
                    + scope
                    + " AND dimensions=1536 ORDER BY embedding <=> %s::vector,chunk_id LIMIT 20",
                    (generations, identifier, generation, str(vector)),
                )
            ).fetchall()
            lexical = await (
                await db.execute(
                    "SELECT metadata FROM v1_chunks WHERE "
                    + scope
                    + " AND (to_tsvector('simple',content) @@ plainto_tsquery('simple',%s) OR EXISTS(SELECT 1 FROM unnest(%s::text[]) term WHERE strpos(content,term)>0)) ORDER BY ts_rank(to_tsvector('simple',content),plainto_tsquery('simple',%s)) DESC,chunk_id LIMIT 20",
                    (generations, identifier, generation, query, list(chinese_terms(query)), query),
                )
            ).fetchall()
            return (
                [Chunk.model_validate(row["metadata"]) for row in lexical],
                [Chunk.model_validate(row["metadata"]) for row in semantic],
            )

    async def bootstrap_reviews(self, owner_id: int) -> None:
        """Local configuration only; repositories stay unverified until explicit sync."""
        if owner_id <= 0:
            raise DomainError("source_owner_not_configured")
        async with await self.connection.connect() as db, db.transaction():
            for repository in ("checkout", "setup-python"):
                source = "review-github-" + repository
                await db.execute(
                    "INSERT INTO v1_sources(source_id,owner_id,acl) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                    (source, owner_id, Jsonb([owner_id])),
                )
                await db.execute(
                    "INSERT INTO v11_repositories(repository_id,source_id,name,policy_version) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (
                        repository,
                        source,
                        "actions/" + repository,
                        "checkout-pr-all-checks.2026-10-09.1" if repository == "checkout" else "analysis_only.v1.1",
                    ),
                )

    async def repositories(self, actor: LocalActor) -> list[dict]:
        """Filter source ACL before returning repository identities."""
        async with await self.connection.connect() as db:
            return await (
                await db.execute(
                    "SELECT r.* FROM v11_repositories r JOIN v1_sources s USING(source_id) WHERE NOT s.deleted AND s.acl @> %s ORDER BY repository_id",
                    (Jsonb([actor.user_id]),),
                )
            ).fetchall()

    async def repository(self, identifier: str, actor: LocalActor) -> dict:
        """Unknown and inaccessible repositories use the same error."""
        for row in await self.repositories(actor):
            if row["repository_id"] == identifier:
                return row
        raise DomainError("source_not_found")

    async def pin_repository(self, identifier: str, actor: LocalActor, identity: dict) -> None:
        """No silent owner transfer or identity replacement after first verification."""
        async with await self.connection.connect() as db, db.transaction():
            row = await (
                await db.execute(
                    "SELECT r.*,s.acl,s.deleted FROM v11_repositories r JOIN v1_sources s USING(source_id) WHERE repository_id=%s FOR UPDATE OF r FOR SHARE OF s",
                    (identifier,),
                )
            ).fetchone()
            if not row or row["deleted"] or actor.user_id not in row["acl"]:
                raise DomainError("source_not_found")
            if row["github_id"] is not None and (row["github_id"], row["github_owner_id"]) != (
                identity["github_id"],
                identity["github_owner_id"],
            ):
                raise DomainError("repository_identity_changed")
            await db.execute(
                "UPDATE v11_repositories SET github_id=%s,github_owner_id=%s,name=%s,verification='verified',verified_at=now() WHERE repository_id=%s",
                (identity["github_id"], identity["github_owner_id"], identity["name"], identifier),
            )

    async def object_row(self, identifier: str, actor: LocalActor, db: Any = None, *, lock: bool = False) -> dict:
        """SQL enforces membership and applicable policy ACL before any row leaves storage."""
        if db is None:
            async with await self.connection.connect() as connection:
                return await self.object_row(identifier, actor, connection)
        row = await (
            await db.execute(
                """SELECT o.*,l.owner_user_id,l.approver_user_id,r.source_id,r.github_id,r.github_owner_id,
                r.name,r.verification,r.policy_version,s.authz_version,
                EXISTS(SELECT 1 FROM v1_commands e WHERE e.resource=o.object_id
                    AND e.contract_version='v11-formal-overlap.1') AS evaluation_readonly
                FROM v11_objects o JOIN local_mvp_records l ON l.change_id=o.object_id
                JOIN v11_repositories r USING(repository_id) JOIN v1_sources s USING(source_id)
                WHERE o.object_id=%s AND %s IN (l.owner_user_id,l.approver_user_id)
                AND NOT s.deleted AND s.acl @> %s
                AND EXISTS(SELECT 1 FROM v1_sources a WHERE a.source_id='markdown-rules' AND NOT a.deleted AND a.acl @> %s)
                AND NOT EXISTS(SELECT 1 FROM v11_rule_sets rs JOIN v1_sources x ON x.source_id=rs.source_id
                    WHERE (rs.applicability->'repositories' @> jsonb_build_array(o.repository_id) OR rs.applicability->'repositories' @> '["*"]'::jsonb)
                    AND (x.deleted OR NOT x.acl @> %s))"""
                + (" FOR UPDATE OF o FOR SHARE OF s,l,r" if lock else ""),
                (identifier, actor.user_id, Jsonb([actor.user_id]), Jsonb([actor.user_id]), Jsonb([actor.user_id])),
            )
        ).fetchone()
        if not row:
            raise DomainError("local_object_not_found")
        if lock:
            session = await (
                await db.execute(
                    "SELECT id FROM session WHERE id=%s AND user_id=%s FOR SHARE", (actor.session_id, actor.user_id)
                )
            ).fetchone()
            if session is None or actor.expires_at <= datetime.now(UTC).timestamp():
                raise DomainError("local_session_invalid")
            sources = await (
                await db.execute(
                    """SELECT s.source_id,s.acl,s.deleted FROM v1_sources s
                WHERE s.source_id IN (%s,'markdown-rules') OR s.source_id IN
                    (SELECT source_id FROM v11_rule_sets WHERE applicability->'repositories' @> %s OR applicability->'repositories' @> '["*"]'::jsonb)
                ORDER BY s.source_id FOR SHARE""",
                    (row["source_id"], Jsonb([row["repository_id"]])),
                )
            ).fetchall()
            if len(sources) < 2 or any(s["deleted"] or actor.user_id not in s["acl"] for s in sources):
                raise DomainError("local_object_not_found")
        return row

    async def add_object(self, identifier: str, actor: LocalActor, binding: ReviewBinding, db: Any) -> None:
        """Called inside creation transaction; bounded across all owners."""
        await db.execute("SELECT pg_advisory_xact_lock(71801101)")
        prior = await (
            await db.execute("SELECT selector,repository_id FROM v11_objects WHERE object_id=%s", (identifier,))
        ).fetchone()
        selector = binding.selector.model_dump(mode="json")
        if prior:
            if prior["selector"] != selector or prior["repository_id"] != binding.repository_id:
                raise DomainError("idempotency_conflict")
            return
        row = await (await db.execute("SELECT COUNT(*) AS count FROM v11_objects")).fetchone()
        if row is None or row["count"] >= 100:
            raise DomainError("capacity_exceeded")
        repository = await self.repository(binding.repository_id, actor)
        await self.source("markdown-rules", actor)
        await db.execute(
            "INSERT INTO v11_objects(object_id,repository_id,kind,selector,selector_key) VALUES(%s,%s,%s,%s,%s)",
            (
                identifier,
                binding.repository_id,
                binding.selector.kind,
                Jsonb(selector),
                canonical_hash([repository["repository_id"], selector]),
            ),
        )

    async def claim_sync(self, identifier: str, actor: LocalActor, command: str, version: int) -> dict:
        """A durable object slot blocks new keys while a collection is uncertain."""
        async with await self.connection.connect() as db, db.transaction():
            row = await self.object_row(identifier, actor, db, lock=True)
            if row["owner_user_id"] != actor.user_id:
                raise DomainError("local_owner_required")
            if row["object_version"] != version:
                raise DomainError("object_version_conflict")
            if row["sync_claim"]:
                raise DomainError("sync_outcome_unknown")
            await db.execute(
                "UPDATE v11_objects SET sync_claim=%s,collection_status='collecting' WHERE object_id=%s",
                (command, identifier),
            )
            return row

    async def publish_snapshot(
        self, identifier: str, actor: LocalActor, command: str, version: int, collection: CollectedReview
    ) -> dict:
        """CI observations do not change immutable code hashes or force reindexing."""
        code = collection.model_dump(mode="json", exclude={"ci", "captured_at", "repository_name"})
        digest = canonical_hash(code)
        snapshot = "snapshot:" + canonical_hash([identifier, digest])
        observation = "observation:" + canonical_hash([identifier, command, collection.ci])
        async with await self.connection.connect() as db, db.transaction():
            row = await self.object_row(identifier, actor, db, lock=True)
            if row["object_version"] != version or row["sync_claim"] != command:
                raise DomainError("sync_not_adopted")
            if (row["github_id"], row["github_owner_id"]) != (collection.github_id, collection.owner_id):
                raise DomainError("repository_identity_changed")
            prior = await (
                await db.execute(
                    "SELECT snapshot_id FROM v11_snapshots WHERE object_id=%s AND content_hash=%s",
                    (identifier, digest),
                )
            ).fetchone()
            if prior is None:
                count = await (
                    await db.execute("SELECT COUNT(*) AS count FROM v11_snapshots WHERE object_id=%s", (identifier,))
                ).fetchone()
                if count is None or count["count"] >= 20:
                    raise DomainError("capacity_exceeded")
                await db.execute(
                    "INSERT INTO v11_snapshots(snapshot_id,object_id,sequence,content_hash,facts,captured_at) VALUES(%s,%s,%s,%s,%s,%s)",
                    (snapshot, identifier, count["count"] + 1, digest, Jsonb(code), collection.captured_at),
                )
                for file in collection.files:
                    await db.execute(
                        "INSERT INTO v11_diff_files(snapshot_id,file_id,path,facts) VALUES(%s,%s,%s,%s)",
                        (snapshot, file.file_id, file.path, Jsonb(file.model_dump(mode="json"))),
                    )
            await db.execute(
                "INSERT INTO v11_observations(observation_id,object_id,snapshot_id,facts,captured_at) VALUES(%s,%s,%s,%s,%s)",
                (
                    observation,
                    identifier,
                    snapshot,
                    Jsonb(collection.ci.model_dump(mode="json")),
                    collection.captured_at,
                ),
            )
            coverage = collection.coverage
            status = "ready" if coverage.content_completeness == "complete" else "partial"
            if coverage.manifest_completeness == "complete" and not collection.files:
                status = "no_changes"
            await db.execute(
                "UPDATE v11_objects SET active_generation_id=CASE WHEN active_snapshot_id=%s THEN active_generation_id ELSE NULL END,active_snapshot_id=%s,object_version=object_version+1,collection_status=%s WHERE object_id=%s",
                (snapshot, snapshot, status, identifier),
            )
            return {
                "snapshot_id": snapshot,
                "observation_id": observation,
                "object_version": version + 1,
                "unchanged": prior is not None,
            }

    @override
    async def reserve(self, actor: LocalActor, command_id: str, attempt: str, amount: Decimal, limit: Decimal) -> None:
        """One designated ledger includes reconciled carryover and unresolved charges."""
        if command_id in RETIRED_SENDS or any(
            command_id.startswith("rules:" + batch + ":") for batch in RETIRED_BATCHES
        ):
            raise DomainError("evaluation_batch_retired")
        if self.evaluation_expires_at is not None and datetime.now(UTC) >= self.evaluation_expires_at:
            raise DomainError("evaluation_approval_expired")
        if self.evaluation_attempts is not None and (
            actor.user_id != self.evaluation_owner or attempt not in self.evaluation_attempts.get(command_id, set())
        ):
            raise DomainError("evaluation_provider_attempt_not_authorized")
        if amount <= 0 or not Decimal("0") < limit <= Decimal("100"):
            raise DomainError("budget_configuration_invalid")
        async with await self.connection.connect() as db, db.transaction():
            await db.execute("SELECT pg_advisory_xact_lock(71800801)")
            await recovery_guard(
                db, command_id, actor.user_id, self.evaluation_recovery_hash, self.evaluation_recovery_boot_id
            )
            prior = await (
                await db.execute(
                    "SELECT 1 FROM v1_budget WHERE owner_id=%s AND command_id=%s AND attempt=%s",
                    (actor.user_id, command_id, attempt),
                )
            ).fetchone()
            if prior:
                raise DomainError("model_attempt_already_reserved")
            carry = await (
                await db.execute(
                    "SELECT COUNT(*) AS count,BOOL_AND(reconciled) AS valid,COALESCE(SUM(total_charge),0) AS total,COALESCE(SUM(smoke_charge),0) AS smoke FROM v11_budget_carryovers"
                )
            ).fetchone()
            command = await (
                await db.execute(
                    "SELECT contract_version,budget_category,resource,status FROM v1_commands WHERE owner_id=%s AND command_id=%s",
                    (actor.user_id, command_id),
                )
            ).fetchone()
            if not carry or not carry["count"] or not carry["valid"]:
                raise DomainError("budget_carryover_unreconciled")
            if (
                not command
                or command["contract_version"] != CONTRACT
                or command["status"] != "started"
                or command["budget_category"] not in {"smoke", "development", "acceptance"}
            ):
                raise DomainError("budget_category_required")
            is_object = await (
                await db.execute("SELECT 1 FROM v11_objects WHERE object_id=%s", (command["resource"],))
            ).fetchone()
            if is_object:
                await self.object_row(command["resource"], actor, db, lock=True)
            elif command["resource"] in {"application-policy", "checkout-reference"}:
                source = await (
                    await db.execute(
                        "SELECT acl,owner_id,deleted FROM v1_sources WHERE source_id='markdown-rules' FOR SHARE"
                    )
                ).fetchone()
                if (
                    not source
                    or source["deleted"]
                    or source["owner_id"] != actor.user_id
                    or actor.user_id not in source["acl"]
                ):
                    raise DomainError("source_owner_required")
            totals = await (
                await db.execute(
                    """SELECT COALESCE(SUM(b.charge),0) AS total,
                COALESCE(SUM(b.charge) FILTER(WHERE c.contract_version=%s),0) AS current,
                COALESCE(SUM(b.charge) FILTER(WHERE c.contract_version=%s AND c.budget_category IN ('smoke','development')),0) AS smoke,
                COALESCE(SUM(b.charge) FILTER(WHERE c.contract_version=%s AND c.budget_category=%s),0) AS category
                FROM v1_budget b JOIN v1_commands c USING(owner_id,command_id)""",
                    (CONTRACT, CONTRACT, CONTRACT, command["budget_category"]),
                )
            ).fetchone()
            category_limit = {"smoke": Decimal("2"), "development": Decimal("3"), "acceptance": Decimal("25")}[
                command["budget_category"]
            ]
            if (
                not totals
                or totals["total"] + carry["total"] + amount > limit
                or totals["current"] + amount > 30
                or totals["category"] + amount > category_limit
                or (
                    self.evaluation_ceiling is not None
                    and totals["total"] + carry["total"] + amount > self.evaluation_ceiling
                )
            ):
                raise DomainError("model_budget_exhausted")
            if (
                command["budget_category"] in {"smoke", "development"}
                and totals["smoke"] + carry["smoke"] + amount > 30
            ):
                raise DomainError("smoke_budget_exhausted")
            if command["budget_category"] == "development" and attempt == "analysis-1":
                count = await (
                    await db.execute(
                        "SELECT COUNT(*) AS count FROM v1_budget b JOIN v1_commands c USING(owner_id,command_id) WHERE c.contract_version=%s AND c.budget_category='development' AND b.attempt='analysis-1'",
                        (CONTRACT,),
                    )
                ).fetchone()
                # Explicit development budget amendment: 17 total calls, never a reset.
                if count and count["count"] >= 17:
                    raise DomainError("development_attempt_limit")
            await db.execute(
                "INSERT INTO v1_budget(owner_id,command_id,attempt,reserved,charge) VALUES(%s,%s,%s,%s,%s)",
                (actor.user_id, command_id, attempt, amount, amount),
            )
