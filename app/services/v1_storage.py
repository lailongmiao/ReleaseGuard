"""Short async PostgreSQL transactions for readonly source and model commands."""

from decimal import Decimal
from math import isfinite
from typing import Any

from psycopg.types.json import Jsonb

from app.schemas.local_mvp import LocalActor
from app.schemas.v1_backend import Chunk, SourceCommand
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_rag import chinese_terms

SOURCE_IDS = ("github-checkout", "markdown-rules")


class V1Store:
    """Authority is PostgreSQL, not model text, process locks, or checkpoints."""

    def __init__(self, connection: LocalPostgresStore | None = None) -> None:
        """Reuse private connection configuration without copying credentials."""
        self.connection = connection or LocalPostgresStore()

    async def event(self, db: Any, actor: LocalActor, command: str, resource: str, action: str, facts: dict) -> None:
        """Atomic append with safe digest; no prompt, secret or raw response."""
        identifier = "v1-event:" + canonical_hash({"owner": actor.user_id, "command": command, "action": action})
        await db.execute(
            "INSERT INTO v1_audit(event_id,owner_id,command_id,resource,action,digest) VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (identifier, actor.user_id, command, resource, action, canonical_hash(facts)),
        )

    async def audit(self, resource: str) -> list[dict]:
        """Caller must first reauthorize the object and its live source ACL."""
        async with await self.connection.connect() as db:
            rows = await (
                await db.execute(
                    "SELECT event_id,command_id,action,digest,created_at FROM v1_audit WHERE resource=%s ORDER BY created_at,event_id",
                    (resource,),
                )
            ).fetchall()
            return [{**row, "created_at": row["created_at"].isoformat()} for row in rows]

    async def bootstrap(self, owner_id: int, checks: tuple[str, ...]) -> None:
        """Explicit server owner and frozen manifest; never overwrite existing ACL."""
        if owner_id <= 0:
            raise DomainError("source_owner_not_configured")
        async with await self.connection.connect() as db:
            async with db.transaction():
                for source in SOURCE_IDS:
                    await db.execute(
                        "INSERT INTO v1_sources(source_id,owner_id,acl,required_checks) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (source, owner_id, Jsonb([owner_id]), Jsonb(list(checks) if source == SOURCE_IDS[0] else [])),
                    )

    async def source(self, source_id: str, actor: LocalActor, *, owner: bool = False) -> dict[str, Any]:
        """A credential or designated-approver role never grants source rights."""
        async with await self.connection.connect() as db:
            cursor = await db.execute("SELECT * FROM v1_sources WHERE source_id=%s", (source_id,))
            row = await cursor.fetchone()
            if row is None or row["deleted"] or actor.user_id not in row["acl"]:
                raise DomainError("source_not_found")
            if owner and row["owner_id"] != actor.user_id:
                raise DomainError("source_owner_required")
            return row

    async def lock_sources(self, db: Any, expected: dict, actor: LocalActor) -> None:
        """Linearize source revocation and analysis adoption in a short PG commit."""
        for identifier in sorted(expected):
            row = await (
                await db.execute(
                    "SELECT version,generation,acl,deleted FROM v1_sources WHERE source_id=%s FOR SHARE", (identifier,)
                )
            ).fetchone()
            value = expected[identifier]
            if (
                not row
                or row["deleted"]
                or actor.user_id not in row["acl"]
                or (row["version"], row["generation"], row["acl"])
                != (value["version"], value["generation"], value["acl"])
            ):
                raise DomainError("analysis_inputs_changed")

    async def sources(self, actor: LocalActor) -> list[dict[str, Any]]:
        """Filter ACL before returning source identities and generation status."""
        async with await self.connection.connect() as db:
            cursor = await db.execute(
                "SELECT * FROM v1_sources WHERE NOT deleted AND acl @> %s ORDER BY source_id",
                (Jsonb([actor.user_id]),),
            )
            return await cursor.fetchall()

    async def generation(self, identifier: str) -> dict:
        """Read immutable generation configuration for version and model checks."""
        async with await self.connection.connect() as db:
            row = await (
                await db.execute("SELECT config FROM v1_generations WHERE generation=%s", (identifier,))
            ).fetchone()
            if row is None:
                raise DomainError("index_generation_missing")
            return row["config"]

    async def grant(
        self, source_id: str, actor: LocalActor, body: SourceCommand, user_id: int, *, revoke: bool
    ) -> dict:
        """Serialize ACL CAS and its command result in one short transaction."""
        digest = canonical_hash({"source": source_id, "body": body, "user": user_id, "revoke": revoke})
        async with await self.connection.connect() as db:
            async with db.transaction():
                cursor = await db.execute("SELECT * FROM v1_sources WHERE source_id=%s FOR UPDATE", (source_id,))
                row = await cursor.fetchone()
                if not row or row["deleted"] or row["owner_id"] != actor.user_id:
                    raise DomainError("source_owner_required")
                prior = await db.execute(
                    "SELECT * FROM v1_commands WHERE owner_id=%s AND command_id=%s", (actor.user_id, body.command_id)
                )
                command = await prior.fetchone()
                if command:
                    if command["digest"] != digest:
                        raise DomainError("idempotency_conflict")
                    return command["result"]
                if body.expected_version != row["version"]:
                    raise DomainError("source_version_conflict")
                users = set(row["acl"])
                if revoke and user_id == row["owner_id"]:
                    raise DomainError("source_owner_revocation_denied")
                if revoke:
                    users.discard(user_id)
                else:
                    valid = await db.execute('SELECT id FROM "user" WHERE id=%s', (user_id,))
                    if await valid.fetchone() is None:
                        raise DomainError("source_user_not_found")
                    users.add(user_id)
                result = {"source_id": source_id, "version": row["version"] + 1, "acl": sorted(users)}
                await db.execute(
                    "UPDATE v1_sources SET acl=%s,version=version+1 WHERE source_id=%s",
                    (Jsonb(sorted(users)), source_id),
                )
                await db.execute(
                    "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,result) VALUES(%s,%s,%s,%s,'completed',%s)",
                    (actor.user_id, body.command_id, digest, source_id, Jsonb(result)),
                )
                await self.event(
                    db, actor, body.command_id, source_id, "source_revoke" if revoke else "source_grant", result
                )
                return result

    async def claim(self, actor: LocalActor, command_id: str, digest: str, resource: str) -> dict | None:
        """Claim before external calls; duplicate in-flight commands never resend."""
        async with await self.connection.connect() as db:
            async with db.transaction():
                await db.execute(
                    "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status) VALUES(%s,%s,%s,%s,'started') ON CONFLICT DO NOTHING",
                    (actor.user_id, command_id, digest, resource),
                )
                row = await (
                    await db.execute(
                        "SELECT * FROM v1_commands WHERE owner_id=%s AND command_id=%s FOR UPDATE",
                        (actor.user_id, command_id),
                    )
                ).fetchone()
                if row is None or row["digest"] != digest:
                    raise DomainError("idempotency_conflict")
                if row["status"] == "completed":
                    return row["result"]
                # xmin is not used as an ownership token. A returned fresh token
                # is installed atomically; any prior token means diagnostic hold.
                if row["result"]:
                    raise DomainError("provider_command_outcome_unknown")
                await db.execute(
                    "UPDATE v1_commands SET result=%s WHERE owner_id=%s AND command_id=%s",
                    (Jsonb({"claimed": True}), actor.user_id, command_id),
                )
                await self.event(db, actor, command_id, resource, "command_claimed", {"digest": digest})
                return None

    async def complete(self, actor: LocalActor, command_id: str, result: dict) -> None:
        """Only an original claimed command is completed, never auto-reclaimed."""
        async with await self.connection.connect() as db:
            async with db.transaction():
                row = await (
                    await db.execute(
                        "UPDATE v1_commands SET status='completed',result=%s WHERE owner_id=%s AND command_id=%s AND status='started' RETURNING resource",
                        (Jsonb(result), actor.user_id, command_id),
                    )
                ).fetchone()
                if row:
                    await self.event(db, actor, command_id, row["resource"], "command_completed", result)

    async def reserve(self, actor: LocalActor, command_id: str, attempt: str, amount: Decimal, limit: Decimal) -> None:
        """Global transaction lock prevents concurrent commands overspending."""
        if amount <= 0 or limit > Decimal("100") or limit <= 0:
            raise DomainError("budget_configuration_invalid")
        async with await self.connection.connect() as db:
            async with db.transaction():
                await db.execute("SELECT pg_advisory_xact_lock(71800801)")
                row = await (await db.execute("SELECT COALESCE(SUM(charge),0) AS spent FROM v1_budget")).fetchone()
                carry = Decimal("0")
                migrated = await (
                    await db.execute("SELECT to_regclass('v11_budget_carryovers') AS relation")
                ).fetchone()
                if migrated and migrated["relation"]:
                    history = await (
                        await db.execute("SELECT COALESCE(SUM(total_charge),0) AS total FROM v11_budget_carryovers")
                    ).fetchone()
                    carry = history["total"] if history else Decimal("0")
                if row is None or row["spent"] + carry + amount > limit:
                    raise DomainError("model_budget_exhausted")
                prior = await (
                    await db.execute(
                        "SELECT attempt FROM v1_budget WHERE owner_id=%s AND command_id=%s AND attempt=%s",
                        (actor.user_id, command_id, attempt),
                    )
                ).fetchone()
                if prior:
                    raise DomainError("model_attempt_already_reserved")
                await db.execute(
                    "INSERT INTO v1_budget(owner_id,command_id,attempt,reserved,charge) VALUES(%s,%s,%s,%s,%s)",
                    (actor.user_id, command_id, attempt, amount, amount),
                )

    async def usage(
        self, actor: LocalActor, command_id: str, attempt: str, usage: dict, charge: Decimal | None
    ) -> None:
        """Unknown usage retains the full conservative charge, not a zero."""
        async with await self.connection.connect() as db:
            async with db.transaction():
                await db.execute("SELECT pg_advisory_xact_lock(71800801)")
                await db.execute(
                    "UPDATE v1_budget SET usage=%s,charge=COALESCE(%s,charge) WHERE owner_id=%s AND command_id=%s AND attempt=%s",
                    (Jsonb(usage), charge, actor.user_id, command_id, attempt),
                )

    async def publish(
        self, actor: LocalActor, source: dict, generation: str, config: dict, chunks: list[Chunk]
    ) -> None:
        """All vectors are staged then activated atomically, with live ACL/CAS."""
        if not chunks or len(chunks) > 500:
            raise DomainError("index_chunk_limit_or_empty")
        dimensions = config["fingerprint"]["dimensions"]
        if any(
            c.generation != generation
            or c.document.source_id != source["source_id"]
            or len(c.vector) != dimensions
            or not any(c.vector)
            or any(not isfinite(v) for v in c.vector)
            or c.text_hash != canonical_hash(c.text)
            or c.text != c.document.content[c.start_char : c.end_char]
            for c in chunks
        ):
            raise DomainError("index_publication_invariant_invalid")
        async with await self.connection.connect() as db:
            async with db.transaction():
                row = await (
                    await db.execute("SELECT * FROM v1_sources WHERE source_id=%s FOR UPDATE", (source["source_id"],))
                ).fetchone()
                if (
                    not row
                    or row["version"] != source["version"]
                    or row["owner_id"] != actor.user_id
                    or row["deleted"]
                ):
                    raise DomainError("source_changed_during_sync")
                await self.connection.authenticate(actor.session_id, actor.expires_at)
                await db.execute(
                    "INSERT INTO v1_generations(generation,source_id,config) VALUES(%s,%s,%s)",
                    (generation, source["source_id"], Jsonb(config)),
                )
                for chunk in chunks:
                    await db.execute(
                        "INSERT INTO v1_chunks(generation,chunk_id,document_id,resource,content,embedding,dimensions,metadata) VALUES(%s,%s,%s,%s,%s,%s::vector,%s,%s)",
                        (
                            generation,
                            chunk.chunk_id,
                            chunk.document.document_id,
                            chunk.document.resource,
                            chunk.text,
                            str(chunk.vector),
                            len(chunk.vector),
                            Jsonb(chunk.model_dump(mode="json", exclude={"vector"})),
                        ),
                    )
                await db.execute(
                    "UPDATE v1_sources SET generation=%s,version=version+1 WHERE source_id=%s",
                    (generation, source["source_id"]),
                )

    async def candidates(
        self, actor: LocalActor, pr_number: int, query: str, vector: list[float] | None
    ) -> tuple[list[Chunk], list[Chunk]]:
        """ACL/resource/active generation are SQL predicates BEFORE ranking."""
        where = "FROM v1_chunks c JOIN v1_sources s ON c.generation=s.generation WHERE NOT s.deleted AND s.acl @> %s AND (s.source_id='markdown-rules' OR s.source_id='github-checkout' AND c.resource=%s) "
        args = (Jsonb([actor.user_id]), f"pr:{pr_number}")
        terms = chinese_terms(query)
        async with await self.connection.connect() as db:
            if terms:
                lexical = await (
                    await db.execute(
                        "SELECT c.metadata "
                        + where
                        + "AND (to_tsvector('simple',c.content) @@ plainto_tsquery('simple',%s) "
                        "OR EXISTS (SELECT 1 FROM unnest(%s::text[]) q(term) WHERE strpos(c.content,q.term)>0)) "
                        "ORDER BY (ts_rank_cd(to_tsvector('simple',c.content),plainto_tsquery('simple',%s)) "
                        "+ (SELECT count(*) FROM unnest(%s::text[]) q(term) WHERE strpos(c.content,q.term)>0)::float/%s) "
                        "DESC,c.chunk_id LIMIT 20",
                        (*args, query, list(terms), query, list(terms), len(terms)),
                    )
                ).fetchall()
            else:
                lexical = await (
                    await db.execute(
                        "SELECT c.metadata "
                        + where
                        + "AND to_tsvector('simple',c.content) @@ plainto_tsquery('simple',%s) "
                        + "ORDER BY ts_rank_cd(to_tsvector('simple',c.content),plainto_tsquery('simple',%s)) DESC,c.chunk_id LIMIT 20",
                        (*args, query, query),
                    )
                ).fetchall()
            vectors = []
            if vector:
                vectors = await (
                    await db.execute(
                        "SELECT c.metadata "
                        + where
                        + "AND c.dimensions=%s ORDER BY c.embedding <=> %s::vector,c.chunk_id LIMIT 20",
                        (*args, len(vector), str(vector)),
                    )
                ).fetchall()

            def restore(rows: list[dict]) -> list[Chunk]:
                return [Chunk.model_validate({**row["metadata"], "vector": []}) for row in rows]

            return restore(lexical), restore(vectors)
