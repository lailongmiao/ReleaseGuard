"""Async PostgreSQL facts and per-object serialization for the local MVP.

Each mutation is committed before its Graph node returns. In-memory domain
algorithms are transient validation engines; authority is the PostgreSQL row.
The session advisory lock dies with the connection/process and never licenses
repeating an uncertain effect. Remote/production execution remains disabled.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

from app.core.config import settings
from app.repositories.in_memory_approval import InMemoryApprovalRepository
from app.repositories.in_memory_evidence import InMemoryEvidenceRepository
from app.repositories.in_memory_execution import InMemoryExecutionRepository
from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.repositories.releaseguard import SnapshotPayload
from app.schemas.approval import ApprovalStoreSnapshot
from app.schemas.evidence import EvidenceSnapshot, EvidenceCitation
from app.schemas.local_mvp import LocalActor
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.sandbox_executor import SyntheticTargetStore


class SavedFacts(BaseModel):
    """Strict versioned local encoding; schema/hash verified on every load."""

    model_config = ConfigDict(extra="forbid")
    version: Literal["local-mvp.v1", "local-mvp.v2", "local-mvp.v3"] = "local-mvp.v1"
    domain: str = ""
    approvals: ApprovalStoreSnapshot = ApprovalStoreSnapshot()
    evidence: list[EvidenceSnapshot] = []
    citations: list[EvidenceCitation] = []
    ledger: str = ""
    target_values: dict[str, int] = {"local-synthetic-register": 0}
    target_versions: dict[str, int] = {"local-synthetic-register": 0}
    target_outcomes: dict[str, bool] = {}
    target_calls: int = 0
    commands: dict[str, dict[str, Any]] = {}
    creation: dict[str, Any] = {}
    product: dict[str, Any] = {}

    @model_serializer(mode="wrap")
    def serialize_facts(self, handler: SerializerFunctionWrapHandler) -> dict:
        """Old aggregate encodings do not silently acquire product authority."""
        values = handler(self)
        if self.version == "local-mvp.v1":
            if self.product:
                raise ValueError("legacy_product_authority_denied")
            values.pop("product", None)
        return values


class LocalUnit:
    """One locked object and authenticated actor, outside Graph checkpoints."""

    def __init__(self, connection: AsyncConnection[DictRow], row: dict, actor: LocalActor) -> None:
        """No caller can supply a serialized unit as authorization."""
        self.connection, self.row, self.actor = connection, row, actor
        self.at = datetime.now(UTC)
        self.domain = InMemoryReleaseGuardRepository()
        self.approvals = InMemoryApprovalRepository()
        self.evidence = InMemoryEvidenceRepository()
        self.ledger = InMemoryExecutionRepository()
        self.facts = SavedFacts()
        self.target = SyntheticTargetStore(self.facts.target_values)

    async def load(self) -> None:
        """Restore only from the locked database authority, with strict hashes."""
        payload = self.row["payload"]
        if payload:
            if set(payload) != {"facts", "hash"} or canonical_hash(payload["facts"]) != payload["hash"]:
                raise DomainError("local_store_integrity_failure")
            self.facts = SavedFacts.model_validate(payload["facts"])
            if self.facts.domain:
                await self.domain.restore_snapshot(SnapshotPayload(data=self.facts.domain))
            await self.approvals.restore_snapshot(self.facts.approvals)
            for snapshot in self.facts.evidence:
                await self.evidence.append_snapshot(snapshot)
            for citation in self.facts.citations:
                await self.evidence.append_citation(citation)
            if self.facts.ledger:
                await self.ledger.restore_snapshot(self.facts.ledger)
            self.target = SyntheticTargetStore(self.facts.target_values)
            self.target._versions = dict(self.facts.target_versions)
            self.target._outcomes = dict(self.facts.target_outcomes)
            self.target.calls = self.facts.target_calls

    async def save(self) -> None:
        """Commit validated facts with CAS; failure prevents node success."""
        self.facts.domain = (await self.domain.export_snapshot()).data
        self.facts.approvals = await self.approvals.export_snapshot()
        self.facts.evidence = list(self.evidence._snapshots.values())
        self.facts.citations = list(self.evidence._citations.values())
        self.facts.ledger = await self.ledger.export_snapshot()
        self.facts.target_values = dict(self.target._values)
        self.facts.target_versions = dict(self.target._versions)
        self.facts.target_outcomes = dict(self.target._outcomes)
        self.facts.target_calls = self.target.calls
        facts = SavedFacts.model_validate_json(self.facts.model_dump_json()).model_dump(mode="json")
        async with self.connection.transaction():
            if self.facts.version == "local-mvp.v3" and self.facts.product.get("binding_hash"):
                scope = review_scope.get()
                if scope is None or "save_guard" not in scope:
                    raise DomainError("review_commit_guard_required")
                await scope["save_guard"](self)
            cursor = await self.connection.execute(
                "UPDATE local_mvp_records SET payload=%s, version=version+1 WHERE change_id=%s AND version=%s RETURNING version",
                (Jsonb({"facts": facts, "hash": canonical_hash(facts)}), self.row["change_id"], self.row["version"]),
            )
            result = await cursor.fetchone()
            if result is None:
                raise DomainError("local_store_version_conflict")
        self.row["version"] = result["version"]


active_unit: ContextVar[LocalUnit | None] = ContextVar("local_mvp_unit", default=None)
review_scope: ContextVar[dict | None] = ContextVar("v11_trusted_review_scope", default=None)


def unit() -> LocalUnit:
    """Graph dependencies require a server-created unit, never checkpoint data."""
    result = active_unit.get()
    if result is None:
        raise DomainError("local_authenticated_unit_required")
    return result


class ScopedRepository:
    """Delegate existing protocols and persist every first local mutation."""

    WRITES = frozenset(
        {
            "create_change",
            "append_revision",
            "create_run",
            "commit_run_update",
            "create",
            "decide",
            "invalidate",
            "append_audit",
            "claim_call",
            "complete_call",
            "append_snapshot",
            "append_citation",
            "claim",
            "append_fact",
        }
    )

    def __init__(self, name: str) -> None:
        """Pin one known local repository, not a caller-provided class."""
        if name not in {"domain", "approvals", "evidence", "ledger"}:
            raise ValueError("local_repository_unknown")
        self.name = name

    def __getattr__(self, name: str) -> Any:
        """Protocol calls remain async; mutable stores never leave their unit."""

        async def invoke(*args: Any, **kwargs: Any) -> Any:
            current = unit()
            result = await getattr(getattr(current, self.name), name)(*args, **kwargs)
            if name in self.WRITES:
                await current.save()
            return result

        return invoke


class LocalPostgresStore:
    """Only private runtime settings, parameterized SQL and additive storage."""

    async def connect(self) -> AsyncConnection[DictRow]:
        """Open an async connection without printing DSNs or credentials."""
        return await AsyncConnection[DictRow].connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=3,
            options="-c statement_timeout=15000 -c lock_timeout=10000",
        )

    @asynccontextmanager
    async def open(
        self, change_id: str, actor: LocalActor, creation: tuple[int, str] | None = None
    ) -> AsyncIterator[LocalUnit]:
        """Serialize processes per object, enforce membership before loading facts."""
        lock_key = int.from_bytes(sha256(change_id.encode()).digest()[:8], "big", signed=True)
        async with await self.connect() as connection:
            await connection.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
            try:
                if creation:
                    approver, digest = creation
                    if approver == actor.user_id:
                        raise DomainError("local_self_approval_denied")
                    cursor = await connection.execute('SELECT id FROM "user" WHERE id=%s', (approver,))
                    if await cursor.fetchone() is None:
                        raise DomainError("local_approver_not_found")
                    await connection.execute(
                        "INSERT INTO local_mvp_records(change_id,owner_user_id,approver_user_id,creation_hash,payload) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (change_id, actor.user_id, approver, digest, Jsonb({})),
                    )
                cursor = await connection.execute("SELECT * FROM local_mvp_records WHERE change_id=%s", (change_id,))
                row = await cursor.fetchone()
                if row is None or actor.user_id not in {row["owner_user_id"], row["approver_user_id"]}:
                    raise DomainError("local_object_not_found")
                if creation and (row["creation_hash"] != creation[1] or row["approver_user_id"] != creation[0]):
                    raise DomainError("idempotency_conflict")
                current = LocalUnit(connection, row, actor)
                await current.load()
                if current.facts.creation and (current.facts.version == "local-mvp.v3") != (
                    review_scope.get() is not None
                ):
                    raise DomainError("local_object_not_found")
                token = active_unit.set(current)
                try:
                    yield current
                finally:
                    active_unit.reset(token)
            finally:
                await connection.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))

    async def authenticate(self, session_id: str, expires_at: float) -> LocalActor:
        """Session deletion revokes access; a user JWT is not a session JWT."""
        if expires_at <= datetime.now(UTC).timestamp():
            raise DomainError("local_session_expired")
        async with await self.connect() as connection:
            cursor = await connection.execute(
                'SELECT s.user_id FROM session s JOIN "user" u ON u.id=s.user_id WHERE s.id=%s',
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise DomainError("local_session_invalid")
            return LocalActor(user_id=row["user_id"], session_id=session_id, expires_at=expires_at)

    async def list_records(self, actor: LocalActor, *, inbox: bool, cursor: str, limit: int) -> list[str]:
        """Filter object membership before pagination; never return global counts."""
        async with await self.connect() as connection:
            result = await connection.execute(
                "SELECT change_id FROM local_mvp_records WHERE "
                "(approver_user_id=%s AND %s OR (owner_user_id=%s OR approver_user_id=%s) AND NOT %s) "
                "AND COALESCE(payload#>>'{facts,version}','') <> 'local-mvp.v3' "
                "AND (COALESCE(payload->'facts'->'product','{}'::jsonb)='{}'::jsonb OR "
                "(SELECT count(*) FROM v1_sources s WHERE s.source_id IN ('github-checkout','markdown-rules') "
                "AND NOT s.deleted AND s.acl @> %s::jsonb)=2) "
                "AND (NOT %s OR EXISTS (SELECT 1 FROM jsonb_array_elements("
                "COALESCE(NULLIF(payload#>>'{facts,domain}',''),'{}')::jsonb->'runs') AS run "
                "WHERE run->>'run_id'=payload#>>'{facts,creation,run_id}' "
                "AND run->>'status'='WAITING_FOR_APPROVAL')) "
                "AND change_id>%s ORDER BY change_id LIMIT %s",
                (
                    actor.user_id,
                    inbox,
                    actor.user_id,
                    actor.user_id,
                    inbox,
                    Jsonb([actor.user_id]),
                    inbox,
                    cursor,
                    limit,
                ),
            )
            return [row["change_id"] for row in await result.fetchall()]
