"""Bounded read queries and permission-bound expiring cursors."""

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from psycopg.types.json import Jsonb
from psycopg import sql

from app.core.config import settings
from app.schemas.local_mvp import LocalActor
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_storage import V11Store


def encode_cursor(actor: LocalActor, scope: dict, acl: str, position: Any) -> str:
    """The only scan position encoded belongs to a row already authorized for this actor."""
    return jwt.encode(
        {
            "sub": str(actor.user_id),
            "scope": canonical_hash(scope),
            "acl": acl,
            "position": position,
            "exp": int((datetime.now(UTC) + timedelta(minutes=15)).timestamp()),
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_cursor(cursor: str | None, actor: LocalActor, scope: dict, acl: str, default: Any) -> Any:
    """Never accept cursors issued to another principal, filter or ACL version."""
    if cursor is None:
        return default
    if len(cursor) > 4096:
        raise DomainError("cursor_invalid")
    try:
        value = jwt.decode(
            cursor,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require_exp": True, "require_sub": True},
        )
        if value["sub"] != str(actor.user_id) or value["scope"] != canonical_hash(scope) or value["acl"] != acl:
            raise ValueError("cursor_scope_invalid")
        return value["position"]
    except (JWTError, ValueError, KeyError, TypeError):
        raise DomainError("cursor_invalid_or_expired") from None


class ReviewQueries:
    """SQL applies ACL and filters before ranking and pagination."""

    def __init__(self, store: V11Store) -> None:
        """Use the same source and object authorization boundary as commands."""
        self.store = store

    async def acl_digest(self, actor: LocalActor, db: Any = None) -> str:
        """Only hash authorized source versions; never expose scanned source identities."""
        if db is None:
            async with await self.store.connection.connect() as connection:
                return await self.acl_digest(actor, connection)
        rows = await (
            await db.execute(
                "SELECT source_id,authz_version FROM v1_sources WHERE NOT deleted AND acl @> %s ORDER BY source_id",
                (Jsonb([actor.user_id]),),
            )
        ).fetchall()
        return canonical_hash(rows)

    async def objects(self, actor: LocalActor, filters: dict, cursor: str | None, limit: int) -> dict:
        """Keyset ordered by immutable creation time and ID, without a global count."""
        acl = await self.acl_digest(actor)
        scope = {"kind": "objects", "filters": filters, "limit": limit}
        position = decode_cursor(cursor, actor, scope, acl, ["1970-01-01T00:00:00+00:00", ""])
        if not isinstance(position, list) or len(position) != 2 or not all(isinstance(p, str) for p in position):
            raise DomainError("cursor_invalid")
        repository, kind, analysis, state = (filters.get(key) for key in ("repository", "kind", "analysis", "state"))
        inbox = bool(filters.get("inbox"))
        async with await self.store.connection.connect() as db, db.transaction():
            rows = await (
                await db.execute(
                    """SELECT o.object_id,o.repository_id,o.kind,o.created_at,
                l.payload#>>'{facts,product,analysis_status}' AS analysis_status,
                domain.run->>'status' AS domain_status
                FROM v11_objects o JOIN local_mvp_records l ON l.change_id=o.object_id
                JOIN v11_repositories r USING(repository_id) JOIN v1_sources s USING(source_id)
                CROSS JOIN LATERAL (SELECT run FROM jsonb_array_elements((l.payload#>>'{facts,domain}')::jsonb->'runs') run
                    WHERE run->>'run_id'=l.payload#>>'{facts,creation,run_id}' LIMIT 1) domain
                WHERE %s IN (l.owner_user_id,l.approver_user_id) AND NOT s.deleted AND s.acl @> %s
                AND EXISTS(SELECT 1 FROM v1_sources a WHERE a.source_id='markdown-rules' AND NOT a.deleted AND a.acl @> %s)
                AND NOT EXISTS(SELECT 1 FROM v11_rule_sets rs JOIN v1_sources a ON a.source_id=rs.source_id
                    WHERE (rs.applicability->'repositories' @> jsonb_build_array(o.repository_id) OR rs.applicability->'repositories' @> '["*"]'::jsonb)
                    AND (a.deleted OR NOT a.acl @> %s))
                AND (%s::text IS NULL OR o.repository_id=%s) AND (%s::text IS NULL OR o.kind=%s)
                AND (%s::text IS NULL OR l.payload#>>'{facts,product,analysis_status}'=%s)
                AND (%s::text IS NULL OR domain.run->>'status'=%s)
                AND (NOT %s OR (l.approver_user_id=%s AND domain.run->>'status'='WAITING_FOR_APPROVAL'))
                AND (o.created_at,o.object_id)>(%s::timestamptz,%s)
                ORDER BY o.created_at,o.object_id LIMIT %s""",
                    (
                        actor.user_id,
                        Jsonb([actor.user_id]),
                        Jsonb([actor.user_id]),
                        Jsonb([actor.user_id]),
                        repository,
                        repository,
                        kind,
                        kind,
                        analysis,
                        analysis,
                        state,
                        state,
                        inbox,
                        actor.user_id,
                        position[0],
                        position[1],
                        limit + 1,
                    ),
                )
            ).fetchall()
            # Locks linearize final authorization against concurrent source revoke.
            for row in rows:
                await self.store.object_row(row["object_id"], actor, db, lock=True)
            if await self.acl_digest(actor, db) != acl:
                raise DomainError("cursor_permissions_changed")
        await self.store.connection.authenticate(actor.session_id, actor.expires_at)
        if await self.acl_digest(actor) != acl:
            raise DomainError("cursor_permissions_changed")
        next_cursor = (
            encode_cursor(actor, scope, acl, [rows[limit - 1]["created_at"].isoformat(), rows[limit - 1]["object_id"]])
            if len(rows) > limit
            else None
        )
        return {"items": rows[:limit], "next_cursor": next_cursor}

    async def history(
        self,
        identifier: str,
        actor: LocalActor,
        kind: str,
        cursor: str | None,
        limit: int,
        snapshot: str | None = None,
    ) -> dict:
        """History positions expose only rows in the reauthorized object's scope."""
        tables = {
            "snapshots": ("v11_snapshots", "snapshot_id"),
            "analyses": ("v11_analyses", "analysis_id"),
            "files": ("v11_diff_files", "file_id"),
        }
        table, key = tables[kind]
        acl = await self.acl_digest(actor)
        scope = {"object": identifier, "kind": kind, "snapshot": snapshot, "limit": limit}
        order = "path" if kind == "files" else "sequence" if kind == "snapshots" else "revision"
        position = decode_cursor(cursor, actor, scope, acl, ["" if kind == "files" else 0, ""])
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not isinstance(position[1], str)
            or (not isinstance(position[0], str) if kind == "files" else type(position[0]) is not int)
        ):
            raise DomainError("cursor_invalid")
        async with await self.store.connection.connect() as db, db.transaction():
            await self.store.object_row(identifier, actor, db, lock=True)
            if kind == "files":
                found = await (
                    await db.execute(
                        "SELECT 1 FROM v11_snapshots WHERE object_id=%s AND snapshot_id=%s", (identifier, snapshot)
                    )
                ).fetchone()
                if not found:
                    raise DomainError("local_object_not_found")
            field, parent = ("snapshot_id", snapshot) if kind == "files" else ("object_id", identifier)
            # Table/column names are server constants, values remain parameters.
            query = sql.SQL("SELECT * FROM {} WHERE {}=%s AND ({},{})>(%s,%s) ORDER BY {},{} LIMIT %s").format(
                sql.Identifier(table),
                sql.Identifier(field),
                sql.Identifier(order),
                sql.Identifier(key),
                sql.Identifier(order),
                sql.Identifier(key),
            )
            rows = await (await db.execute(query, (parent, position[0], position[1], limit + 1))).fetchall()
            if await self.acl_digest(actor, db) != acl:
                raise DomainError("cursor_permissions_changed")
        summaries = []
        for row in rows[:limit]:
            summaries.append(
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"facts", "report", "dependencies", "owner_id", "command_id", "input_hash"}
                }
            )
        return {
            "items": summaries,
            "next_cursor": encode_cursor(actor, scope, acl, [rows[limit - 1][order], rows[limit - 1][key]])
            if len(rows) > limit
            else None,
        }
