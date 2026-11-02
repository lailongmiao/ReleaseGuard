"""Explicit V1.1 API; legacy request types cannot address review records."""

import re
from collections.abc import AsyncIterator, Coroutine, Callable
from typing import Annotated, Any, Literal, override
from uuid import uuid4
from psycopg.types.json import Jsonb

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from slowapi.errors import RateLimitExceeded

from app.schemas.inheritance import ApprovalHistoryPage, ReportCIView
from app.services.inheritance_queries import history, report_ci
from app.api.v1.releaseguard import Actor, Runtime
from app.core.limiter import limiter
from app.core.config import settings
from app.schemas.v11_backend import (
    ReviewAnalyze,
    ReviewCapabilities,
    ReviewResult,
    ReviewAnalysis,
    ReviewApproval,
    ReviewCommand,
    ReviewCreate,
    ReviewPage,
    ReviewPrepare,
    ReviewRevision,
    ReviewSearch,
    ReviewView,
    SyncInput,
)
from app.services.local_mvp_storage import review_scope, unit
from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard.normalization import canonical_hash
from app.services.v11_backend import V11Backend
from app.services.v11_queries import ReviewQueries, decode_cursor, encode_cursor
from app.services.v11_policy import CAPABILITIES
from app.schemas.v1_backend import SourceGrant, GrantResult


class ReviewRoute(APIRoute):
    """Safe error envelopes apply to validation and authentication as well as business errors."""

    @override
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        """Convert only known failures; unexpected faults remain server failures."""
        original = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await original(request)
            except RateLimitExceeded:
                return JSONResponse(
                    status_code=429,
                    content={
                        "code": "local_rate_limited",
                        "request_id": str(uuid4()),
                        "retryable": False,
                        "details": {},
                    },
                )
            except DomainError as error:
                code = error.code
                status = (
                    404
                    if code in {"local_object_not_found", "source_not_found"}
                    else (
                        403
                        if code
                        in {
                            "local_owner_required",
                            "local_approver_required",
                            "source_owner_required",
                            "local_self_approval_denied",
                        }
                        else (
                            401
                            if code.startswith("local_session_")
                            else (
                                503
                                if code
                                in {
                                    "source_unavailable",
                                    "source_timeout_or_unavailable",
                                    "rate_limited",
                                    "review_runtime_unavailable",
                                    "review_real_reads_disabled",
                                }
                                else 409
                            )
                        )
                    )
                )
                return JSONResponse(
                    status_code=status,
                    content={"code": code, "request_id": str(uuid4()), "retryable": False, "details": {}},
                )
            except RequestValidationError as error:
                details = [{"path": list(item["loc"]), "type": item["type"]} for item in error.errors()[:8]]
                return JSONResponse(
                    status_code=422,
                    content={
                        "code": "review_input_invalid",
                        "request_id": str(uuid4()),
                        "retryable": False,
                        "details": {"errors": details},
                    },
                )
            except HTTPException as error:
                code = (
                    error.detail
                    if isinstance(error.detail, str) and re.fullmatch(r"[a-z_]+", error.detail)
                    else "request_rejected"
                )
                return JSONResponse(
                    status_code=error.status_code,
                    headers=error.headers,
                    content={"code": code, "request_id": str(uuid4()), "retryable": False, "details": {}},
                )

        return handle


async def review_runtime(runtime: Runtime) -> AsyncIterator[V11Backend]:
    """Only server route dispatch sets scope; request payloads cannot enable v3."""
    product = runtime.product_runtime
    if product is None or product.reviews is None:
        raise HTTPException(503, "review_runtime_unavailable")
    token = review_scope.set({"save_guard": product.reviews.commit_guard})
    try:
        yield product.reviews
    finally:
        review_scope.reset(token)


Backend = Annotated[V11Backend, Depends(review_runtime)]
Limit = Annotated[int, Query(ge=1, le=50)]
Cursor = Annotated[str | None, Query(max_length=4096)]
router = APIRouter(route_class=ReviewRoute)


@router.get("/capabilities", response_model=ReviewCapabilities)
@limiter.limit("120/minute")
async def capabilities(request: Request, current: Actor, runtime: Backend) -> ReviewCapabilities:
    """Expose current phase restrictions without source or Provider access."""
    return CAPABILITIES


@router.get("/runtime")
@limiter.limit("120/minute")
async def runtime_identity(request: Request, current: Actor, runtime: Backend) -> dict:
    """Authorized acceptance preflight verifies the designated ledger without secrets."""
    identity = await runtime.store.real_budget_identity(current)
    return {
        **identity,
        "database_name": settings.POSTGRES_DB,
        "database_port": settings.POSTGRES_PORT,
        "provider": "real"
        if runtime.provider.provenance == "provider" and runtime.provider.enabled
        else "disabled_or_synthetic",
        "sources": "real"
        if not runtime.injected_source and settings.V11_REAL_READS_ENABLED
        else "disabled_or_synthetic",
    }


@router.get("/sources", response_model=ReviewPage)
@limiter.limit("120/minute")
async def sources(request: Request, current: Actor, runtime: Backend) -> dict:
    """V1.1 source ACL administration is explicit and separate from legacy indexes."""
    return {
        "items": [
            row
            for row in await runtime.store.sources(current)
            if row["source_id"] in {"markdown-rules", "review-github-checkout", "review-github-setup-python"}
        ]
    }


@router.post("/sources/{source_id}/grant", response_model=GrantResult)
@limiter.limit("30/minute")
async def grant_source(
    request: Request,
    source_id: Literal["markdown-rules", "review-github-checkout", "review-github-setup-python"],
    body: SourceGrant,
    current: Actor,
    runtime: Backend,
) -> dict:
    """Source owner explicitly grants a subject; membership alone grants no source access."""
    return await runtime.store.grant(source_id, current, body, body.user_id, revoke=False)


@router.post("/sources/{source_id}/revoke", response_model=GrantResult)
@limiter.limit("30/minute")
async def revoke_source(
    request: Request,
    source_id: Literal["markdown-rules", "review-github-checkout", "review-github-setup-python"],
    body: SourceGrant,
    current: Actor,
    runtime: Backend,
) -> dict:
    """Reuse the original owner CAS and command ledger while invalidating ACL versions."""
    return await runtime.store.grant(source_id, current, body, body.user_id, revoke=True)


@router.get("/repositories", response_model=ReviewPage)
@limiter.limit("120/minute")
async def repositories(request: Request, current: Actor, runtime: Backend) -> dict:
    """List authorized whitelist metadata without credentials or private sources."""
    rows = await runtime.store.repositories(current)
    return {
        "items": [
            {
                **row,
                "supported_types": ["pr", "commit", "branch_compare"],
                "rehearsal": "pr_only" if row["repository_id"] == "checkout" else "analysis_only",
                "capabilities": CAPABILITIES.model_dump(mode="json"),
            }
            for row in rows
        ]
    }


@router.post("/repositories/{repository_id}/verify")
@limiter.limit("5/minute")
async def verify_repository(
    request: Request, repository_id: Literal["checkout", "setup-python"], current: Actor, runtime: Backend
) -> dict:
    """Source owner explicitly verifies public repository identity before collection."""
    return await runtime.verify_repository(repository_id, current)


async def discovery(
    repository: str,
    kind: str,
    current: Any,
    runtime: V11Backend,
    cursor: str | None,
    limit: int,
    state: str = "all",
    sha: str | None = None,
) -> dict:
    """Live upstream cursor chains are limited to ten authenticated pages."""
    row = await runtime.store.repository(repository, current)
    if row["verification"] != "verified":
        raise DomainError("repository_unverified")
    runtime.reads_enabled()
    queries = ReviewQueries(runtime.store)
    acl = await queries.acl_digest(current)
    scope = {"repository": repository, "kind": kind, "limit": limit, "state": state, "sha": sha}
    page = decode_cursor(cursor, current, scope, acl, 1)
    if type(page) is not int or not 1 <= page <= 10:
        raise DomainError("cursor_invalid")
    items, more = await runtime.github.discover(repository, row, kind, page, limit, state, sha)
    await runtime.local.store.authenticate(current.session_id, current.expires_at)
    await runtime.store.repository(repository, current)
    if await queries.acl_digest(current) != acl:
        raise DomainError("cursor_permissions_changed")
    return {
        "items": items,
        "next_cursor": encode_cursor(current, scope, acl, page + 1) if more and page < 10 else None,
        "consistency": "upstream_live",
    }


@router.get("/repositories/{repository_id}/pulls", response_model=ReviewPage)
@limiter.limit("20/minute")
async def pulls(
    request: Request,
    repository_id: Literal["checkout", "setup-python"],
    current: Actor,
    runtime: Backend,
    state: Literal["open", "closed", "all"] = "all",
    cursor: Cursor = None,
    limit: Limit = 20,
) -> dict:
    """Discover PRs in a fixed creation order within the authorized base repository."""
    return await discovery(repository_id, "pulls", current, runtime, cursor, limit, state)


@router.get("/repositories/{repository_id}/commits", response_model=ReviewPage)
@limiter.limit("20/minute")
async def commits(
    request: Request,
    repository_id: Literal["checkout", "setup-python"],
    current: Actor,
    runtime: Backend,
    cursor: Cursor = None,
    limit: Limit = 20,
) -> dict:
    """Default-branch discovery returns only complete commit identities."""
    return await discovery(repository_id, "commits", current, runtime, cursor, limit)


@router.get("/repositories/{repository_id}/commits/{sha}", response_model=ReviewPage)
@limiter.limit("20/minute")
async def commit(
    request: Request, repository_id: Literal["checkout", "setup-python"], sha: str, current: Actor, runtime: Backend
) -> dict:
    """Expose actual parent identities before explicit merge-parent selection."""
    if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise HTTPException(422, "commit_sha_invalid")
    return await discovery(repository_id, "commits", current, runtime, None, 1, sha=sha)


@router.get("/repositories/{repository_id}/branches", response_model=ReviewPage)
@limiter.limit("20/minute")
async def branches(
    request: Request,
    repository_id: Literal["checkout", "setup-python"],
    current: Actor,
    runtime: Backend,
    cursor: Cursor = None,
    limit: Limit = 20,
) -> dict:
    """Branch names are discovery observations, not already pinned comparisons."""
    return await discovery(repository_id, "branches", current, runtime, cursor, limit)


@router.get("/rules", response_model=ReviewPage)
@limiter.limit("120/minute")
async def rules(request: Request, current: Actor, runtime: Backend) -> dict:
    """Expose rule-set versions only after their source ACL is checked."""
    async with await runtime.store.connection.connect() as db, db.transaction():
        rows = await (
            await db.execute(
                "SELECT r.rule_set_id,r.version,r.generation,r.applicability,r.content_hash FROM v11_rule_sets r JOIN v1_sources s USING(source_id) WHERE NOT s.deleted AND s.acl @> %s ORDER BY rule_set_id FOR SHARE OF s",
                (Jsonb([current.user_id]),),
            )
        ).fetchall()
        return {"items": rows}


@router.post("/rules/{rule_set_id}/sync", response_model=ReviewResult, response_model_exclude_unset=True)
@limiter.limit("5/minute")
async def sync_rules(request: Request, rule_set_id: str, body: SyncInput, current: Actor, runtime: Backend) -> dict:
    """Only the source owner can publish an explicitly selected rule-set generation."""
    return await runtime.sync_rules(rule_set_id, current, body)


@router.post("", response_model=ReviewView)
@limiter.limit("60/minute")
async def create(request: Request, body: ReviewCreate, current: Actor, runtime: Backend) -> ReviewView:
    """Create a separate immutable selector, even if another user chose the same PR."""
    return await runtime.create(body, current)


@router.get("", response_model=ReviewPage)
@limiter.limit("120/minute")
async def objects(
    request: Request,
    current: Actor,
    runtime: Backend,
    cursor: Cursor = None,
    limit: Limit = 20,
    repository: Literal["checkout", "setup-python"] | None = None,
    kind: Literal["pr", "commit", "branch_compare"] | None = None,
    analysis: str | None = None,
    state: str | None = None,
) -> dict:
    """Filter authorized objects before keyset pagination."""
    return await ReviewQueries(runtime.store).objects(
        current, {"repository": repository, "kind": kind, "analysis": analysis, "state": state}, cursor, limit
    )


@router.get("/approvals", response_model=ReviewPage)
@limiter.limit("120/minute")
async def approvals(
    request: Request, current: Actor, runtime: Backend, cursor: Cursor = None, limit: Limit = 20
) -> dict:
    """An inbox role never bypasses repository or rule-source access."""
    return await ReviewQueries(runtime.store).objects(current, {"inbox": True}, cursor, limit)


@router.get("/creation-result", response_model=ReviewView)
@limiter.limit("60/minute")
async def creation_result(
    request: Request,
    current: Actor,
    runtime: Backend,
    idempotency_key: Annotated[str, Query(min_length=1, max_length=128)],
) -> ReviewView:
    """Read a creation outcome scoped to the authenticated owner; never repeat creation."""
    identifier = "local-change:" + canonical_hash({"actor": current.user_id, "key": idempotency_key})
    return await runtime.view(identifier, current)


@router.get("/management-commands/{command_id}")
@limiter.limit("60/minute")
async def management_command(request: Request, command_id: str, current: Actor, runtime: Backend) -> dict:
    """Read only the caller's source/rule command after current resource authorization."""
    async with await runtime.store.connection.connect() as db, db.transaction():
        row = await (
            await db.execute(
                "SELECT resource,status,result FROM v1_commands WHERE owner_id=%s AND command_id=%s",
                (current.user_id, command_id),
            )
        ).fetchone()
        if not row:
            raise DomainError("local_object_not_found")
        if row["resource"] in {"markdown-rules", "review-github-checkout", "review-github-setup-python"}:
            await runtime.store.source(row["resource"], current, owner=True)
        else:
            rule = await (
                await db.execute("SELECT rule_set_id FROM v11_rule_sets WHERE rule_set_id=%s", (row["resource"],))
            ).fetchone()
            if not rule:
                raise DomainError("local_object_not_found")
            await runtime.store.source("markdown-rules", current, owner=True)
        attempts = await (
            await db.execute(
                "SELECT attempt,reserved,charge,usage FROM v1_budget WHERE owner_id=%s AND command_id=%s ORDER BY attempt LIMIT 16",
                (current.user_id, command_id),
            )
        ).fetchall()
        return {
            "status": "unknown" if row["status"] == "started" else row["status"],
            "result": row["result"],
            "attempts": attempts,
        }


@router.get("/{object_id}", response_model=ReviewView)
@limiter.limit("120/minute")
async def get_review(request: Request, object_id: str, current: Actor, runtime: Backend) -> ReviewView:
    """Analysis completion and deterministic approval eligibility are separate fields."""
    return await runtime.view(object_id, current)


@router.post("/{object_id}/sync", response_model=ReviewResult, response_model_exclude_unset=True)
@limiter.limit("5/minute")
async def sync(request: Request, object_id: str, body: SyncInput, current: Actor, runtime: Backend) -> dict:
    """Pin code and checks with bounded GETs and an independent object generation."""
    return await runtime.sync(object_id, current, body)


@router.post("/{object_id}/analyze", response_model=ReviewResult, response_model_exclude_unset=True)
@limiter.limit("10/minute")
async def analyze(request: Request, object_id: str, body: ReviewAnalyze, current: Actor, runtime: Backend) -> dict:
    """Qualified readonly input may be analyzed despite failed or missing CI."""
    return await runtime.analyze(object_id, current, body)


@router.post("/{object_id}/search", response_model=ReviewResult, response_model_exclude_unset=True)
@limiter.limit("20/minute")
async def search(request: Request, object_id: str, body: ReviewSearch, current: Actor, runtime: Backend) -> dict:
    """Budgeted query fixes its object, snapshot and generation."""
    return await runtime.search(object_id, current, body)


@router.get("/{object_id}/snapshots", response_model=ReviewPage)
@limiter.limit("120/minute")
async def snapshots(
    request: Request, object_id: str, current: Actor, runtime: Backend, cursor: Cursor = None, limit: Limit = 20
) -> dict:
    """Immutable snapshot history remains bounded and authorized."""
    return await ReviewQueries(runtime.store).history(object_id, current, "snapshots", cursor, limit)


@router.get("/{object_id}/snapshots/{snapshot_id}")
@limiter.limit("120/minute")
async def snapshot(request: Request, object_id: str, snapshot_id: str, current: Actor, runtime: Backend) -> dict:
    """Return snapshot facts excluding file bodies, which have separate bounded reads."""
    row = await runtime.store.snapshot(object_id, snapshot_id, current)
    return {**row, "facts": {key: value for key, value in row["facts"].items() if key != "files"}}


@router.get("/{object_id}/snapshots/{snapshot_id}/files", response_model=ReviewPage)
@limiter.limit("120/minute")
async def files(
    request: Request,
    object_id: str,
    snapshot_id: str,
    current: Actor,
    runtime: Backend,
    cursor: Cursor = None,
    limit: Limit = 20,
) -> dict:
    """File IDs never act as local paths."""
    return await ReviewQueries(runtime.store).history(object_id, current, "files", cursor, limit, snapshot_id)


@router.get("/{object_id}/snapshots/{snapshot_id}/files/{file_id}")
@limiter.limit("120/minute")
async def file(
    request: Request, object_id: str, snapshot_id: str, file_id: str, current: Actor, runtime: Backend
) -> dict:
    """Only lines actually collected at this immutable snapshot can be returned."""
    row = await runtime.store.snapshot(object_id, snapshot_id, current)
    for value in row["facts"]["files"]:
        if value["file_id"] == file_id:
            return value
    raise DomainError("local_object_not_found")


@router.get("/{object_id}/analyses", response_model=ReviewPage)
@limiter.limit("120/minute")
async def analyses(
    request: Request, object_id: str, current: Actor, runtime: Backend, cursor: Cursor = None, limit: Limit = 20
) -> dict:
    """Previous reports remain available without conferring current authority."""
    return await ReviewQueries(runtime.store).history(object_id, current, "analyses", cursor, limit)


@router.get("/{object_id}/analyses/{analysis_id}", response_model=ReviewAnalysis, response_model_exclude_unset=True)
@limiter.limit("120/minute")
async def analysis(request: Request, object_id: str, analysis_id: str, current: Actor, runtime: Backend) -> dict:
    """Every report read rechecks membership, source access and staleness."""
    return await runtime.analysis(object_id, analysis_id, current)


@router.post("/{object_id}/revision", response_model=ReviewResult, response_model_exclude_unset=True)
@limiter.limit("60/minute")
async def revision(request: Request, object_id: str, body: ReviewRevision, current: Actor, runtime: Backend) -> dict:
    """New user intent retains old analyses, unknown claims and approval history."""
    return await runtime.command(object_id, current, body, "revise")


@router.get("/{object_id}/analyses/{analysis_id}/rejection")
@limiter.limit("120/minute")
async def rejection(request: Request, object_id: str, analysis_id: str, current: Actor, runtime: Backend) -> dict:
    """Explicit local owner diagnostic, excluded from ordinary reports/errors/command results."""
    if not settings.LOCAL_MVP_ONLY or request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
        raise DomainError("local_object_not_found")
    return await runtime.store.rejection(object_id, analysis_id, current)


@router.post("/{object_id}/prepare", response_model=ReviewResult, response_model_exclude_unset=True)
@limiter.limit("30/minute")
async def prepare(request: Request, object_id: str, body: ReviewPrepare, current: Actor, runtime: Backend) -> dict:
    """Prepare only a currently eligible checkout PR through the strict Evidence Gate."""
    return await runtime.command(object_id, current, body, "prepare")


@router.post("/{object_id}/approval", response_model=ReviewResult, response_model_exclude_unset=True)
@limiter.limit("30/minute")
async def approval(request: Request, object_id: str, body: ReviewApproval, current: Actor, runtime: Backend) -> dict:
    """Record an independently authenticated approval bound to current evidence."""
    return await runtime.command(object_id, current, body, "decide")


@router.post("/{object_id}/run", response_model=ReviewResult, response_model_exclude_unset=True)
@limiter.limit("30/minute")
async def run(request: Request, object_id: str, body: ReviewCommand, current: Actor, runtime: Backend) -> dict:
    """Revalidate current evidence and authorization before a target-free Dry Run."""
    return await runtime.command(object_id, current, body, "execute")


@router.post("/{object_id}/revoke", response_model=ReviewResult, response_model_exclude_unset=True)
@limiter.limit("30/minute")
async def revoke(request: Request, object_id: str, body: ReviewCommand, current: Actor, runtime: Backend) -> dict:
    """Owner revocation retains the original immutable approval trail."""
    return await runtime.command(object_id, current, body, "revoke")


@router.get("/{object_id}/receipt")
@limiter.limit("120/minute")
async def receipt(request: Request, object_id: str, current: Actor, runtime: Backend) -> dict:
    """Receipt is loaded from authoritative saved domain facts."""
    await runtime.store.object_row(object_id, current)
    async with runtime.local.store.open(object_id, current):
        state = await runtime.local.run()
        return {
            "receipt": state.receipt.model_dump(mode="json") if state.receipt else None,
            "target_write_calls": unit().target.calls,
        }


@router.get("/{object_id}/audit")
@limiter.limit("120/minute")
async def audit(request: Request, object_id: str, current: Actor, runtime: Backend) -> dict:
    """Return bounded safe command attribution without provider envelopes."""
    async with await runtime.store.connection.connect() as db, db.transaction():
        await runtime.store.object_row(object_id, current, db, lock=True)
        rows = await (
            await db.execute(
                "SELECT event_id,command_id,action,digest,created_at FROM v1_audit WHERE resource=%s ORDER BY created_at DESC,event_id DESC LIMIT 101",
                (object_id,),
            )
        ).fetchall()
        return {"items": list(reversed(rows[:100])), "truncated": len(rows) > 100}


@router.get("/{object_id}/commands/{command_id}")
@limiter.limit("120/minute")
async def command(request: Request, object_id: str, command_id: str, current: Actor, runtime: Backend) -> dict:
    """Diagnostics expose attempts and unknown fees, never a retry action."""
    async with await runtime.store.connection.connect() as db, db.transaction():
        await runtime.store.object_row(object_id, current, db, lock=True)
        row = await (
            await db.execute(
                "SELECT owner_id,status,result FROM v1_commands WHERE resource=%s AND command_id=%s AND owner_id=%s",
                (object_id, command_id, current.user_id),
            )
        ).fetchone()
        if not row:
            raise DomainError("local_object_not_found")
        attempts = await (
            await db.execute(
                "SELECT attempt,reserved,charge,usage FROM v1_budget WHERE owner_id=%s AND command_id=%s ORDER BY attempt LIMIT 16",
                (current.user_id, command_id),
            )
        ).fetchall()
        return {
            "status": "unknown" if row["status"] == "started" else row["status"],
            "result": row["result"],
            "attempts": attempts,
        }


@router.get("/{object_id}/approval-history", response_model=ApprovalHistoryPage)
@limiter.limit("120/minute")
async def approval_history(
    request: Request,
    object_id: str,
    current: Actor,
    runtime: Backend,
    cursor: Cursor = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ApprovalHistoryPage:
    """Same original approval projection with V1.1 membership and source policy."""
    await runtime.store.object_row(object_id, current)
    return await history(runtime.local, object_id, current, cursor, limit)


@router.get("/{object_id}/analyses/{analysis_id}/ci-observation", response_model=ReportCIView)
@limiter.limit("120/minute")
async def analysis_ci(
    request: Request, object_id: str, analysis_id: str, current: Actor, runtime: Backend
) -> ReportCIView:
    """No collection: return only the explicitly bound saved observation."""
    return await report_ci(runtime.store, object_id, analysis_id, current)
