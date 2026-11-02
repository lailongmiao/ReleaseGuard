"""Fixed readonly source and scoped RAG API; no arbitrary URL or credential API."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.api.v1.releaseguard import Actor, Runtime, rejection
from app.core.limiter import limiter
from app.schemas.v1_backend import (
    SearchInput,
    SourceCommand,
    SourceGrant,
    SourceList,
    SourceView,
    SourceResult,
    GrantResult,
    SearchResult,
)
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_backend import V1Backend
from app.services.v1_storage import SOURCE_IDS

router = APIRouter()


def backend(runtime: Any) -> V1Backend:
    """Only startup-wired dependencies can enable product source capability."""
    if runtime.product_runtime is None:
        raise HTTPException(503, "v1_backend_disabled")
    return runtime.product_runtime


@router.get("/sources", response_model=SourceList)
@limiter.limit("120/minute")
async def sources(request: Request, current: Actor, runtime: Runtime) -> dict:
    """List only the caller's explicitly authorized sources."""
    return await backend(runtime).sources(current)


@router.get("/sources/{source_id}", response_model=SourceView)
@limiter.limit("120/minute")
async def source(request: Request, source_id: str, current: Actor, runtime: Runtime) -> dict:
    """Safe status without credentials or global source counts."""
    try:
        if source_id not in SOURCE_IDS:
            raise DomainError("source_not_found")
        row = await backend(runtime).store.source(source_id, current)
        return {key: value for key, value in row.items() if key != "authz_version"}
    except DomainError as error:
        raise rejection(error) from None


@router.post("/sources/{source_id}/sync", response_model=SourceResult, response_model_exclude_unset=True)
@limiter.limit("5/minute")
async def sync(request: Request, source_id: str, body: SourceCommand, current: Actor, runtime: Runtime) -> dict:
    """Bounded readonly collection and local indexing by the source owner."""
    try:
        return await backend(runtime).sync(source_id, current, body)
    except DomainError as error:
        raise rejection(error) from None


@router.post("/sources/{source_id}/grant", response_model=GrantResult)
@limiter.limit("30/minute")
async def grant(request: Request, source_id: str, body: SourceGrant, current: Actor, runtime: Runtime) -> dict:
    """An explicit source-owner action, distinct from assigning an approver."""
    try:
        if source_id not in SOURCE_IDS:
            raise DomainError("source_not_found")
        return await backend(runtime).store.grant(source_id, current, body, body.user_id, revoke=False)
    except DomainError as error:
        raise rejection(error) from None


@router.post("/sources/{source_id}/revoke", response_model=GrantResult)
@limiter.limit("30/minute")
async def revoke(request: Request, source_id: str, body: SourceGrant, current: Actor, runtime: Runtime) -> dict:
    """Revoke live rights and invalidate any older source-bound analysis."""
    try:
        if source_id not in SOURCE_IDS:
            raise DomainError("source_not_found")
        return await backend(runtime).store.grant(source_id, current, body, body.user_id, revoke=True)
    except DomainError as error:
        raise rejection(error) from None


@router.post("/search", response_model=SearchResult, response_model_exclude_unset=True)
@limiter.limit("20/minute")
async def search(request: Request, body: SearchInput, current: Actor, runtime: Runtime) -> dict:
    """Object/source scoped exact hybrid retrieval under a persistent budget."""
    try:
        return await backend(runtime).search(current, body)
    except DomainError as error:
        raise rejection(error) from None
