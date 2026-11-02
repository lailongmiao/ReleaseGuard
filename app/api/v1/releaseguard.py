"""Authenticated local product API; real targets/production remain disabled."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import settings
from app.core.limiter import limiter
from app.schemas.inheritance import ApprovalHistoryPage, LegacyCommandView
from app.services.inheritance_queries import history, legacy_command
from app.schemas.approval import ApprovalCommand
from app.schemas.local_mvp import LocalActor, LocalCommand, LocalCreate, LocalPrepare, LocalRevision, LocalView
from app.schemas.v1_backend import AnalyzeInput, SourceResult, AnalysisView, ChangePage, EvidenceView
from app.services.local_mvp import LocalMVPService
from app.services.local_mvp_policy import principal_id
from app.services.releaseguard.state_machine import DomainError

router = APIRouter()
security = HTTPBearer(auto_error=False)


def service(request: Request) -> LocalMVPService:
    """Runtime explicitly opts into development/test; absent wiring fails closed."""
    result = getattr(request.app.state, "local_mvp", None)
    if result is None:
        raise HTTPException(503, "local_mvp_disabled")
    return result


async def actor(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    runtime: Annotated[LocalMVPService, Depends(service)],
) -> LocalActor:
    """Verify signed session token AND current database session, every request."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "local_session_required", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require_exp": True, "require_sub": True},
        )
        if not isinstance(payload["sub"], str) or not isinstance(payload["exp"], (int, float)):
            raise ValueError("local_session_invalid")
        return await runtime.store.authenticate(payload["sub"], payload["exp"])
    except (JWTError, ValueError):
        raise HTTPException(401, "local_session_invalid", headers={"WWW-Authenticate": "Bearer"}) from None


Runtime = Annotated[LocalMVPService, Depends(service)]
Actor = Annotated[LocalActor, Depends(actor)]


def rejection(error: DomainError) -> HTTPException:
    """Only stable safe codes reach the client; unrelated objects are hidden."""
    code = error.code
    if code in {"local_object_not_found", "source_not_found"}:
        return HTTPException(404, code)
    if code in {
        "local_owner_required",
        "local_approver_required",
        "local_requester_mismatch",
        "local_self_approval_denied",
        "local_environment_denied",
        "approval_actor_denied",
        "source_owner_required",
    }:
        return HTTPException(403, code)
    return HTTPException(409, code)


@router.post("/changes/{change_id}/analyze", response_model=SourceResult, response_model_exclude_unset=True)
@limiter.limit("10/minute")
async def analyze(
    request: Request, change_id: str, body: AnalyzeInput, current: Actor, runtime: Runtime
) -> dict[str, Any]:
    """A model suggests grounded analysis; never approves or executes."""
    if runtime.product_runtime is None:
        raise HTTPException(503, "v1_backend_disabled")
    try:
        return await runtime.product_runtime.analyze(change_id, current, body)
    except DomainError as error:
        raise rejection(error) from None


@router.get("/changes/{change_id}/analysis", response_model=AnalysisView, response_model_exclude_unset=True)
@limiter.limit("120/minute")
async def analysis(request: Request, change_id: str, current: Actor, runtime: Runtime) -> dict[str, Any]:
    """Saved analysis with live source authorization and explicit staleness."""
    if runtime.product_runtime is None:
        raise HTTPException(503, "v1_backend_disabled")
    try:
        return await runtime.product_runtime.analysis(change_id, current)
    except DomainError as error:
        raise rejection(error) from None


@router.get("/identity")
@limiter.limit("120/minute")
async def identity(request: Request, current: Actor) -> dict[str, Any]:
    """Opaque user identity for request construction; never return a credential."""
    return {
        "user_id": current.user_id,
        "principal_id": principal_id(current.user_id),
        "production_execution": "disabled",
    }


@router.post("/changes", response_model=LocalView)
@limiter.limit("60/minute")
async def create(request: Request, body: LocalCreate, current: Actor, runtime: Runtime) -> LocalView:
    """Create a normalized change using an authenticated requester/key."""
    try:
        return await runtime.create(body, current)
    except DomainError as error:
        raise rejection(error) from None


@router.get("/changes", response_model=ChangePage)
@limiter.limit("120/minute")
async def changes(
    request: Request,
    current: Actor,
    runtime: Runtime,
    cursor: Annotated[str, Query(max_length=128)] = "",
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> dict[str, Any]:
    """Live authorized pages; an empty page with a cursor is not the end."""
    try:
        return await runtime.listing(current, inbox=False, cursor=cursor, limit=limit)
    except DomainError as error:
        raise rejection(error) from None


@router.get("/approvals", response_model=ChangePage)
@limiter.limit("120/minute")
async def approvals(
    request: Request,
    current: Actor,
    runtime: Runtime,
    cursor: Annotated[str, Query(max_length=128)] = "",
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> dict[str, Any]:
    """Pending before pagination; follow next_cursor even after an empty page."""
    try:
        return await runtime.listing(current, inbox=True, cursor=cursor, limit=limit)
    except DomainError as error:
        raise rejection(error) from None


@router.get("/changes/{change_id}/evidence", response_model=EvidenceView)
@limiter.limit("120/minute")
async def evidence(request: Request, change_id: str, current: Actor, runtime: Runtime) -> dict[str, Any]:
    """Authenticated saved evidence and immutable citation references."""
    try:
        return await runtime.evidence_view(change_id, current)
    except DomainError as error:
        raise rejection(error) from None


@router.get("/changes/{change_id}", response_model=LocalView)
@limiter.limit("120/minute")
async def get(request: Request, change_id: str, current: Actor, runtime: Runtime) -> LocalView:
    """Owner/designated approver can read saved state after restart."""
    try:
        return await runtime.get(change_id, current)
    except DomainError as error:
        raise rejection(error) from None


@router.post("/changes/{change_id}/prepare", response_model=LocalView)
@limiter.limit("60/minute")
async def prepare(request: Request, change_id: str, body: LocalPrepare, current: Actor, runtime: Runtime) -> LocalView:
    """Collect synthetic CI/history and enter the existing Graph approval wait."""
    try:
        return await runtime.command(change_id, current, body, "prepare")
    except DomainError as error:
        raise rejection(error) from None


@router.post("/changes/{change_id}/approval", response_model=LocalView)
@limiter.limit("60/minute")
async def decide(
    request: Request, change_id: str, body: ApprovalCommand, current: Actor, runtime: Runtime
) -> LocalView:
    """Designated approver sends typed intent; service rechecks every seal."""
    try:
        return await runtime.command(change_id, current, body, "decide")
    except DomainError as error:
        raise rejection(error) from None


@router.post("/changes/{change_id}/revision", response_model=LocalView)
@limiter.limit("60/minute")
async def revise(request: Request, change_id: str, body: LocalRevision, current: Actor, runtime: Runtime) -> LocalView:
    """Owner supplements the request; old approval cannot cross revisions."""
    try:
        return await runtime.command(change_id, current, body, "revise")
    except DomainError as error:
        raise rejection(error) from None


@router.post("/changes/{change_id}/revoke", response_model=LocalView)
@limiter.limit("60/minute")
async def revoke(request: Request, change_id: str, body: LocalCommand, current: Actor, runtime: Runtime) -> LocalView:
    """Owner revocation is saved before any future protected call."""
    try:
        return await runtime.command(change_id, current, body, "revoke")
    except DomainError as error:
        raise rejection(error) from None


@router.post("/changes/{change_id}/run", response_model=LocalView)
@limiter.limit("60/minute")
async def run(request: Request, change_id: str, body: LocalCommand, current: Actor, runtime: Runtime) -> LocalView:
    """Explicitly run only the approved simulation or isolated integer sandbox."""
    try:
        return await runtime.command(change_id, current, body, "execute")
    except DomainError as error:
        raise rejection(error) from None


@router.get("/changes/{change_id}/receipt")
@limiter.limit("120/minute")
async def receipt(request: Request, change_id: str, current: Actor, runtime: Runtime) -> dict[str, Any]:
    """No fabricated success receipt for an unexecuted/unknown result."""
    try:
        result = await runtime.get(change_id, current)
        return {
            "receipt": result.state.receipt.model_dump(mode="json") if result.state.receipt else None,
            "synthetic": True,
            "production_execution": "disabled",
            "source_provenance": result.source_provenance,
            "analysis_provenance": result.analysis_provenance,
            "execution_mode": result.execution_mode,
        }
    except DomainError as error:
        raise rejection(error) from None


@router.get("/changes/{change_id}/audit")
@limiter.limit("120/minute")
async def audit(request: Request, change_id: str, current: Actor, runtime: Runtime) -> dict[str, Any]:
    """Query persisted bounded audit facts with object authorization."""
    try:
        return await runtime.audit(change_id, current)
    except DomainError as error:
        raise rejection(error) from None


@router.get("/changes/{change_id}/approval-history", response_model=ApprovalHistoryPage)
@limiter.limit("120/minute")
async def approval_history(
    request: Request,
    change_id: str,
    current: Actor,
    runtime: Runtime,
    cursor: Annotated[str | None, Query(max_length=4096)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ApprovalHistoryPage:
    """Read original approval facts without changing their validity or seals."""
    try:
        return await history(runtime, change_id, current, cursor, limit)
    except DomainError as error:
        raise rejection(error) from None


@router.get("/changes/{change_id}/commands/{command_id}", response_model=LegacyCommandView)
@limiter.limit("120/minute")
async def original_command(
    request: Request, change_id: str, command_id: str, current: Actor, runtime: Runtime
) -> LegacyCommandView:
    """Only persisted original actor attribution authorizes a result lookup."""
    try:
        return await legacy_command(runtime, change_id, command_id, current)
    except DomainError as error:
        raise rejection(error) from None
