"""Rate-limited safe runtime probes, independent of business authorization."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel

from app.core.config import settings
from app.core.limiter import limiter
from app.services.runtime_readiness import RuntimeReadinessResult, RuntimeReadinessService

router = APIRouter(tags=["Runtime"])


class LivenessResult(BaseModel):
    """Liveness is solely the ASGI process, not dependency readiness."""

    status: str = "alive"


def get_readiness(request: Request) -> RuntimeReadinessService:
    """Retrieve lifespan-injected probes without constructing business adapters."""
    return request.app.state.runtime_readiness


@router.get("/live", response_model=LivenessResult)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["health"][0])
async def live(request: Request) -> LivenessResult:
    """Return process liveness only."""
    return LivenessResult()


@router.get("/ready", response_model=RuntimeReadinessResult)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["health"][0])
async def ready(
    request: Request,
    response: Response,
    service: Annotated[RuntimeReadinessService, Depends(get_readiness)],
) -> RuntimeReadinessResult:
    """Return 503 when a required dependency or migration is missing."""
    result = await service.check()
    response.status_code = 200 if result.ready else 503
    return result
