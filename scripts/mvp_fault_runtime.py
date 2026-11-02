"""TEST ONLY: server-side faults for a separately authorized synthetic project.

Mount this module only into the isolated test container. It is never wired by
the product entrypoint, API input or Graph state.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator

from fastapi import FastAPI

from app.core.config import Environment, settings
from app.main import app
from app.services.local_mvp_storage import unit
from app.services.tools.execution_adapter import TargetObservation


class SyntheticCrash:
    """Stop at one saved claim/effect window without granting retry authority."""

    def __init__(self, point: str) -> None:
        """Pin the fault in server configuration, not an HTTP request."""
        self.point = point

    async def hit(self, point: str) -> None:
        """Raise an explicit test exception at the selected window."""
        if point == self.point:
            raise RuntimeError("synthetic_fault")


class WrongIndependentReader:
    """Observe an intentionally wrong synthetic value to test verification."""

    source_pin = "independent-register.v1"

    async def observe(self, resource_id: str, *, at: datetime) -> TargetObservation:
        """Return wrong observation, with the real isolated target untouched."""
        result = await unit().target.read(resource_id, at=at)
        return result.model_copy(update={"value": 0})


original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def fault_lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Require explicit provider-free nonproduction mode before any injection."""
    if not (
        settings.LOCAL_MVP_ONLY
        and settings.LOCAL_MVP_ENABLED
        and settings.ENVIRONMENT in {Environment.DEVELOPMENT, Environment.TEST}
    ):
        raise RuntimeError("mvp_fault_runtime_requires_isolated_local_mode")
    point = os.environ.get("MVP_TEST_FAULT", "")
    if point not in {"after_claim", "after_send", "verification"}:
        raise RuntimeError("mvp_fault_runtime_invalid_point")
    async with original_lifespan(application):
        service = application.state.local_mvp
        if point == "verification":
            service.verification.reader = WrongIndependentReader()
        else:
            service.execution.faults = SyntheticCrash(point)
        yield


app.router.lifespan_context = fault_lifespan
