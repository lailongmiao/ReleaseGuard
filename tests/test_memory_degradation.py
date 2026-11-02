"""Memory off/empty/unavailable behavior stays explicit and non-blocking."""

import asyncio

from app.core import config as config_module
from app.core.cache import cache_service
from app.schemas.memory import MemoryStatus
from app.services.memory import MemoryService


def test_memory_disabled_returns_structured_status(monkeypatch) -> None:
    """Disable memory without constructing Mem0 or touching a provider."""
    monkeypatch.setattr(config_module.settings, "MEMORY_ENABLED", False)
    service = MemoryService()

    result = asyncio.run(service.search_with_status("user-1", "private query"))

    assert result.status is MemoryStatus.DISABLED
    assert result.content == ""


def test_memory_provider_failure_returns_unavailable(monkeypatch) -> None:
    """Convert provider failures to a safe degradation result."""
    monkeypatch.setattr(config_module.settings, "MEMORY_ENABLED", True)

    async def no_cache(key: str):
        return None

    async def failing_memory():
        raise RuntimeError("fixture provider unavailable")

    monkeypatch.setattr(cache_service, "get", no_cache)
    service = MemoryService()
    monkeypatch.setattr(service, "_get_memory", failing_memory)

    result = asyncio.run(service.search_with_status("user-1", "private query"))

    assert result.status is MemoryStatus.UNAVAILABLE
    assert result.degradation_reason == "memory_provider_unavailable"
    assert result.error_category == "memory_provider_unavailable"
