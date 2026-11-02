"""Long-term memory service using mem0 and pgvector with explicit degradation states."""

from time import perf_counter

from mem0 import AsyncMemory

from app.core.cache import (
    cache_key,
    cache_service,
)
from app.core.config import settings
from app.core.logging import logger
from app.schemas.memory import MemorySearchResult, MemoryStatus
from app.services.llm.contracts import LLMRole
from app.services.llm.registry import LLMRegistry


class MemoryService:
    """Service for managing long-term memory using mem0 and pgvector."""

    def __init__(self):
        """Initialize the memory service."""
        self._memory: AsyncMemory | None = None

    async def _get_memory(self) -> AsyncMemory:
        if self._memory is None:
            self._memory = await AsyncMemory.from_config(config_dict=self._config_dict())
        return self._memory

    @staticmethod
    def _config_dict() -> dict[str, object]:
        """Build Mem0 config from the independent Memory and Embedding roles."""
        embedding = LLMRegistry.get_config("", LLMRole.EMBEDDING)
        dimensions = embedding.capabilities.embedding_dimensions
        if dimensions is None:
            raise ValueError("embedding_dimensions must be configured for Mem0 pgvector")
        memory_llm = LLMRegistry.get("", role=LLMRole.MEMORY_LLM)
        embedding_adapter = LLMRegistry.get_embeddings()
        return {
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "collection_name": settings.LONG_TERM_MEMORY_COLLECTION_NAME,
                    "embedding_model_dims": dimensions,
                    "dbname": settings.POSTGRES_DB,
                    "user": settings.POSTGRES_USER,
                    "password": settings.POSTGRES_PASSWORD,
                    "host": settings.POSTGRES_HOST,
                    "port": settings.POSTGRES_PORT,
                },
            },
            "llm": {"provider": "langchain", "config": {"model": memory_llm}},
            "embedder": {"provider": "langchain", "config": {"model": embedding_adapter}},
        }

    async def initialize(self) -> None:
        """Pre-warm the mem0 AsyncMemory instance and its pgvector connection pool.

        Call once at startup so the first search() or add() doesn't pay the
        ~130ms from_config + pgvector.list_cols() cold-init cost.
        """
        if not settings.MEMORY_ENABLED:
            logger.info("memory_service_disabled")
            return
        await self._get_memory()
        logger.info("memory_service_initialized")

    async def search_with_status(self, user_id: str | None, query: str) -> MemorySearchResult:
        """Search memory and preserve disabled, empty, and unavailable states.

        Checks cache first; on miss, queries mem0 and caches the result.
        """
        started = perf_counter()
        if not settings.MEMORY_ENABLED:
            return MemorySearchResult.disabled("memory_disabled")
        if user_id is None:
            return MemorySearchResult.disabled("anonymous_session")
        try:
            # Check cache first
            key = cache_key("memory", str(user_id), query)
            cached = await cache_service.get(key)
            if cached is not None:
                logger.debug("memory_search_cache_hit", user_id=user_id)
                return MemorySearchResult(
                    status=MemoryStatus.ENABLED,
                    content=cached,
                    latency_ms=_latency_ms(started),
                )

            memory = await self._get_memory()
            results = await memory.search(user_id=str(user_id), query=query)
            result = "\n".join(f"* {item['memory']}" for item in results.get("results", []))

            # Cache successful results
            if result:
                await cache_service.set(key, result)

            if not result:
                return MemorySearchResult.empty(_latency_ms(started))
            return MemorySearchResult(
                status=MemoryStatus.ENABLED,
                content=result,
                latency_ms=_latency_ms(started),
            )
        except Exception as exc:
            error_category = getattr(getattr(exc, "code", None), "value", "memory_provider_unavailable")
            logger.exception(
                "memory_search_failed",
                error_category=error_category,
                user_id=user_id,
            )
            return MemorySearchResult.unavailable(
                "memory_provider_unavailable",
                error_category=error_category,
                latency_ms=_latency_ms(started),
            )

    async def search(self, user_id: str | None, query: str) -> str:
        """Compatibility wrapper returning only the formatted memory text."""
        result = await self.search_with_status(user_id, query)
        return result.content

    async def add(self, user_id: str | None, messages: list[dict], metadata: dict | None = None) -> None:
        """Add messages to long-term memory for a user.

        No-op when ``user_id`` is ``None`` (see ``search`` for rationale).
        """
        if user_id is None or not settings.MEMORY_ENABLED:
            return
        try:
            memory = await self._get_memory()
            await memory.add(messages, user_id=str(user_id), metadata=metadata)
            logger.info("long_term_memory_updated_successfully", user_id=user_id)
        except Exception as e:
            logger.exception("failed_to_update_long_term_memory", user_id=user_id, error=str(e))


memory_service = MemoryService()


def _latency_ms(started: float) -> float:
    """Return a bounded monotonic duration for memory diagnostics."""
    return round((perf_counter() - started) * 1000, 3)
