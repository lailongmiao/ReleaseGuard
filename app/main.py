"""This file contains the main application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from asgi_correlation_id import CorrelationIdMiddleware

from app.api.v1.api import api_router
from app.api.runtime import router as runtime_router
from app.api.v1.chatbot import agent
from app.core.cache import cache_service, InMemoryCacheService
from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import logger
from app.core.metrics import setup_metrics
from app.core.middleware import (
    LoggingContextMiddleware,
    MetricsMiddleware,
    ProfilingMiddleware,
)
from app.core.observability import langfuse_init
from app.services.memory import memory_service
from app.services.local_mvp import LocalMVPService
from app.services.runtime_readiness import RuntimeReadinessService, check_database
from app.services.v1_backend import V1Backend
from app.services.v11_backend import V11Backend

# Load environment variables
load_dotenv()
langfuse_init()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info(
        "application_startup",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        api_prefix=settings.API_V1_STR,
    )

    app.state.optional_components = {
        "cache": "degraded",
        "memory": "degraded" if settings.MEMORY_ENABLED else "disabled",
    }
    app.state.runtime_readiness = RuntimeReadinessService(
        check_database, lambda: agent._graph is not None, lambda: app.state.optional_components.copy()
    )
    # Initialize cache service (connects to Valkey if configured)
    try:
        await cache_service.initialize()
        app.state.optional_components["cache"] = (
            "degraded" if settings.VALKEY_HOST and isinstance(cache_service, InMemoryCacheService) else "ready"
        )
    except Exception:
        logger.exception("cache_initialization_failed")

    # Pre-warm the LangGraph agent: create graph + connection pool at startup
    # to avoid cold-start latency on the first request
    try:
        if settings.LOCAL_MVP_ENABLED:
            local_mvp = LocalMVPService(agent.create_graph)
            if settings.V1_BACKEND_ENABLED:
                product = V1Backend(local_mvp)
                if settings.V1_SOURCE_OWNER_ID:
                    await product.store.bootstrap(settings.V1_SOURCE_OWNER_ID, settings.V1_REQUIRED_CHECKS)
                if settings.V11_BACKEND_ENABLED:
                    reviews = V11Backend(product, evaluation=getattr(app.state, "review_evaluation", None))
                    if settings.V1_SOURCE_OWNER_ID:
                        await reviews.bootstrap(settings.V1_SOURCE_OWNER_ID)
            if agent._graph is not None:
                raise RuntimeError("local_mvp_requires_fresh_graph")
            agent.releaseguard_repository = local_mvp.domain
            agent.local_mvp_dependencies = local_mvp.dependencies()
            app.state.local_mvp = local_mvp
        await agent.create_graph()
        logger.info("graph_pre_warmed")
    except Exception:
        logger.exception("graph_pre_warm_failed")

    # Pre-warm mem0 AsyncMemory: initializes pgvector connection and schema check
    # so the first search() cache miss or add() doesn't pay the ~130ms cold-init cost
    try:
        await memory_service.initialize()
        app.state.optional_components["memory"] = "ready" if settings.MEMORY_ENABLED else "disabled"
    except Exception:
        logger.exception("memory_service_pre_warm_failed")

    yield

    # Cleanup on shutdown
    await cache_service.close()
    if agent._connection_pool:
        await agent._connection_pool.close()
        logger.info("connection_pool_closed")
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set up Prometheus metrics
setup_metrics(app)

# Add logging context middleware (must be added before other middleware to capture context)
app.add_middleware(LoggingContextMiddleware)

# Add custom metrics middleware
app.add_middleware(MetricsMiddleware)

# Add profiling middleware (DEBUG only — saves HTML to /tmp on slow requests)
if settings.DEBUG:
    app.add_middleware(ProfilingMiddleware)

# Add correlation ID middleware — must be outermost so request_id is set before all others
app.add_middleware(CorrelationIdMiddleware)

# Set up rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # pyright: ignore[reportArgumentType]


# Add validation exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors from request data.

    Args:
        request: The request that caused the validation error
        exc: The validation error

    Returns:
        JSONResponse: A formatted error response
    """
    # Log the validation error
    logger.warning(
        "validation_error",
        error_count=len(exc.errors()),
    )

    # Format the errors to be more user-friendly
    formatted_errors = []
    for error in exc.errors():
        loc = " -> ".join([str(loc_part) for loc_part in error["loc"] if loc_part != "body"])
        formatted_errors.append({"field": loc, "message": error["msg"]})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": formatted_errors},
    )


# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(runtime_router)


@app.get("/")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["root"][0])
async def root(request: Request):
    """Root endpoint returning basic API information."""
    logger.info("root_endpoint_called")
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "environment": settings.ENVIRONMENT.value,
        "swagger_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/health")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["health"][0])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint with environment-specific information.

    Returns:
        JSONResponse: Health status payload, with HTTP 503 when the
        database is unreachable so load balancers can drop the instance.
    """
    logger.info("health_check_called")

    # Check database connectivity
    db_healthy = (await check_database()).database

    response = {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "components": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
        "timestamp": datetime.now().isoformat(),
    }

    # If DB is unhealthy, set the appropriate status code
    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(content=response, status_code=status_code)
