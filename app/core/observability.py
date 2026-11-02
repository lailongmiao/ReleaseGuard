"""Observability module for the application."""

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from app.core.config import settings
from app.core.logging import logger
from app.core.telemetry import mask_trace


def langfuse_init():
    """Initialize Langfuse."""
    Langfuse(
        tracing_enabled=settings.LANGFUSE_TRACING_ENABLED,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        environment=settings.ENVIRONMENT.value,
        debug=False,
        mask=mask_trace,
    )

    logger.info("langfuse_configured", enabled=settings.LANGFUSE_TRACING_ENABLED)


def get_langfuse_callback_handler() -> CallbackHandler:
    """Create a Langfuse CallbackHandler for tracking LLM interactions.

    Returns:
        CallbackHandler: Configured Langfuse callback handler.
    """
    langfuse_init()
    return CallbackHandler(public_key=settings.LANGFUSE_PUBLIC_KEY or None)


langfuse_callback_handler = get_langfuse_callback_handler()
