"""Prometheus metrics configuration for the application.

This module sets up and configures Prometheus metrics for monitoring the application.
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
from starlette_prometheus import metrics

from app.core.logging import logger

if TYPE_CHECKING:
    from app.services.llm.contracts import UsageInfo
    from app.schemas.releaseguard import ChangeRunState


class ReleaseGuardMetrics:
    """Bounded labels; observe only a first committed fact, never a replay."""

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        """Allow isolated registries in tests and fresh counters on process restart."""
        self.events = Counter(
            "releaseguard_facts_total", "First committed local facts", ["category", "outcome"], registry=registry
        )
        self.runs = Counter("change_runs_total", "First registered local runs", registry=registry)
        self.states = Counter(
            "releaseguard_states_total",
            "Committed states from validated domain enums",
            ["status", "risk"],
            registry=registry,
        )
        self.tokens = Counter("releaseguard_llm_tokens_total", "Reported tokens only", ["role"], registry=registry)
        self.cost = Counter("releaseguard_llm_cost_usd_total", "Reported cost only", ["role"], registry=registry)
        self.availability = Counter(
            "releaseguard_llm_measurements_total",
            "Missing data stays unknown",
            ["role", "measurement", "quality"],
            registry=registry,
        )

    def fact(self, category: str, outcome: str) -> None:
        """Reject arbitrary/high-cardinality labels before creating a time series."""
        categories = {"state", "gate", "approval", "tool", "execution", "recovery", "verification", "receipt"}
        outcomes = {"recorded", "pending", "denied", "unknown", "completed", "failed", "passed", "expired", "revoked"}
        if category not in categories or outcome not in outcomes:
            raise ValueError("metric_label_invalid")
        self.events.labels(category, outcome).inc()

    def usage(self, role: Literal["chat", "memory", "embedding", "naming", "judge"], usage: UsageInfo) -> None:
        """Unknown usage/cost never increments a consumption counter."""
        for measurement, value in (("tokens", usage.total_tokens), ("cost", usage.cost_usd)):
            quality = "known" if value is not None else "unknown"
            self.availability.labels(role, measurement, quality).inc()
            if value is not None:
                counter = self.tokens if measurement == "tokens" else self.cost
                counter.labels(role).inc(value)


releaseguard_metrics = ReleaseGuardMetrics()


def observe_fact(category: str, outcome: str = "recorded", /, **references: object) -> None:
    """Observability failures cannot alter authority or cause another tool call."""
    try:
        releaseguard_metrics.fact(category, outcome)
        logger.info("releaseguard_fact_observed", category=category, outcome=outcome, **references)
    except Exception:
        logger.exception("telemetry_observation_failed")


def observe_usage(role: str, usage: UsageInfo) -> None:
    """The fixed role vocabulary excludes untrusted model/resource labels."""
    try:
        role = "memory" if role == "memory_llm" else role
        if role in {"chat", "memory", "embedding", "naming", "judge"}:
            releaseguard_metrics.usage(role, usage)  # pyright: ignore[reportArgumentType]
    except Exception:
        logger.exception("telemetry_observation_failed")


def observe_run(**references: object) -> None:
    """Count first registration only; repositories suppress replay beforehand."""
    try:
        releaseguard_metrics.runs.inc()
    except Exception:
        logger.exception("telemetry_observation_failed")
    observe_fact("state", **references)


def observe_state(state: ChangeRunState) -> None:
    """Observe repository-validated enums; object identities never enter labels."""
    try:
        risk = state.risk_assessment.level.value if state.risk_assessment else "UNKNOWN"
        releaseguard_metrics.states.labels(state.status.value, risk).inc()
    except Exception:
        logger.exception("telemetry_observation_failed")


# Request metrics
http_requests_total = Counter("http_requests_total", "Total number of HTTP requests", ["method", "endpoint", "status"])

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", ["method", "endpoint"]
)

# Database metrics
db_connections = Gauge("db_connections", "Number of active database connections")

# Custom business metrics
orders_processed = Counter("orders_processed_total", "Total number of orders processed")

llm_inference_duration_seconds = Histogram(
    "llm_inference_duration_seconds",
    "Time spent processing LLM inference",
    ["model"],
    buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
)


llm_stream_duration_seconds = Histogram(
    "llm_stream_duration_seconds",
    "Time spent processing LLM stream inference",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

session_names_generated_total = Counter(
    "session_names_generated_total",
    "Total session names generated by the LLM",
    ["status"],  # "success" | "error"
)


def setup_metrics(app):
    """Set up Prometheus metrics middleware and endpoints.

    Args:
        app: FastAPI application instance
    """
    # HTTP collection is owned by our route-template MetricsMiddleware.
    app.add_route("/metrics", metrics)
