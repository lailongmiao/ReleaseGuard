"""Safe legacy facade; unscoped fetching and implicit external writes are removed."""

import asyncio
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.core.config import settings
from evals.helpers import generate_report, initialize_metrics_summary, initialize_report
from evals.metrics import metrics
from evals.releaseguard.langfuse_io import LangfuseIO


class Evaluator:
    """Preserve run/report usage, with explicitly scoped read-only external access."""

    def __init__(self, *, langfuse_io: LangfuseIO | None = None) -> None:
        """Configuration alone never creates a client or proves verification."""
        self.io = langfuse_io or LangfuseIO(None, None)
        self.verification_status = "unverified"
        self.report: dict[str, Any] = initialize_report(settings.EVALUATION_LLM)
        self.report.update(
            verification_status="unverified",
            quality_status="unverified",
            judge_provider=settings.JUDGE_LLM_PROVIDER,
            judge_model=settings.JUDGE_LLM_MODEL,
        )
        initialize_metrics_summary(self.report, metrics)

    async def run(self, generate_report_file: bool = True, dry_run: bool = True) -> dict[str, Any]:
        """Default off; live scoring uses the versioned typed Phase 7 API/CLI."""
        started = perf_counter()
        self.report["dry_run"] = dry_run
        result = await self.io.read(at=datetime.now(timezone.utc))
        self.report["total_traces"] = len(result.traces)
        self.report["read_status"] = result.status.value
        self.report["reason_code"] = result.reason_code
        self.report["unverified_traces"] = len(result.traces)
        self.report["duration_seconds"] = perf_counter() - started
        # A successful read does not evaluate quality and does not authorize writes.
        if generate_report_file:
            await asyncio.to_thread(generate_report, self.report)
        return self.report
