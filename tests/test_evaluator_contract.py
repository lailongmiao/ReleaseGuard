"""Evaluation judge portability and unverified external-state behavior."""

import asyncio

from evals.evaluator import Evaluator


def test_evaluation_dry_run_is_unverified_without_langfuse() -> None:
    """Dry run does not require Langfuse credentials or write scores."""
    evaluator = Evaluator()

    asyncio.run(evaluator.run(generate_report_file=False, dry_run=True))

    assert evaluator.report["dry_run"] is True
    assert evaluator.report["verification_status"] == "unverified"
