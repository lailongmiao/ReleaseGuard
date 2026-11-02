"""Compatible legacy command flags, using rich and safe external-off defaults."""

import argparse
import asyncio
from typing import Any

from rich.console import Console
from rich.table import Table

from evals.evaluator import Evaluator

console = Console()


def display_summary(report: dict[str, Any]) -> None:
    """No zero-denominator success and no required report-path key."""
    table = Table(title="Evaluation summary")
    table.add_column("Fact")
    table.add_column("Value")
    for key in ("total_traces", "read_status", "quality_status", "verification_status", "reason_code"):
        table.add_row(key, str(report.get(key, "unknown")))
    console.print(table)
    if report.get("generate_report_path"):
        console.print(report["generate_report_path"])


async def run_evaluation(generate_report: bool = True) -> dict[str, Any]:
    """Legacy quick mode never implicitly authorizes trace reads or score writes."""
    evaluator = Evaluator()
    report = await evaluator.run(generate_report_file=generate_report, dry_run=True)
    display_summary(report)
    return report


def main() -> None:
    """Accept existing Makefile flags, honor no-report in every mode."""
    parser = argparse.ArgumentParser(description="External-off evaluation; versioned runner: evals.releaseguard.cli")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.interactive and console.input("Run with external access off? [Y/n] ").lower() in {"n", "no"}:
        return
    asyncio.run(run_evaluation(generate_report=not args.no_report))


if __name__ == "__main__":
    main()
