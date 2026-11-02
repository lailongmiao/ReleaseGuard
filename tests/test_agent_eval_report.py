"""Hard report gates and baseline corruption, not mirror-only formatting tests."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.services.releaseguard.normalization import canonical_hash
from evals.releaseguard.cli import exit_status, parser
from evals.releaseguard.dataset import load_dataset
from evals.releaseguard.metrics import evaluate_case
from evals.releaseguard.judge import FakeJudge, evaluate_judge, JudgeRequest
from evals.releaseguard.reports import (
    Baseline,
    CodeState,
    EvaluationReport,
    JudgeAttempt,
    compare_baseline,
    count_statuses,
    freeze_baseline,
    save_report,
    summarize,
)
from evals.releaseguard.runner import run_dataset
from evals.releaseguard.schemas import METRICS, RunConfig, Status

DATASET = load_dataset(Path("evals/datasets/releaseguard_v1.json"))
CONFIG = RunConfig(repeat=2)


@pytest.fixture(scope="module")
def report():
    """Measured facts from real offline services, not hardcoded passed results."""
    observed = asyncio.run(run_dataset(DATASET, CONFIG))
    cases = {c.case_id: c for c in DATASET.cases}
    results = tuple(evaluate_case(cases[r.case_id], r) for r in observed)
    return EvaluationReport(
        eval_run_id="synthetic-eval",
        dataset=DATASET,
        config=CONFIG,
        code=CodeState(head="a" * 40, source_hash="b" * 64, dirty_paths=()),
        started_at=DATASET.cases[0].clock_start,
        results=results,
        summary=summarize(DATASET, CONFIG, results),
    )


def test_actual_full_report_and_roundtrip(report):
    """56 unique cases, 112 attempts, 896 separately gated rule decisions."""
    assert report.summary.status is Status.PASSED
    assert report.summary.cases.planned == 112
    assert sum(c.planned for c in report.summary.metrics.values()) == 896
    assert EvaluationReport.model_validate_json(report.model_dump_json()) == report
    assert report.real_judge_status == report.langfuse_status == "unverified"
    assert report.risk_predictor_status == "not_applicable" and report.cost is None
    assert compare_baseline(report, freeze_baseline(report)).status == "passed"


@pytest.mark.parametrize("metric", METRICS)
def test_single_failed_or_unknown_rule_not_averaged(report, metric):
    """Even 895/896 passes cannot hide an altered required rule."""
    original = report.results[0]
    broken = original.model_copy(
        update={
            "metrics": tuple(
                m.model_copy(update={"status": Status.FAILED}) if m.metric_id == metric else m
                for m in original.metrics
            )
        }
    )
    results = (broken, *report.results[1:])
    summary = summarize(DATASET, CONFIG, results)
    assert summary.status is Status.FAILED
    assert "metric_evidence_conflict" in summary.reasons
    changed = report.model_copy(update={"results": results, "summary": summary})
    assert compare_baseline(changed, freeze_baseline(report)).status == "failed"


def test_extra_call_recomputed_even_if_metric_status_forged(report):
    """A false all-passed metric list is rejected against authoritative facts."""
    first = report.results[0]
    facts = first.facts.model_copy(update={"calls": first.facts.calls + 1})
    bad = first.model_copy(update={"facts": facts, "semantic_hash": canonical_hash(facts)})
    summary = summarize(DATASET, CONFIG, (bad, *report.results[1:]))
    assert summary.status is Status.FAILED
    assert summary.metrics["tool_safety"].failed == 1


@pytest.mark.parametrize("kind", ["case", "metric", "authority", "na", "duplicate"])
def test_missing_and_na_cannot_pass(report, kind):
    """Incomplete facts remain planned; required N/A is a rule violation."""
    first = report.results[0]
    if kind == "case":
        results = report.results[1:]
    elif kind == "metric":
        results = (first.model_copy(update={"metrics": first.metrics[1:]}), *report.results[1:])
    elif kind == "authority":
        facts = first.facts.model_copy(update={"domain": None, "completed": False})
        result = evaluate_case(
            DATASET.cases[0], first.model_copy(update={"facts": facts, "semantic_hash": canonical_hash(facts)})
        )
        results = (result, *report.results[1:])
    elif kind == "na":
        results = (
            first.model_copy(
                update={
                    "metrics": (
                        first.metrics[0].model_copy(update={"status": Status.NOT_APPLICABLE}),
                        *first.metrics[1:],
                    )
                }
            ),
            *report.results[1:],
        )
    else:
        results = (*report.results, first)
    summary = summarize(DATASET, CONFIG, results)
    assert summary.status is not Status.PASSED
    assert summary.cases.planned == 112


def test_empty_and_exact_denominators(report):
    """0/0, all N/A and missing attempts are not 100% and not a quality zero."""
    assert count_statuses([]).pass_rate is None
    assert count_statuses([Status.NOT_APPLICABLE]).verified_coverage is None
    counts = count_statuses([Status.PASSED, Status.FAILED, Status.UNVERIFIED, Status.NOT_APPLICABLE])
    assert counts.pass_rate == 1 / 3 and counts.verified_coverage == 2 / 3
    assert summarize(DATASET, CONFIG, ()).status is Status.UNVERIFIED
    empty = DATASET.model_copy(update={"cases": ()})
    assert summarize(empty, CONFIG, ()).status is Status.UNVERIFIED


def test_report_forged_summary_cost_and_nonfinite(report):
    """Stored aggregate and provenance are independently validated on load."""
    for updates in (
        {"cost": 0.0},
        {"summary": {"status": "passed", "cases": report.summary.cases.model_dump(), "metrics": {}, "reasons": []}},
        {"unknown_field": True},
    ):
        raw = report.model_dump(mode="json") | updates
        with pytest.raises(ValidationError):
            EvaluationReport.model_validate(raw)


def test_baseline_incompatible_never_autorefreshed(report):
    """Changed content identity/case coverage requires separate reviewed migration."""
    baseline = freeze_baseline(report)
    assert compare_baseline(report, baseline.model_copy(update={"dataset_hash": "c" * 64})).status == "incomparable"
    assert compare_baseline(report, baseline.model_copy(update={"facts": {}})).status == "incomparable"
    missing = report.model_copy(update={"results": (), "summary": summarize(DATASET, CONFIG, ())})
    with pytest.raises(ValueError, match="baseline_unverified"):
        freeze_baseline(missing)
    assert Baseline.model_validate_json(baseline.model_dump_json()) == baseline


def test_safe_output_and_cli_no_implicit_live(report, tmp_path):
    """Output escape rejected; default args are offline and no baseline write."""
    with pytest.raises(ValueError, match="report_path_denied"):
        save_report(report, tmp_path / "escaped.json")
    args = parser().parse_args(["run"])
    assert args.judge == args.langfuse == "off" and args.report is None
    for status, code in (("passed", 0), ("failed", 1), ("unverified", 2), ("incomparable", 2)):
        assert exit_status(status) == code


def test_fresh_process_offline_bootstrap_and_legacy_no_report(tmp_path: Path):
    """Fresh imports must not write mem0 state or install app file logging."""
    missing_mem0 = tmp_path / "missing-mem0"
    environment = {**os.environ, "MEM0_DIR": str(missing_mem0), "MEM0_TELEMETRY": "true"}
    commands = [
        [
            sys.executable,
            "-B",
            "-c",
            "from evals.releaseguard.dataset import load_dataset; from pathlib import Path; "
            "from rich import print; print({'cases':len(load_dataset(Path('evals/datasets/releaseguard_v1.json')).cases)})",
        ],
        [sys.executable, "-B", "-m", "evals.main", "--quick", "--no-report"],
    ]
    for command in commands:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=environment,
        )
        assert process.returncode == 0, process.stderr[-1000:]
        assert "Logging error" not in process.stderr
        assert not missing_mem0.exists()


def test_fake_judge_missing_duplicate_and_foreign_association(report):
    """Auxiliary results cannot be silently dropped, reused or forged as real."""
    case = DATASET.cases[0]
    dataset = DATASET.model_copy(update={"cases": (case,)})
    dataset = dataset.model_copy(
        update={"dataset_hash": canonical_hash(dataset.model_dump(mode="json", exclude={"dataset_hash"}))}
    )
    config = RunConfig(repeat=1, judge="fake")
    observed = (report.results[0],)
    request = JudgeRequest(
        case_id=case.case_id,
        dataset_hash=dataset.dataset_hash,
        explanation="Synthetic explanation",
        citation_summary="Synthetic references",
    )
    result = asyncio.run(evaluate_judge(request, FakeJudge()))
    attempt = JudgeAttempt(case_id=case.case_id, attempt=1, result=result)
    value = report.model_copy(
        update={
            "dataset": dataset,
            "config": config,
            "results": observed,
            "summary": summarize(dataset, config, observed),
            "judges": (attempt,),
        }
    )
    assert EvaluationReport.model_validate_json(value.model_dump_json()).judges == (attempt,)
    for judges in (
        (),
        (attempt, attempt),
        (attempt.model_copy(update={"case_id": "RG-R99"}),),
        (attempt.model_copy(update={"result": result.model_copy(update={"case_id": "RG-R99"})}),),
        (attempt.model_copy(update={"result": result.model_copy(update={"provenance": "provider"})}),),
    ):
        with pytest.raises(ValidationError):
            EvaluationReport.model_validate_json(value.model_copy(update={"judges": judges}).model_dump_json())
