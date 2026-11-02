"""Traceable safety reports, exact denominators and explicitly frozen baselines."""

import asyncio
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from app.services.releaseguard.normalization import canonical_hash
from evals.releaseguard.judge import FakeJudge, JudgeRequest, JudgeResult, evaluate_judge
from evals.releaseguard.metrics import evaluate_case
from evals.releaseguard.runner import run_dataset
from evals.releaseguard.schemas import METRICS, CaseResult, Dataset, Envelope, Hash, Name, RunConfig, Status

ROOT = Path(__file__).resolve().parents[2]


class CodeState(Envelope):
    """Hash explicit source files, never ignored credentials or raw Git diff."""

    head: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_hash: Hash
    dirty_paths: tuple[str, ...]


def code_state() -> CodeState:
    """Read only tracked/nonignored sources and status; report paths are excluded."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    paths = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    fingerprints = {}
    for name in sorted(set(paths)):
        path = ROOT / name
        if path.is_file() and (
            name in {"pyproject.toml", "uv.lock"}
            or name.startswith(("app/", "evals/", "tests/"))
            and path.suffix == ".py"
            or name.startswith("evals/datasets/")
            and path.suffix == ".json"
        ):
            fingerprints[name] = sha256(path.read_bytes()).hexdigest()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return CodeState(
        head=head, source_hash=canonical_hash(fingerprints), dirty_paths=tuple(sorted(s[3:] for s in status))
    )


class Counts(Envelope):
    """All planned attempts remain in the applicable denominator."""

    planned: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    unverified: int = Field(ge=0)
    not_applicable: int = Field(default=0, ge=0)
    pass_rate: float | None = None
    verified_coverage: float | None = None


def count_statuses(values: list[Status]) -> Counts:
    """Zero applicable denominator stays unknown, never perfect or zero quality."""
    passed, failed = values.count(Status.PASSED), values.count(Status.FAILED)
    unverified, na = values.count(Status.UNVERIFIED), values.count(Status.NOT_APPLICABLE)
    denominator = len(values) - na
    return Counts(
        planned=len(values),
        passed=passed,
        failed=failed,
        unverified=unverified,
        not_applicable=na,
        pass_rate=passed / denominator if denominator else None,
        verified_coverage=(passed + failed) / denominator if denominator else None,
    )


class Summary(Envelope):
    """Averages cannot grant exemptions to any hard safety failure."""

    status: Status
    cases: Counts
    metrics: dict[str, Counts]
    reasons: tuple[Name, ...]


def summarize(dataset: Dataset, config: RunConfig, results: tuple[CaseResult, ...]) -> Summary:
    """Recompute rules from actual facts even if stored metric statuses were forged."""
    indexed = {(r.case_id, r.attempt): r for r in results}
    metric_statuses: dict[str, list[Status]] = {m: [] for m in METRICS}
    case_statuses = []
    reasons: set[str] = set()
    planned = {(c.case_id, attempt) for c in dataset.cases for attempt in range(1, config.repeat + 1)}
    if len(indexed) != len(results) or not set(indexed) <= planned:
        reasons.add("unexpected_or_duplicate_attempt")
    for case in dataset.cases:
        hashes = set()
        for attempt in range(1, config.repeat + 1):
            result = indexed.get((case.case_id, attempt))
            statuses = []
            if result is None:
                reasons.add("missing_attempt")
            recalculated = evaluate_case(case, result) if result is not None else None
            for metric in case.required_metrics:
                status = Status.UNVERIFIED
                if result is not None:
                    assert recalculated is not None
                    actual = next((m for m in result.metrics if m.metric_id == metric), None)
                    recomputed = next(m for m in recalculated.metrics if m.metric_id == metric)
                    if actual is None:
                        reasons.add("missing_metric")
                    elif actual.status is Status.NOT_APPLICABLE:
                        status = Status.FAILED
                        reasons.add("required_metric_not_applicable")
                    elif actual != recomputed or len({m.metric_id for m in result.metrics}) != len(result.metrics):
                        status = Status.FAILED
                        reasons.add("metric_evidence_conflict")
                    else:
                        status = recomputed.status
                    if result.semantic_hash != canonical_hash(result.facts):
                        status = Status.FAILED
                        reasons.add("semantic_hash_conflict")
                statuses.append(status)
                metric_statuses[metric].append(status)
            case_statuses.append(
                Status.FAILED
                if Status.FAILED in statuses
                else Status.UNVERIFIED
                if Status.UNVERIFIED in statuses
                else Status.PASSED
            )
            if result:
                hashes.add(result.semantic_hash)
        if len(hashes) > 1:
            reasons.add("repeat_nondeterministic")
    counts = count_statuses(case_statuses)
    status = (
        Status.FAILED
        if counts.failed or reasons & {"unexpected_or_duplicate_attempt", "repeat_nondeterministic"}
        else Status.UNVERIFIED
        if not counts.planned or counts.unverified
        else Status.PASSED
    )
    return Summary(
        status=status,
        cases=counts,
        metrics={m: count_statuses(s) for m, s in metric_statuses.items()},
        reasons=tuple(sorted(reasons)),
    )


class EvaluationReport(Envelope):
    """Full synthetic authority, all attempts, measured timing and separate layers."""

    schema_version: Literal["report.v1"] = "report.v1"
    eval_run_id: Name
    dataset: Dataset
    config: RunConfig
    code: CodeState
    started_at: datetime
    results: tuple[CaseResult, ...]
    summary: Summary
    judges: tuple["JudgeAttempt", ...] = ()
    real_judge_status: Literal["unverified"] = "unverified"
    langfuse_status: Literal["unverified"] = "unverified"
    production_execution: Literal["disabled"] = "disabled"
    risk_predictor_status: Literal["not_applicable"] = "not_applicable"
    risk_predictor_reason: Literal["no_business_risk_assessor"] = "no_business_risk_assessor"
    latency_provenance: Literal["measured_offline_wall"] = "measured_offline_wall"
    cost: float | None = None
    cost_provenance: Literal["unknown"] = "unknown"

    @model_validator(mode="after")
    def validate_summary(self) -> "EvaluationReport":
        """Reject forged aggregate, stale dataset identity and fabricated cost."""
        if canonical_hash(self.dataset.model_dump(mode="json", exclude={"dataset_hash"})) != self.dataset.dataset_hash:
            raise ValueError("report_dataset_hash_conflict")
        if self.summary != summarize(self.dataset, self.config, self.results) or self.cost is not None:
            raise ValueError("report_summary_conflict")
        if self.config.judge == "off" and self.judges or any(r.result.provenance == "provider" for r in self.judges):
            raise ValueError("report_judge_provenance_conflict")
        if self.config.judge == "fake":
            expected = {(r.case_id, r.attempt) for r in self.results}
            actual = {(j.case_id, j.attempt) for j in self.judges}
            if (
                actual != expected
                or len(actual) != len(self.judges)
                or any(j.case_id != j.result.case_id or j.result.provenance != "fake" for j in self.judges)
            ):
                raise ValueError("report_judge_association_conflict")
        return self


class JudgeAttempt(Envelope):
    """Tie auxiliary output to a specific observed case attempt."""

    case_id: Name
    attempt: int = Field(strict=True, ge=1)
    result: JudgeResult


async def build_report(dataset: Dataset, config: RunConfig) -> EvaluationReport:
    """Observe before measuring; repeated scenarios instantiate their own stores."""
    code = await asyncio.to_thread(code_state)
    started = datetime.now(timezone.utc)
    observed = await run_dataset(dataset, config)
    cases = {c.case_id: c for c in dataset.cases}
    results = tuple(evaluate_case(cases[r.case_id], r) for r in observed)
    judges = []
    if config.judge == "fake":
        transport = FakeJudge()
        for result in results:
            # This is a derived structured diagnostic, not a claimed business LLM explanation.
            judged = await evaluate_judge(
                JudgeRequest(
                    case_id=result.case_id,
                    dataset_hash=dataset.dataset_hash,
                    explanation="Synthetic observed diagnostic: "
                    + ",".join(result.facts.diagnostics or ("contract_recorded",)),
                    citation_summary="Synthetic persisted citation identifiers only.",
                ),
                transport,
            )
            judges.append(JudgeAttempt(case_id=result.case_id, attempt=result.attempt, result=judged))
    run_id = (
        "eval-"
        + canonical_hash(
            {
                "dataset": dataset.dataset_hash,
                "source": code.source_hash,
                "started": started.isoformat(),
                "config": config,
            }
        )[:24]
    )
    return EvaluationReport(
        eval_run_id=run_id,
        dataset=dataset,
        config=config,
        code=code,
        started_at=started,
        results=results,
        summary=summarize(dataset, config, results),
        judges=tuple(judges),
    )


def save_report(report: EvaluationReport, path: Path) -> None:
    """Only safe local output directory, outside the business case I/O guard."""
    target, allowed = path.resolve(), (ROOT / "evals/reports").resolve()
    if not target.is_relative_to(allowed) or target.suffix != ".json":
        raise ValueError("report_path_denied")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")


class Baseline(Envelope):
    """Reviewed compact first-pass facts; cannot be automatically refreshed."""

    schema_version: Literal["baseline.v1"] = "baseline.v1"
    dataset_hash: Hash
    dataset_version: Name
    oracle_version: Name
    metric_version: Literal["rules.v1"] = "rules.v1"
    code: CodeState
    facts: dict[str, Hash]
    status: Literal["passed"] = "passed"


def freeze_baseline(report: EvaluationReport) -> Baseline:
    """Only measured passing repeated runs may propose a baseline for review."""
    if report.summary.status is not Status.PASSED or report.config.repeat < 2:
        raise ValueError("baseline_unverified")
    return Baseline(
        dataset_hash=report.dataset.dataset_hash,
        dataset_version=report.dataset.version,
        oracle_version=report.dataset.oracle_version,
        code=report.code,
        facts={r.case_id: r.semantic_hash for r in report.results},
    )


class Comparison(Envelope):
    """Incompatible manifests must be reviewed; regression is a hard failure."""

    status: Literal["passed", "failed", "incomparable", "unverified"]
    reasons: tuple[Name, ...]


def compare_baseline(report: EvaluationReport, baseline: Baseline) -> Comparison:
    """No average score, changed oracle, extra call or missing fact can pass."""
    if (report.dataset.dataset_hash, report.dataset.version, report.dataset.oracle_version) != (
        baseline.dataset_hash,
        baseline.dataset_version,
        baseline.oracle_version,
    ):
        return Comparison(status="incomparable", reasons=("baseline_version_conflict",))
    if set(baseline.facts) != {c.case_id for c in report.dataset.cases}:
        return Comparison(status="incomparable", reasons=("baseline_case_set_conflict",))
    summary = summarize(report.dataset, report.config, report.results)
    if summary.status is not Status.PASSED:
        return Comparison(
            status="failed" if summary.status is Status.FAILED else "unverified", reasons=("hard_gate_not_passed",)
        )
    if any(r.semantic_hash != baseline.facts[r.case_id] for r in report.results):
        return Comparison(status="failed", reasons=("baseline_facts_regression",))
    return Comparison(status="passed", reasons=("all_case_facts_unchanged",))
