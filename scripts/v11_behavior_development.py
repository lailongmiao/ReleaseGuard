"""Explicit two-group paid development probe; synthetic sources never become product facts."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from jose import jwt
from rich.console import Console

from app.core.config import settings
from app.schemas.v11_backend import ReviewCitation
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_policy import INDEX_POLICY, load_rules
from app.services.v11_provider import SYSTEM, ReviewProvider, build_context
from app.services.v11_storage import V11Store
from evals.v11_backend.behavior_cases import cases, evidence
from scripts.v11_stack import configuration

ROOT = Path(__file__).resolve().parents[1]


async def preflight(store: V11Store) -> dict:
    """This turn can spend at most one additional yuan and four development model sends."""
    await store.require_real_budget()
    async with await store.connection.connect() as db, db.transaction():
        await db.execute("SET TRANSACTION READ ONLY")
        row = await (
            await db.execute(
                "SELECT COALESCE(SUM(b.charge),0) AS charge,COUNT(*) FILTER(WHERE b.attempt='analysis-1') AS calls "
                "FROM v1_budget b JOIN v1_commands c USING(owner_id,command_id) WHERE c.budget_category='development'"
            )
        ).fetchone()
    if not row or row["calls"] >= 12 or row["charge"] + Decimal("0.1312") > Decimal("1.31397960"):
        raise DomainError("repair2_development_limit")
    if settings.V11_BUDGET_CATEGORY != "development" or Decimal(str(settings.V1_MODEL_BUDGET_CNY)) > Decimal(
        "24.91547198"
    ):
        raise DomainError("repair2_runtime_budget_limit_required")
    return {"development_calls_before": row["calls"], "development_charge_before": str(row["charge"])}


async def run(private: Path, group: str, output: Path) -> None:
    """Exactly one persisted claim and one real Provider send; no JSON repair or retry."""
    if output.exists():
        raise ValueError("development_report_exists")
    connection = LocalPostgresStore()
    store = V11Store(connection)
    tokens = json.loads((private / "sessions.json").read_text(encoding="utf-8"))
    claims = jwt.decode(tokens["1"], settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    actor = await connection.authenticate(claims["sub"], claims["exp"])
    before = await preflight(store)
    await store.source("markdown-rules", actor)
    selected_cases = [case for case in cases() if case["group"] == group]
    facts, citations, roles = evidence(selected_cases)
    _, entries = load_rules("checkout")
    mandatory = []
    for entry, text in entries:
        if entry["role"] != "application_policy":
            continue
        documents = [e for e, _ in entries if e["rule_set_id"] == entry["rule_set_id"]]
        mandatory.append(
            ReviewCitation(
                citation_id="rule:" + canonical_hash([entry["filename"], entry["sha256"]]),
                source_role="application_policy",
                kind="rule",
                snapshot_id="synthetic-snapshot",
                generation_id="rules:"
                + canonical_hash([entry["rule_set_id"], canonical_hash(documents), INDEX_POLICY]),
                text=text,
                text_hash=canonical_hash(text),
                locator="knowledge:" + entry["filename"] + "#" + entry["version_condition"],
            )
        )
    query = (
        "逐文件对照旧、新实现：哪些行为保持，哪些默认值或可选路径确实变化？引用两侧依据，不从函数新增推导行为新增。"
        if group == "A"
        else "逐文件解释修改：区分文档措辞、执行条件变化及未变上下文。指出有依据的重要变化，不将相同赋值自动视为等价。"
    )
    user, selected = build_context(
        query,
        mandatory,
        [],
        citations,
        {
            "coverage": facts["coverage"],
            "index_complete": True,
            "environment": "development",
            "provenance": "synthetic_independent_development_not_github",
        },
        roles,
    )
    command = "repair2-independent-" + group
    digest = canonical_hash([SYSTEM, user, actor.user_id])
    prior = await store.claim_review(actor, command, digest, "application-policy", "development", digest)
    if prior is not None:
        raise DomainError("development_prior_result_no_resend")
    report = {
        "at": datetime.now(UTC).isoformat(),
        "provenance": "synthetic_sources_real_business_provider",
        "not_product_graph_evidence": True,
        "group": group,
        "budget_before": before,
        "system_hash": canonical_hash(SYSTEM),
        "input": json.loads(user),
        "citations": [c.model_dump(mode="json") for c in selected],
        "facts": facts,
        "status": "started",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        result, telemetry = await ReviewProvider(store, enabled=True).review(user, selected, actor, command, command)
        report.update(status="completed", output=result.model_dump(mode="json"), telemetry=telemetry)
    except DomainError as error:
        reason = str(error)
        report.update(status="unknown" if "unknown" in reason else "failed", reason=reason)
    await store.complete(actor, command, {"status": report["status"], "provenance": report["provenance"]})
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Console().print({"group": group, "status": report["status"], "report": str(output)})


def main() -> None:
    """Use the existing operator launch to load the designated private environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--group", choices=["A", "B"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    _, values = configuration(args.compose_command, offline=False)
    if any(str(getattr(settings, key)) != values[key] for key in ("POSTGRES_PORT", "POSTGRES_DB")):
        raise ValueError("designated_environment_required")
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run(args.compose_command.parent, args.group, args.report))


if __name__ == "__main__":
    main()
