"""Freeze two local evaluation candidates from preserved evidence; never issue send permits."""

import json
from decimal import Decimal
from pathlib import Path

from rich.console import Console

from app.schemas.v11_backend import ReviewCitation
from app.services.releaseguard.normalization import canonical_hash
from app.services.v11_policy import PROMPT_VERSION, citation_line_roles, token_count
from app.services.v11_provider import SYSTEM, build_context
from evals.v1_backend.provenance import code_state

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evals" / "reports"
QUESTIONS = {
    "A": "说明 README 和 action.yml 中 filter 与 sparse-checkout 的关系怎样改写，引用实际可见内容。",
    "B": "对照旧、新代码，判断 sparse-checkout 对应的默认 filter 是否新增、处理阶段是否迁移，以及 none 关闭选项与执行条件如何变化。引用旧、新实现依据，区分实现与文档，不推断实际运行结果。",
}


def freeze(prefix: str = "v11_eval_ready") -> dict:
    """Use the product selector unchanged for both questions, with all archived diff candidates."""
    raw = json.loads((REPORTS / "v11_eval_candidate_archived_inputs.json").read_text(encoding="utf-8"))
    old = json.loads((REPORTS / "v11_eval_candidate_frozen_payload.json").read_text(encoding="utf-8"))
    sent = json.loads(old["user"])
    assert QUESTIONS["A"] == sent["query"]
    assert canonical_hash(raw["facts"]) == raw["snapshot_hash"]
    citations = [ReviewCitation.model_validate(c) for c in old["citations"]]
    diffs = [ReviewCitation.model_validate(c) for c in raw["index_config"]["citations"]]
    roles = citation_line_roles(diffs, raw["facts"], raw["object_id"])
    mandatory = [c for c in citations if c.kind == "ci" or c.source_role == "application_policy"]
    references = [c for c in citations if c.kind == "rule" and c.source_role != "application_policy"]
    result = {
        "source_hash": code_state()["source_hash"],
        "prompt_version": PROMPT_VERSION,
        "system_hash": canonical_hash(SYSTEM),
        "authorized_to_send": False,
        "comparison_contract": "v11-behavior-comparison.2",
        "evidence_policy": "v11-evaluation-structured.1; signed retain_unvalidated required",
        "recommended_case": "A",
        "call_timeout_seconds": 30,
        "provenance": "product_context_builder_archived_snapshot_and_rule_selection_not_live_retrieval",
        "cases": {},
    }
    for case, query in QUESTIONS.items():
        user, selected = build_context(query, mandatory, references, diffs, sent["SERVER_OBSERVATIONS"], roles)
        payload = {
            "case": case,
            "object_id": raw["object_id"],
            "system": SYSTEM,
            "user": user,
            "facts": raw["facts"],
            "citations": [c.model_dump(mode="json") for c in selected],
        }
        amount = (Decimal(len((SYSTEM + user).encode()) + 64) * 2 + Decimal(8192) * 8) / 1000000
        parsed = json.loads(user)
        result["cases"][case] = {
            "query": query,
            "payload_hash": canonical_hash(payload),
            "input_bytes_with_margin": len((SYSTEM + user).encode()) + 64,
            "input_tokens_with_margin": token_count(SYSTEM + user) + 64,
            "max_cny": str(amount),
            "diff_bytes": sum(len(c.text.encode()) for c in selected if c.kind == "diff"),
            "diff_tokens": sum(token_count(c.text) for c in selected if c.kind == "diff"),
            "model_coverage_complete": parsed["SERVER_OBSERVATIONS"]["model_coverage_complete"],
            "optional_references": sum(c.source_role == "usage_reference" for c in selected),
            "omitted_navigation_entries": parsed["comparison_index"].get("omitted_navigation_entries", 0),
            "sources": [
                {
                    "id": c.citation_id,
                    "path": c.path,
                    "kind": c.kind,
                    "side": c.side,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in selected
            ],
        }
        assert result["cases"][case]["model_coverage_complete"]
        with (REPORTS / f"{prefix}_{case}.json").open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
    with (REPORTS / f"{prefix}_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    manifest = freeze()
    Console().print(
        {
            case: {k: v for k, v in value.items() if k not in {"sources", "query"}}
            for case, value in manifest["cases"].items()
        }
    )
