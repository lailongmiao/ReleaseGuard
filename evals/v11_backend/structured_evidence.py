"""Private evaluation-only projection of parsed answers, never an adopted report."""

import json
from hashlib import sha256
from typing import Any

from app.services.releaseguard.normalization import canonical_hash
from app.services.v11_rejection import redact_text
from evals.v11_backend.complete_evidence import retain_text

STRUCTURED_LIMIT = 524288
CLAIM_FIELDS = {"text", "kind", "citation_ids", "severity"}
COMPARISON_FIELDS = {"section", "index", "classification", "before", "after", "before_ids", "after_ids"}
ROOT_FIELDS = {
    "summary",
    "observations",
    "risks",
    "validation_suggestions",
    "unknowns",
    "candidate_steps",
    "model_risk",
    "comparisons",
    "comparison_contract",
}


def structured_evidence(parsed: Any, sent_ids: set[str], *, public_tokens: set[str] | None = None) -> dict:
    """Retain every supported answer field; explicitly disclose redaction and capacity loss.

    Unknown fields (including reasoning/headers) are never retained. Limits exceed every
    valid wire field/list; malformed but parseable answers remain inspectable within them.
    """
    changes: list[dict] = []

    def project(value: Any, path: str, fields: set[str] | None = None, references: bool = False) -> Any:
        if isinstance(value, str):
            if references:
                clean = value if value in sent_ids else "unavailable:" + sha256(value.encode()).hexdigest()
            else:
                if public_tokens is None:
                    clean = redact_text(value, limit=4000)
                else:
                    clean, reasons = retain_text(value, public_tokens, limit=4000)
                    changes.extend({"path": path, "reason": reason} for reason in reasons)
            if clean != value and (references or public_tokens is None):
                changes.append({"path": path, "reason": "redacted_or_truncated"})
            return clean
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, list):
            if len(value) > 64:
                changes.append({"path": path, "reason": "list_truncated", "original_count": len(value)})
            return [project(v, f"{path}/{i}", fields, references) for i, v in enumerate(value[:64])]
        if isinstance(value, dict) and fields is not None:
            unknown = set(value) - fields
            if unknown:
                changes.append({"path": path, "reason": "unsupported_fields_omitted", "count": len(unknown)})
            return {
                key: project(v, path + "/" + key, references=key in {"citation_ids", "before_ids", "after_ids"})
                for key, v in value.items()
                if key in fields
            }
        changes.append({"path": path, "reason": "unsupported_structure_omitted"})
        return None

    answer: dict = {}
    if isinstance(parsed, dict):
        if set(parsed) - ROOT_FIELDS:
            changes.append(
                {"path": "", "reason": "unsupported_fields_omitted", "count": len(set(parsed) - ROOT_FIELDS)}
            )
        for key, value in parsed.items():
            if key not in ROOT_FIELDS:
                continue
            fields = (
                CLAIM_FIELDS
                if key in {"observations", "risks"}
                else (
                    COMPARISON_FIELDS
                    if key == "comparisons"
                    else {"text", "performed"}
                    if key == "validation_suggestions"
                    else None
                )
            )
            answer[key] = project(value, "/" + key, fields)
    else:
        changes.append({"path": "", "reason": "non_object_answer"})
    result = {
        "version": "v11-evaluation-structured.1" if public_tokens is None else "v11-evaluation-structured.2",
        "status": "unvalidated",
        "not_business_analysis": True,
        "original_parsed_hash": canonical_hash(parsed),
        "answer": answer,
        "transformations": changes[:256],
        "transformation_count": len(changes),
        "review_complete": not changes,
        "not_semantic_certification": True,
    }
    if len(json.dumps(result, ensure_ascii=False).encode()) > STRUCTURED_LIMIT - 4096:
        result.update(answer=None, review_complete=False, body_omitted_capacity=True)
    return result
