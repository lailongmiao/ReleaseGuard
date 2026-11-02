"""Bounded, redacted claim diagnostics; never a model response or authorization."""

import json
import re
from hashlib import sha256
from typing import Any

from app.core.config import settings
from app.schemas.v11_backend import ReviewCitation
from app.services.releaseguard.state_machine import DomainError

DIAGNOSTIC_KEY = "_rejection_diagnostic"
DIAGNOSTIC_LIMIT = 16384


class ReviewProviderFailure(DomainError):
    """Keep transport diagnostics on the same restricted failure path as rejected claims."""

    def __init__(self, code: str, diagnostic: dict | None) -> None:
        """Preserve the original domain code and never add details to exception text."""
        super().__init__(code)
        self.diagnostic = diagnostic


class ReviewValidationError(DomainError):
    """Exception text remains a safe code; private detail never enters str/repr."""

    def __init__(self, code: str, validator: str, section: str | None = None, index: int | None = None) -> None:
        """Attach location without adding claim text to exception arguments."""
        super().__init__(code)
        self.validator = validator
        self.section = section
        self.index = index
        self.diagnostic: dict | None = None


def redact_text(text: str, *, limit: int = 1000) -> str:
    """Remove configured credentials and credential/opaque identity shaped strings before truncation."""
    for name, value in vars(settings).items():
        if any(word in name for word in ("API_KEY", "SECRET", "PASSWORD", "READ_TOKEN")):
            secret = value.get_secret_value() if hasattr(value, "get_secret_value") else value
            if isinstance(secret, str) and secret:
                text = text.replace(secret, "[redacted]")
    patterns = (
        r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-]*PRIVATE KEY-----|$)",
        r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9_./+=-]+",
        r"(?i)\b(?:api[_-]?key|token|password|secret|authorization)[\"']?\s*[=:]\s*[^\s,;}]+",
        r"\b(?:sk-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]+",
        r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        r"https?://[^\s`<>]+",
        r"\b(?:local-change|local-user|snapshot|generation|analysis|diff|rule|ci):[A-Za-z0-9:_.-]+",
        r"(?<![A-Za-z0-9])[A-Za-z0-9_+/=-]{40,}(?![A-Za-z0-9])",
    )
    for pattern in patterns:
        text = re.sub(pattern, "[redacted]", text)
    return "".join(c for c in text if c in "\n\t" or c.isprintable())[: min(limit, 4000)]


def rejection_diagnostic(error: ReviewValidationError, wire: Any, citations: list[ReviewCitation], sent: dict) -> dict:
    """Retain only the rejected claim and related comparisons, never other claims or reasoning."""
    available = {c.citation_id: c for c in citations}
    sources = {c["citation_id"]: c for c in sent.get("sources", [])}

    def reference(key: str) -> dict:
        citation = available.get(key)
        source = sources.get(key)
        if citation is None or source is None or source.get("text") != citation.text:
            return {
                "citation_id": "unavailable:" + sha256(key.encode()).hexdigest()[:24],
                "kind": None,
                "actually_sent": False,
            }
        return {
            "citation_id": citation.citation_id,
            "kind": citation.kind,
            "source_role": citation.source_role,
            "actually_sent": True,
            "side": citation.side,
            "start_line": citation.start_line,
            "end_line": citation.end_line,
        }

    claim = None
    if error.section in {"observations", "risks"} and error.index is not None:
        items = getattr(wire, error.section)
        if 0 <= error.index < len(items):
            item = items[error.index]
            claim = {
                "kind": item.kind,
                "text": redact_text(item.text),
                "references": [reference(key) for key in item.citation_ids],
            }
    comparisons = [
        {
            "classification": item.classification,
            "before": redact_text(item.before),
            "after": redact_text(item.after),
            "before_references": [
                reference(key)
                for key in getattr(
                    item,
                    "before_ids",
                    [
                        key
                        for key in getattr(wire, item.section)[item.index].citation_ids
                        if key in available and available[key].kind == "diff" and available[key].side == "LEFT"
                    ],
                )
            ],
            "after_references": [
                reference(key)
                for key in getattr(
                    item,
                    "after_ids",
                    [
                        key
                        for key in getattr(wire, item.section)[item.index].citation_ids
                        if key in available and available[key].kind == "diff" and available[key].side == "RIGHT"
                    ],
                )
            ],
        }
        for item in wire.comparisons
        if item.section == error.section and item.index == error.index
    ][:2]
    result = {
        "version": "v11-rejection.1",
        "validator": error.validator,
        "code": error.code,
        "section": error.section,
        "index": error.index,
        "claim": claim,
        "comparisons": comparisons,
        "redacted": True,
        "not_semantic_certification": True,
    }
    if len(json.dumps(result, ensure_ascii=False).encode()) > DIAGNOSTIC_LIMIT:
        result["claim"] = None
        result["comparisons"] = []
        result["body_omitted_capacity"] = True
    return result
