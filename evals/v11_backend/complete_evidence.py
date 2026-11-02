"""Evaluation-only bounded text retention with source-verified public identifiers."""

import json
import re
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from app.core.config import settings
from app.schemas.v11_backend import ReviewCitation
from app.services.releaseguard.normalization import canonical_hash
from app.services.v11_policy import model_sources

TEXT_LIMIT = 131072
SECRET_PATTERNS = (
    r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-]*PRIVATE KEY-----|$)",
    r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9_./+=-]+",
    r"(?i)\b(?:api[_-]?key|token|password|secret|authorization)[\"']?\s*[=:]\s*[^\s,;}]+",
    r"\b(?:sk-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]+",
    r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    r"\bAIza[A-Za-z0-9_-]{35}\b",
    r"\bxox[baprs]-[A-Za-z0-9-]+",
    r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]*@[^\s\"'<>]+",
)
OPAQUE = re.compile(
    r"\b(?:local-change|local-user|snapshot|generation|analysis|diff|rule|ci):[A-Za-z0-9:_.-]+|(?<![A-Za-z0-9])[A-Za-z0-9_+/=-]{40,}(?![A-Za-z0-9])"
)


# A URL token includes RFC punctuation. Boundaries are resolved against an exact
# source value, never by mutating the URL with rstrip(). Chinese prose delimiters
# and quotes are delimiters; percent-encoded delimiters remain part of the URL.
URL_TOKEN = re.compile(r"https?://[^\s\\\"<>`，。；！？、（）【】《》“”‘’]+")
URL_SUFFIX = re.compile(r"[.,;:!?()\[\]{}']*")
SENSITIVE_QUERY = re.compile(
    r"(?i)(?:token|secret|password|passwd|credential|authorization|api.?key|signature|sig|session|cookie|auth|key)"
)


def safe_public_url(value: str) -> bool:
    """Public locator syntax is necessary, not sufficient: source identity is separate.

    Decode for detection only. Never normalize the retained spelling or trust a
    source exception over credential detection. Ambiguous nested encoding fails closed.
    """
    decoded = value
    for _ in range(4):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    if unquote(decoded) != decoded or any(ord(c) < 32 for c in decoded):
        return False
    try:
        url = urlsplit(decoded)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            return False
        if any(SENSITIVE_QUERY.search(key) for key, _ in parse_qsl(url.query, keep_blank_values=True)):
            return False
        return not any(re.search(pattern, decoded) for pattern in SECRET_PATTERNS) and not any(
            secret in decoded for secret in configured_secrets()
        )
    except ValueError:
        return False


def public_values(citations: list[ReviewCitation]) -> set[str]:
    """Allow typed metadata, locator/SRI/commit fields and rule Markdown links.

    No repository, package or filename exceptions. Arbitrary source strings are
    not public values. Caller supplies the server-validated, actually sent citations.
    """
    values: set[str] = set()
    for citation in citations:
        if citation.text_hash != canonical_hash(citation.text):
            continue
        values.update(
            v
            for v in (citation.citation_id, citation.sha, citation.text_hash, citation.hunk_id, citation.file_id)
            if v
        )
        values.update(
            v.rsplit(":", 1)[-1] for v in list(values) if re.fullmatch(r"[a-f0-9]{64}", v.rsplit(":", 1)[-1])
        )
        if citation.kind == "rule":
            # Markdown link destination is an explicit locator field. Balanced
            # parentheses inside a destination are supported, not stripped.
            for match in re.finditer(r"\]\((https?://)", citation.text):
                start, cursor, depth = match.start(1), match.end(), 1
                while cursor < len(citation.text) and depth:
                    depth += (citation.text[cursor] == "(") - (citation.text[cursor] == ")")
                    cursor += 1
                value = citation.text[start : cursor - 1]
                if depth == 0 and safe_public_url(value):
                    values.add(value)
        if citation.kind != "diff":
            continue
        values.update(
            re.findall(
                r"\bexpect\([A-Za-z_$][\w$.]*\.(?:commit|sha|headSha|baseSha)\)\.to(?:Be|Equal)\(['\"]([a-f0-9]{40})['\"]\)",
                citation.text,
            )
        )
        for field, literal in re.findall(r'"([A-Za-z_]+)"\s*:\s*("(?:[^"\\\n]|\\.)*")', citation.text):
            try:
                value = json.loads(literal)
            except ValueError:
                continue
            if field in {"resolved", "url", "download_url", "homepage", "repository_url", "documentation_url"}:
                if safe_public_url(value):
                    values.add(value)
            elif field == "integrity" and re.fullmatch(r"sha(?:256|384|512)-[A-Za-z0-9+/]+={0,2}", value):
                values.update((value, value.split("-", 1)[1]))
            elif field in {"commit", "sha", "head_sha", "base_sha"} and re.fullmatch(r"[a-f0-9]{40}", value):
                values.add(value)
    return values


def retain_public_tokens(text: str, allowed: set[str]) -> str:
    """Process URLs atomically, avoiding re-redaction of hashes inside public URLs."""
    urls = sorted((v for v in allowed if v.startswith(("http://", "https://"))), key=len, reverse=True)
    parts: list[str] = []
    end = 0
    for match in URL_TOKEN.finditer(text):
        parts.append(OPAQUE.sub(lambda m: m[0] if m[0] in allowed else "[redacted]", text[end : match.start()]))
        token = match[0]
        valid = safe_public_url(token) and any(
            token.startswith(url) and URL_SUFFIX.fullmatch(token[len(url) :]) for url in urls
        )
        parts.append(token if valid else "[redacted]")
        end = match.end()
    parts.append(OPAQUE.sub(lambda m: m[0] if m[0] in allowed else "[redacted]", text[end:]))
    return "".join(parts)


def configured_secrets() -> list[str]:
    """Configured credentials stay private even when URL-encoded in a known source."""
    result = []
    for name, value in vars(settings).items():
        if any(word in name for word in ("API_KEY", "SECRET", "PASSWORD", "READ_TOKEN")):
            secret = value.get_secret_value() if hasattr(value, "get_secret_value") else value
            if isinstance(secret, str) and secret:
                result.append(secret)
    return result


def retain_text(text: str, allowed: set[str], *, limit: int = TEXT_LIMIT) -> tuple[str, list[str]]:
    """Separate redaction and truncation; neither silently certifies complete review."""
    clean = text
    for secret in configured_secrets():
        clean = clean.replace(secret, "[redacted]")
    for pattern in SECRET_PATTERNS:
        clean = re.sub(pattern, "[redacted]", clean)
    clean = retain_public_tokens(clean, allowed)
    clean = "".join(c for c in clean if c in "\n\t" or c.isprintable())
    reasons = ["redacted"] if clean != text else []
    if len(clean) > limit:
        clean = clean[:limit]
        reasons.append("truncated")
    return clean, reasons


def input_evidence(user: str, citations: list[ReviewCitation], observations: dict) -> tuple[dict, set[str]]:
    """Retain exact compiled input, validating its source projection before exemptions."""
    allowed = public_values(citations)
    # These hashes are typed server comparison identities, never arbitrary input fields.
    comparison = observations.get("comparison", {})
    for key in ("head_sha", "base_tip_sha", "diff_base_sha", "merge_base_sha", "parent_sha"):
        value = comparison.get(key)
        if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{40}", value):
            allowed.add(value)
    clean, reasons = retain_text(user, allowed)
    result: dict[str, Any] = {
        "version": "v11-evaluation-input.2",
        "status": "unvalidated",
        "user": clean,
        "input_hash": canonical_hash(user),
        "input_redacted": "redacted" in reasons,
        "input_truncated": "truncated" in reasons,
        "transformations": reasons,
        "review_complete": not reasons,
        "review_impact": "incomplete" if reasons else "none",
    }
    try:
        decoded = json.loads(user)
        sources = decoded["sources"]
        expected = {c["citation_id"]: c for c in model_sources(citations)}
        valid = (
            len(sources) == len(expected)
            and {s["citation_id"] for s in sources} == set(expected)
            and all(
                s["citation_id"] in expected
                and {k: v for k, v in s.items() if k != "line_roles"} == expected[s["citation_id"]]
                for s in sources
            )
            and not (set(decoded) - {"query", "sources", "comparison_index", "SERVER_OBSERVATIONS"})
        )
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        result.update(
            user=None,
            review_complete=False,
            review_impact="incomplete",
            failure_code="evaluation_input_identity_invalid",
        )
    elif reasons:
        result["failure_code"] = "evaluation_input_incomplete"
    return result, allowed
