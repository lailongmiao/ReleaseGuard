"""Actual failed-batch answer plus independent security and length counterexamples."""

import json

import pytest

from app.services.releaseguard.normalization import canonical_hash
from app.services.v11_policy import model_sources
from app.services.v11_provider import build_context
from evals.v11_backend.behavior_cases import cases, evidence
from evals.v11_backend.complete_evidence import TEXT_LIMIT, input_evidence, public_values, retain_text
from evals.v11_backend.structured_evidence import structured_evidence
from tests.test_v11_behavior_comparison import wire


def recorded():
    """Build deterministic public evidence from the frozen synthetic corpus."""
    case = cases()[1]
    _, citations, _ = evidence([case])
    return wire(case, citations).model_dump(mode="json"), citations


def test_saved_answer_survives_public_values_without_loss():
    """A complete synthetic answer remains exact under evidence v2."""
    answer, citations = recorded()
    saved = structured_evidence(answer, {c.citation_id for c in citations}, public_tokens=public_values(citations))
    assert saved["review_complete"], saved["transformations"]
    assert saved["answer"] == answer
    assert saved["original_parsed_hash"] == canonical_hash(answer)


@pytest.mark.parametrize("length", [3999, 4000, 4001, 32768, TEXT_LIMIT])
def test_long_synthetic_input_keeps_exact_unicode_and_actual_sources(length):
    """Independent generated input; this is explicitly not recovered historical input."""
    _, citations = recorded()
    sources = model_sources(citations) if length > 4001 else []
    base = json.dumps({"query": "", "sources": sources}, ensure_ascii=False)
    user = json.dumps({"query": "中" * (length - len(base)), "sources": sources}, ensure_ascii=False)
    citations = citations if sources else []
    retained, _ = input_evidence(user, citations, {})
    if len(user) <= TEXT_LIMIT:
        assert retained["review_complete"], retained["transformations"]
        assert retained["user"] == user
    else:
        assert retained["input_truncated"] and not retained["review_complete"]


@pytest.mark.parametrize(
    "secret",
    [
        "token=private-value",
        "https://user:password@example.org/a",
        "https://registry.npmjs.org/a/-/a.tgz?token=private",
        "sk-synthetic-private-value",
        "Bearer synthetic-secret",
        "postgresql://example:private-value@localhost/database",
        "AKIA" + "Z" * 16,
        "AIza" + "z" * 35,
        "xoxb-synthetic-private-value",
        "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----",
    ],
)
def test_source_presence_does_not_exempt_credentials(secret):
    """A source cannot launder credentials into the public-value set."""
    _, citations = recorded()
    altered = citations[0].model_copy(update={"text": '"resolved": "' + secret + '"'})
    altered.text_hash = canonical_hash(altered.text)
    clean, reasons = retain_text(secret, public_values([altered]))
    assert reasons and secret not in clean


def test_untrusted_hash_and_oversized_answer_remain_incomplete():
    """Unknown opaque values and lost fields must prevent complete review."""
    clean, reasons = retain_text("A" * 64, set())
    assert reasons == ["redacted"] and "A" * 64 not in clean
    saved = structured_evidence({"summary": "中" * 4001}, set(), public_tokens=set())
    assert not saved["review_complete"] and saved["transformations"][0]["reason"] == "truncated"
    saved = structured_evidence({"summary": "ok", "reasoning_content": "secret"}, set(), public_tokens=set())
    assert not saved["review_complete"] and "secret" not in json.dumps(saved)


def test_input_unknown_fields_and_duplicate_sources_fail_closed():
    """Neither an unauthorized field nor duplicated source identity can be retained as complete."""
    _, citations = recorded()
    for raw in (
        {"sources": model_sources(citations), "headers": {"Cookie": "do-not-save"}},
        {"sources": [model_sources(citations)[0]] * len(citations)},
    ):
        saved, _ = input_evidence(json.dumps(raw), citations, {})
        assert not saved["review_complete"] and saved["user"] is None
        assert "do-not-save" not in json.dumps(saved)


def test_archived_inputs_and_commit_assertion_semantics():
    """Synthetic archived inputs remain reviewable; secret assertions gain no exemption."""
    for case in cases()[:2]:
        facts, citations, roles = evidence([case])
        user, selected = build_context(
            "Compare",
            [],
            [],
            citations,
            {"coverage": facts["coverage"], "index_complete": True},
            roles,
        )
        saved, _ = input_evidence(user, selected, json.loads(user)["SERVER_OBSERVATIONS"])
        assert saved["review_complete"] and saved["user"] == user
    _, citations = recorded()
    text = "expect(settings.secret).toBe('" + "f" * 40 + "')"
    citation = next(c for c in citations if c.kind == "diff").model_copy(
        update={"text": text, "text_hash": canonical_hash(text)}
    )
    clean, reasons = retain_text(text, public_values([citation]))
    assert reasons and "f" * 40 not in clean


@pytest.mark.parametrize(
    "url",
    [
        "https://public.example/pkg.tgz",
        "https://public.example/a(b),c;d!e.tgz?version=1&platform=linux#part",
        "https://public.example/a%28b%29%2Ec?name=%E4%B8%AD",
        "https://public.example/a.",
        "https://public.example/it's(a),b",
        "https://public.example/a?value=x,y;z!",
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        "English {}.",
        "中文{}。",
        "({})",
        "（{}）",
        '"{}"',
        "“{}”",
        "[source]({})",
        "`{}`",
        "{}，结束",
        "{}; next",
        "{}!",
        "{}?",
    ],
)
def test_public_locator_round_trip_across_languages_and_boundaries(url, wrapper):
    """Preserve exact public evidence across independent URL and prose combinations."""
    _, citations = recorded()
    text = json.dumps({"download_url": url}, ensure_ascii=False)
    citation = next(c for c in citations if c.kind == "diff").model_copy(
        update={"path": "config/resources.json", "text": text, "text_hash": canonical_hash(text)}
    )
    allowed = public_values([citation])
    original = wrapper.format(url)
    assert retain_text(original, allowed) == (original, [])
    answer = {"summary": original, "comparisons": [{"before": original, "after": original}]}
    result = structured_evidence(answer, set(), public_tokens=allowed)
    restored = json.loads(json.dumps(result, ensure_ascii=False))
    assert restored["review_complete"] and restored["answer"] == answer
    assert restored["original_parsed_hash"] == canonical_hash(answer)


@pytest.mark.parametrize(
    "suffix",
    [
        "/unknown",
        ".evil/path",
        "?token=private",
        "?access_token=private",
        "?%61pi_key=private",
        "?session_id=private",
        "/sk-private-example",
    ],
)
def test_public_prefix_does_not_launder_secrets_or_unknown_locator(suffix):
    """A matching public prefix never exempts an unknown locator or secret query."""
    url = "https://public.example/resource"
    original = url + suffix
    clean, reasons = retain_text(original, {url, original} if "?" in suffix else {url})
    assert reasons and original not in clean


def test_typed_public_identifiers_are_not_filename_whitelists():
    """Typed SRI and commit fields work outside a particular lockfile or repository."""
    _, citations = recorded()
    text = json.dumps({"integrity": "sha512-" + "A" * 86 + "==", "commit": "f" * 40})
    citation = next(c for c in citations if c.kind == "diff").model_copy(
        update={"path": "metadata.json", "text": text, "text_hash": canonical_hash(text)}
    )
    assert retain_text(text, public_values([citation])) == (text, [])
    invalid = citation.model_copy(update={"text_hash": "0" * 64})
    assert retain_text(text, public_values([invalid]))[1]


def test_nested_unauthorized_fields_and_control_characters_disclose_loss():
    """Unsupported nested fields and removed controls cannot claim complete review."""
    answer = {"observations": [{"text": "safe", "headers": {"authorization": "private"}}]}
    saved = structured_evidence(answer, set(), public_tokens=set())
    assert not saved["review_complete"] and "private" not in json.dumps(saved)
    assert retain_text("中\x00文", set()) == ("中文", ["redacted"])
