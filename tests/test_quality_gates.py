"""Document and scan gates reject missing/unsafe evidence rather than skip it."""

from pathlib import Path

from scripts.check_docs import check_documents
from scripts.phase8_sensitive_probe import identities, reviewed_identities


def test_document_gate_rejects_path_anchor_fence_and_format(tmp_path: Path) -> None:
    """Fixture errors exercise real gate decisions without changing repository docs."""
    path = tmp_path / "test.md"
    path.write_text("# Heading\n\n[missing](missing.md)\n[bad](#missing)\ntrailing  \n```text\n", encoding="utf-8")
    result = check_documents(tmp_path, [path])
    assert {error[2] for error in result.errors} == {
        "missing_target",
        "missing_anchor",
        "trailing_whitespace",
        "unclosed_fence",
    }


def test_new_scan_candidates_are_not_implicitly_accepted(tmp_path: Path) -> None:
    """A new fingerprint/path remains unexplained even if metadata looks familiar."""
    head = {"results": {"fixture.py": [{"type": "Secret Keyword", "hashed_secret": "a" * 40}]}}
    new = {"results": {"other.py": [{"type": "Secret Keyword", "hashed_secret": "a" * 40}]}}
    assert len(identities(new) - reviewed_identities(tmp_path, head)) == 1
