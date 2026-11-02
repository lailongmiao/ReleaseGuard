"""Exact diff identity, boundary disclosure and unsupported-content regressions."""

import pytest

from app.services.releaseguard.state_machine import DomainError
from app.services.v11_diff import bounded_hunks, parse_patch, read_file, rebuild_patch, safe_path


def test_coordinates_and_newline_facts() -> None:
    """Deletion belongs to old SHA and additions to new SHA, preserving CRLF."""
    patch = "@@ -8,2 +8,2 @@\n context\r\n-old\n\\ No newline at end of file\n+new\r"
    lines = parse_patch(patch)[0].lines
    assert [(line.old_line, line.new_line) for line in lines] == [(8, 8), (9, None), (None, 9)]
    assert lines[0].text == "context\r" and lines[1].no_final_newline
    saved, text, complete = bounded_hunks(parse_patch(patch), 1000)
    assert complete and parse_patch(text) == saved


@pytest.mark.parametrize("patch", ["@@ -1,2 +1 @@\n-x\n+y", "@@ -0 +1 @@\n-x\n+y", "@@ -1 +1 @@\n-x\n+y\n+z"])
def test_invalid_lengths_cannot_make_citations(patch: str) -> None:
    """Truncation and impossible positions are not accepted as full patches."""
    with pytest.raises(DomainError):
        parse_patch(patch)


@pytest.mark.parametrize("path", ["../x", "/x", "x\\y", "x//y", "https://x", "x\x00y"])
def test_unsafe_paths(path: str) -> None:
    """Never turn untrusted file names into filesystem or arbitrary URL inputs."""
    with pytest.raises(DomainError):
        safe_path(path)


def test_unicode_identity_and_reconstruction() -> None:
    """Non-LF Unicode separators are payload characters, not Git line breaks."""
    assert safe_path("源码/É.py") == "源码/É.py"
    patch = rebuild_patch("a\u2028b\r\nlast", "a\u2028c\r\nnext")
    lines = parse_patch(patch)[0].lines
    assert [line.text for line in lines] == ["a\u2028b\r", "last", "a\u2028c\r", "next"]
    assert lines[1].no_final_newline and lines[3].no_final_newline


def test_patch_capacity_and_missing_classification() -> None:
    """N-1/N/N+1 are explicit; absent patch never means binary by default."""
    raw = {"filename": "x", "status": "modified", "additions": 1, "deletions": 1, "patch": "@@ -1 +1 @@\n-x\n+y"}
    size = len(raw["patch"].encode()) + 1
    assert not read_file(raw, size - 1).complete
    assert read_file(raw, size).complete and read_file(raw, size + 1).complete
    del raw["patch"]
    assert read_file(raw, 999).reasons == ["patch_missing_unclassified"]
    raw.update(status="renamed", additions=0, deletions=0, previous_filename="old")
    renamed = read_file(raw, 0)
    assert renamed.complete and not renamed.hunks and renamed.old_path == "old"
