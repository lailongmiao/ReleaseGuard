"""Strict selectors and explicit v3 additions preserve historical serialization."""

import pytest
from pydantic import ValidationError

from app.schemas.v11_backend import BranchSelector, CommitSelector, Comparison, PRSelector, ReviewBinding
from app.services.local_mvp_storage import SavedFacts
from app.services.releaseguard.normalization import canonical_hash


@pytest.mark.parametrize("number", [True, "1", 0, -1])
def test_pr_number_is_strict(number: object) -> None:
    """Coercion cannot silently choose another upstream target."""
    with pytest.raises(ValidationError):
        PRSelector.model_validate({"pr_number": number})


@pytest.mark.parametrize("branch", ["HEAD~1", "refs/heads/main", "x..y", "x.lock", "a//b", "a@{1}", "a\nb"])
def test_only_exact_branch_names(branch: str) -> None:
    """Revision expressions and invalid ref names are rejected."""
    with pytest.raises(ValidationError):
        BranchSelector(base_branch=branch, head_branch="main")


def test_selector_and_root_identity() -> None:
    """Complete SHA and matching logical empty-tree representation are mandatory."""
    with pytest.raises(ValidationError):
        CommitSelector(commit_sha="abc123")
    with pytest.raises(ValidationError):
        ReviewBinding.model_validate({"repository_id": "arbitrary", "selector": {"kind": "pr", "pr_number": 1}})
    with pytest.raises(ValidationError):
        Comparison(kind="commit", head_sha="a" * 40, comparison="parent")
    root = Comparison(kind="commit", head_sha="a" * 40, comparison="empty_tree", base_kind="empty_tree")
    assert root.parent_sha is None


def test_old_facts_roundtrip_has_no_v3_authority() -> None:
    """Golden legacy payloads keep their original keys and canonical hashes."""
    v1 = {
        "version": "local-mvp.v1",
        "domain": "",
        "approvals": {"version": "approval-store.v1", "entries": [], "audit_events": [], "calls": []},
        "evidence": [],
        "citations": [],
        "ledger": "",
        "target_values": {"local-synthetic-register": 0},
        "target_versions": {"local-synthetic-register": 0},
        "target_outcomes": {},
        "target_calls": 0,
        "commands": {},
        "creation": {},
    }
    for facts in (v1, {**v1, "version": "local-mvp.v2", "product": {"binding": {"pr_number": 1}}}):
        restored = SavedFacts.model_validate(facts).model_dump(mode="json")
        assert restored == facts and canonical_hash(restored) == canonical_hash(facts)
    assert SavedFacts(version="local-mvp.v3").version == "local-mvp.v3"
