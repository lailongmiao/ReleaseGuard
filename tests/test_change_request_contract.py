"""Offline request normalization tests using synthetic business inputs."""

from datetime import UTC, datetime
from hashlib import sha256

import pytest
from pydantic import ValidationError

from app.schemas.releaseguard import (
    ChangeIdentity,
    ChangeRequest,
    ChangeRequestInput,
    ChangeWindow,
    CreateChangeRequest,
    RunIdentity,
    build_change_request,
    request_input_hash,
)
from app.services.releaseguard.normalization import canonical_json


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def request_body(**overrides: object) -> ChangeRequestInput:
    """Build a synthetic request input with no external dependencies."""
    values: dict[str, object] = {
        "requester_id": "synthetic-requester",
        "service": "SyntheticService",
        "environment": "staging",
        "title": "合成变更",
        "description": "Synthetic contract fixture",
        "change_window": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T01:00:00Z", "timezone": "UTC"},
        "rollback_plan": "Restore synthetic version",
        "verification_plan": "Check synthetic evidence",
    }
    values.update(overrides)
    return ChangeRequestInput.model_validate(values)


def request_fixture(**overrides: object) -> ChangeRequest:
    """Construct a valid request from a typed command."""
    return build_change_request(
        CreateChangeRequest(
            identity=ChangeIdentity(change_id="synthetic-change", revision=1),
            body=request_body(**overrides),
            idempotency_key="synthetic-key",
            occurred_at=NOW,
        )
    )


def test_fixed_canonical_vector_and_roundtrip() -> None:
    """Compare canonical bytes with an independently specified input vector."""
    expected = (
        '{"change_diff_ref":null,"change_window":{"end":"2026-01-01T01:00:00.000000Z",'
        '"start":"2026-01-01T00:00:00.000000Z"},"description":"Synthetic contract fixture",'
        '"environment":"staging","requester_id":"synthetic-requester","risk_acceptor_id":null,'
        '"rollback_plan":"Restore synthetic version","service":"SyntheticService","target_version":null,'
        '"title":"合成变更","verification_plan":"Check synthetic evidence"}'
    )
    request = request_fixture()
    assert request.request_hash == sha256(expected.encode("utf-8")).hexdigest()
    assert ChangeRequest.model_validate_json(request.model_dump_json()) == request
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_normalization_preserves_semantics_and_excludes_generated_fields() -> None:
    """Equivalent formatting and generated metadata do not affect the hash."""
    original = request_fixture()
    equivalent = request_fixture(environment=" STAGING ", title=" 合成变更 ", target_version=" ")
    assert original.request_hash == equivalent.request_hash
    for field in ("service", "title", "description", "requester_id", "rollback_plan", "verification_plan"):
        assert request_input_hash(request_body(**{field: "DifferentCase"})) != original.request_hash
    changed = original.model_dump()
    changed.update(
        change_id="other-generated", revision=2, idempotency_key="other-key", updated_at="2026-01-02T00:00:00Z"
    )
    assert ChangeRequest.model_validate(changed).request_hash == original.request_hash
    assert (
        RunIdentity(change_id=" synthetic-change ", run_id="synthetic-run", revision=1).change_id == original.change_id
    )


def test_equivalent_timezone_hash() -> None:
    """An equivalent window in a different IANA timezone has identical hash."""
    assert (
        request_fixture().request_hash
        == request_fixture(
            change_window={
                "start": "2026-01-01T08:00:00+08:00",
                "end": "2026-01-01T09:00:00+08:00",
                "timezone": "Asia/Singapore",
            }
        ).request_hash
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"service": " "},
        {"environment": "invalid"},
        {"rollback_plan": ""},
        {"verification_plan": ""},
        {"requester_id": "bad\x00id"},
        {"environment": "production"},
        {"change_window": {"start": "2026-01-01T01:00:00Z", "end": "2026-01-01T00:00:00Z", "timezone": "UTC"}},
        {"change_window": {"start": "2026-01-01T00:00:00", "end": "2026-01-01T01:00:00", "timezone": "UTC"}},
        {
            "change_window": {
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-01T01:00:00Z",
                "timezone": "Asia/Singapore",
            }
        },
    ],
)
def test_invalid_request_inputs(overrides: dict[str, object]) -> None:
    """Reject missing operational plans, invalid windows and ambiguous identities."""
    with pytest.raises(ValidationError):
        request_fixture(**overrides)


def test_forged_hash_and_unknown_fields_are_rejected() -> None:
    """Restoration cannot accept a forged hash or a hidden input field."""
    values = request_fixture().model_dump()
    values["request_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="hash_mismatch"):
        ChangeRequest.model_validate(values)
    with pytest.raises(ValidationError):
        request_body(unexpected="synthetic")


def test_non_utc_window_can_be_reused_and_roundtripped() -> None:
    """Nested validation is idempotent and keeps strict input offset checks."""
    window = ChangeWindow.model_validate(
        {"start": "2026-01-01T08:00:00+08:00", "end": "2026-01-01T09:00:00+08:00", "timezone": "Asia/Singapore"}
    )
    body = request_body(change_window=window)
    assert request_input_hash(body) == request_fixture().request_hash
    request = request_fixture(change_window=window)
    assert ChangeRequest.model_validate(request.model_dump()) == request
    assert ChangeRequest.model_validate_json(request.model_dump_json()) == request
    assert request.change_window.start.hour == 8


def test_multiline_plans_preserve_content_and_normalize_line_endings() -> None:
    """Business plans support lines and tabs while identities remain strict."""
    request = request_fixture(rollback_plan="  Step one\r\n\tStep two  ")
    assert request.rollback_plan == "Step one\n\tStep two"
    assert request.request_hash == request_fixture(rollback_plan="Step one\n\tStep two").request_hash
    with pytest.raises(ValidationError):
        request_fixture(requester_id="synthetic\nrequester")
    with pytest.raises(ValidationError):
        request_fixture(rollback_plan="synthetic\x00plan")


def test_dst_window_uses_instant_order_instead_of_local_clock_order() -> None:
    """The repeated hour at DST fall-back still has an unambiguous UTC order."""
    window = ChangeWindow.model_validate(
        {"start": "2026-11-01T01:45:00-04:00", "end": "2026-11-01T01:15:00-05:00", "timezone": "America/New_York"}
    )
    assert ChangeWindow.model_validate_json(window.model_dump_json()) == window


@pytest.mark.parametrize("revision", [0, -1, True, 1.0, "1"])
def test_revision_requires_a_positive_integer(revision: object) -> None:
    """Coercion cannot silently convert booleans or floats into business versions."""
    with pytest.raises(ValidationError):
        ChangeIdentity.model_validate({"change_id": "synthetic-change", "revision": revision})
