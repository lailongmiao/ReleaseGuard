"""Canonical, provider-independent hashes for domain inputs."""

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


def canonical_json(value: Any) -> str:
    """Encode a normalized input with stable keys, UTC times and Unicode."""
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Any) -> str:
    """Hash canonical UTF-8 bytes without retaining or logging input text."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalize(value: Any) -> Any:
    """Convert typed inputs into canonical JSON values, preserving list order."""
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="python"))
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ValueError("timestamp_requires_timezone")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_normalize(item) for item in value), key=canonical_json)
    return value
