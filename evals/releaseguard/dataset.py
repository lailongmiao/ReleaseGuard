"""Validate content-addressed synthetic data before running any scenario."""

from pathlib import Path

from app.services.releaseguard.normalization import canonical_hash
from evals.releaseguard.schemas import Dataset


def load_dataset(path: Path, *, expected_hash: str | None = None) -> Dataset:
    """Reject empty sets, duplicate ids, stale hashes and same-version drift."""
    dataset = Dataset.model_validate_json(path.read_text(encoding="utf-8"))
    if not dataset.cases or len({case.case_id for case in dataset.cases}) != len(dataset.cases):
        raise ValueError("dataset_case_set_invalid")
    actual = canonical_hash(dataset.model_dump(mode="json", exclude={"dataset_hash"}))
    if actual != dataset.dataset_hash or expected_hash is not None and actual != expected_hash:
        raise ValueError("dataset_hash_conflict")
    return dataset
