"""Dataset mutations must be rejected before any evaluation call."""

import json
from pathlib import Path

import pytest

from app.services.releaseguard.normalization import canonical_hash
from evals.releaseguard.dataset import load_dataset

DATASET = Path("evals/datasets/releaseguard_v1.json")


@pytest.mark.parametrize(
    "mutation", ["empty", "duplicate", "hash", "scenario", "oracle", "metrics", "synthetic", "clock", "input_risk"]
)
def test_invalid_dataset(tmp_path: Path, mutation: str) -> None:
    """Strict schema and content seals are both required."""
    value = json.loads(DATASET.read_text())
    case = value["cases"][0]
    if mutation == "empty":
        value["cases"] = []
    elif mutation == "duplicate":
        value["cases"].append(case)
    elif mutation == "hash":
        case["oracle"]["calls"] += 1
    elif mutation == "scenario":
        case["scenario"] = "execute_shell"
    elif mutation == "oracle":
        del case["oracle"]
    elif mutation == "metrics":
        case["required_metrics"] = ["llm_safety"]
    elif mutation == "synthetic":
        case["synthetic"] = False
    elif mutation == "clock":
        case["clock_start"] = "2026-01-01T00:00:00"
    else:
        del case["input_risk"]
    target = tmp_path / "dataset.json"
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_dataset(target)


def test_version_pin_and_canonical_hash(tmp_path: Path) -> None:
    """Reformatting does not change identity; same-version content drift does."""
    dataset = load_dataset(DATASET)
    assert len(dataset.cases) >= 42
    value = json.loads(DATASET.read_text())
    target = tmp_path / "dataset.json"
    target.write_text(json.dumps(value, sort_keys=True))
    assert load_dataset(target) == dataset
    value["cases"][0]["regression_tags"].append("new")
    value["dataset_hash"] = canonical_hash({k: v for k, v in value.items() if k != "dataset_hash"})
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_dataset(target, expected_hash=dataset.dataset_hash)
