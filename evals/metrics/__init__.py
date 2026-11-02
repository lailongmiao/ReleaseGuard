"""Metrics for evals."""

from hashlib import sha256
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
metrics: list[dict[str, str]] = [
    {
        "name": file.stem,
        "prompt": file.read_text(encoding="utf-8"),
        "prompt_hash": sha256(file.read_bytes()).hexdigest(),
        "direction": "lower_better" if file.stem in {"hallucination", "toxicity"} else "higher_better",
    }
    for file in sorted(PROMPTS_DIR.glob("*.md"))
]
