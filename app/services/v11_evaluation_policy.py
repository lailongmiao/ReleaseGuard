"""The four explicitly approved formal positions are not business recovery."""

from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard.normalization import canonical_hash

RETIRED_BATCH = "v11-formal-contract2-evidence2-01"
RETIRED_BATCHES = ("v11-formal-contract2-01", RETIRED_BATCH)
FORMAL_BATCH = "v11-formal-contract2-evidence3-01"
OVERLAP_CASES = frozenset({"real-03-zh", "real-03-en"})
DATASET_HASH = "db0f3fb90ea95067d0180174f733ced7e535bfbae4b4cbe957f368c40d7d9c7b"
LINK_CONTRACT = "v11-formal-overlap.1"
READONLY_REASON = "evaluation_permanently_readonly"
OLD_COMMANDS = (
    "eval:v11-d1-evidence-01",
    "eval:v11-d1-contract2-behavior-01",
    "smoke-analysis-c77c6d75ff2e4ec8e8351cc1",
)
RETIRED_SENDS = frozenset(
    prefix + canonical_hash([batch, DATASET_HASH, f"real-{number:02}-{language}", repeat])
    for batch in RETIRED_BATCHES
    for number in range(1, 21)
    for language in ("zh", "en")
    for repeat in (1, 2)
    for prefix in ("analysis:", "sync:", "rag:")
)
RETIREMENT_MARKER = "formal-retired:" + RETIRED_BATCH


def require_business_object(obj: dict) -> None:
    """Independent of global capability state and any operator server lifetime."""
    if obj.get("evaluation_readonly"):
        raise DomainError(READONLY_REASON)
