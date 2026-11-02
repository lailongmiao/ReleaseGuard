"""Async Evidence snapshot and Citation repository boundary."""

from typing import Protocol

from app.schemas.evidence import EvidenceCitation, EvidenceSnapshot


class EvidenceRepository(Protocol):
    """Immutable snapshot boundary kept separate from read-only search."""

    async def append_snapshot(self, snapshot: EvidenceSnapshot) -> EvidenceSnapshot:
        """Append or exactly replay one verified snapshot."""
        ...

    async def get_snapshot(self, snapshot_id: str) -> EvidenceSnapshot:
        """Read one snapshot by its deterministic identifier."""
        ...

    async def find_snapshots(self, evidence_id: str) -> list[EvidenceSnapshot]:
        """Read all versions of one evidence identity in stable order."""
        ...

    async def append_citation(self, citation: EvidenceCitation) -> EvidenceCitation:
        """Append or exactly replay a citation after snapshot validation."""
        ...

    async def get_citation(self, citation_id: str) -> EvidenceCitation:
        """Read one immutable citation."""
        ...
