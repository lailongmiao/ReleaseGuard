"""Offline immutable Evidence repository with deterministic replay semantics."""

import asyncio
from typing import override

from app.repositories.evidence import EvidenceRepository
from app.schemas.evidence import EvidenceCitation, EvidenceItem, EvidenceSnapshot
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError


class InMemoryEvidenceRepository(EvidenceRepository):
    """Single-process fixture; it is not a durable production store."""

    def __init__(self) -> None:
        """Initialize isolated snapshot and citation maps."""
        self._lock = asyncio.Lock()
        self._snapshots: dict[str, EvidenceSnapshot] = {}
        self._by_evidence: dict[str, set[str]] = {}
        self._citations: dict[str, EvidenceCitation] = {}

    @override
    async def append_snapshot(self, snapshot: EvidenceSnapshot) -> EvidenceSnapshot:
        """Append a verified snapshot or return its byte-equivalent replay."""
        snapshot = EvidenceSnapshot.model_validate_json(snapshot.model_dump_json())
        if snapshot.item.collection_status.value != "verified":
            raise DomainError("untrusted_evidence")
        async with self._lock:
            existing = self._snapshots.get(snapshot.snapshot_id)
            if existing is not None:
                if existing != snapshot:
                    raise DomainError("evidence_conflict")
                return existing.model_copy(deep=True)
            for existing_id in self._by_evidence.get(snapshot.item.evidence_id, set()):
                existing_snapshot = self._snapshots[existing_id]
                if existing_snapshot.item.source_version == snapshot.item.source_version:
                    raise DomainError("evidence_conflict")
            self._snapshots[snapshot.snapshot_id] = snapshot
            self._by_evidence.setdefault(snapshot.item.evidence_id, set()).add(snapshot.snapshot_id)
            return snapshot.model_copy(deep=True)

    @override
    async def get_snapshot(self, snapshot_id: str) -> EvidenceSnapshot:
        """Return an isolated snapshot or a stable not-found error."""
        async with self._lock:
            snapshot = self._snapshots.get(snapshot_id)
            if snapshot is None:
                raise DomainError("evidence_not_found")
            return snapshot.model_copy(deep=True)

    @override
    async def find_snapshots(self, evidence_id: str) -> list[EvidenceSnapshot]:
        """Return versions in deterministic snapshot-id order."""
        async with self._lock:
            result = [
                self._snapshots[key].model_copy(deep=True) for key in sorted(self._by_evidence.get(evidence_id, set()))
            ]
            if not result:
                raise DomainError("evidence_not_found")
            return result

    @override
    async def append_citation(self, citation: EvidenceCitation) -> EvidenceCitation:
        """Validate the referenced snapshot and append an immutable citation."""
        citation = EvidenceCitation.model_validate_json(citation.model_dump_json())
        async with self._lock:
            snapshot = self._snapshots.get(citation.snapshot_id)
            if snapshot is None:
                raise DomainError("evidence_not_found")
            item = snapshot.item
            if (
                item.evidence_id != citation.evidence_id
                or item.source_version != citation.source_version
                or item.content_hash != citation.content_hash
                or item.captured_at != citation.captured_at
            ):
                raise DomainError("citation_snapshot_mismatch")
            existing = self._citations.get(citation.citation_id)
            if existing is not None:
                if existing != citation:
                    raise DomainError("citation_conflict")
                return existing.model_copy(deep=True)
            self._citations[citation.citation_id] = citation
            return citation.model_copy(deep=True)

    @override
    async def get_citation(self, citation_id: str) -> EvidenceCitation:
        """Return an isolated citation or a stable not-found error."""
        async with self._lock:
            citation = self._citations.get(citation_id)
            if citation is None:
                raise DomainError("citation_not_found")
            return citation.model_copy(deep=True)


def snapshot_id_for_item(item: EvidenceItem) -> str:
    """Return the deterministic snapshot id for a validated EvidenceItem."""
    return canonical_hash(
        {
            "evidence_id": item.evidence_id,
            "source_uri": item.source_uri,
            "source_version": item.source_version,
            "content_hash": item.content_hash,
        }
    )
