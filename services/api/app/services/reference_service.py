"""Reference-library lookup backed by the loaded FAISS index metadata."""

from __future__ import annotations

from pathlib import Path

from app.schemas.analysis import Reference
from ml.retrieval.faiss_store import ReferenceIndex, ReferenceRecord


class ReferenceService:
    def __init__(self, index: ReferenceIndex, data_dir: Path) -> None:
        self._index_version = index.version
        self._records = {record.reference_id: record for record in index.records}
        self._data_dir = data_dir.resolve()

    def get(self, reference_id: str) -> Reference | None:
        record = self._records.get(reference_id)
        if record is None:
            return None
        return Reference(
            reference_id=record.reference_id,
            image_id=record.image_id,
            class_name=record.class_name,
            fragment_type=record.fragment_type,
            dataset=record.dataset,
            source=record.source,
            index_version=self._index_version,
            image_url=reference_image_url(record),
        )

    def image_path(self, reference_id: str) -> Path | None:
        """Resolve a reference image inside DATA_DIR; anything escaping it is treated as missing."""
        record = self._records.get(reference_id)
        if record is None:
            return None
        path = (self._data_dir / record.image_path).resolve()
        return path if path.is_relative_to(self._data_dir) and path.is_file() else None


def reference_image_url(record: ReferenceRecord) -> str:
    return f"/reference/{record.reference_id}/image"
