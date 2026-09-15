"""SQLite persistence for completed analyses and their normalised sample images.

SQLite needs no server, credentials or migrations tool, which suits a local-first,
single-node screening station. Each call opens its own connection, so the store is safe to
use from FastAPI's worker threads.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from app.schemas.analysis import AnalysisResponse, AnalysisSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    decision_status TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS analyses_created_at ON analyses (created_at DESC);
"""


class AnalysisStore:
    def __init__(self, database_path: Path, sample_dir: Path) -> None:
        self._database_path = database_path
        self._sample_dir = sample_dir
        database_path.parent.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def sample_path(self, analysis_id: str) -> Path:
        return self._sample_dir / f"{analysis_id}.png"

    def save(self, analysis: AnalysisResponse, sample_png: bytes) -> None:
        self.sample_path(analysis.id).write_bytes(sample_png)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO analyses (id, created_at, filename, decision_status, predicted_class, confidence,"
                " result_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    analysis.id,
                    analysis.created_at,
                    analysis.sample.filename,
                    analysis.decision.status,
                    analysis.prediction.class_name,
                    analysis.prediction.confidence,
                    analysis.model_dump_json(),
                ),
            )

    def get(self, analysis_id: str) -> AnalysisResponse | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT result_json FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        return AnalysisResponse.model_validate_json(row["result_json"]) if row else None

    def list_recent(self, limit: int) -> list[AnalysisSummary]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, created_at, filename, decision_status, predicted_class, confidence"
                " FROM analyses ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [AnalysisSummary(**dict(row), image_url=f"/analyses/{row['id']}/image") for row in rows]
