"""Track pipeline progress using SQLite for checkpointing."""

import json
import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PipelineState:
    """Track pipeline progress using SQLite for checkpointing."""

    STAGES = ("discovered", "downloaded", "extracted", "chunked", "embedded")

    def __init__(self, db_path: str = "pipeline_state.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                work_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL DEFAULT 'discovered',
                pdf_path TEXT,
                num_sections INTEGER DEFAULT 0,
                num_chunks INTEGER DEFAULT 0,
                metadata TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_papers_stage ON papers(stage);

            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                papers_processed INTEGER DEFAULT 0,
                papers_failed INTEGER DEFAULT 0,
                config TEXT
            );
        """)
        self.conn.commit()

    def mark_discovered(self, work_id: str, metadata: dict):
        """Mark a paper as discovered."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT OR IGNORE INTO papers (work_id, stage, metadata, created_at, updated_at)
               VALUES (?, 'discovered', ?, ?, ?)""",
            (work_id, json.dumps(metadata), now, now),
        )
        self.conn.commit()

    def mark_downloaded(self, work_id: str, pdf_path: str):
        """Mark a paper as downloaded."""
        self._update_stage(work_id, "downloaded", pdf_path=pdf_path)

    def mark_extracted(self, work_id: str, num_sections: int):
        """Mark a paper as extracted."""
        self._update_stage(work_id, "extracted", num_sections=num_sections)

    def mark_chunked(self, work_id: str, num_chunks: int):
        """Mark a paper as chunked."""
        self._update_stage(work_id, "chunked", num_chunks=num_chunks)

    def mark_embedded(self, work_id: str, num_chunks: int):
        """Mark a paper as embedded (final stage)."""
        self._update_stage(work_id, "embedded", num_chunks=num_chunks)

    def mark_failed(self, work_id: str, stage: str, error: str):
        """Mark a paper as failed at a specific stage."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE papers SET stage = ?, error = ?, updated_at = ?
               WHERE work_id = ?""",
            (f"failed_{stage}", error, now, work_id),
        )
        self.conn.commit()

    def is_processed(self, work_id: str) -> bool:
        """Check if a paper has already been fully processed."""
        row = self.conn.execute(
            "SELECT stage FROM papers WHERE work_id = ?", (work_id,)
        ).fetchone()
        return row is not None and row["stage"] == "embedded"

    def get_pending(self, stage: str, limit: int = 500) -> list[dict]:
        """Get papers pending at a given stage.

        Returns papers whose current stage is the one before the requested stage.
        """
        stage_index = self.STAGES.index(stage)
        if stage_index == 0:
            return []
        prev_stage = self.STAGES[stage_index - 1]

        rows = self.conn.execute(
            """SELECT work_id, pdf_path, metadata FROM papers
               WHERE stage = ? LIMIT ?""",
            (prev_stage, limit),
        ).fetchall()

        return [
            {
                "work_id": row["work_id"],
                "pdf_path": row["pdf_path"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }
            for row in rows
        ]

    def get_stats(self) -> dict:
        """Get pipeline progress statistics."""
        rows = self.conn.execute(
            "SELECT stage, COUNT(*) as count FROM papers GROUP BY stage"
        ).fetchall()
        stats = {row["stage"]: row["count"] for row in rows}
        stats["total"] = sum(stats.values())
        return stats

    def batch_mark_discovered(self, papers: list[tuple[str, dict]]):
        """Batch mark papers as discovered."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.executemany(
            """INSERT OR IGNORE INTO papers (work_id, stage, metadata, created_at, updated_at)
               VALUES (?, 'discovered', ?, ?, ?)""",
            [(work_id, json.dumps(meta), now, now) for work_id, meta in papers],
        )
        self.conn.commit()

    def start_run(self, config: dict = None) -> int:
        """Start a new pipeline run."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO pipeline_runs (started_at, config) VALUES (?, ?)",
            (now, json.dumps(config or {})),
        )
        self.conn.commit()
        return cursor.lastrowid

    def end_run(self, run_id: int, papers_processed: int, papers_failed: int):
        """End a pipeline run."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE pipeline_runs SET ended_at = ?, papers_processed = ?, papers_failed = ?
               WHERE id = ?""",
            (now, papers_processed, papers_failed, run_id),
        )
        self.conn.commit()

    def _update_stage(self, work_id: str, stage: str, **kwargs):
        """Update paper stage with optional additional fields."""
        now = datetime.now(timezone.utc).isoformat()
        sets = ["stage = ?", "updated_at = ?"]
        vals = [stage, now]

        if "pdf_path" in kwargs:
            sets.append("pdf_path = ?")
            vals.append(kwargs["pdf_path"])
        if "num_sections" in kwargs:
            sets.append("num_sections = ?")
            vals.append(kwargs["num_sections"])
        if "num_chunks" in kwargs:
            sets.append("num_chunks = ?")
            vals.append(kwargs["num_chunks"])

        vals.append(work_id)
        self.conn.execute(
            f"UPDATE papers SET {', '.join(sets)} WHERE work_id = ?",
            vals,
        )
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()
