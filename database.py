from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class HistoryStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path

    def ensure_database(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT NOT NULL,
                    input_summary TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    explanation TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add(self, task: str, input_summary: str, prediction: str, confidence: float, explanation: str) -> int:
        self.ensure_database()
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (task, input_summary, prediction, confidence, explanation, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task, input_summary, prediction, confidence, explanation, created_at),
            )
            return int(cursor.lastrowid)

    def latest(self, limit: int = 12) -> list[dict[str, object]]:
        self.ensure_database()
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, task, input_summary, prediction, confidence, explanation, created_at
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self) -> None:
        self.ensure_database()
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DELETE FROM predictions")
