"""Persist session and feedback to SQLite."""
import json
import sqlite3
from pathlib import Path
from typing import Any

from src.config import MEMORY_DB_PATH


def _ensure_db():
    p = Path(MEMORY_DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_input TEXT,
            parsed_question TEXT,
            retrieved_context TEXT,
            solution TEXT,
            verifier_outcome TEXT,
            user_feedback TEXT,
            feedback_comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def store(
    original_input: str,
    parsed_question: dict,
    retrieved_context: list[dict],
    solution: dict,
    verifier_outcome: dict,
    user_feedback: str | None = None,
    feedback_comment: str | None = None,
):
    """Save one session (and optional feedback)."""
    conn = _ensure_db()
    conn.execute(
        """INSERT INTO sessions (
            original_input, parsed_question, retrieved_context, solution,
            verifier_outcome, user_feedback, feedback_comment
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            original_input,
            json.dumps(parsed_question),
            json.dumps(retrieved_context),
            json.dumps(solution),
            json.dumps(verifier_outcome),
            user_feedback,
            feedback_comment,
        ),
    )
    conn.commit()
    conn.close()


def get_recent(limit: int = 50) -> list[dict[str, Any]]:
    """Get recent sessions for display. Ensures the sessions table exists first."""
    conn = _ensure_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
