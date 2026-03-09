"""Persist session and feedback to SQLite and Chroma (Chroma enables cloud persistence and semantic similar-problem retrieval)."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import MEMORY_DB_PATH

# Chroma metadata value size limit (truncate to stay safe)
_META_MAX = 40000


def _truncate(s: str, max_len: int = _META_MAX) -> str:
    if not s or len(s) <= max_len:
        return s or ""
    return s[: max_len - 3] + "..."


def _get_memory_collection():
    """Get or create Chroma collection for sessions (same client as RAG; works with Chroma Cloud)."""
    from src.rag.vector_store import get_client, get_or_create_collection
    client = get_client()
    return get_or_create_collection(client, name="math_mentor_memory")


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
    """Save one session (and optional feedback) to SQLite and to Chroma for semantic retrieval."""
    # 1. SQLite (always, for get_recent and fallback)
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

    # 2. Chroma (for semantic similar-problem search; persists on Chroma Cloud)
    try:
        from src.rag.embedder import embed
        problem_text = (parsed_question.get("problem_text") or original_input or "").strip() or " "
        embeddings = embed([problem_text])
        if not embeddings:
            return
        coll = _get_memory_collection()
        now = datetime.now(timezone.utc).isoformat()
        meta = {
            "original_input": _truncate(original_input or ""),
            "parsed_question": _truncate(json.dumps(parsed_question)),
            "retrieved_context": _truncate(json.dumps(retrieved_context)),
            "solution": _truncate(json.dumps(solution)),
            "verifier_outcome": _truncate(json.dumps(verifier_outcome)),
            "user_feedback": (user_feedback or "")[:500],
            "feedback_comment": _truncate((feedback_comment or ""), 2000),
            "created_at": now,
        }
        doc_id = f"session_{now[:19].replace(':', '-')}_{uuid.uuid4().hex[:8]}"
        coll.add(
            ids=[doc_id],
            embeddings=embeddings,
            documents=[problem_text[:10000]],
            metadatas=[meta],
        )
    except Exception:
        pass


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
