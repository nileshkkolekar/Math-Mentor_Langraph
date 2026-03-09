"""Retrieve similar solved problems from memory (by embedding problem_text)."""
import json
from typing import Any

from src.memory.store import get_recent


def retrieve_similar(parsed: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    """
    Return past sessions with similar problem_text for pattern reuse.
    Simplified: return recent sessions; full version would embed and query vector store.
    """
    recent = get_recent(limit=limit * 5)
    topic = parsed.get("topic", "")
    out = []
    for row in recent:
        try:
            p = json.loads(row.get("parsed_question") or "{}")
            if topic and p.get("topic") != topic:
                continue
            out.append({
                "original_input": row.get("original_input"),
                "parsed_question": p,
                "solution": json.loads(row.get("solution") or "{}"),
                "user_feedback": row.get("user_feedback"),
            })
            if len(out) >= limit:
                break
        except Exception:
            continue
    return out
