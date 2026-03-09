"""Retrieve similar solved problems from memory (Chroma for semantic search when available, else SQLite recent+topic)."""
import json
from typing import Any

from src.memory.store import get_recent


def _retrieve_similar_from_chroma(parsed: dict[str, Any], limit: int) -> list[dict[str, Any]] | None:
    """Use Chroma memory collection for semantic similar-problem retrieval. Returns None on failure or empty."""
    try:
        from src.rag.embedder import embed
        from src.rag.vector_store import get_client, get_or_create_collection
        problem_text = (parsed.get("problem_text") or "").strip()
        if not problem_text:
            return None
        client = get_client()
        coll = get_or_create_collection(client, name="math_mentor_memory")
        # Query by embedding similarity
        q_emb = embed([problem_text])
        if not q_emb:
            return None
        results = coll.query(
            query_embeddings=q_emb,
            n_results=limit,
            include=["metadatas"],
        )
        metas = results.get("metadatas", [[]])
        if not metas or not metas[0]:
            return None
        out = []
        for m in metas[0]:
            try:
                parsed_q = json.loads(m.get("parsed_question") or "{}")
                sol = json.loads(m.get("solution") or "{}")
                out.append({
                    "original_input": m.get("original_input", ""),
                    "parsed_question": parsed_q,
                    "solution": sol,
                    "user_feedback": m.get("user_feedback"),
                })
            except Exception:
                continue
        return out[:limit] if out else None
    except Exception:
        return None


def retrieve_similar(parsed: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    """
    Return past sessions with similar problem_text for pattern reuse.
    Uses Chroma (semantic similarity) when available; falls back to SQLite recent + topic filter.
    """
    similar = _retrieve_similar_from_chroma(parsed, limit)
    if similar is not None and len(similar) > 0:
        return similar
    # Fallback: SQLite get_recent + topic filter
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
