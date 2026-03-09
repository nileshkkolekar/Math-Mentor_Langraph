"""RAG retriever: query vector store and return chunks."""
from typing import Any

from src.config import RAG_TOP_K, KNOWLEDGE_BASE_DIR, CHROMA_PERSIST_DIR
from src.rag.vector_store import get_client, get_or_create_collection, query_collection

# Lazy init
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = get_or_create_collection(client)
    return _collection


def retrieve(parsed: dict[str, Any], top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Use parsed problem (problem_text + topic) to retrieve relevant chunks.
    Returns list of {"text", "source"}. If collection is empty, returns [].
    """
    top_k = top_k or RAG_TOP_K
    query = parsed.get("problem_text", "") or ""
    topic = parsed.get("topic", "")
    if topic:
        query = f"{topic}: {query}"
    coll = _get_collection()
    try:
        return query_collection(coll, query, top_k=top_k)
    except Exception:
        return []
