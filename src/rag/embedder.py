"""Embedding: OpenAI API or optional sentence-transformers fallback."""
from typing import Any

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_PROVIDER


def _embed_openai(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Use OpenAI Embeddings API. Requires OPENAI_API_KEY."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    model = model or EMBEDDING_MODEL
    out = []
    batch_size = 100
    fallback_model = "text-embedding-ada-002"  # older model if 3-small is not available
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=model)
        except Exception as e:
            if "invalid model" in str(e).lower() and model != fallback_model:
                resp = client.embeddings.create(input=batch, model=fallback_model)
            else:
                raise
        # Preserve order (API may return by index)
        by_index = {d.index: d.embedding for d in resp.data}
        out.extend(by_index[j] for j in range(len(batch)))
    return out


def _embed_sentence_transformers(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Use local sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_numpy=True).tolist()


def embed(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Return list of embedding vectors. Uses OpenAI if EMBEDDING_PROVIDER=openai, else sentence-transformers."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts, model=model_name or EMBEDDING_MODEL)
    return _embed_sentence_transformers(texts, model_name=model_name)
