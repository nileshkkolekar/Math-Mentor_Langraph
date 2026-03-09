"""Vector store using Chroma."""
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_PERSIST_DIR
from src.rag.embedder import embed


def get_client(persist_dir: Path | None = None):
    persist_dir = persist_dir or CHROMA_PERSIST_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))


def get_or_create_collection(client, name: str = "math_mentor"):
    return client.get_or_create_collection(name=name, metadata={"description": "Math knowledge chunks"})


def add_chunks(collection, chunks: list[dict], embedding_model: str | None = None):
    """chunks: list of {text, source}. Embed and add to collection."""
    if not chunks:
        return
    texts = [c["text"] for c in chunks]
    embeddings = embed(texts, embedding_model)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c.get("source", "unknown")} for c in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def query_collection(collection, query_text: str, top_k: int = 5, embedding_model: str | None = None):
    """Return list of {text, source} for top_k similar chunks."""
    if not query_text.strip():
        return []
    q_emb = embed([query_text], embedding_model)[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas"])
    out = []
    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        out.append({"text": doc, "source": meta.get("source", "unknown")})
    return out
