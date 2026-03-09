"""Build RAG index from knowledge_base directory. Run from repo root: python scripts/build_rag.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import KNOWLEDGE_BASE_DIR, CHROMA_PERSIST_DIR
from src.rag.chunker import chunk_directory
from src.rag.vector_store import get_client, get_or_create_collection, add_chunks

def main():
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    chunks = chunk_directory(KNOWLEDGE_BASE_DIR)
    if not chunks:
        print("No documents found in", KNOWLEDGE_BASE_DIR)
        return
    print(f"Chunked {len(chunks)} chunks from knowledge base.")
    client = get_client()
    coll = get_or_create_collection(client)
    add_chunks(coll, chunks)
    print("Index built at", CHROMA_PERSIST_DIR)

if __name__ == "__main__":
    main()
