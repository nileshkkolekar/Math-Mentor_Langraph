"""Chunk documents for RAG."""
from pathlib import Path
import re
from typing import Any


def chunk_file(path: Path, content: str, chunk_size: int = 400, overlap: int = 50) -> list[dict[str, Any]]:
    """Split content into overlapping chunks. Preserve formula blocks when possible."""
    chunks = []
    # Split by double newline first to keep paragraphs together
    blocks = re.split(r"\n\s*\n", content)
    current = []
    current_len = 0
    source = path.name
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        blen = len(block) + 1
        if current_len + blen > chunk_size and current:
            text = "\n\n".join(current)
            chunks.append({"text": text, "source": source})
            # overlap: keep last part of current
            overlap_text = current[-1][-overlap:] if overlap else ""
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)
        current.append(block)
        current_len += blen
    if current:
        text = "\n\n".join(current)
        chunks.append({"text": text, "source": source})
    return chunks


def chunk_directory(knowledge_dir: Path, extensions: tuple = (".md", ".txt")) -> list[dict[str, Any]]:
    """Chunk all matching files in directory."""
    all_chunks = []
    for path in knowledge_dir.rglob("*"):
        if path.suffix.lower() not in extensions:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            all_chunks.extend(chunk_file(path, content))
        except Exception:
            continue
    return all_chunks
