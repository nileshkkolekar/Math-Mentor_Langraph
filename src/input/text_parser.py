"""Text input: normalize and pass through."""
from typing import Any


def parse_text(raw: str) -> dict[str, Any]:
    """Normalize text input. No HITL for plain text."""
    text = (raw or "").strip()
    return {
        "text": text,
        "confidence": 1.0,
        "needs_hitl": False,
        "source": "text",
    }
