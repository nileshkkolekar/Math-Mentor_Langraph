"""Image text extraction: Vision LLM when API key is set, else EasyOCR. Returns text and confidence; triggers HITL if low confidence."""
import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import OPENAI_API_KEY, OPENAI_MODEL_VISION

# Lazy init EasyOCR reader (used when no API key or as fallback)
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _image_media_type(image_path: str | Path | bytes) -> str:
    """Return 'image/png' or 'image/jpeg' for API."""
    if isinstance(image_path, bytes):
        if image_path[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if image_path[:2] == b"\xff\xd8":
            return "image/jpeg"
        return "image/png"
    s = str(image_path).lower()
    if s.endswith(".jpg") or s.endswith(".jpeg"):
        return "image/jpeg"
    return "image/png"


def _extract_with_vision(image_bytes: bytes, media_type: str) -> dict[str, Any]:
    """Use OpenAI vision API to extract text from image. Returns same shape as parse_image."""
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image exactly as it appears. "
                            "Preserve line breaks, math notation, and structure. "
                            "If it is a math problem or equation, write it clearly.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        text = (resp.choices[0].message.content or "").strip()
        confidence = 0.9 if len(text) > 10 else 0.7
        return {"text": text, "confidence": confidence, "needs_hitl": confidence < 0.7, "source": "image"}
    except Exception:
        return None


def _extract_with_ocr(img) -> dict[str, Any]:
    """EasyOCR path. img is numpy array (BGR)."""
    reader = _get_reader()
    results = reader.readtext(img)
    if not results:
        return {"text": "", "confidence": 0.0, "needs_hitl": True, "source": "image"}
    texts = []
    confidences = []
    for (_, text, conf) in results:
        texts.append(text)
        confidences.append(float(conf))
    full_text = " ".join(texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {"text": full_text, "confidence": avg_conf, "needs_hitl": avg_conf < 0.7, "source": "image"}


def parse_image(image_path: str | Path | bytes, confidence_threshold: float = 0.7) -> dict[str, Any]:
    """
    Extract text from image. Uses Vision LLM when OPENAI_API_KEY is set, else EasyOCR.
    Return extracted text and confidence; set needs_hitl=True if confidence < confidence_threshold.
    """
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        image_bytes = Path(image_path).read_bytes() if Path(image_path).exists() else None
    else:
        nparr = np.frombuffer(image_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_bytes = image_path
    if img is None:
        return {
            "text": "",
            "confidence": 0.0,
            "needs_hitl": True,
            "source": "image",
            "error": "Could not load image",
        }
    media_type = _image_media_type(image_path)
    if image_bytes is not None and OPENAI_API_KEY:
        out = _extract_with_vision(image_bytes, media_type)
        if out is not None:
            out["needs_hitl"] = out["confidence"] < confidence_threshold
            return out
    out = _extract_with_ocr(img)
    out["needs_hitl"] = out["confidence"] < confidence_threshold
    return out
