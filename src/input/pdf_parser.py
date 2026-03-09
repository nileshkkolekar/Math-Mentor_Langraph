"""PDF text extraction: normal text for text-based PDFs, vision LLM for image-based pages."""
from pathlib import Path
from typing import Any
import base64
import tempfile

import fitz  # PyMuPDF

from src.config import OPENAI_API_KEY, OPENAI_MODEL_VISION

# Minimum non-whitespace characters per page to consider it "text-based"
MIN_TEXT_CHARS_PER_PAGE = 40


def _is_image_based_page(page) -> bool:
    """Return True if the page has too little extractable text (treat as image-based)."""
    text = page.get_text().strip()
    non_white = "".join(text.split())
    return len(non_white) < MIN_TEXT_CHARS_PER_PAGE


def _page_to_png_bytes(page, dpi: int = 150) -> bytes:
    """Render a PDF page to PNG bytes for vision API."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _extract_text_with_vision(png_bytes: bytes, api_key: str, model: str) -> str:
    """Use OpenAI vision API to extract text from a page image."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    resp = client.chat.completions.create(
        model=model,
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
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_pdf(
    pdf_path: str | Path | bytes,
    confidence_threshold: float = 0.75,
) -> dict[str, Any]:
    """
    Extract text from a PDF. If pages are text-based, use normal extraction.
    If a page is image-based (no/minimal text layer), render to image and use vision LLM.
    Returns: { "text", "confidence", "needs_hitl", "source": "pdf" }.
    """
    if isinstance(pdf_path, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_path)
            path = f.name
        try:
            return _parse_pdf_path(path, confidence_threshold)
        finally:
            Path(path).unlink(missing_ok=True)
    return _parse_pdf_path(str(pdf_path), confidence_threshold)


def _parse_pdf_path(path: str, confidence_threshold: float) -> dict[str, Any]:
    doc = fitz.open(path)
    try:
        parts = []
        used_vision = False
        for i in range(len(doc)):
            page = doc[i]
            if _is_image_based_page(page):
                if OPENAI_API_KEY:
                    png_bytes = _page_to_png_bytes(page)
                    text = _extract_text_with_vision(png_bytes, OPENAI_API_KEY, OPENAI_MODEL_VISION)
                    used_vision = True
                else:
                    # Fallback: OCR the rendered page (e.g. EasyOCR)
                    try:
                        import cv2
                        import numpy as np
                        from src.input.image_parser import parse_image
                        pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
                        img_bytes = pix.tobytes("png")
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            result = parse_image(img_bytes, confidence_threshold=0.5)
                            text = result.get("text", "")
                        else:
                            text = page.get_text().strip()
                    except Exception:
                        text = page.get_text().strip()
                parts.append(text)
            else:
                parts.append(page.get_text().strip())
        full_text = "\n\n".join(p for p in parts if p).strip()
        # Confidence: lower if we used vision (LLM can hallucinate) or if very short
        confidence = 0.85 if not used_vision else 0.8
        if len(full_text) < 20:
            confidence = 0.5
        needs_hitl = confidence < confidence_threshold
        return {
            "text": full_text or "",
            "confidence": confidence,
            "needs_hitl": needs_hitl,
            "source": "pdf",
        }
    finally:
        doc.close()
