"""Parser Agent: raw text -> structured problem with topic, variables, constraints."""
import json
from typing import Any

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL


def parse(raw_text: str) -> dict[str, Any]:
    """
    Convert raw input into structured problem.
    Returns dict with problem_text, topic, variables, constraints, needs_clarification.
    """
    if not OPENAI_API_KEY:
        # Fallback when no API key: minimal structure
        return {
            "problem_text": (raw_text or "").strip(),
            "topic": "algebra",
            "variables": [],
            "constraints": [],
            "needs_clarification": False,
        }
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = """You are a math problem parser. Given raw text from OCR or speech, output a structured JSON object with:
- problem_text: cleaned, unambiguous problem statement
- topic: one of "algebra", "probability", "calculus", "linear_algebra"
- variables: list of variable names (e.g. ["x", "n"])
- constraints: list of constraints (e.g. ["x > 0", "n is natural number"])
- needs_clarification: true only if information is missing or ambiguous

Output ONLY valid JSON, no markdown or explanation."""
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": raw_text or ""},
            ],
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        # Strip markdown code block if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        out = json.loads(content)
        return {
            "problem_text": out.get("problem_text", raw_text),
            "topic": out.get("topic", "algebra"),
            "variables": out.get("variables", []),
            "constraints": out.get("constraints", []),
            "needs_clarification": bool(out.get("needs_clarification", False)),
        }
    except Exception:
        return {
            "problem_text": (raw_text or "").strip(),
            "topic": "algebra",
            "variables": [],
            "constraints": [],
            "needs_clarification": False,
        }
