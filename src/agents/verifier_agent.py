"""Verifier / Critic Agent: check correctness, units, domain; output confidence."""
import json
import re
from typing import Any

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL_VERIFIER, VERIFIER_CONFIDENCE_THRESHOLD
from src.tools.calculator import evaluate


def _try_arithmetic_check(problem_text: str, final_answer: str) -> tuple[bool | None, float | None]:
    """
    If the problem looks like a single arithmetic expression, evaluate it and compare to final_answer.
    Returns (True if match, False if mismatch, None if not applicable), (expected value or None).
    """
    text = (problem_text or "").strip().lower()
    # Map common words to operators
    text = re.sub(r"\bminus\b", "-", text)
    text = re.sub(r"\bplus\b", "+", text)
    text = re.sub(r"\btimes\b", "*", text)
    text = re.sub(r"\bdivided by\b", "/", text)
    # Extract possible expression: digits and + - * / ( )
    expr = re.sub(r"[^\d+\-*/().\s]", "", text).strip()
    if not expr or len(expr) < 3:
        return None, None
    try:
        result = evaluate(expr)
        if result.get("error"):
            return None, None
        expected_str = result.get("value", "").strip()
        # Normalize final_answer for comparison (e.g. "-1" vs "-1.0")
        try:
            expected_val = float(expected_str)
            answer_val = float(re.sub(r"[^\d.\-]", "", final_answer.strip()) or "nan")
            if abs(expected_val - answer_val) < 1e-6:
                return True, expected_val
            return False, expected_val
        except ValueError:
            if expected_str == final_answer.strip():
                return True, None
            return None, None
    except Exception:
        return None, None


def verify(
    parsed: dict[str, Any],
    solution: dict[str, Any],
) -> dict[str, Any]:
    """
    Returns: { "correct": bool, "confidence": float, "issues": list[str] }.
    If confidence < VERIFIER_CONFIDENCE_THRESHOLD, caller should trigger HITL.
    """
    problem = parsed.get("problem_text", "")
    steps = solution.get("steps", [])
    final_answer = solution.get("final_answer", "")
    if not OPENAI_API_KEY:
        return {"correct": True, "confidence": 0.8, "issues": [], "needs_hitl": False}

    # For simple arithmetic, check numerically so we don't mark correct answers wrong
    arithmetic_ok, expected_val = _try_arithmetic_check(problem, final_answer)
    if arithmetic_ok is True:
        return {"correct": True, "confidence": 0.95, "issues": [], "needs_hitl": False}
    if arithmetic_ok is False:
        return {
            "correct": False,
            "confidence": 0.9,
            "issues": [f"Arithmetic check: expected {expected_val}, got final answer '{final_answer}'."],
            "needs_hitl": False,
        }

    client = OpenAI(api_key=OPENAI_API_KEY)
    extra = f"\n\n(If the problem is simple arithmetic, the correct numerical answer is the only criterion. Set correct=true if the final answer value is right, and issues=[] if there are no real errors.)"
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_VERIFIER,
            messages=[
                {"role": "system", "content": "You are a math solution verifier. Your job is to decide if the FINAL ANSWER is mathematically correct. Set correct=true only when the final answer value is wrong; set correct=false only when the stated final answer does not match the correct result. Be consistent: if the correct result is X and the solver gave X, then correct must be true and issues must be empty or only about presentation. Respond with JSON only: {\"correct\": true or false, \"confidence\": 0.0-1.0, \"issues\": [list of real issues or empty]}."},
                {"role": "user", "content": f"Problem: {problem}\n\nSolution steps: {steps}\n\nFinal answer: {final_answer}{extra}"},
            ],
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        out = json.loads(content)
        correct = bool(out.get("correct", True))
        confidence = float(out.get("confidence", 0.8))
        issues = out.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        needs_hitl = confidence < VERIFIER_CONFIDENCE_THRESHOLD
        return {
            "correct": correct,
            "confidence": confidence,
            "issues": issues,
            "needs_hitl": needs_hitl,
        }
    except Exception:
        return {"correct": True, "confidence": 0.5, "issues": [], "needs_hitl": True}
