"""Explainer / Tutor Agent: step-by-step student-friendly explanation."""
from typing import Any

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL


def explain(
    parsed: dict[str, Any],
    solution: dict[str, Any],
    verification: dict[str, Any],
) -> str:
    """Produce a clear, step-by-step explanation for the student."""
    problem = parsed.get("problem_text", "")
    steps = solution.get("steps", [])
    final_answer = solution.get("final_answer", "")
    if not OPENAI_API_KEY:
        return "## Explanation\n\n" + "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) + f"\n\n**Answer:** {final_answer}"
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a friendly math tutor. Explain the solution in simple, step-by-step language suitable for a JEE aspirant. Use markdown. Do not repeat the raw steps verbatim; add intuition and clarity."},
                {"role": "user", "content": f"Problem: {problem}\n\nSolution steps:\n" + "\n".join(steps) + f"\n\nFinal answer: {final_answer}"},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip() or "No explanation generated."
    except Exception as e:
        return "## Explanation\n\n" + "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) + f"\n\n**Answer:** {final_answer}\n\n(Explanation generation failed: {e})"
