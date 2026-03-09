"""Intent Router Agent: classify problem type and route strategy."""
from typing import Any

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL


def route(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Classify problem and return routing info: topic, subtype, strategy_hint.
    """
    topic = parsed.get("topic", "algebra")
    problem_text = parsed.get("problem_text", "")
    if not OPENAI_API_KEY:
        return {"topic": topic, "subtype": "general", "strategy_hint": "Use RAG and step-by-step solution."}
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a math problem classifier. Given a structured problem, respond with JSON: {\"topic\": \"algebra\"|\"probability\"|\"calculus\"|\"linear_algebra\", \"subtype\": brief subtype, \"strategy_hint\": one line hint for solving. Output only JSON."},
                {"role": "user", "content": f"Topic: {topic}\nProblem: {problem_text}"},
            ],
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        import json
        out = json.loads(content)
        return {
            "topic": out.get("topic", topic),
            "subtype": out.get("subtype", "general"),
            "strategy_hint": out.get("strategy_hint", "Use RAG and step-by-step solution."),
        }
    except Exception:
        return {"topic": topic, "subtype": "general", "strategy_hint": "Use RAG and step-by-step solution."}
