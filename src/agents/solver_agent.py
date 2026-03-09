"""Solver Agent: use RAG context + calculator to produce solution steps and final answer."""
from typing import Any

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.tools.calculator import evaluate, differentiate, limit


def solve(
    parsed: dict[str, Any],
    route_info: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    similar_problems: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Produce solution steps and final answer using RAG context and optional similar problems.
    retrieved_chunks: list of {"text", "source"} from RAG.
    similar_problems: optional list of past {parsed_question, solution, user_feedback} for pattern reuse.
    """
    problem = parsed.get("problem_text", "")
    topic = route_info.get("topic", "algebra")
    hint = route_info.get("strategy_hint", "")
    context_block = "\n\n".join(
        f"[{c.get('source', 'doc')}]\n{c.get('text', c.get('content', ''))}" for c in retrieved_chunks
    )
    if not context_block:
        context_block = "No additional context."
    if similar_problems:
        similar_block = "\n\n".join(
            f"Similar problem (feedback: {p.get('user_feedback', 'N/A')}): {p.get('parsed_question', {}).get('problem_text', '')[:200]}... Solution: {str(p.get('solution', {}).get('final_answer', ''))}"
            for p in similar_problems[:3]
        )
        context_block = context_block + "\n\n--- Similar solved problems for pattern reuse ---\n" + similar_block
    if not OPENAI_API_KEY:
        return {
            "steps": ["No API key configured. Add OPENAI_API_KEY to solve."],
            "final_answer": "N/A",
            "tool_calls": [],
        }
    client = OpenAI(api_key=OPENAI_API_KEY)
    tools_desc = (
        "You may use these tools: evaluate(expression), differentiate(expr, symbol), limit(expr, symbol, point). "
        "When you need to compute, output a line like: TOOL: evaluate('2+3') or TOOL: differentiate('x^2','x')."
    )
    prompt = f"""You are a JEE-style math tutor. Solve the problem step by step.

Knowledge base context:
{context_block}

Strategy: {hint}

Problem: {problem}

{tools_desc}

Output your response as:
STEPS:
1. First step...
2. Second step...
...
FINAL_ANSWER: <exact answer>"""
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        # Parse STEPS and FINAL_ANSWER
        steps = []
        final_answer = ""
        in_steps = True
        for line in content.split("\n"):
            if line.strip().upper().startswith("FINAL_ANSWER"):
                in_steps = False
                final_answer = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
            elif in_steps and line.strip():
                steps.append(line.strip())
        if not steps:
            steps = [content]
        if not final_answer:
            final_answer = steps[-1] if steps else "N/A"
        return {"steps": steps, "final_answer": final_answer, "raw_response": content, "tool_calls": []}
    except Exception as e:
        return {
            "steps": [f"Error: {e}"],
            "final_answer": "Error",
            "tool_calls": [],
        }
