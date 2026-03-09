"""Orchestrator: run Parser -> Router -> RAG -> Solver -> Verifier -> Explainer with HITL hooks."""
from typing import Any, Callable

from src.agents.parser_agent import parse
from src.agents.router_agent import route
from src.agents.solver_agent import solve
from src.agents.verifier_agent import verify
from src.agents.explainer_agent import explain
from src.memory.retriever import retrieve_similar


def run_pipeline(
    raw_text: str,
    get_retrieved_chunks: Callable[[dict], list[dict]],
    *,
    parsed_override: dict | None = None,
    skip_verify_hitl: bool = False,
    solution_override: dict | None = None,
    verification_override: dict | None = None,
) -> dict[str, Any]:
    """
    Execute full pipeline. get_retrieved_chunks(parsed) returns list of {"text", "source"}.
    If parsed_override is set, skip parser and use this.
    If solution_override and verification_override are set (e.g. after HITL Approve), skip solve/verify and only run explain.
    """
    trace = []
    # 1. Parse
    if parsed_override is not None:
        parsed = parsed_override
        trace.append({"step": "parser", "output": "user_confirmed", "parsed": parsed})
    else:
        parsed = parse(raw_text)
        trace.append({"step": "parser", "output": parsed})
        if parsed.get("needs_clarification"):
            return {
                "trace": trace,
                "parsed": parsed,
                "route_info": None,
                "retrieved": [],
                "solution": None,
                "verification": None,
                "explanation": None,
                "hitl_required": "parser",
            }
    # 2. Route
    route_info = route(parsed)
    trace.append({"step": "router", "output": route_info})
    # 3. Retrieve
    retrieved = get_retrieved_chunks(parsed)
    similar = retrieve_similar(parsed)
    trace.append({"step": "retrieve", "output": {"num_chunks": len(retrieved), "similar_problems": len(similar)}})

    if solution_override is not None and verification_override is not None:
        # HITL Approve: use existing solution and verification, only run Explainer
        solution = solution_override
        verification = verification_override
        trace.append({"step": "solver", "output": "user_approved"})
        trace.append({"step": "verifier", "output": "user_approved"})
    else:
        # 4. Solve
        solution = solve(parsed, route_info, retrieved, similar_problems=similar)
        trace.append({"step": "solver", "output": {"steps_count": len(solution.get("steps", [])), "final_answer": solution.get("final_answer")}})
        # 5. Verify
        verification = verify(parsed, solution)
        trace.append({"step": "verifier", "output": verification})
        if verification.get("needs_hitl") and not skip_verify_hitl:
            return {
                "trace": trace,
                "parsed": parsed,
                "route_info": route_info,
                "retrieved": retrieved,
                "solution": solution,
                "verification": verification,
                "explanation": None,
                "hitl_required": "verifier",
            }
    # 6. Explain
    explanation = explain(parsed, solution, verification)
    trace.append({"step": "explainer", "output": "ok"})
    return {
        "trace": trace,
        "parsed": parsed,
        "route_info": route_info,
        "retrieved": retrieved,
        "solution": solution,
        "verification": verification,
        "explanation": explanation,
        "hitl_required": None,
    }
