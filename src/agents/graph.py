"""LangGraph pipeline: Parser -> Router -> Retrieve -> Solver -> Verifier -> Explainer with HITL."""
from typing import Annotated, Any, Callable, Literal, TypedDict

from langgraph.graph import StateGraph

from src.agents.parser_agent import parse
from src.agents.router_agent import route
from src.agents.solver_agent import solve
from src.agents.verifier_agent import verify
from src.agents.explainer_agent import explain
from src.memory.retriever import retrieve_similar


def _trace_reducer(left: list, right: list) -> list:
    """Append right to left (for state updates)."""
    return left + (right or [])


# State: TypedDict-like dict. trace uses reducer to append.


def _parser_node(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("parsed_override") is not None:
        parsed = state["parsed_override"]
        return {"parsed": parsed, "trace": [{"step": "parser", "output": "user_confirmed", "parsed": parsed}]}
    parsed = parse(state.get("raw_text", "") or "")
    hitl = "parser" if parsed.get("needs_clarification") else None
    return {"parsed": parsed, "trace": [{"step": "parser", "output": parsed}], "hitl_required": hitl}


def _after_parser(state: dict[str, Any]) -> Literal["router", "__end__"]:
    if state.get("hitl_required") == "parser":
        return "__end__"
    return "router"


def _router_node(state: dict[str, Any]) -> dict[str, Any]:
    route_info = route(state["parsed"])
    return {"route_info": route_info, "trace": [{"step": "router", "output": route_info}]}


def _retrieve_node(state: dict[str, Any], *, get_retrieved_chunks: Callable[[dict], list]) -> dict[str, Any]:
    retrieved = get_retrieved_chunks(state["parsed"])
    similar = retrieve_similar(state["parsed"])
    return {"retrieved": retrieved, "similar": similar, "trace": [{"step": "retrieve", "output": {"num_chunks": len(retrieved), "similar_problems": len(similar)}}]}


def _solver_node(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("solution_override") is not None and state.get("verification_override") is not None:
        return {"solution": state["solution_override"], "verification": state["verification_override"], "trace": [{"step": "solver", "output": "user_approved"}, {"step": "verifier", "output": "user_approved"}]}
    solution = solve(state["parsed"], state["route_info"], state["retrieved"], similar_problems=state.get("similar") or [])
    return {"solution": solution, "trace": [{"step": "solver", "output": {"steps_count": len(solution.get("steps", [])), "final_answer": solution.get("final_answer")}}]}


def _verifier_node(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("solution_override") is not None and state.get("verification_override") is not None:
        return {}  # already set in solver_node path
    verification = verify(state["parsed"], state["solution"])
    hitl = "verifier" if verification.get("needs_hitl") else None
    return {"verification": verification, "trace": [{"step": "verifier", "output": verification}], "hitl_required": hitl}


def _after_verifier(state: dict[str, Any]) -> Literal["explainer", "__end__"]:
    if state.get("hitl_required") == "verifier":
        return "__end__"
    return "explainer"


def _explainer_node(state: dict[str, Any]) -> dict[str, Any]:
    explanation = explain(state["parsed"], state["solution"], state["verification"])
    return {"explanation": explanation, "trace": [{"step": "explainer", "output": "ok"}], "hitl_required": None}


def build_graph(get_retrieved_chunks: Callable[[dict], list]):
    """Build and compile the Math Mentor LangGraph. get_retrieved_chunks(parsed) returns list of {text, source}."""
    class State(TypedDict, total=False):
        raw_text: str
        parsed_override: dict | None
        solution_override: dict | None
        verification_override: dict | None
        parsed: dict | None
        route_info: dict | None
        retrieved: list
        similar: list
        solution: dict | None
        verification: dict | None
        explanation: str | None
        trace: Annotated[list, _trace_reducer]
        hitl_required: str | None

    builder = StateGraph(State)

    builder.add_node("parser", _parser_node)
    builder.add_node("router", _router_node)
    builder.add_node("retrieve", lambda s: _retrieve_node(s, get_retrieved_chunks=get_retrieved_chunks))
    builder.add_node("solver", _solver_node)
    builder.add_node("verifier", _verifier_node)
    builder.add_node("explainer", _explainer_node)

    builder.set_entry_point("parser")
    builder.add_conditional_edges("parser", _after_parser, path_map={"router": "router", "__end__": "__end__"})
    builder.add_edge("router", "retrieve")
    builder.add_edge("retrieve", "solver")
    builder.add_edge("solver", "verifier")
    builder.add_conditional_edges("verifier", _after_verifier, path_map={"explainer": "explainer", "__end__": "__end__"})
    builder.add_edge("explainer", "__end__")

    return builder.compile()


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
    Run the LangGraph pipeline (Parser → Router → Retrieve → Solver → Verifier → Explainer).
    Returns dict with trace, parsed, route_info, retrieved, solution, verification, explanation, hitl_required.
    """
    graph = build_graph(get_retrieved_chunks)
    initial: dict[str, Any] = {
        "raw_text": raw_text,
        "parsed_override": parsed_override,
        "solution_override": solution_override,
        "verification_override": verification_override,
        "trace": [],
    }
    result = graph.invoke(initial)
    # Normalize to expected return shape for the app
    return {
        "trace": result.get("trace", []),
        "parsed": result.get("parsed"),
        "route_info": result.get("route_info"),
        "retrieved": result.get("retrieved", []),
        "solution": result.get("solution"),
        "verification": result.get("verification"),
        "explanation": result.get("explanation"),
        "hitl_required": result.get("hitl_required"),
    }
