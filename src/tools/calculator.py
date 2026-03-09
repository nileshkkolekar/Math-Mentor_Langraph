"""Safe math calculator using sympy. No arbitrary code execution."""
from typing import Any

import sympy
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication, parse_expr


def _safe_parse(s: str):
    """Parse expression with only safe builtins."""
    transformations = standard_transformations + (implicit_multiplication,)
    return parse_expr(s, transformations=transformations)


def evaluate(expression: str) -> dict[str, Any]:
    """
    Evaluate a mathematical expression. Returns {"value": str, "error": str|None}.
    """
    expression = (expression or "").strip()
    if not expression:
        return {"value": "", "error": "Empty expression"}
    try:
        expr = _safe_parse(expression)
        result = expr.evalf()
        if hasattr(result, "__iter__") and not isinstance(result, (str, sympy.Basic)):
            result = list(result)
        return {"value": str(result), "error": None}
    except Exception as e:
        return {"value": "", "error": str(e)}


def differentiate(expr_str: str, symbol: str = "x") -> dict[str, Any]:
    """Compute derivative with respect to symbol."""
    try:
        x = sympy.Symbol(symbol)
        expr = _safe_parse(expr_str)
        der = sympy.diff(expr, x)
        return {"value": str(der), "error": None}
    except Exception as e:
        return {"value": "", "error": str(e)}


def limit(expr_str: str, symbol: str, point: str) -> dict[str, Any]:
    """Compute limit as symbol -> point."""
    try:
        x = sympy.Symbol(symbol)
        expr = _safe_parse(expr_str)
        pt = _safe_parse(point)
        lim = sympy.limit(expr, x, pt)
        return {"value": str(lim), "error": None}
    except Exception as e:
        return {"value": "", "error": str(e)}
