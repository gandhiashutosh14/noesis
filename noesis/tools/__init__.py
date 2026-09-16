"""
NOESIS — Tool implementations
=============================

Three tools the agent runtime can call:

  * calculator   — safe arithmetic evaluator
  * text_search  — in-process small corpus search (no network)
  * code_runner  — sandboxed Python evaluator for short snippets
"""
from __future__ import annotations

import ast
import math
import operator
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# calculator — safe arithmetic via AST walk
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UN_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_NAMES: Dict[str, Any] = {
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
    "log10": math.log10, "exp": math.exp, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
    "min": min, "max": max,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN_OPS:
        return _UN_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _NAMES:
        fn = _NAMES[node.func.id]
        if not callable(fn):
            raise ValueError(f"Not callable: {node.func.id}")
        return fn(*[_eval_node(a) for a in node.args])
    raise ValueError(f"Disallowed expression: {ast.dump(node)}")


async def calculator_evaluate(*, expression: str, **kwargs: Any) -> Dict[str, Any]:
    expr = (expression or "").strip()
    if not expr:
        return {"ok": False, "error": "empty expression"}
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "result": result, "expression": expr}


# ---------------------------------------------------------------------------
# text_search — in-memory corpus
# ---------------------------------------------------------------------------
_DEFAULT_CORPUS: List[Dict[str, str]] = [
    {"id": "p1", "title": "Thompson sampling",
     "body": "Thompson sampling draws one sample from each arm's Beta posterior and picks the arm with the highest sample. Uncertain arms get explored because their posteriors are wide; proven arms get exploited because their posteriors are narrow and high."},
    {"id": "p2", "title": "Upper confidence bound",
     "body": "UCB scores each arm by its mean reward plus an exploration bonus that shrinks with the number of pulls. The constant c trades off exploration against exploitation."},
    {"id": "p3", "title": "Precision and recall",
     "body": "Precision is true positives divided by all predicted positives; recall is true positives divided by all actual positives. Raising a decision threshold usually raises precision and lowers recall."},
    {"id": "p4", "title": "Brier score",
     "body": "The Brier score is the mean squared difference between a forecast probability and the realised outcome. Lower is better, and it rewards calibration as well as discrimination."},
    {"id": "p5", "title": "Cognitive Momentum",
     "body": "A reward formulation that measures productive movement toward truth under uncertainty rather than only final correctness."},
    {"id": "p6", "title": "Self-distillation gate",
     "body": "SDAR treats teacher signals as a gated auxiliary objective. Negative teacher rejections are softly attenuated rather than forced through."},
]


async def text_search(*, query: str, k: int = 3,
                      corpus: Optional[List[Dict[str, str]]] = None,
                      **kwargs: Any) -> Dict[str, Any]:
    corpus = corpus or _DEFAULT_CORPUS
    q = query or ""
    q_tokens = set(re.findall(r"[a-z0-9]+", q.lower()))
    if not q_tokens:
        return {"ok": False, "error": "empty query", "passages": []}
    scored: List[Dict[str, Any]] = []
    for doc in corpus:
        text = (doc.get("title", "") + " " + doc.get("body", "")).lower()
        tokens = set(re.findall(r"[a-z0-9]+", text))
        overlap = len(q_tokens & tokens)
        if overlap == 0:
            continue
        tf = sum(text.count(t) for t in q_tokens)
        score = overlap + 0.05 * tf
        scored.append({"id": doc["id"], "title": doc.get("title", ""),
                       "body": doc.get("body", ""), "score": score})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return {"ok": True, "passages": scored[:k], "query": q, "matched": len(scored)}


# ---------------------------------------------------------------------------
# code_runner — restricted Python execution
# ---------------------------------------------------------------------------
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Attribute,
    ast.With, ast.AsyncWith, ast.AsyncFor, ast.AsyncFunctionDef,
)
_ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "pow", "print", "range",
    "round", "set", "sorted", "str", "sum", "tuple", "zip",
}


def _safe_builtins() -> Dict[str, Any]:
    import builtins
    return {k: getattr(builtins, k) for k in _ALLOWED_BUILTINS if hasattr(builtins, k)}


async def code_run_python(*, code: str, **kwargs: Any) -> Dict[str, Any]:
    code = (code or "").strip()
    if not code:
        return {"ok": False, "error": "empty code"}
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return {"ok": False, "error": f"SyntaxError: {e}"}
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            return {"ok": False, "error": f"disallowed: {type(node).__name__}"}
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return {"ok": False, "error": f"dunder access disallowed: {node.id}"}
    namespace: Dict[str, Any] = {"__builtins__": _safe_builtins(), "result": None}
    try:
        exec(compile(tree, "<noesis-sandbox>", "exec"), namespace)  # noqa: S102
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "result": namespace.get("result"),
            "namespace_keys": [k for k in namespace if not k.startswith("__") and k != "result"]}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass
class ToolHandle:
    name: str
    description: str
    fn: Any
    cost_estimate: float
    latency_estimate_ms: int
    enabled: bool = True

    async def invoke(self, **kwargs: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            result = await self.fn(**kwargs)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "cost": self.cost_estimate}
        if not isinstance(result, dict):
            result = {"ok": True, "result": result}
        result.setdefault("ok", True)
        result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        result["cost"] = self.cost_estimate
        return result


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolHandle] = {}

    def register(self, handle: ToolHandle) -> None:
        self._tools[handle.name] = handle

    def get(self, name: str) -> ToolHandle:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    def list_enabled(self) -> List[ToolHandle]:
        return [t for t in self._tools.values() if t.enabled]

    def descriptions_for_prompt(self) -> str:
        rows = [f"  - {t.name}: {t.description}" for t in self.list_enabled()]
        return "\n".join(rows) if rows else "  (no tools)"


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolHandle(
        name="calculator",
        description="Evaluate a math expression. Use for arithmetic. Args: {expression: str}.",
        fn=calculator_evaluate, cost_estimate=0.01, latency_estimate_ms=10,
    ))
    reg.register(ToolHandle(
        name="text_search",
        description="Search an in-process text corpus. Args: {query: str, k: int=3}.",
        fn=text_search, cost_estimate=0.02, latency_estimate_ms=30,
    ))
    reg.register(ToolHandle(
        name="code_runner",
        description="Run a short Python snippet (no imports, no attribute access). Args: {code: str}.",
        fn=code_run_python, cost_estimate=0.05, latency_estimate_ms=80,
    ))
    return reg
