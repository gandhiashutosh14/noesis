"""
NOESIS — Strategy Book
======================

A *strategy* is a bundle of decisions the runtime defers to policy:

    (prompt_template_id, prompt_version, reasoning_depth,
     model_endpoint, tool_preference)

The book exposes a small, finite set of strategies. The selector
(`policy/selector.py`) picks one per (task_type, complexity_bucket)
based on stored stats.

Why finite: with infinite strategies, exploration becomes futile. Finite
strategies let Thompson sampling actually converge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Strategy:
    """A complete recipe the runtime can execute."""
    strategy_id: str
    prompt_template_id: str
    prompt_version: str
    reasoning_depth: str                   # "shallow" | "standard" | "deep"
    model_endpoint: str
    tool_preference: tuple = ()            # ordered tuple of tool names

    def signature(self) -> str:
        return (f"{self.strategy_id}|{self.prompt_template_id}@{self.prompt_version}|"
                f"{self.reasoning_depth}|{self.model_endpoint}|"
                f"tools={'-'.join(self.tool_preference)}")


# ---------------------------------------------------------------------------
class StrategyBook:
    """In-memory catalogue, seeded from sensible defaults at startup."""

    def __init__(self) -> None:
        self._strategies: Dict[str, Strategy] = {}

    def register(self, strategy: Strategy) -> None:
        self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> Strategy:
        return self._strategies[strategy_id]

    def all(self) -> List[Strategy]:
        return list(self._strategies.values())

    def __contains__(self, strategy_id: str) -> bool:
        return strategy_id in self._strategies

    def __len__(self) -> int:
        return len(self._strategies)


# ---------------------------------------------------------------------------
def build_default_strategy_book(default_endpoint: str = "mock_default") -> StrategyBook:
    """The seed set of strategies. They differ along the axes the policy
    update mechanism can actually move: depth, model, tool preference,
    prompt version.
    """
    book = StrategyBook()

    book.register(Strategy(
        strategy_id="baseline_shallow",
        prompt_template_id="planner.default",
        prompt_version="1.0.0",
        reasoning_depth="shallow",
        model_endpoint=default_endpoint,
        tool_preference=("calculator", "text_search"),
    ))
    book.register(Strategy(
        strategy_id="baseline_standard",
        prompt_template_id="planner.default",
        prompt_version="1.0.0",
        reasoning_depth="standard",
        model_endpoint=default_endpoint,
        tool_preference=("calculator", "text_search", "code_runner"),
    ))
    book.register(Strategy(
        strategy_id="deep_with_reflection",
        prompt_template_id="planner.default",
        prompt_version="1.0.0",
        reasoning_depth="deep",
        model_endpoint=default_endpoint,
        tool_preference=("text_search", "code_runner", "calculator"),
    ))
    book.register(Strategy(
        strategy_id="search_first",
        prompt_template_id="planner.default",
        prompt_version="1.0.0",
        reasoning_depth="standard",
        model_endpoint=default_endpoint,
        tool_preference=("text_search", "calculator"),
    ))
    book.register(Strategy(
        strategy_id="compute_first",
        prompt_template_id="planner.default",
        prompt_version="1.0.0",
        reasoning_depth="standard",
        model_endpoint=default_endpoint,
        tool_preference=("calculator", "code_runner"),
    ))

    return book
