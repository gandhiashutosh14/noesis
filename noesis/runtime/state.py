"""
NOESIS — Agent State
====================

The state space the runtime observes and the action space it chooses from.
These are explicit dataclasses, not implicit dictionaries, so:

  * The runtime can serialize state into trajectories with no information loss.
  * The judges can see exactly what the agent saw at decision time.
  * The reward function can compute Δ-features cleanly (entropy reduction,
    depth-appropriateness, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class StepKind(str, Enum):
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    REFLECT = "reflect"
    REVISE = "revise"
    SYNTHESIZE = "synthesize"
    ERROR = "error"


class FailureMode(str, Enum):
    NONE = "none"
    TOOL_ERROR = "tool_error"
    LLM_REFUSED = "llm_refused"
    PARSING_ERROR = "parsing_error"
    BUDGET_EXCEEDED = "budget_exceeded"
    REFLECTION_LOOP = "reflection_loop"


# ---------------------------------------------------------------------------
# Trajectory step
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryStep:
    step_id: str
    kind: StepKind
    description: str
    started_at: str
    finished_at: str
    latency_ms: int
    cost: float
    # Per-step decision context — captured at the time the agent chose to take this step.
    chosen_action: str = ""               # name of the action enum value
    chosen_tool: Optional[str] = None
    chosen_model: Optional[str] = None
    reasoning_depth: Optional[str] = None
    # Outputs
    output: Any = None                    # tool result, llm text, or structured payload
    error: Optional[str] = None
    failure_mode: FailureMode = FailureMode.NONE
    # Self-reported confidence at the time
    confidence: float = 0.5
    # Counter-step memory: did this step revise an earlier one?
    revises_step_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "kind": self.kind.value,
            "description": self.description,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "chosen_action": self.chosen_action,
            "chosen_tool": self.chosen_tool,
            "chosen_model": self.chosen_model,
            "reasoning_depth": self.reasoning_depth,
            "output": self.output,
            "error": self.error,
            "failure_mode": self.failure_mode.value,
            "confidence": self.confidence,
            "revises_step_id": self.revises_step_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryStep":
        return cls(
            step_id=d["step_id"],
            kind=StepKind(d["kind"]),
            description=d.get("description", ""),
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            latency_ms=int(d.get("latency_ms", 0)),
            cost=float(d.get("cost", 0.0)),
            chosen_action=d.get("chosen_action", ""),
            chosen_tool=d.get("chosen_tool"),
            chosen_model=d.get("chosen_model"),
            reasoning_depth=d.get("reasoning_depth"),
            output=d.get("output"),
            error=d.get("error"),
            failure_mode=FailureMode(d.get("failure_mode", "none")),
            confidence=float(d.get("confidence", 0.5)),
            revises_step_id=d.get("revises_step_id"),
        )


# ---------------------------------------------------------------------------
# AgentState — what the runtime carries through execution
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    # Task identity
    task_id: str
    task_type: str
    task_description: str
    expected_answer: Optional[str] = None
    complexity: float = 0.5               # [0,1]

    # Strategy in force for this run
    strategy_id: str = "default"
    prompt_template_id: str = "planner.default"
    prompt_version: str = "1.0.0"
    reasoning_depth: str = "standard"     # ReasoningDepth value
    model_endpoint: str = ""

    # Memory / context
    retrieved_reflections: List[Dict[str, Any]] = field(default_factory=list)

    # Live execution
    plan: Optional[Dict[str, Any]] = None
    steps: List[TrajectoryStep] = field(default_factory=list)
    current_step_index: int = 0
    reflection_passes: int = 0

    # Budgets
    token_budget_remaining: int = 8000
    cost_budget_remaining: float = 0.50    # USD, soft cap
    latency_budget_remaining_ms: int = 60000

    # Outcome
    final_answer: Optional[str] = None
    status: str = "running"               # "running" | "complete" | "error" | "aborted"
    failure_mode: FailureMode = FailureMode.NONE
    final_confidence: float = 0.5

    # Metadata
    started_at: str = ""
    finished_at: str = ""

    # Internal: trajectory id stable across the run
    trajectory_id: str = field(default_factory=lambda: str(uuid4()))

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        self.cost_budget_remaining -= step.cost
        self.latency_budget_remaining_ms -= step.latency_ms

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "expected_answer": self.expected_answer,
            "complexity": self.complexity,
            "strategy_id": self.strategy_id,
            "prompt_template_id": self.prompt_template_id,
            "prompt_version": self.prompt_version,
            "reasoning_depth": self.reasoning_depth,
            "model_endpoint": self.model_endpoint,
            "retrieved_reflections": self.retrieved_reflections,
            "plan": self.plan,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "reflection_passes": self.reflection_passes,
            "token_budget_remaining": self.token_budget_remaining,
            "cost_budget_remaining": self.cost_budget_remaining,
            "latency_budget_remaining_ms": self.latency_budget_remaining_ms,
            "final_answer": self.final_answer,
            "status": self.status,
            "failure_mode": self.failure_mode.value,
            "final_confidence": self.final_confidence,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "trajectory_id": self.trajectory_id,
        }
