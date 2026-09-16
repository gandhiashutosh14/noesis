"""
NOESIS — Reward Shaping
=======================

The composition layer: all sub-rewards in, one scalar out.

The composition is intentionally explicit (no learned weights, no
black-box blending). Cognitive Momentum + Gated Teacher carry the
identity of the system; correctness/efficiency/novelty are the
supporting weights; safety is the kill-switch.

Persists the result to memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

from noesis.config.schema import GatedTeacherConfig, RewardWeights
from noesis.judges.jury import JuryVerdict
from noesis.memory.store import MemoryStore
from noesis.rewards.cognitive_momentum import (
    CognitiveMomentumBreakdown,
    compute_cognitive_momentum,
)
from noesis.rewards.components import ComponentRewards, compute_components
from noesis.rewards.gated_teacher import (
    GatedTeacherBreakdown,
    compute_gated_teacher,
    load_or_init_gate,
)
from noesis.runtime.state import AgentState


@dataclass
class ShapedReward:
    """The complete reward output for a single trajectory."""
    trajectory_id: str
    task_id: str
    task_type: str
    cognitive_momentum: CognitiveMomentumBreakdown
    gated_teacher: GatedTeacherBreakdown
    components: ComponentRewards
    final_scalar: float
    safety_violation: bool = False
    notes: str = ""

    def to_persistence_dict(self) -> Dict[str, Any]:
        return {
            "reward_id": str(uuid4()),
            "trajectory_id": self.trajectory_id,
            "cognitive_momentum": self.cognitive_momentum.final_value,
            "gated_teacher": self.gated_teacher.final_value,
            "components": {
                "cognitive_momentum": self.cognitive_momentum.to_dict(),
                "gated_teacher": self.gated_teacher.to_dict(),
                "components": self.components.to_dict(),
                "notes": self.notes,
                "safety_violation": self.safety_violation,
            },
            "final_scalar": self.final_scalar,
        }


# ---------------------------------------------------------------------------
def shape_reward(
    state: AgentState,
    verdict: JuryVerdict,
    memory: MemoryStore,
    weights: RewardWeights,
    gate_cfg: GatedTeacherConfig,
    experiment_name: str,
) -> ShapedReward:
    """Compose all sub-rewards into one scalar and persist.

    Order matters:
      1. Compute gated_teacher first — it produces the per-step gates
         that Cognitive Momentum consumes.
      2. Compute CMR with those gates.
      3. Compute the supporting components.
      4. Compose. If safety fails, zero everything (but keep the
         breakdown for diagnostics).
      5. Persist.
    """
    # 1. Gate parameters from memory (per experiment), initialised lazily
    alpha, beta = load_or_init_gate(memory, experiment_name, gate_cfg)

    # 2. Gated-teacher signal — produces gates that CMR uses
    gt = compute_gated_teacher(
        state, verdict, gate_cfg, weights, alpha=alpha, beta=beta,
    )

    # 3. Cognitive Momentum, consuming the gates
    cm = compute_cognitive_momentum(state, verdict, gt.step_gates, weights)

    # 4. Supporting components
    comps = compute_components(state, verdict, memory)

    # 5. Compose
    safety_violation = not comps.safety_pass
    if safety_violation:
        # The safety_violation_penalty is the magnitude of zeroing; we
        # treat any safety failure as a hard zero of the positive reward,
        # with a small explanatory note.
        final_scalar = 0.0
        notes = "safety violation: reward zeroed"
    else:
        positive = (
            weights.correctness * comps.correctness
            + weights.efficiency * comps.efficiency
            + weights.novelty * comps.novelty
            + cm.final_value         # already weights-applied internally
            + gt.final_value         # already weights-applied internally
        )
        final_scalar = max(0.0, min(1.0, positive))
        notes = ""

    reward = ShapedReward(
        trajectory_id=state.trajectory_id,
        task_id=state.task_id,
        task_type=state.task_type,
        cognitive_momentum=cm,
        gated_teacher=gt,
        components=comps,
        final_scalar=final_scalar,
        safety_violation=safety_violation,
        notes=notes,
    )

    # 6. Persist
    memory.upsert_reward(reward.to_persistence_dict())
    return reward
