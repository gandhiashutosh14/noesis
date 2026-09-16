"""
NOESIS — Gated Teacher Reward (SDAR-translated)
================================================

The framework's second named contribution.

SDAR (Lu et al., 2025) is a token-level training method: it treats
teacher signals as a *gated auxiliary objective* — strengthening
distillation on teacher-endorsed positive-gap tokens, softly attenuating
negative teacher rejections. Crucial insight: trusting every teacher
signal makes multi-turn agents unstable; filtering them through a gate
stabilises training.

We do no gradient training. But the same insight maps cleanly into
preference learning over agent trajectories:

  * Token-level teacher signals → step-level judge signals
  * The sigmoid gate over (judge_confidence + agreement) → here
  * Asymmetric treatment of positive vs negative signals → here
  * Online update of gate parameters from observed outcomes → here

The gate is:

    σ(αⱼ) = sigmoid( α · (confidenceⱼ + agreementⱼ) − β )

with hard floors on confidence and agreement (below them, the signal is
zeroed regardless of σ) and an asymmetric attenuation on negative
signals (a step that the jury scored *below* the trajectory consensus
gets its weight reduced by `negative_attenuation` — SDAR's key
mechanism: never let a noisy teacher push the student in a bad
direction with full force).

α, β are not learned by backprop. They are stored in the SQLite
`gate_params` table per experiment and updated online by a small
gradient-free rule (`update_gate_params`) using the realised reward as
the supervision signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from noesis.config.schema import GatedTeacherConfig, RewardWeights
from noesis.judges.jury import JuryVerdict
from noesis.memory.store import MemoryStore
from noesis.runtime.state import AgentState


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------
@dataclass
class GatedTeacherBreakdown:
    """The gating outputs CMR also wants to see (step_gates) plus the
    final gated-teacher scalar."""
    step_gates: List[float] = field(default_factory=list)     # σ(αⱼ) per step
    step_raw_signals: List[float] = field(default_factory=list)   # signal before gating
    step_gated_signals: List[float] = field(default_factory=list)  # after gating
    final_value: float = 0.0
    alpha_used: float = 0.0
    beta_used: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_gates": self.step_gates,
            "step_raw_signals": self.step_raw_signals,
            "step_gated_signals": self.step_gated_signals,
            "final_value": self.final_value,
            "alpha_used": self.alpha_used,
            "beta_used": self.beta_used,
        }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def compute_gated_teacher(
    state: AgentState,
    verdict: JuryVerdict,
    gate_cfg: GatedTeacherConfig,
    reward_weights: RewardWeights,
    *,
    alpha: float,
    beta: float,
) -> GatedTeacherBreakdown:
    """Compute the gated-teacher signal for a trajectory.

    `alpha`/`beta` are passed in so the caller (the self-improve loop)
    controls when they update. The function itself is pure.
    """
    if not state.steps:
        return GatedTeacherBreakdown(alpha_used=alpha, beta_used=beta)

    consensus_by_id = {row["step_id"]: row for row in verdict.per_step_consensus}
    overall = verdict.consensus_overall

    gates: List[float] = []
    raw_signals: List[float] = []
    gated_signals: List[float] = []

    for s in state.steps:
        row = consensus_by_id.get(s.step_id, {})
        consensus_score = float(row.get("consensus_score", 0.5))
        confidence = float(row.get("confidence", 0.0))
        agreement = float(row.get("agreement", 0.0))

        # 1. Build the raw teacher signal: per-step consensus centered on
        #    the trajectory's overall consensus. Positive = step scored
        #    above the trajectory average; negative = step scored below.
        raw = consensus_score - overall  # ∈ approx [-1, 1]

        # 2. Hard floors — below them, the signal is uninformative
        if confidence < gate_cfg.confidence_floor:
            gate = 0.0
        elif agreement < gate_cfg.agreement_floor:
            gate = 0.0
        else:
            # 3. Sigmoid gate
            gate = _sigmoid(alpha * (confidence + agreement) - beta)

        # 4. Asymmetric attenuation: SDAR's central insight. Negative
        #    signals (steps the jury thinks dragged the trajectory down)
        #    must not push the policy with full force.
        gated = gate * raw
        if raw < 0:
            gated *= gate_cfg.negative_attenuation

        gates.append(gate)
        raw_signals.append(raw)
        gated_signals.append(gated)

    # The final scalar is the gated-signal mean, mapped to [0,1] for
    # composition with the rest of the reward.
    if gated_signals:
        mean_signed = sum(gated_signals) / len(gated_signals)
        # mean_signed ∈ approx [-1, 1]; map to [0,1]
        final_unit = max(0.0, min(1.0, (mean_signed + 1.0) / 2.0))
    else:
        final_unit = 0.5
    final_weighted = reward_weights.gated_teacher * final_unit

    return GatedTeacherBreakdown(
        step_gates=gates,
        step_raw_signals=raw_signals,
        step_gated_signals=gated_signals,
        final_value=final_weighted,
        alpha_used=alpha,
        beta_used=beta,
    )


# ---------------------------------------------------------------------------
# Online update of (α, β)
# ---------------------------------------------------------------------------
def update_gate_params(
    *,
    current_alpha: float,
    current_beta: float,
    breakdown: GatedTeacherBreakdown,
    realised_reward: float,
    target_reward: float,
    gate_cfg: GatedTeacherConfig,
) -> Tuple[float, float]:
    """Gradient-free online update.

    Intuition:
      * If realised_reward > target_reward, the gate was useful — keep its
        shape. Slight nudge to *sharpen* (raise α a touch, lower β a touch).
      * If realised_reward < target_reward, the gate let noise through —
        soften it (lower α, raise β).

    The deltas are scaled by `learning_rate` and clipped so α/β stay in
    sane ranges. Conservative on purpose: NOESIS prefers slow gate drift
    over reactive thrashing.
    """
    error = realised_reward - target_reward
    # Normalise error to [-1, 1]
    error_clipped = max(-1.0, min(1.0, error * 2.0))

    lr = gate_cfg.learning_rate
    delta_alpha = lr * error_clipped * 1.0
    delta_beta = -lr * error_clipped * 1.0

    new_alpha = max(0.5, min(20.0, current_alpha + delta_alpha))
    new_beta = max(-5.0, min(10.0, current_beta + delta_beta))
    return new_alpha, new_beta


# ---------------------------------------------------------------------------
# Memory glue: load/persist (α,β) bound to an experiment
# ---------------------------------------------------------------------------
def load_or_init_gate(memory: MemoryStore, experiment_name: str,
                      gate_cfg: GatedTeacherConfig) -> Tuple[float, float]:
    """Read (α,β) from the store; initialise them if absent."""
    found = memory.get_gate(experiment_name)
    if found is not None:
        return found
    a, b = gate_cfg.initial_alpha, gate_cfg.initial_beta
    memory.set_gate(experiment_name, a, b)
    return a, b


def persist_gate(memory: MemoryStore, experiment_name: str,
                 alpha: float, beta: float) -> None:
    memory.set_gate(experiment_name, alpha, beta)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    # Numerically stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
