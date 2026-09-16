"""
NOESIS — Cognitive Momentum Reward
==================================

The framework's first named contribution.

Most reward functions measure where the trajectory *ended*. Cognitive
Momentum measures whether the trajectory *moved productively toward truth
under uncertainty*.

Formally, for trajectory τ with steps s₁..sₙ:

    CM(τ) = Σⱼ σ(αⱼ) · Δhⱼ · κⱼ
            − Σⱼ 𝟙[thrash(sⱼ)] · λ_thrash
            + 𝟙[productive_recovery(τ)] · λ_rec
            + calibration_bonus(τ)

where:

  σ(αⱼ)      — the SDAR-inspired gate over judge signals at step j
                (computed externally; passed in as `step_gates`)
  Δhⱼ        — entropy reduction over the agent's answer distribution
                at step j (proxied here by judge-consensus delta and
                confidence delta — both per-step features we have)
  κⱼ         — depth-appropriateness: penalises deep reasoning on
                trivial steps and shallow reasoning on hard steps
  thrash(sⱼ) — step j is a backtrack NOT preceded by a reflection step;
                i.e., the agent revised without first acknowledging why
  productive_recovery(τ) — there exists j s.t. step j is a reflect/revise
                AND a downstream step shows measurable improvement
  calibration_bonus(τ) — small reward when the agent's self-reported
                confidence aligns with the jury's actual consensus

The function does *not* require differentiability — NOESIS does no
gradient training. CM enters the system as a shaping signal: it is
combined with the gated teacher reward and other components in
`rewards/shaping.py` to form the scalar that updates the policy.

The intuition is borrowed from the "credit assignment" problem in
reinforcement learning: sparse end-of-episode feedback fails on
multi-step agents because it never says *which* step earned or lost the
reward. CM distributes credit *across* the trajectory rather than
concentrating it at the final step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from noesis.config.schema import RewardWeights
from noesis.judges.jury import JuryVerdict
from noesis.runtime.state import AgentState, StepKind


@dataclass
class CognitiveMomentumBreakdown:
    """Per-component breakdown — surfaced for explainability + policy update."""
    momentum_term: float = 0.0
    depth_appropriateness_term: float = 0.0
    productive_recovery_term: float = 0.0
    calibration_term: float = 0.0
    thrash_penalty_term: float = 0.0
    final_value: float = 0.0
    # Auxiliary signals stored for the policy layer
    per_step_momentum: List[float] = field(default_factory=list)
    per_step_kappa: List[float] = field(default_factory=list)
    thrash_step_ids: List[str] = field(default_factory=list)
    productive_recovery_observed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "momentum_term": self.momentum_term,
            "depth_appropriateness_term": self.depth_appropriateness_term,
            "productive_recovery_term": self.productive_recovery_term,
            "calibration_term": self.calibration_term,
            "thrash_penalty_term": self.thrash_penalty_term,
            "final_value": self.final_value,
            "per_step_momentum": self.per_step_momentum,
            "per_step_kappa": self.per_step_kappa,
            "thrash_step_ids": self.thrash_step_ids,
            "productive_recovery_observed": self.productive_recovery_observed,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_cognitive_momentum(
    state: AgentState,
    verdict: JuryVerdict,
    step_gates: List[float],
    weights: RewardWeights,
) -> CognitiveMomentumBreakdown:
    """Compute CMR over a completed trajectory.

    Parameters
    ----------
    state : AgentState
        The finished trajectory.
    verdict : JuryVerdict
        Output of the jury (used for entropy-proxy + calibration term).
    step_gates : list of floats in [0,1]
        σ(αⱼ) per step, computed by `rewards/gated_teacher.py`. Length
        equal to len(state.steps). The gate attenuates step-level signal
        before it enters CMR.
    weights : RewardWeights
        Component weights from config.

    Returns
    -------
    CognitiveMomentumBreakdown
        All sub-terms and the final scalar.
    """
    n_steps = len(state.steps)
    if n_steps == 0:
        return CognitiveMomentumBreakdown()

    # Pad/truncate gates defensively
    if len(step_gates) < n_steps:
        step_gates = list(step_gates) + [0.5] * (n_steps - len(step_gates))
    elif len(step_gates) > n_steps:
        step_gates = step_gates[:n_steps]

    # 1. Momentum term: Σⱼ σ(αⱼ) · Δhⱼ · κⱼ
    deltas = _entropy_reduction_proxy(state, verdict)
    kappas = _depth_appropriateness_per_step(state)
    momentum_components = [
        float(g) * float(d) * float(k)
        for g, d, k in zip(step_gates, deltas, kappas)
    ]
    momentum_sum = sum(momentum_components)
    # Normalize to [0,1] by step count — keeps the term comparable across
    # short and long trajectories. We divide by n_steps so the average step
    # contributes ~ 1/n; with σ,Δh,κ each in [0,1] that's a sane bound.
    momentum_norm = momentum_sum / max(1, n_steps)

    # 2. Depth-appropriateness term (separately surfaced for interpretability)
    depth_term = sum(kappas) / max(1, n_steps)

    # 3. Productive recovery
    productive_recovery = _detect_productive_recovery(state, verdict)
    recovery_term = 1.0 if productive_recovery else 0.0

    # 4. Calibration: align self-confidence with consensus_overall
    calibration_term = _calibration_score(state, verdict)

    # 5. Thrash penalty
    thrash_step_ids = _detect_thrash_steps(state)
    thrash_term = float(len(thrash_step_ids)) / max(1, n_steps)

    # Compose final CM scalar — this is the *CM-only* value, not the
    # blended reward. Blending with other components happens in shaping.py.
    final = (
        weights.momentum * momentum_norm
        + weights.depth_appropriateness * depth_term
        + weights.productive_recovery * recovery_term
        + weights.calibration * calibration_term
        - weights.thrash_penalty * thrash_term
    )

    return CognitiveMomentumBreakdown(
        momentum_term=weights.momentum * momentum_norm,
        depth_appropriateness_term=weights.depth_appropriateness * depth_term,
        productive_recovery_term=weights.productive_recovery * recovery_term,
        calibration_term=weights.calibration * calibration_term,
        thrash_penalty_term=weights.thrash_penalty * thrash_term,
        final_value=final,
        per_step_momentum=momentum_components,
        per_step_kappa=kappas,
        thrash_step_ids=thrash_step_ids,
        productive_recovery_observed=productive_recovery,
    )


# ---------------------------------------------------------------------------
# Per-step features
# ---------------------------------------------------------------------------
def _entropy_reduction_proxy(state: AgentState, verdict: JuryVerdict) -> List[float]:
    """Proxy Δhⱼ.

    True entropy reduction would require sampling N candidate answers per
    step and measuring distributional change. We can't afford that. Two
    correlated, cheaper signals replace it:

      a) Per-step jury consensus score (high score = the agent moved
         toward a defensible answer at that step).
      b) The step's own confidence rise vs. the prior step.

    Δhⱼ ≈ 0.5 · jury_consensus_j + 0.5 · clip(conf_j - conf_{j-1}, -1, 1) → [0,1]
    """
    consensus_by_id = {row["step_id"]: row["consensus_score"]
                       for row in verdict.per_step_consensus}
    deltas: List[float] = []
    prev_conf = 0.5
    for s in state.steps:
        consensus = float(consensus_by_id.get(s.step_id, 0.5))
        conf_delta = max(-1.0, min(1.0, s.confidence - prev_conf))
        # Map conf_delta from [-1,1] to [0,1]
        conf_delta_unit = (conf_delta + 1.0) / 2.0
        delta = 0.5 * consensus + 0.5 * conf_delta_unit
        deltas.append(max(0.0, min(1.0, delta)))
        prev_conf = s.confidence
    return deltas


def _depth_appropriateness_per_step(state: AgentState) -> List[float]:
    """κⱼ — penalises mismatched reasoning depth.

    Approach: estimate per-step "warranted depth" from the step's kind and
    the task complexity, then compare to the actual reasoning depth
    declared on the state. Mismatch reduces κⱼ.

    Returns one κⱼ ∈ [0,1] per step.
    """
    # Map reasoning_depth string to a 0..2 scale
    depth_map = {"shallow": 0, "standard": 1, "deep": 2}
    declared = depth_map.get(state.reasoning_depth, 1)

    out: List[float] = []
    for s in state.steps:
        # Heuristic per step kind: planning + synthesis + reflect want
        # depth >= 1; tool calls are fine at depth 0; errors should not
        # have been at depth 0.
        if s.kind in (StepKind.PLAN, StepKind.SYNTHESIZE, StepKind.REFLECT):
            warranted = 1 if state.complexity < 0.7 else 2
        elif s.kind == StepKind.REVISE:
            warranted = 2  # revising warrants depth
        elif s.kind == StepKind.TOOL_CALL:
            warranted = 0
        else:
            warranted = 1
        # κ = 1 when warranted == declared; decays with |distance|
        distance = abs(warranted - declared)
        kappa = max(0.0, 1.0 - 0.4 * distance)
        # Errors at any depth lose appropriateness
        if s.error:
            kappa *= 0.5
        out.append(kappa)
    return out


def _detect_productive_recovery(state: AgentState, verdict: JuryVerdict) -> bool:
    """A productive_recovery is observed when:
       1) the trajectory contains a REFLECT or REVISE step at index k, AND
       2) some step at index > k achieves a higher per-step consensus than
          the steps preceding the reflection.
    """
    if not state.steps:
        return False
    consensus_by_id = {row["step_id"]: row["consensus_score"]
                       for row in verdict.per_step_consensus}

    reflect_indices = [
        i for i, s in enumerate(state.steps)
        if s.kind in (StepKind.REFLECT, StepKind.REVISE)
    ]
    if not reflect_indices:
        return False

    first_reflect = reflect_indices[0]
    if first_reflect >= len(state.steps) - 1:
        return False

    pre_consensus = [consensus_by_id.get(state.steps[i].step_id, 0.5)
                     for i in range(first_reflect)]
    post_consensus = [consensus_by_id.get(state.steps[i].step_id, 0.5)
                      for i in range(first_reflect + 1, len(state.steps))]
    if not pre_consensus or not post_consensus:
        return False
    pre_mean = sum(pre_consensus) / len(pre_consensus)
    post_max = max(post_consensus)
    return post_max > pre_mean + 0.05  # require meaningful improvement


def _calibration_score(state: AgentState, verdict: JuryVerdict) -> float:
    """Reward when self-reported confidence aligns with jury consensus.

    Returns 1.0 for a perfectly aligned trajectory, 0.0 for one that is
    catastrophically miscalibrated.
    """
    if not state.steps:
        return 0.5
    # Use the final answer's confidence (state.final_confidence) and the
    # jury's overall consensus as the two scalars to compare.
    gap = abs(state.final_confidence - verdict.consensus_overall)
    return max(0.0, 1.0 - 2.0 * gap)  # 0.5 gap → 0; 0 gap → 1


def _detect_thrash_steps(state: AgentState) -> List[str]:
    """Thrash = a REVISE step (or a TOOL_CALL with a `revises_step_id`)
    that was NOT preceded by a REFLECT step within the prior 2 steps.

    Backtracking without reflection is the failure mode CMR is built to
    suppress — it implies random retry rather than diagnostic recovery.
    """
    thrash_ids: List[str] = []
    for i, s in enumerate(state.steps):
        is_revise = s.kind == StepKind.REVISE or s.revises_step_id is not None
        if not is_revise:
            continue
        window = state.steps[max(0, i - 2): i]
        has_reflect = any(prev.kind == StepKind.REFLECT for prev in window)
        if not has_reflect:
            thrash_ids.append(s.step_id)
    return thrash_ids
