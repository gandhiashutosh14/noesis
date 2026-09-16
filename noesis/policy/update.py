"""
NOESIS — Policy Update
======================

The end-of-task hook. Given (state, verdict, reward, decision), update:

  * strategy_stats          — Beta-distribution params + running reward stats
  * prompt_versions         — performance_score of the prompt used
  * gate_params             — (α, β) of the sigmoid gate, via the
                              gradient-free rule in rewards/gated_teacher.py
  * judge_reliability       — running Brier score per (persona, task_type)
  * preference_pairs        — formed via policy/preference_store.py
  * reflection_notes        — extracted from the adversarial critique

Each of these is a small, deterministic operation. The function is
composed not because the steps need each other but because they all
fire after the same event: a trajectory has been evaluated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

from noesis.config.schema import GatedTeacherConfig, NoesisConfig, RLConfig
from noesis.judges.jury import JuryVerdict
from noesis.memory.store import MemoryStore
from noesis.policy.preference_store import form_preference_pairs
from noesis.policy.selector import SelectionDecision
from noesis.rewards.gated_teacher import (
    persist_gate,
    update_gate_params,
)
from noesis.rewards.shaping import ShapedReward
from noesis.runtime.state import AgentState


@dataclass
class PolicyUpdateSummary:
    """What changed. Surfaced for logging + the regression guard."""
    strategy_id: str
    task_type: str
    reward_final: float
    new_alpha: float
    new_beta: float
    preference_pairs_formed: int
    reflection_notes_added: int


# ---------------------------------------------------------------------------
def update_policy_after_task(
    *,
    state: AgentState,
    verdict: JuryVerdict,
    reward: ShapedReward,
    decision: SelectionDecision,
    config: NoesisConfig,
    memory: MemoryStore,
    recent_trajectory_ids_same_type: List[str],
) -> PolicyUpdateSummary:
    """The single call the self-improve loop makes after each task."""

    # 1. Strategy stats. "Success" defined as final_scalar above a soft floor
    #    that rises slowly with attempt count — a curriculum effect: early
    #    on, 0.4 is OK; over time, 0.5+ is required to count as a win.
    success_floor = 0.4 + min(0.15, len(recent_trajectory_ids_same_type) * 0.005)
    success = reward.final_scalar >= success_floor
    memory.update_strategy_stats(
        decision.strategy.strategy_id,
        state.task_type,
        reward.final_scalar,
        success,
    )

    # 2. Prompt version performance — running mean
    memory.update_prompt_performance(
        decision.strategy.prompt_template_id,
        decision.strategy.prompt_version,
        reward.final_scalar,
    )

    # 3. Gate parameters — only update when we have agreement signal
    new_alpha, new_beta = update_gate_params(
        current_alpha=reward.gated_teacher.alpha_used,
        current_beta=reward.gated_teacher.beta_used,
        breakdown=reward.gated_teacher,
        realised_reward=reward.final_scalar,
        target_reward=0.6,        # informal baseline; calibrated empirically
        gate_cfg=config.gate,
    )
    persist_gate(memory, config.experiment.experiment_name, new_alpha, new_beta)

    # 4. Judge reliability — per (persona, task_type). We use a tiny Brier
    #    contribution per judge: (their score - realised_reward) squared.
    for js in verdict.per_judge:
        brier = (js.overall_score - reward.final_scalar) ** 2
        memory.update_judge_reliability(js.persona_id, state.task_type, brier)

    # 5. Preference pairs over recent trajectories on this task type
    pair_ids: List = []
    if recent_trajectory_ids_same_type:
        pair_ids = form_preference_pairs(
            memory=memory,
            task_type=state.task_type,
            recent_trajectory_ids=recent_trajectory_ids_same_type,
            rl_cfg=config.rl,
        )

    # 6. Reflection notes — only when the adversarial critique looks useful
    notes_added = 0
    crit = (verdict.adversarial_critique or "").strip()
    if crit and len(crit) > 40 and reward.final_scalar < 0.6:
        failure_mode = _classify_failure_mode(crit, state)
        memory.insert_reflection({
            "note_id": str(uuid4()),
            "task_type": state.task_type,
            "trigger_failure_mode": failure_mode,
            "body": crit[:1200],
            "usefulness_score": 0.0,        # rises as it gets retrieved + helps
            "retrieve_count": 0,
        })
        notes_added = 1

    return PolicyUpdateSummary(
        strategy_id=decision.strategy.strategy_id,
        task_type=state.task_type,
        reward_final=reward.final_scalar,
        new_alpha=new_alpha,
        new_beta=new_beta,
        preference_pairs_formed=len(pair_ids),
        reflection_notes_added=notes_added,
    )


# ---------------------------------------------------------------------------
# Failure-mode heuristic — keep it transparent, no LLM call.
# ---------------------------------------------------------------------------
_FAILURE_PATTERNS = [
    ("tool_misuse", re.compile(r"\b(wrong tool|misused tool|incorrect arg)", re.I)),
    ("hallucination", re.compile(r"\b(hallucinat|invented|unsupported|fabricat)", re.I)),
    ("incomplete", re.compile(r"\b(incomplete|did not finish|missed|skipped step)", re.I)),
    ("over_reasoning", re.compile(r"\b(over[- ]?engineered|too verbose|over[- ]?thought)", re.I)),
    ("under_reasoning", re.compile(r"\b(shallow|surface[- ]?level|no analysis)", re.I)),
    ("budget_blown", re.compile(r"\b(too slow|over budget|expensive|cost)", re.I)),
]


def _classify_failure_mode(critique: str, state: AgentState) -> str:
    for label, pat in _FAILURE_PATTERNS:
        if pat.search(critique):
            return label
    if state.failure_mode.value != "none":
        return state.failure_mode.value
    return "unspecified"
