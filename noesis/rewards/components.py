"""
NOESIS — Reward Components
==========================

The "supporting cast" of sub-rewards that compose with Cognitive Momentum
and the Gated Teacher signal:

  * correctness   — match against expected_answer if present; else proxy
                    by consensus_overall × (1 − contested_penalty)
  * efficiency    — cost & latency vs a complexity-aware budget baseline
  * novelty       — Jaccard distance vs prior trajectories on same task type
  * safety        — derived from the safety rubric dimension; gates the
                    whole reward (a safety hit zeroes the final scalar)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from noesis.judges.jury import JuryVerdict
from noesis.memory.store import MemoryStore
from noesis.runtime.state import AgentState


@dataclass
class ComponentRewards:
    correctness: float = 0.0          # [0,1]
    efficiency: float = 0.0           # [0,1]
    novelty: float = 0.0              # [0,1]
    safety_pass: bool = True          # False → zero the whole reward
    safety_raw: float = 1.0           # raw safety rubric score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correctness": self.correctness,
            "efficiency": self.efficiency,
            "novelty": self.novelty,
            "safety_pass": self.safety_pass,
            "safety_raw": self.safety_raw,
        }


# ---------------------------------------------------------------------------
def compute_components(
    state: AgentState,
    verdict: JuryVerdict,
    memory: MemoryStore,
    *,
    safety_threshold: float = 0.4,
) -> ComponentRewards:
    return ComponentRewards(
        correctness=_correctness(state, verdict),
        efficiency=_efficiency(state),
        novelty=_novelty(state, memory),
        safety_raw=float(verdict.consensus_rubric.get("safety", 1.0)),
        safety_pass=float(verdict.consensus_rubric.get("safety", 1.0)) >= safety_threshold,
    )


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
def _correctness(state: AgentState, verdict: JuryVerdict) -> float:
    """If we have an expected answer, compare against it. Otherwise fall
    back to the jury's correctness rubric dimension, with a small penalty
    when the verdict is contested.
    """
    if state.expected_answer:
        return _answer_match(state.final_answer or "", state.expected_answer)
    raw = float(verdict.consensus_rubric.get("correctness", 0.5))
    penalty = 0.15 if verdict.contested else 0.0
    return max(0.0, min(1.0, raw - penalty))


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _answer_match(predicted: str, expected: str) -> float:
    """Tiered matcher: numeric first (high precision), then normalized
    substring, finally token Jaccard."""
    p, e = predicted.strip(), expected.strip()
    if not p:
        return 0.0
    if not e:
        return 0.0

    # 1. Numeric match
    p_nums = _NUM_RE.findall(p)
    e_nums = _NUM_RE.findall(e)
    if e_nums:
        if p_nums and abs(float(p_nums[0]) - float(e_nums[0])) < 1e-6:
            return 1.0
        # Numeric mismatch is informative — heavy penalty
        return 0.1

    # 2. Normalized substring containment
    p_norm = _normalize(p)
    e_norm = _normalize(e)
    if e_norm in p_norm or p_norm in e_norm:
        return 0.85

    # 3. Token Jaccard
    p_tokens = set(p_norm.split())
    e_tokens = set(e_norm.split())
    if not e_tokens:
        return 0.0
    inter = len(p_tokens & e_tokens)
    union = len(p_tokens | e_tokens)
    return inter / union if union else 0.0


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower())


# ---------------------------------------------------------------------------
# efficiency
# ---------------------------------------------------------------------------
def _efficiency(state: AgentState) -> float:
    """Reward fast, cheap trajectories — but baseline-adjusted for
    complexity so a deep-reasoning task isn't penalised for using budget.

    Baselines (USD cost, ms latency) are linear in complexity:
        cost_budget    = 0.10 + 0.40 · complexity
        latency_budget = 5000 + 25000 · complexity

    score = 0.5 · cost_ratio_inv + 0.5 · latency_ratio_inv,
        clipped to [0, 1].
    """
    if not state.steps:
        return 0.5
    total_cost = sum(s.cost for s in state.steps)
    total_latency = sum(s.latency_ms for s in state.steps)
    cost_budget = 0.10 + 0.40 * state.complexity
    latency_budget = 5000 + 25000 * state.complexity
    cost_ratio = total_cost / cost_budget if cost_budget > 0 else 1.0
    lat_ratio = total_latency / latency_budget if latency_budget > 0 else 1.0
    # Invert: under-budget = high efficiency
    cost_score = max(0.0, 1.0 - cost_ratio)
    lat_score = max(0.0, 1.0 - lat_ratio)
    return 0.5 * cost_score + 0.5 * lat_score


# ---------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------
def _novelty(state: AgentState, memory: MemoryStore) -> float:
    """Jaccard *distance* of this trajectory's step-kind+tool fingerprint
    against the most recent trajectories on the same task type.

    Returns ~1 if no prior trajectories exist (everything is novel) and
    ~0 if this trajectory exactly matches a prior one.
    """
    fingerprint = _trajectory_fingerprint(state)
    if not fingerprint:
        return 0.5

    priors = memory.trajectories_for_task(state.task_id)
    # Only consider trajectories other than this one
    other = [t for t in priors if t.get("trajectory_id") != state.trajectory_id]
    if not other:
        # Look at recent task-type-level trajectories instead — but we don't
        # have an index for that here; the registry is per-task. Default
        # to a moderate novelty score so the system doesn't over-reward
        # first attempts.
        return 0.7

    sims: List[float] = []
    for t in other[:10]:
        prior_fp = _trajectory_fingerprint_from_dict(t)
        if not prior_fp:
            continue
        inter = len(fingerprint & prior_fp)
        union = len(fingerprint | prior_fp)
        sim = inter / union if union else 0.0
        sims.append(sim)
    if not sims:
        return 0.7
    max_sim = max(sims)
    return max(0.0, 1.0 - max_sim)  # distance


def _trajectory_fingerprint(state: AgentState) -> set:
    return {f"{s.kind.value}:{s.chosen_tool or '-'}" for s in state.steps}


def _trajectory_fingerprint_from_dict(t: Dict[str, Any]) -> set:
    steps = t.get("steps", []) or []
    if isinstance(steps, str):
        return set()
    fp: set = set()
    for s in steps:
        if not isinstance(s, dict):
            continue
        fp.add(f"{s.get('kind', '-')}:{s.get('chosen_tool', '-') or '-'}")
    return fp
