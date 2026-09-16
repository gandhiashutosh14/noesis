"""
NOESIS — Preference Store
=========================

DPO-style preference learning. For each task type, we look at the set of
trajectories produced and form (winner, loser) pairs where the reward
gap exceeds `preference_margin`. These pairs are stored and consumed by
the prompt-evolution and policy-update layers.

We do *not* run DPO training (no weights). The pairs serve as:

  * Targets for prompt mutation — when a losing prompt has a stable
    pattern of failure, the mutator generates a variant that addresses
    it directly.
  * Evidence for strategy selection — strategies that produce more
    winners get more weight (via the Beta posteriors in `strategy_stats`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

from noesis.config.schema import RLConfig
from noesis.memory.store import MemoryStore


@dataclass
class PreferencePair:
    pair_id: str
    task_type: str
    winner_trajectory_id: str
    loser_trajectory_id: str
    reward_gap: float

    def to_dict(self) -> Dict:
        return {
            "pair_id": self.pair_id,
            "task_type": self.task_type,
            "winner_trajectory_id": self.winner_trajectory_id,
            "loser_trajectory_id": self.loser_trajectory_id,
            "reward_gap": self.reward_gap,
        }


# ---------------------------------------------------------------------------
def form_preference_pairs(
    *,
    memory: MemoryStore,
    task_type: str,
    recent_trajectory_ids: List[str],
    rl_cfg: RLConfig,
    max_pairs: int = 8,
) -> List[PreferencePair]:
    """Look at the recent trajectories on this task type, build winner/
    loser pairs, persist them.

    Only forms pairs when both trajectories have rewards persisted and
    the gap exceeds `preference_margin`.
    """
    # Load rewards for each candidate trajectory
    scored: List[Dict] = []
    for tid in recent_trajectory_ids:
        r = memory.get_reward(tid)
        if r is None:
            continue
        scored.append({"trajectory_id": tid, "final_scalar": float(r["final_scalar"])})

    if len(scored) < 2:
        return []

    # Sort descending by reward
    scored.sort(key=lambda d: d["final_scalar"], reverse=True)

    pairs: List[PreferencePair] = []
    # Pair the top-N winners against the bottom-N losers
    n = len(scored)
    # We don't pair every winner against every loser — that explodes
    # and biases toward redundant signal. Pair top-half-best against
    # bottom-half-worst, one to one.
    half = n // 2
    if half == 0:
        return []
    winners = scored[:half]
    losers = list(reversed(scored[half:]))  # worst-first
    for w, l in zip(winners, losers):
        gap = w["final_scalar"] - l["final_scalar"]
        if gap < rl_cfg.preference_margin:
            continue
        if w["trajectory_id"] == l["trajectory_id"]:
            continue
        pair = PreferencePair(
            pair_id=str(uuid4()),
            task_type=task_type,
            winner_trajectory_id=w["trajectory_id"],
            loser_trajectory_id=l["trajectory_id"],
            reward_gap=gap,
        )
        memory.insert_preference(pair.to_dict())
        pairs.append(pair)
        if len(pairs) >= max_pairs:
            break

    return pairs


def list_recent_pairs(memory: MemoryStore, task_type: str,
                      limit: int = 50) -> List[Dict]:
    return memory.preferences_for_type(task_type, limit=limit)
