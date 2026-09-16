"""
NOESIS — Regression Guard
=========================

After each self-improvement cycle, compare current rolling reward mean
against the most recent policy snapshot's baseline. If we regressed by
more than `regression_guard_threshold` over a window of `rollback_window`
cycles, restore the latest snapshot.

The guard is intentionally simple. Sophisticated guards (bootstrap CIs,
sequential hypothesis tests) sound impressive on paper but produce
unstable behaviour under low sample counts — which is exactly the
regime NOESIS operates in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from noesis.config.schema import RLConfig
from noesis.memory.store import MemoryStore


@dataclass
class GuardOutcome:
    triggered: bool
    rolling_mean: float
    baseline_reward: float
    regression_gap: float
    snapshot_id_used: Optional[str]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "rolling_mean": self.rolling_mean,
            "baseline_reward": self.baseline_reward,
            "regression_gap": self.regression_gap,
            "snapshot_id_used": self.snapshot_id_used,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
def take_snapshot(
    memory: MemoryStore,
    experiment_name: str,
    baseline_reward: float,
    policy_payload: Dict[str, Any],
) -> str:
    snapshot_id = str(uuid4())
    memory.insert_snapshot({
        "snapshot_id": snapshot_id,
        "experiment_name": experiment_name,
        "baseline_reward": baseline_reward,
        "payload": policy_payload,
    })
    return snapshot_id


# ---------------------------------------------------------------------------
def check_regression(
    *,
    memory: MemoryStore,
    rl_cfg: RLConfig,
    experiment_name: str,
    recent_rewards: List[float],
) -> GuardOutcome:
    """Compare recent rolling-mean reward against the latest snapshot's
    baseline. Trigger if the gap exceeds the configured threshold.
    """
    if len(recent_rewards) < max(2, rl_cfg.rollback_window // 2):
        return GuardOutcome(
            triggered=False, rolling_mean=0.0, baseline_reward=0.0,
            regression_gap=0.0, snapshot_id_used=None,
            reason="insufficient samples for guard",
        )

    rolling = sum(recent_rewards) / len(recent_rewards)
    snapshots = memory.latest_snapshots(experiment_name, limit=1)
    if not snapshots:
        return GuardOutcome(
            triggered=False, rolling_mean=rolling, baseline_reward=0.0,
            regression_gap=0.0, snapshot_id_used=None,
            reason="no snapshot to compare against",
        )

    baseline = float(snapshots[0]["baseline_reward"])
    gap = baseline - rolling
    triggered = gap > rl_cfg.regression_guard_threshold

    return GuardOutcome(
        triggered=triggered,
        rolling_mean=rolling,
        baseline_reward=baseline,
        regression_gap=gap,
        snapshot_id_used=snapshots[0]["snapshot_id"] if triggered else None,
        reason="regression detected" if triggered else "no regression",
    )


# ---------------------------------------------------------------------------
def apply_rollback(memory: MemoryStore, snapshot_id: str) -> bool:
    """Restore the policy fragments stored in the snapshot.

    The payload schema is intentionally permissive: it just stores what
    the loop chose to capture (strategy stats snapshot, gate params,
    active prompt versions). Restoring is best-effort.
    """
    snapshots = memory.latest_snapshots("", limit=200)  # all experiments
    target = next((s for s in snapshots if s["snapshot_id"] == snapshot_id), None)
    if not target:
        return False
    payload = target.get("payload", {}) or {}

    # Restore active prompt versions
    for entry in payload.get("active_prompts", []):
        try:
            memory.set_active_prompt(entry["template_id"], entry["version"])
        except Exception:
            continue

    # Restore gate params
    gp = payload.get("gate")
    if gp:
        try:
            memory.set_gate(target["experiment_name"], float(gp["alpha"]), float(gp["beta"]))
        except Exception:
            pass

    # Strategy stats are not literally restored (overwriting them would
    # corrupt counts of subsequent attempts). Instead the rolling guard
    # just slows down further drift by keeping the snapshot pinned.
    return True
