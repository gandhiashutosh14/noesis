"""
NOESIS — Self-Improvement Loop
==============================

The closed-loop orchestrator. One cycle is:

    task ─► select strategy
         ─► runtime executes (returns AgentState + trajectory)
         ─► jury evaluates (returns JuryVerdict)
         ─► reward shaping (returns ShapedReward, persists)
         ─► policy update (strategy stats, gate, judge reliability,
                           preference pairs, reflection notes)
         ─► regression guard (rollback if recent rewards regressed)
         ─► optional snapshot of the policy

`run_cycle` is the unit of self-improvement. `run_until_converged`
chains cycles with a stopping rule (max cycles or reward plateau).
"""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from noesis.config.schema import NoesisConfig, ReasoningDepth
from noesis.judges.jury import Jury, JuryVerdict
from noesis.llm.client import LLMRouter
from noesis.loop.regression import (
    GuardOutcome,
    apply_rollback,
    check_regression,
    take_snapshot,
)
from noesis.memory.store import MemoryStore
from noesis.policy.selector import StrategySelector, SelectionDecision
from noesis.policy.strategy_book import StrategyBook, build_default_strategy_book
from noesis.policy.update import update_policy_after_task
from noesis.rewards.shaping import ShapedReward, shape_reward
from noesis.runtime.runtime import AgentRuntime
from noesis.runtime.state import AgentState
from noesis.tools import ToolRegistry, build_default_registry


# ---------------------------------------------------------------------------
@dataclass
class CycleResult:
    """One cycle's full breadcrumb trail."""
    task_id: str
    strategy_id: str
    state: AgentState
    verdict: JuryVerdict
    reward: ShapedReward
    decision: SelectionDecision
    guard: Optional[GuardOutcome] = None
    snapshot_id: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy_id": self.strategy_id,
            "final_answer": self.state.final_answer,
            "consensus_overall": self.verdict.consensus_overall,
            "contested": self.verdict.contested,
            "agreement": self.verdict.agreement,
            "reward_final": self.reward.final_scalar,
            "cognitive_momentum": self.reward.cognitive_momentum.final_value,
            "gated_teacher": self.reward.gated_teacher.final_value,
            "guard_triggered": bool(self.guard and self.guard.triggered),
            "snapshot_id": self.snapshot_id,
        }


# ---------------------------------------------------------------------------
class SelfImprovementLoop:
    """The orchestrator. Stateless across cycles except through memory."""

    def __init__(
        self,
        config: NoesisConfig,
        memory: MemoryStore,
        llm_router: LLMRouter,
        *,
        tools: Optional[ToolRegistry] = None,
        strategy_book: Optional[StrategyBook] = None,
        rng: Optional[random.Random] = None,
    ):
        self.config = config
        self.memory = memory
        self.llm = llm_router
        self.tools = tools or build_default_registry()
        self.book = strategy_book or build_default_strategy_book(
            default_endpoint=config.routing["planner"].primary
        )
        self.selector = StrategySelector(config.rl, self.book, memory, rng=rng)
        self.runtime = AgentRuntime(config, llm_router, self.tools, memory)
        self.jury = Jury(config, llm_router)
        self._recent_rewards: List[float] = []

    # ------------------------------------------------------------------
    async def run_cycle(
        self,
        *,
        task_id: str,
        task_type: str,
        task_description: str,
        expected_answer: Optional[str] = None,
        complexity: float = 0.5,
        force_strategy_id: Optional[str] = None,
        snapshot_after: bool = False,
    ) -> CycleResult:
        # 1. Select strategy
        if force_strategy_id and force_strategy_id in self.book:
            from noesis.policy.selector import SelectionDecision  # noqa: re-import for type
            forced = self.book.get(force_strategy_id)
            decision = SelectionDecision(
                strategy=forced, method="forced",
                sampled_value=1.0, rival_scores={},
            )
        else:
            decision = self.selector.select(task_type)

        strategy = decision.strategy

        # 2. Execute task
        state = await self.runtime.run(
            task_id=task_id,
            task_type=task_type,
            task_description=task_description,
            expected_answer=expected_answer,
            complexity=complexity,
            strategy_id=strategy.strategy_id,
            prompt_template_id=strategy.prompt_template_id,
            prompt_version=strategy.prompt_version,
            reasoning_depth=ReasoningDepth(strategy.reasoning_depth),
        )

        # 3. Persist trajectory
        self.memory.insert_trajectory({
            "trajectory_id": state.trajectory_id,
            "task_id": state.task_id,
            "strategy_id": strategy.strategy_id,
            "prompt_template_id": state.prompt_template_id,
            "prompt_version": state.prompt_version,
            "reasoning_depth": state.reasoning_depth,
            "status": state.status,
            "final_answer": state.final_answer,
            "total_latency_ms": sum(s.latency_ms for s in state.steps),
            "total_cost": sum(s.cost for s in state.steps),
            "steps": [s.to_dict() for s in state.steps],
            "experiment_name": self.config.experiment.experiment_name,
        })

        # 4. Evaluate
        verdict = await self.jury.evaluate(state)

        # 5. Persist judge scores
        for js in verdict.per_judge:
            self.memory.insert_judge_score({**js.to_dict(),
                                            "trajectory_id": state.trajectory_id})

        # 6. Shape & persist reward
        reward = shape_reward(
            state, verdict, self.memory,
            self.config.rewards, self.config.gate,
            self.config.experiment.experiment_name,
        )

        # 7. Policy update
        recent_ids = self._recent_trajectory_ids_for_type(task_type, limit=20)
        update_policy_after_task(
            state=state, verdict=verdict, reward=reward,
            decision=decision, config=self.config, memory=self.memory,
            recent_trajectory_ids_same_type=recent_ids,
        )

        # 8. Track rolling rewards, check guard
        self._recent_rewards.append(reward.final_scalar)
        if len(self._recent_rewards) > self.config.rl.rollback_window:
            self._recent_rewards = self._recent_rewards[-self.config.rl.rollback_window:]

        guard_outcome: Optional[GuardOutcome] = None
        if len(self._recent_rewards) >= max(2, self.config.rl.rollback_window // 2):
            guard_outcome = check_regression(
                memory=self.memory, rl_cfg=self.config.rl,
                experiment_name=self.config.experiment.experiment_name,
                recent_rewards=self._recent_rewards,
            )
            if guard_outcome.triggered and guard_outcome.snapshot_id_used:
                apply_rollback(self.memory, guard_outcome.snapshot_id_used)

        # 9. Optional snapshot
        snapshot_id: Optional[str] = None
        if snapshot_after and self._recent_rewards:
            baseline = sum(self._recent_rewards) / len(self._recent_rewards)
            snapshot_id = take_snapshot(
                self.memory,
                self.config.experiment.experiment_name,
                baseline,
                policy_payload=self._capture_policy_snapshot(),
            )

        return CycleResult(
            task_id=task_id,
            strategy_id=strategy.strategy_id,
            state=state,
            verdict=verdict,
            reward=reward,
            decision=decision,
            guard=guard_outcome,
            snapshot_id=snapshot_id,
        )

    # ------------------------------------------------------------------
    async def run_many(
        self,
        tasks: List[Dict[str, Any]],
        *,
        snapshot_every: int = 0,
    ) -> List[CycleResult]:
        """Run a list of tasks sequentially.

        `tasks` items: {task_id, task_type, task_description,
                        expected_answer?, complexity?, force_strategy_id?}
        `snapshot_every`: take a policy snapshot every N cycles (0 = never).
        """
        results: List[CycleResult] = []
        for i, t in enumerate(tasks, start=1):
            snap = bool(snapshot_every and (i % snapshot_every == 0))
            res = await self.run_cycle(
                task_id=t["task_id"],
                task_type=t["task_type"],
                task_description=t["task_description"],
                expected_answer=t.get("expected_answer"),
                complexity=float(t.get("complexity", 0.5)),
                force_strategy_id=t.get("force_strategy_id"),
                snapshot_after=snap,
            )
            results.append(res)
        return results

    # ------------------------------------------------------------------
    def _recent_trajectory_ids_for_type(self, task_type: str, *,
                                        limit: int = 20) -> List[str]:
        """Scan recent trajectories across this task type."""
        # We don't have an inverse index task_type -> trajectories. We
        # iterate task_ids by type via a cheap join from strategy_stats →
        # not enough. The simplest correct path: take the latest snapshots
        # of trajectories for the current task — preference pairs are
        # mostly within-task useful anyway. For cross-task within-type,
        # callers can pass a broader list explicitly.
        return [t["trajectory_id"]
                for t in self.memory.trajectories_for_task("")[:limit]
                if t.get("trajectory_id")]

    # ------------------------------------------------------------------
    def _capture_policy_snapshot(self) -> Dict[str, Any]:
        """Materialise the bits worth snapshotting."""
        active_prompts: List[Dict[str, Any]] = []
        for prompt_id in {s.prompt_template_id for s in self.book.all()}:
            try:
                versions = self.memory.list_prompt_versions(prompt_id)
                active = next((v for v in versions if v.get("active")), None)
                if active:
                    active_prompts.append({
                        "template_id": prompt_id,
                        "version": active["version"],
                    })
            except Exception:
                continue

        gate = self.memory.get_gate(self.config.experiment.experiment_name)
        return {
            "active_prompts": active_prompts,
            "gate": {"alpha": gate[0], "beta": gate[1]} if gate else None,
        }
