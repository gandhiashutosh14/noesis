"""
NOESIS — Strategy Selector
==========================

Three exploration policies — pick one per experiment via config:

  * thompson         — sample from Beta(α, β) per strategy, pick max
  * ucb              — mean + c·sqrt(ln(N)/n_i), pick max
  * epsilon_greedy   — ε of the time random, else pick highest mean

Reads `strategy_stats` from memory and writes nothing — selection is
read-only. Updates happen in `policy/update.py`.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from noesis.config.schema import ExplorationStrategy, RLConfig
from noesis.memory.store import MemoryStore
from noesis.policy.strategy_book import Strategy, StrategyBook


@dataclass
class SelectionDecision:
    """What was picked and why — surfaced for observability."""
    strategy: Strategy
    method: str
    sampled_value: float
    rival_scores: Dict[str, float]


class StrategySelector:
    def __init__(self, rl_cfg: RLConfig, book: StrategyBook,
                 memory: MemoryStore, *, rng: Optional[random.Random] = None):
        self.cfg = rl_cfg
        self.book = book
        self.memory = memory
        self.rng = rng or random.Random()

    # ------------------------------------------------------------------
    def select(self, task_type: str) -> SelectionDecision:
        method = self.cfg.exploration_strategy
        stats = {s["strategy_id"]: s for s in self.memory.get_strategy_stats(task_type)}

        candidates = self.book.all()
        if not candidates:
            raise RuntimeError("StrategyBook is empty")

        if method == ExplorationStrategy.THOMPSON:
            return self._thompson(candidates, stats)
        if method == ExplorationStrategy.UCB:
            return self._ucb(candidates, stats)
        return self._epsilon_greedy(candidates, stats)

    # ------------------------------------------------------------------
    def _thompson(self, candidates: List[Strategy],
                  stats: Dict[str, Dict]) -> SelectionDecision:
        scores: Dict[str, float] = {}
        best: Optional[Strategy] = None
        best_score = -1.0
        for c in candidates:
            row = stats.get(c.strategy_id)
            alpha = float(row["alpha"]) if row else 1.0
            beta = float(row["beta"]) if row else 1.0
            sample = self.rng.betavariate(alpha, beta)
            scores[c.strategy_id] = sample
            if sample > best_score:
                best_score = sample
                best = c
        assert best is not None
        return SelectionDecision(
            strategy=best, method="thompson",
            sampled_value=best_score, rival_scores=scores,
        )

    # ------------------------------------------------------------------
    def _ucb(self, candidates: List[Strategy],
             stats: Dict[str, Dict]) -> SelectionDecision:
        total_attempts = sum(int(stats.get(c.strategy_id, {}).get("attempts", 0))
                             for c in candidates)
        if total_attempts == 0:
            # Cold start — uniform random pick
            chosen = self.rng.choice(candidates)
            return SelectionDecision(
                strategy=chosen, method="ucb_coldstart",
                sampled_value=0.0,
                rival_scores={c.strategy_id: 0.0 for c in candidates},
            )

        c_const = self.cfg.ucb_c
        scores: Dict[str, float] = {}
        best: Optional[Strategy] = None
        best_score = -1e9
        for c in candidates:
            row = stats.get(c.strategy_id)
            attempts = int(row["attempts"]) if row else 0
            if attempts == 0:
                ucb = 1e6  # force exploration of unseen strategies
            else:
                mean = float(row["reward_sum"]) / attempts
                explore = c_const * math.sqrt(math.log(max(1, total_attempts)) / attempts)
                ucb = mean + explore
            scores[c.strategy_id] = ucb
            if ucb > best_score:
                best_score = ucb
                best = c
        assert best is not None
        return SelectionDecision(
            strategy=best, method="ucb",
            sampled_value=best_score, rival_scores=scores,
        )

    # ------------------------------------------------------------------
    def _epsilon_greedy(self, candidates: List[Strategy],
                        stats: Dict[str, Dict]) -> SelectionDecision:
        if self.rng.random() < self.cfg.epsilon:
            chosen = self.rng.choice(candidates)
            return SelectionDecision(
                strategy=chosen, method="epsilon_explore",
                sampled_value=0.0,
                rival_scores={c.strategy_id: 0.0 for c in candidates},
            )
        scores: Dict[str, float] = {}
        best: Optional[Strategy] = None
        best_score = -1.0
        for c in candidates:
            row = stats.get(c.strategy_id)
            attempts = int(row["attempts"]) if row else 0
            mean = (float(row["reward_sum"]) / attempts) if attempts else 0.0
            scores[c.strategy_id] = mean
            if mean > best_score:
                best_score = mean
                best = c
        assert best is not None
        return SelectionDecision(
            strategy=best, method="epsilon_exploit",
            sampled_value=best_score, rival_scores=scores,
        )
