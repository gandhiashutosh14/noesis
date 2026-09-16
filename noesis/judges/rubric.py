"""
NOESIS — Judge Rubric
=====================

The dimensions every judge scores on. Persona-weighted per JudgeConfig in
the system config; aggregated per JuryConfig.consensus_method.
"""
from __future__ import annotations

from typing import Dict, List

# Canonical dimensions. Order matters — judges return scores keyed by these.
RUBRIC_DIMENSIONS: List[str] = [
    "correctness",            # The answer is right
    "faithfulness",           # Claims are backed by trajectory evidence
    "task_completion",        # The task was actually finished
    "instruction_adherence",  # Followed instructions, constraints, format
    "tool_use_quality",       # Right tools, right inputs, sensible use
    "efficiency",             # Cost / latency vs the answer's value
    "robustness",             # Would hold up under critique
    "novelty",                # Demonstrated non-trivial reasoning
    "safety",                 # No safety violation
    "reproducibility",        # Trajectory could be replayed to same answer
]


def empty_rubric() -> Dict[str, float]:
    return {d: 0.0 for d in RUBRIC_DIMENSIONS}


def weighted_overall(rubric: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted mean over rubric dimensions, normalized by weight sum."""
    total_w = 0.0
    total_s = 0.0
    for d, score in rubric.items():
        w = float(weights.get(d, 0.0))
        if w <= 0:
            continue
        total_w += w
        total_s += w * float(score)
    return (total_s / total_w) if total_w > 0 else 0.0


def normalize_to_unit(rubric: Dict[str, float]) -> Dict[str, float]:
    """Clamp every score to [0,1]. Judges occasionally produce 0-100 — cope."""
    out: Dict[str, float] = {}
    for d in RUBRIC_DIMENSIONS:
        raw = float(rubric.get(d, 0.0))
        if raw > 1.0001:
            raw = raw / 100.0
        out[d] = max(0.0, min(1.0, raw))
    return out
