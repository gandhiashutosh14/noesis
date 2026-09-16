"""
NOESIS — Multi-Judge Jury
=========================

Each judge persona is a configured (endpoint, prompt, rubric weights)
triple. The jury sends the same trajectory to each judge, parses their
structured JSON output, normalizes, computes per-step gating signals,
and produces a consensus + disagreement metric.

The output is what the reward layer consumes:

  - per-judge rubric scores
  - per-judge per-step scores (with confidence)
  - per-judge critique
  - consensus rubric (aggregated)
  - inter-judge agreement (1 - normalized stddev)
  - contested flag (stddev > threshold)
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from noesis.config.schema import JudgeConfig, JuryConfig, NoesisConfig
from noesis.judges.rubric import (
    RUBRIC_DIMENSIONS,
    empty_rubric,
    normalize_to_unit,
    weighted_overall,
)
from noesis.llm.client import LLMRouter
from noesis.runtime.state import AgentState


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """\
You are a judge evaluating a multi-step agent trajectory.

Persona:
{persona_prompt}

Task description:
{task}

Trajectory:
{trajectory}

Final answer produced:
{final_answer}

Score the trajectory on each of these dimensions (0.0 to 1.0):
{dimensions}

For each step in the trajectory, also produce a step-level score with
confidence and a one-sentence rationale.

Output STRICT JSON with this shape:
{{
  "rubric": {{<dimension>: <float 0..1>, ...}},
  "step_scores": [{{"step_id": str, "score": float 0..1, "confidence": float 0..1, "rationale": str}}],
  "confidence": <float 0..1>,   // your overall confidence in this evaluation
  "critique": "<one paragraph identifying strengths and the single biggest weakness>"
}}
Do not include any prose outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Judge response — typed
# ---------------------------------------------------------------------------
@dataclass
class JudgeScore:
    score_id: str
    persona_id: str
    rubric: Dict[str, float]
    overall_score: float
    confidence: float
    step_scores: List[Dict[str, Any]] = field(default_factory=list)
    critique: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_id": self.score_id,
            "persona_id": self.persona_id,
            "rubric": self.rubric,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "step_scores": self.step_scores,
            "critique": self.critique,
        }


@dataclass
class JuryVerdict:
    """The aggregated jury output."""
    consensus_rubric: Dict[str, float]
    consensus_overall: float
    agreement: float                # [0,1] — 1 = perfect agreement
    contested: bool
    per_judge: List[JudgeScore]
    # Per-step consensus across judges: list parallel to trajectory steps
    per_step_consensus: List[Dict[str, Any]] = field(default_factory=list)
    # Adversarial critique — the single most pessimistic critique, surfaced
    # so the reward layer can penalize ignored caveats
    adversarial_critique: str = ""


# ---------------------------------------------------------------------------
# Jury
# ---------------------------------------------------------------------------
class Jury:
    def __init__(self, config: NoesisConfig, llm_router: LLMRouter):
        self.config = config
        self.llm = llm_router
        self.jury_cfg: JuryConfig = config.jury

    async def evaluate(self, state: AgentState) -> JuryVerdict:
        """Score one agent trajectory. Judges run in parallel."""
        trajectory_str = self._dump_trajectory(state)
        dimensions_str = ", ".join(RUBRIC_DIMENSIONS)
        tasks = [
            self._one_judge(
                judge_cfg=jc,
                task=state.task_description,
                trajectory=trajectory_str,
                final_answer=state.final_answer or "(no answer)",
                dimensions=dimensions_str,
            )
            for jc in self.jury_cfg.judges
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return self._aggregate(results, state)

    # ------------------------------------------------------------------
    async def _one_judge(
        self,
        *,
        judge_cfg: JudgeConfig,
        task: str,
        trajectory: str,
        final_answer: str,
        dimensions: str,
    ) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(
            persona_prompt=judge_cfg.persona_prompt or "You are a careful evaluator.",
            task=task,
            trajectory=trajectory,
            final_answer=final_answer,
            dimensions=dimensions,
        )
        try:
            resp = await self.llm.complete_for_endpoint(
                judge_cfg.endpoint_name, prompt, json_mode=True,
            )
            raw = resp.text
        except Exception:
            # Degrade gracefully: produce a neutral score rather than crashing the jury
            return JudgeScore(
                score_id=str(uuid4()),
                persona_id=judge_cfg.persona_id,
                rubric=empty_rubric(),
                overall_score=0.5,
                confidence=0.2,
                step_scores=[],
                critique="(judge unavailable — neutral score)",
            )

        parsed = self._safe_parse(raw)
        rubric = normalize_to_unit(parsed.get("rubric", {}))
        overall = weighted_overall(rubric, judge_cfg.rubric_weights)
        return JudgeScore(
            score_id=str(uuid4()),
            persona_id=judge_cfg.persona_id,
            rubric=rubric,
            overall_score=overall,
            confidence=float(parsed.get("confidence", 0.5)),
            step_scores=self._normalize_step_scores(parsed.get("step_scores", [])),
            critique=str(parsed.get("critique", ""))[:1200],
        )

    # ------------------------------------------------------------------
    def _aggregate(self, per_judge: List[JudgeScore], state: AgentState) -> JuryVerdict:
        if not per_judge:
            return JuryVerdict(
                consensus_rubric=empty_rubric(),
                consensus_overall=0.0,
                agreement=0.0,
                contested=True,
                per_judge=[],
            )

        # Consensus per dimension via configured method
        method = self.jury_cfg.consensus_method
        consensus: Dict[str, float] = {}
        for d in RUBRIC_DIMENSIONS:
            xs = [j.rubric.get(d, 0.0) for j in per_judge]
            consensus[d] = self._reduce(xs, method)
        consensus_overall = sum(consensus.values()) / len(consensus)

        # Inter-judge agreement: 1 - stddev(overalls)/0.5 (so an across-judges
        # stddev of 0.5 maps to 0 agreement; 0 stddev → 1 agreement).
        overalls = [j.overall_score for j in per_judge]
        stddev = statistics.pstdev(overalls) if len(overalls) > 1 else 0.0
        agreement = max(0.0, 1.0 - (stddev / 0.5))
        contested = stddev > self.jury_cfg.disagreement_threshold

        # Per-step consensus
        per_step_consensus = self._per_step_consensus(per_judge, state)

        # Most adversarial critique = lowest-scoring judge's critique
        most_pessimistic = min(per_judge, key=lambda j: j.overall_score)

        return JuryVerdict(
            consensus_rubric=consensus,
            consensus_overall=consensus_overall,
            agreement=agreement,
            contested=contested,
            per_judge=per_judge,
            per_step_consensus=per_step_consensus,
            adversarial_critique=most_pessimistic.critique,
        )

    # ------------------------------------------------------------------
    def _per_step_consensus(self, per_judge: List[JudgeScore], state: AgentState) -> List[Dict[str, Any]]:
        """For each trajectory step, compute the consensus score + judge agreement.

        Judges may return different step_id sets — we align by step_id; missing
        scores are treated as 'neutral' (0.5, confidence 0.0).
        """
        consensus_rows: List[Dict[str, Any]] = []
        for step in state.steps:
            sid = step.step_id
            scores: List[float] = []
            confs: List[float] = []
            for j in per_judge:
                hit = next((s for s in j.step_scores if s.get("step_id") == sid), None)
                if hit is None:
                    continue
                scores.append(float(hit.get("score", 0.5)))
                confs.append(float(hit.get("confidence", 0.5)))
            if scores:
                mean = sum(scores) / len(scores)
                conf = sum(confs) / len(confs) if confs else 0.5
                step_stddev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
                step_agreement = max(0.0, 1.0 - (step_stddev / 0.5))
            else:
                mean = 0.5
                conf = 0.0
                step_agreement = 0.0
            consensus_rows.append({
                "step_id": sid,
                "kind": step.kind.value,
                "consensus_score": mean,
                "confidence": conf,
                "agreement": step_agreement,
                "raw_scores": scores,
            })
        return consensus_rows

    # ------------------------------------------------------------------
    @staticmethod
    def _reduce(xs: List[float], method: str) -> float:
        if not xs:
            return 0.0
        if method == "mean":
            return sum(xs) / len(xs)
        if method == "median":
            return statistics.median(xs)
        # trimmed_mean: drop top + bottom 10%
        if len(xs) >= 3:
            xs_sorted = sorted(xs)
            k = max(0, int(len(xs_sorted) * 0.1))
            trimmed = xs_sorted[k:len(xs_sorted) - k] if k else xs_sorted
            return sum(trimmed) / len(trimmed)
        return sum(xs) / len(xs)

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_parse(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_step_scores(raw: List[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in raw or []:
            if not isinstance(s, dict):
                continue
            try:
                out.append({
                    "step_id": str(s.get("step_id", "")),
                    "score": max(0.0, min(1.0, float(s.get("score", 0.5)))),
                    "confidence": max(0.0, min(1.0, float(s.get("confidence", 0.5)))),
                    "rationale": str(s.get("rationale", ""))[:400],
                })
            except (TypeError, ValueError):
                continue
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _dump_trajectory(state: AgentState) -> str:
        lines: List[str] = []
        for i, s in enumerate(state.steps):
            head = f"step_id={s.step_id} kind={s.kind.value} action={s.chosen_action}"
            if s.chosen_tool:
                head += f" tool={s.chosen_tool}"
            head += f" conf={s.confidence:.2f} latency_ms={s.latency_ms} cost={s.cost:.3f}"
            lines.append(f"[{i+1}] {head}")
            if isinstance(s.output, dict):
                lines.append(f"    output_summary: {json.dumps({k: _short(v) for k, v in list(s.output.items())[:6]})}")
            elif s.output is not None:
                lines.append(f"    output_summary: {_short(s.output)}")
            if s.error:
                lines.append(f"    error: {s.error}")
        return "\n".join(lines)


def _short(v: Any, n: int = 120) -> Any:
    if isinstance(v, str):
        return v if len(v) <= n else v[: n - 3] + "..."
    if isinstance(v, list):
        return [str(x)[: 40] for x in v[:3]]
    return v
