"""
NOESIS — Evidence report
========================

Recomputes, from the rows in the SQLite store, what the loop did: how many
trajectories ran, what steps they took, what each judge said, how far the
judges were apart, which steps were thrash, where tools failed, and what the
rewards were. Wherever a stored scalar can be recomputed from the underlying
records, the recomputation is compared with what was stored and any
discrepancy is reported, not hidden.

It reads the store only. It runs no model. The provider that produced the
rows is read from the steps themselves (`chosen_model`), so a report over a
mock-provider run says so in its own header.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

AGREEMENT_SCALE = 0.5           # jury.py: agreement = 1 - stddev(overalls) / 0.5
DEFAULT_CONTESTED_THRESHOLD = 0.25   # defaults.yaml: jury.disagreement_threshold


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _loads(text: Any, default: Any) -> Any:
    if not isinstance(text, str):
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def thrash_steps(steps: List[Dict[str, Any]]) -> List[str]:
    """Step ids of revise steps not immediately preceded by a reflect step (the CMR definition of thrash)."""
    out = []
    for i, s in enumerate(steps):
        if s.get("kind") == "revise" and (i == 0 or steps[i - 1].get("kind") != "reflect"):
            out.append(s.get("step_id"))
    return out


def repeated_revisions(steps: List[Dict[str, Any]]) -> Dict[str, int]:
    """Steps revised more than once: {revised step id: number of revisions}."""
    counts = Counter(s.get("revises_step_id") for s in steps if s.get("kind") == "revise" and s.get("revises_step_id"))
    return {k: v for k, v in counts.items() if v > 1}


def build_report(db_path: str, *, contested_threshold: float = DEFAULT_CONTESTED_THRESHOLD) -> Dict[str, Any]:
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        tasks = {t["task_id"]: t for t in _rows(conn, "SELECT * FROM tasks")}
        trajectories = _rows(conn, "SELECT * FROM trajectories ORDER BY created_at, trajectory_id")
        scores = _rows(conn, "SELECT * FROM judge_scores ORDER BY created_at, score_id")
        rewards = {r["trajectory_id"]: r for r in _rows(conn, "SELECT * FROM rewards")}
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("tasks", "trajectories", "judge_scores", "rewards", "preference_pairs", "strategy_stats",
                            "prompt_versions", "reflection_notes", "gate_params", "judge_reliability", "policy_snapshots")}
    finally:
        conn.close()

    scores_by_traj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in scores:
        scores_by_traj[s["trajectory_id"]].append(s)

    per_trajectory: List[Dict[str, Any]] = []
    discrepancies: List[str] = []
    models_seen: Counter = Counter()
    judge_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    strategy_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for t in trajectories:
        steps = _loads(t.get("steps_json"), [])
        kinds = Counter(s.get("kind") for s in steps)
        for s in steps:
            if s.get("chosen_model"):
                models_seen[s["chosen_model"]] += 1
        tool_calls = [s for s in steps if s.get("kind") == "tool_call"]
        tool_errors = [s for s in tool_calls if s.get("error") or (s.get("failure_mode") not in (None, "none"))]
        thrash = thrash_steps(steps)
        repeats = repeated_revisions(steps)

        js = scores_by_traj.get(t["trajectory_id"], [])
        overalls = [float(s["overall_score"]) for s in js]
        stddev = statistics.pstdev(overalls) if len(overalls) > 1 else 0.0
        agreement = max(0.0, 1.0 - stddev / AGREEMENT_SCALE) if overalls else 0.0
        contested = stddev > contested_threshold if overalls else True
        mean_overall = statistics.fmean(overalls) if overalls else 0.0
        for s in js:
            judge_rows[s["persona_id"]].append({"overall": float(s["overall_score"]), "confidence": float(s["confidence"]),
                                                "deviation": float(s["overall_score"]) - mean_overall})

        reward = rewards.get(t["trajectory_id"])
        components = _loads(reward.get("components_json"), {}) if reward else {}
        stored_thrash = list((components.get("cognitive_momentum") or {}).get("thrash_step_ids") or [])
        stored_gates = (components.get("gated_teacher") or {}).get("step_gates")
        if reward and sorted(stored_thrash) != sorted(thrash):
            discrepancies.append(f"{t['trajectory_id'][:8]}: stored thrash steps {stored_thrash} vs recomputed {thrash}")
        if isinstance(stored_gates, list) and len(stored_gates) != len(steps):
            discrepancies.append(f"{t['trajectory_id'][:8]}: {len(stored_gates)} step gates stored for {len(steps)} steps")

        row = {
            "trajectory_id": t["trajectory_id"],
            "task_id": t["task_id"],
            "task_type": tasks.get(t["task_id"], {}).get("task_type", "?"),
            "strategy_id": t["strategy_id"],
            "status": t["status"],
            "steps": len(steps),
            "kinds": dict(kinds),
            "tool_calls": len(tool_calls),
            "tool_errors": len(tool_errors),
            "thrash_steps": thrash,
            "repeated_revisions": repeats,
            "judges": len(js),
            "judge_mean_overall": round(mean_overall, 4),
            "judge_stddev": round(stddev, 4),
            "agreement": round(agreement, 4),
            "contested": contested,
            "reward_final": round(float(reward["final_scalar"]), 4) if reward else None,
            "cognitive_momentum": round(float(reward["cognitive_momentum"]), 4) if reward else None,
            "gated_teacher": round(float(reward["gated_teacher"]), 4) if reward else None,
            "latency_ms": int(t.get("total_latency_ms") or 0),
        }
        per_trajectory.append(row)
        strategy_rows[t["strategy_id"]].append(row)

    judges = []
    for persona, rows in sorted(judge_rows.items()):
        judges.append({"persona_id": persona, "n": len(rows),
                       "mean_overall": round(statistics.fmean(r["overall"] for r in rows), 4),
                       "mean_confidence": round(statistics.fmean(r["confidence"] for r in rows), 4),
                       "mean_deviation_from_jury_mean": round(statistics.fmean(r["deviation"] for r in rows), 4)})
    strategies = []
    for sid, rows in sorted(strategy_rows.items()):
        with_reward = [r for r in rows if r["reward_final"] is not None]
        strategies.append({"strategy_id": sid, "n": len(rows),
                           "mean_reward": round(statistics.fmean(r["reward_final"] for r in with_reward), 4) if with_reward else None,
                           "mean_steps": round(statistics.fmean(r["steps"] for r in rows), 2)})

    observed = {
        "trajectories": len(per_trajectory),
        "contested": sum(1 for r in per_trajectory if r["contested"]),
        "with_thrash": sum(1 for r in per_trajectory if r["thrash_steps"]),
        "thrash_steps_total": sum(len(r["thrash_steps"]) for r in per_trajectory),
        "tool_calls": sum(r["tool_calls"] for r in per_trajectory),
        "tool_errors": sum(r["tool_errors"] for r in per_trajectory),
        "with_repeated_revisions": sum(1 for r in per_trajectory if r["repeated_revisions"]),
        "status_histogram": dict(Counter(r["status"] for r in per_trajectory)),
    }
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "db_path": str(db_path),
        "table_counts": counts,
        "models_seen_in_steps": dict(models_seen),
        "experiments": sorted({t.get("experiment_name") or "default" for t in trajectories}),
        "observed": observed,
        "judges": judges,
        "strategies": strategies,
        "trajectories": per_trajectory,
        "discrepancies": discrepancies,
    }


def render_markdown(report: Dict[str, Any], note: Optional[str] = None) -> str:
    o = report["observed"]
    models = ", ".join(f"{m} ({n} steps)" for m, n in report["models_seen_in_steps"].items()) or "none recorded"
    lines = ["# NOESIS evidence report", "",
             f"Generated {report['generated']} from `{report['db_path']}`, experiments {report['experiments']}. "
             f"Models named in the stored steps: {models}. Everything below is recomputed from the `trajectories`, "
             "`judge_scores` and `rewards` tables; no model was called to produce this report.", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += ["## Store", "", "| Table | Rows |", "|---|---|"]
    lines += [f"| {t} | {n} |" for t, n in report["table_counts"].items()]
    lines += ["", "## Observed behaviour", "", "| Measure | Value |", "|---|---|",
              f"| Trajectories | {o['trajectories']} |",
              f"| Status histogram | {o['status_histogram']} |",
              f"| Contested verdicts (judge stddev above threshold) | {o['contested']} |",
              f"| Trajectories with thrash (revise not preceded by reflect) | {o['with_thrash']} ({o['thrash_steps_total']} steps) |",
              f"| Tool calls / tool errors | {o['tool_calls']} / {o['tool_errors']} |",
              f"| Trajectories that revised the same step more than once | {o['with_repeated_revisions']} |",
              "", "## Judges (recomputed from judge_scores)", "",
              "| Judge | n | Mean overall | Mean confidence | Mean deviation from jury mean |", "|---|---|---|---|---|"]
    lines += [f"| {j['persona_id']} | {j['n']} | {j['mean_overall']} | {j['mean_confidence']} | {j['mean_deviation_from_jury_mean']:+} |"
              for j in report["judges"]]
    lines += ["", "## Strategies", "", "| Strategy | n | Mean reward | Mean steps |", "|---|---|---|---|"]
    lines += [f"| {s['strategy_id']} | {s['n']} | {s['mean_reward']} | {s['mean_steps']} |" for s in report["strategies"]]
    lines += ["", "## Trajectories", "",
              "| # | Task | Type | Strategy | Status | Steps | Kinds | Tools (err) | Thrash | Judges | Mean | Stddev | Agreement | Contested | Reward | CM | GT |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(report["trajectories"], 1):
        kinds = " ".join(f"{k}:{v}" for k, v in sorted(r["kinds"].items()))
        lines.append(f"| {i} | {r['task_id']} | {r['task_type']} | {r['strategy_id']} | {r['status']} | {r['steps']} | {kinds} | "
                     f"{r['tool_calls']} ({r['tool_errors']}) | {len(r['thrash_steps'])} | {r['judges']} | {r['judge_mean_overall']} | "
                     f"{r['judge_stddev']} | {r['agreement']} | {'yes' if r['contested'] else 'no'} | {r['reward_final']} | "
                     f"{r['cognitive_momentum']} | {r['gated_teacher']} |")
    lines += ["", "## Stored vs recomputed", ""]
    if report["discrepancies"]:
        lines += ["Discrepancies between stored reward components and recomputation from the step records:", ""]
        lines += [f"- {d}" for d in report["discrepancies"]]
    else:
        lines += ["Thrash step ids stored in each reward's cognitive-momentum breakdown equal the ids recomputed from the "
                  "step records, and the number of stored step gates equals the number of steps, for every trajectory."]
    lines += ["", "## What this report does and does not show", "",
              "It shows what the loop recorded and that the stored reward breakdowns agree with the step records. "
              "Under the mock provider the judge scores and rewards are deterministic fixtures: they do not measure task "
              "quality, judge calibration or improvement over cycles, and nothing here should be read as evidence of any of those."]
    return "\n".join(lines) + "\n"
