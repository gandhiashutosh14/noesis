"""
NOESIS — Command-line Interface
================================

Three subcommands:

  noesis run-task      — run a single task; print the cycle summary
  noesis self-improve  — run many tasks from a JSON file; print rewards
  noesis show-stats    — dump strategy / prompt / gate state from memory

Designed to be the smallest possible surface that exercises every layer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from noesis.config.loader import ConfigurationError, load_config
from noesis.llm.client import LLMRouter
from noesis.loop.self_improve import SelfImprovementLoop
from noesis.memory.store import MemoryStore


# ---------------------------------------------------------------------------
def _build_loop(config_path: str | None) -> SelfImprovementLoop:
    try:
        config = load_config(Path(config_path)) if config_path else load_config()
    except ConfigurationError as e:
        print(f"[noesis] configuration error: {e}", file=sys.stderr)
        sys.exit(2)
    memory = MemoryStore(config.memory.sqlite_path)
    llm = LLMRouter(config)
    return SelfImprovementLoop(config, memory, llm)


# ---------------------------------------------------------------------------
async def _cmd_run_task(args: argparse.Namespace) -> None:
    loop = _build_loop(args.config)
    res = await loop.run_cycle(
        task_id=args.task_id,
        task_type=args.task_type,
        task_description=args.description,
        expected_answer=args.expected_answer,
        complexity=args.complexity,
        snapshot_after=args.snapshot,
    )
    print(json.dumps(res.summary(), indent=2, default=str))


# ---------------------------------------------------------------------------
async def _cmd_self_improve(args: argparse.Namespace) -> None:
    loop = _build_loop(args.config)
    tasks = _load_tasks(args.tasks_path)
    if not tasks:
        print("[noesis] no tasks loaded", file=sys.stderr)
        sys.exit(1)

    if args.cycles > 1:
        # Repeat the task list `cycles` times so the policy has multiple
        # passes to converge on each task type.
        tasks = tasks * args.cycles

    results = await loop.run_many(tasks, snapshot_every=args.snapshot_every)
    summaries = [r.summary() for r in results]

    print(json.dumps({
        "cycles": len(results),
        "mean_reward": sum(r.reward.final_scalar for r in results) / max(1, len(results)),
        "results": summaries,
    }, indent=2, default=str))


# ---------------------------------------------------------------------------
def _cmd_show_stats(args: argparse.Namespace) -> None:
    loop = _build_loop(args.config)
    out: Dict[str, Any] = {
        "strategies": [],
        "prompts": [],
        "gate": None,
        "judge_reliability": [],
        "snapshots": [],
    }
    # Strategy stats + per-judge reliability (running Brier-based score) by task type
    if args.task_type:
        out["strategies"] = loop.memory.get_strategy_stats(args.task_type)
        out["judge_reliability"] = [
            {
                "persona_id": judge.persona_id,
                "task_type": args.task_type,
                "reliability": loop.memory.judge_reliability_score(judge.persona_id, args.task_type),
            }
            for judge in loop.config.jury.judges
        ]
    # Prompt versions for each registered template
    for prompt_id in {s.prompt_template_id for s in loop.book.all()}:
        out["prompts"].append({
            "template_id": prompt_id,
            "versions": loop.memory.list_prompt_versions(prompt_id),
        })
    # Gate
    out["gate"] = loop.memory.get_gate(loop.config.experiment.experiment_name)
    # Recent snapshots
    out["snapshots"] = loop.memory.latest_snapshots(
        loop.config.experiment.experiment_name, limit=5,
    )
    print(json.dumps(out, indent=2, default=str))


# ---------------------------------------------------------------------------
def _load_tasks(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.is_file():
        with p.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "tasks" in data:
            return list(data["tasks"])
        if isinstance(data, list):
            return data
        return []
    if p.is_dir():
        tasks: List[Dict[str, Any]] = []
        for f in sorted(p.glob("*.json")):
            try:
                with f.open("r") as fh:
                    d = json.load(fh)
                if isinstance(d, dict):
                    tasks.append(d)
            except Exception as e:
                print(f"[noesis] skipping {f}: {e}", file=sys.stderr)
        return tasks
    return []


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(prog="noesis", description="NOESIS self-improving agent framework")
    p.add_argument("--config", help="Path to config YAML (defaults to bundled).")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run-task", help="Run a single task through the loop")
    run.add_argument("--task-id", required=True)
    run.add_argument("--task-type", required=True)
    run.add_argument("--description", required=True)
    run.add_argument("--expected-answer")
    run.add_argument("--complexity", type=float, default=0.5)
    run.add_argument("--snapshot", action="store_true", help="snapshot policy after the cycle")

    imp = sub.add_parser("self-improve", help="Run many tasks; let policy update across them")
    imp.add_argument("--tasks", dest="tasks_path", required=True,
                     help="Path to a JSON file (list or {tasks: [...]}) or a directory of JSONs")
    imp.add_argument("--cycles", type=int, default=1,
                     help="Repeat the task list N times (default 1)")
    imp.add_argument("--snapshot-every", type=int, default=0,
                     help="Snapshot policy every N cycles (0 = never)")

    stats = sub.add_parser("show-stats", help="Inspect what the policy has learned")
    stats.add_argument("--task-type", help="If set, print strategy stats for this task type")

    args = p.parse_args()

    if args.cmd == "run-task":
        asyncio.run(_cmd_run_task(args))
    elif args.cmd == "self-improve":
        asyncio.run(_cmd_self_improve(args))
    elif args.cmd == "show-stats":
        _cmd_show_stats(args)


if __name__ == "__main__":
    main()
