"""
The evidence report must be recomputed from the store and must agree with what the loop stored.
Runs the loop with the mock provider into a temporary store, then checks the report against the rows.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from noesis.config.loader import load_config  # noqa: E402
from noesis.llm.client import LLMRouter  # noqa: E402
from noesis.loop.self_improve import SelfImprovementLoop  # noqa: E402
from noesis.memory.store import MemoryStore  # noqa: E402
from noesis.report import build_report, render_markdown  # noqa: E402
from noesis.report.evidence import repeated_revisions, thrash_steps  # noqa: E402


def _populate(tmpdir: str) -> str:
    config = load_config()
    config.memory.sqlite_path = Path(tmpdir) / "noesis_state.db"
    config.experiment.experiment_name = "report_test"
    loop = SelfImprovementLoop(config, MemoryStore(config.memory.sqlite_path), LLMRouter(config))

    async def run():
        await loop.run_many([
            {"task_id": "r1", "task_type": "arithmetic", "task_description": "Compute 6 * 7", "expected_answer": "42", "complexity": 0.2},
            {"task_id": "r2", "task_type": "arithmetic", "task_description": "Compute 9 + 10", "expected_answer": "19", "complexity": 0.2},
            {"task_id": "r3", "task_type": "knowledge_lookup", "task_description": "What is Thompson sampling?",
             "expected_answer": "samples each arm's Beta posterior", "complexity": 0.4},
        ], snapshot_every=0)
    asyncio.run(run())
    return str(config.memory.sqlite_path)


def test_thrash_and_repeat_definitions():
    steps = [{"step_id": "a", "kind": "plan"}, {"step_id": "b", "kind": "revise", "revises_step_id": "a"},
             {"step_id": "c", "kind": "reflect"}, {"step_id": "d", "kind": "revise", "revises_step_id": "a"},
             {"step_id": "e", "kind": "revise", "revises_step_id": "a"}]
    assert thrash_steps(steps) == ["b", "e"]          # d follows a reflect, so it is not thrash
    assert repeated_revisions(steps) == {"a": 3}
    assert thrash_steps([]) == [] and repeated_revisions([{"kind": "plan"}]) == {}


def test_report_is_recomputed_from_the_store_and_agrees_with_it():
    with tempfile.TemporaryDirectory(prefix="noesis_report_") as tmp:
        db = _populate(tmp)
        report = build_report(db)

        conn = sqlite3.connect(db)
        n_traj = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        n_scores = conn.execute("SELECT COUNT(*) FROM judge_scores").fetchone()[0]
        stored = {tid: json.loads(c) for tid, c in conn.execute("SELECT trajectory_id, components_json FROM rewards")}
        conn.close()

        assert report["table_counts"]["trajectories"] == n_traj == 3
        assert len(report["trajectories"]) == n_traj and sum(r["judges"] for r in report["trajectories"]) == n_scores
        assert sum(j["n"] for j in report["judges"]) == n_scores and len(report["judges"]) == 3
        assert report["models_seen_in_steps"] and all("mock" in m for m in report["models_seen_in_steps"])
        assert report["experiments"] == ["report_test"]
        # Recomputed thrash equals the ids the reward stored, and gates equal steps: the report says so in its own words.
        assert report["discrepancies"] == []
        for row in report["trajectories"]:
            comp = stored[row["trajectory_id"]]
            assert sorted(comp["cognitive_momentum"]["thrash_step_ids"]) == sorted(row["thrash_steps"])
            assert len(comp["gated_teacher"]["step_gates"]) == row["steps"]
            assert 0.0 <= row["agreement"] <= 1.0 and row["judges"] == 3
        text = render_markdown(report, "test note")
        assert "> test note" in text and "| trajectories | 3 |" in text and "no model was called" in text
        assert "do not measure task quality" in text
