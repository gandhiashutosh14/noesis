"""
NOESIS — Persistent Memory (SQLite-backed)
==========================================

Everything the system needs to remember across runs lives here. Tasks,
trajectories, judge scores, rewards, preference pairs, strategy stats,
prompt versions, reflection notes.

We pick SQLite because it is boring, reliable, has ACID guarantees, and
ships with Python. Production deployments can swap in Postgres by
implementing the same interface; nothing else changes.

The schema is defined inline (in this file) rather than in a separate
.sql so the module is self-contained — `python -m noesis.memory.store
--init` is enough to bring it up.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS tasks (
  task_id           TEXT PRIMARY KEY,
  task_type         TEXT NOT NULL,
  complexity        REAL NOT NULL DEFAULT 0.5,
  description       TEXT NOT NULL,
  expected_answer   TEXT,
  created_at        TEXT NOT NULL,
  metadata          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type);

CREATE TABLE IF NOT EXISTS trajectories (
  trajectory_id     TEXT PRIMARY KEY,
  task_id           TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  strategy_id       TEXT NOT NULL,
  prompt_template_id TEXT NOT NULL,
  prompt_version    TEXT NOT NULL,
  reasoning_depth   TEXT NOT NULL,
  status            TEXT NOT NULL,
  final_answer      TEXT,
  total_latency_ms  INTEGER NOT NULL DEFAULT 0,
  total_cost        REAL NOT NULL DEFAULT 0.0,
  steps_json        TEXT NOT NULL,
  created_at        TEXT NOT NULL,
  experiment_name   TEXT NOT NULL DEFAULT 'default'
);
CREATE INDEX IF NOT EXISTS idx_traj_task ON trajectories(task_id);
CREATE INDEX IF NOT EXISTS idx_traj_strategy ON trajectories(strategy_id);
CREATE INDEX IF NOT EXISTS idx_traj_status ON trajectories(status);

CREATE TABLE IF NOT EXISTS judge_scores (
  score_id          TEXT PRIMARY KEY,
  trajectory_id     TEXT NOT NULL REFERENCES trajectories(trajectory_id) ON DELETE CASCADE,
  persona_id        TEXT NOT NULL,
  rubric_json       TEXT NOT NULL,
  overall_score     REAL NOT NULL,
  confidence        REAL NOT NULL DEFAULT 0.5,
  step_scores_json  TEXT NOT NULL DEFAULT '[]',
  critique          TEXT NOT NULL DEFAULT '',
  created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_score_traj ON judge_scores(trajectory_id);

CREATE TABLE IF NOT EXISTS rewards (
  reward_id         TEXT PRIMARY KEY,
  trajectory_id     TEXT NOT NULL UNIQUE REFERENCES trajectories(trajectory_id) ON DELETE CASCADE,
  cognitive_momentum REAL NOT NULL,
  gated_teacher     REAL NOT NULL,
  components_json   TEXT NOT NULL,
  final_scalar      REAL NOT NULL,
  created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preference_pairs (
  pair_id           TEXT PRIMARY KEY,
  task_type         TEXT NOT NULL,
  winner_trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id) ON DELETE CASCADE,
  loser_trajectory_id  TEXT NOT NULL REFERENCES trajectories(trajectory_id) ON DELETE CASCADE,
  reward_gap        REAL NOT NULL,
  created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pref_type ON preference_pairs(task_type);

CREATE TABLE IF NOT EXISTS strategy_stats (
  strategy_id       TEXT NOT NULL,
  task_type         TEXT NOT NULL,
  attempts          INTEGER NOT NULL DEFAULT 0,
  successes         INTEGER NOT NULL DEFAULT 0,
  reward_sum        REAL NOT NULL DEFAULT 0.0,
  reward_sumsq      REAL NOT NULL DEFAULT 0.0,
  -- Beta-distribution params for Thompson sampling (a=successes+1, b=failures+1)
  -- Stored explicitly so we can use a smoothed running estimate, not just counts.
  alpha             REAL NOT NULL DEFAULT 1.0,
  beta              REAL NOT NULL DEFAULT 1.0,
  last_used_at      TEXT,
  PRIMARY KEY (strategy_id, task_type)
);

CREATE TABLE IF NOT EXISTS prompt_versions (
  template_id       TEXT NOT NULL,
  version           TEXT NOT NULL,
  role              TEXT NOT NULL,
  body              TEXT NOT NULL,
  parent_version    TEXT,
  active            INTEGER NOT NULL DEFAULT 0,
  performance_score REAL NOT NULL DEFAULT 0.0,
  attempts          INTEGER NOT NULL DEFAULT 0,
  created_at        TEXT NOT NULL,
  PRIMARY KEY (template_id, version)
);

CREATE TABLE IF NOT EXISTS reflection_notes (
  note_id           TEXT PRIMARY KEY,
  task_type         TEXT NOT NULL,
  trigger_failure_mode TEXT NOT NULL,
  body              TEXT NOT NULL,
  usefulness_score  REAL NOT NULL DEFAULT 0.0,
  retrieve_count    INTEGER NOT NULL DEFAULT 0,
  created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refl_type ON reflection_notes(task_type);

CREATE TABLE IF NOT EXISTS gate_params (
  experiment_name   TEXT PRIMARY KEY,
  alpha             REAL NOT NULL,
  beta              REAL NOT NULL,
  updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS judge_reliability (
  persona_id        TEXT NOT NULL,
  task_type         TEXT NOT NULL,
  -- Calibration tracked as (predicted_score, observed_outcome) pairs reduced
  -- to a running brier score. Lower is better. Used by the gate.
  brier_sum         REAL NOT NULL DEFAULT 0.0,
  n                 INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (persona_id, task_type)
);

CREATE TABLE IF NOT EXISTS policy_snapshots (
  snapshot_id       TEXT PRIMARY KEY,
  experiment_name   TEXT NOT NULL,
  created_at        TEXT NOT NULL,
  baseline_reward   REAL NOT NULL,
  payload_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snap_exp ON policy_snapshots(experiment_name, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """SQLite-backed store. Thread-safe via a module-level lock per path."""

    _locks: Dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, sqlite_path: Path | str) -> None:
        self.path = Path(sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with MemoryStore._locks_guard:
            self._lock = MemoryStore._locks.setdefault(str(self.path.resolve()), threading.Lock())
        self._init_schema()

    # ------------------------------------------------------------------
    # connection / schema
    # ------------------------------------------------------------------
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Per-call connection. SQLite is fast enough at our scale, and this
        keeps the API trivially thread-safe under the lock."""
        with self._lock:
            conn = sqlite3.connect(self.path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------
    # tasks
    # ------------------------------------------------------------------
    def upsert_task(self, task_id: str, task_type: str, description: str,
                    *, complexity: float = 0.5,
                    expected_answer: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO tasks(task_id, task_type, complexity, description, expected_answer, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_id) DO UPDATE SET
                     task_type=excluded.task_type, complexity=excluded.complexity,
                     description=excluded.description, expected_answer=excluded.expected_answer,
                     metadata=excluded.metadata""",
                (task_id, task_type, complexity, description, expected_answer,
                 _now(), json.dumps(metadata or {})),
            )

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            return _row_to_dict(row, {"metadata"})

    # ------------------------------------------------------------------
    # trajectories
    # ------------------------------------------------------------------
    def insert_trajectory(self, traj: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO trajectories
                   (trajectory_id, task_id, strategy_id, prompt_template_id, prompt_version,
                    reasoning_depth, status, final_answer, total_latency_ms, total_cost,
                    steps_json, created_at, experiment_name)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    traj["trajectory_id"], traj["task_id"], traj["strategy_id"],
                    traj["prompt_template_id"], traj["prompt_version"],
                    traj["reasoning_depth"], traj["status"], traj.get("final_answer"),
                    int(traj.get("total_latency_ms", 0)), float(traj.get("total_cost", 0.0)),
                    json.dumps(traj.get("steps", [])), _now(),
                    traj.get("experiment_name", "default"),
                ),
            )

    def get_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM trajectories WHERE trajectory_id = ?",
                               (trajectory_id,)).fetchone()
            return _row_to_dict(row, {"steps_json"})

    def trajectories_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trajectories WHERE task_id = ? ORDER BY created_at DESC",
                (task_id,),
            ).fetchall()
            return [_row_to_dict(r, {"steps_json"}) for r in rows]

    # ------------------------------------------------------------------
    # judge scores
    # ------------------------------------------------------------------
    def insert_judge_score(self, score: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO judge_scores
                   (score_id, trajectory_id, persona_id, rubric_json, overall_score,
                    confidence, step_scores_json, critique, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    score["score_id"], score["trajectory_id"], score["persona_id"],
                    json.dumps(score["rubric"]), float(score["overall_score"]),
                    float(score.get("confidence", 0.5)),
                    json.dumps(score.get("step_scores", [])),
                    score.get("critique", ""),
                    _now(),
                ),
            )

    def judge_scores_for(self, trajectory_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM judge_scores WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchall()
            return [_row_to_dict(r, {"rubric_json", "step_scores_json"}) for r in rows]

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------
    def upsert_reward(self, reward: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO rewards
                   (reward_id, trajectory_id, cognitive_momentum, gated_teacher,
                    components_json, final_scalar, created_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(trajectory_id) DO UPDATE SET
                     cognitive_momentum=excluded.cognitive_momentum,
                     gated_teacher=excluded.gated_teacher,
                     components_json=excluded.components_json,
                     final_scalar=excluded.final_scalar""",
                (
                    reward["reward_id"], reward["trajectory_id"],
                    float(reward["cognitive_momentum"]), float(reward["gated_teacher"]),
                    json.dumps(reward["components"]), float(reward["final_scalar"]),
                    _now(),
                ),
            )

    def get_reward(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM rewards WHERE trajectory_id = ?",
                               (trajectory_id,)).fetchone()
            return _row_to_dict(row, {"components_json"})

    # ------------------------------------------------------------------
    # preference pairs
    # ------------------------------------------------------------------
    def insert_preference(self, pair: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO preference_pairs
                   (pair_id, task_type, winner_trajectory_id, loser_trajectory_id, reward_gap, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (pair["pair_id"], pair["task_type"], pair["winner_trajectory_id"],
                 pair["loser_trajectory_id"], float(pair["reward_gap"]), _now()),
            )

    def preferences_for_type(self, task_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM preference_pairs WHERE task_type = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (task_type, limit),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # strategy stats — used by exploration / exploitation
    # ------------------------------------------------------------------
    def update_strategy_stats(self, strategy_id: str, task_type: str,
                              reward: float, success: bool) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM strategy_stats WHERE strategy_id=? AND task_type=?",
                (strategy_id, task_type),
            ).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO strategy_stats
                       (strategy_id, task_type, attempts, successes, reward_sum,
                        reward_sumsq, alpha, beta, last_used_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (strategy_id, task_type, 1, int(success), reward, reward * reward,
                     1.0 + float(success), 1.0 + float(not success), _now()),
                )
            else:
                attempts = row["attempts"] + 1
                successes = row["successes"] + (1 if success else 0)
                alpha = row["alpha"] + (1.0 if success else 0.0)
                beta = row["beta"] + (0.0 if success else 1.0)
                conn.execute(
                    """UPDATE strategy_stats SET
                       attempts=?, successes=?, reward_sum=reward_sum+?,
                       reward_sumsq=reward_sumsq+?, alpha=?, beta=?, last_used_at=?
                       WHERE strategy_id=? AND task_type=?""",
                    (attempts, successes, reward, reward * reward, alpha, beta,
                     _now(), strategy_id, task_type),
                )

    def get_strategy_stats(self, task_type: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM strategy_stats WHERE task_type = ?",
                (task_type,),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # prompt versions
    # ------------------------------------------------------------------
    def upsert_prompt_version(self, *, template_id: str, version: str, role: str,
                              body: str, parent_version: Optional[str] = None,
                              active: bool = False) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO prompt_versions
                   (template_id, version, role, body, parent_version, active, created_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(template_id, version) DO UPDATE SET
                     body=excluded.body, parent_version=excluded.parent_version, active=excluded.active""",
                (template_id, version, role, body, parent_version, int(active), _now()),
            )

    def update_prompt_performance(self, template_id: str, version: str,
                                  reward: float) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT performance_score, attempts FROM prompt_versions WHERE template_id=? AND version=?",
                (template_id, version),
            ).fetchone()
            if row is None:
                return
            attempts = row["attempts"] + 1
            # Running mean
            new_score = (row["performance_score"] * row["attempts"] + reward) / attempts
            conn.execute(
                "UPDATE prompt_versions SET performance_score=?, attempts=? WHERE template_id=? AND version=?",
                (new_score, attempts, template_id, version),
            )

    def get_prompt_version(self, template_id: str, version: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_versions WHERE template_id=? AND version=?",
                (template_id, version),
            ).fetchone()
            return _row_to_dict(row)

    def list_prompt_versions(self, template_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM prompt_versions WHERE template_id=? ORDER BY created_at DESC",
                (template_id,),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    def set_active_prompt(self, template_id: str, version: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE prompt_versions SET active=0 WHERE template_id=?",
                (template_id,),
            )
            conn.execute(
                "UPDATE prompt_versions SET active=1 WHERE template_id=? AND version=?",
                (template_id, version),
            )

    # ------------------------------------------------------------------
    # reflection notes
    # ------------------------------------------------------------------
    def insert_reflection(self, note: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reflection_notes
                   (note_id, task_type, trigger_failure_mode, body, usefulness_score,
                    retrieve_count, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (note["note_id"], note["task_type"], note["trigger_failure_mode"],
                 note["body"], float(note.get("usefulness_score", 0.0)),
                 int(note.get("retrieve_count", 0)), _now()),
            )

    def retrieve_reflections(self, task_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM reflection_notes WHERE task_type=?
                   ORDER BY usefulness_score DESC, created_at DESC LIMIT ?""",
                (task_type, limit),
            ).fetchall()
            ids = [r["note_id"] for r in rows]
            for nid in ids:
                conn.execute(
                    "UPDATE reflection_notes SET retrieve_count=retrieve_count+1 WHERE note_id=?",
                    (nid,),
                )
            return [_row_to_dict(r) for r in rows]

    def update_reflection_usefulness(self, note_id: str, delta: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE reflection_notes SET usefulness_score=usefulness_score+? WHERE note_id=?",
                (float(delta), note_id),
            )

    # ------------------------------------------------------------------
    # gate params (alpha, beta evolve online)
    # ------------------------------------------------------------------
    def get_gate(self, experiment_name: str) -> Optional[Tuple[float, float]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT alpha, beta FROM gate_params WHERE experiment_name=?",
                (experiment_name,),
            ).fetchone()
            return (float(row["alpha"]), float(row["beta"])) if row else None

    def set_gate(self, experiment_name: str, alpha: float, beta: float) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO gate_params(experiment_name, alpha, beta, updated_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(experiment_name) DO UPDATE SET
                     alpha=excluded.alpha, beta=excluded.beta, updated_at=excluded.updated_at""",
                (experiment_name, float(alpha), float(beta), _now()),
            )

    # ------------------------------------------------------------------
    # judge reliability
    # ------------------------------------------------------------------
    def update_judge_reliability(self, persona_id: str, task_type: str,
                                 brier_delta: float) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT brier_sum, n FROM judge_reliability WHERE persona_id=? AND task_type=?",
                (persona_id, task_type),
            ).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO judge_reliability(persona_id, task_type, brier_sum, n)
                       VALUES (?,?,?,?)""",
                    (persona_id, task_type, float(brier_delta), 1),
                )
            else:
                conn.execute(
                    """UPDATE judge_reliability SET brier_sum=brier_sum+?, n=n+1
                       WHERE persona_id=? AND task_type=?""",
                    (float(brier_delta), persona_id, task_type),
                )

    def judge_reliability_score(self, persona_id: str, task_type: str) -> float:
        """Returns a [0,1] reliability score (1 = perfectly calibrated)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT brier_sum, n FROM judge_reliability WHERE persona_id=? AND task_type=?",
                (persona_id, task_type),
            ).fetchone()
            if row is None or row["n"] == 0:
                return 0.7  # neutral prior
            mean_brier = row["brier_sum"] / row["n"]
            # Brier is in [0, 1]; lower is better.
            return max(0.0, min(1.0, 1.0 - mean_brier))

    # ------------------------------------------------------------------
    # policy snapshots
    # ------------------------------------------------------------------
    def insert_snapshot(self, snapshot: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO policy_snapshots
                   (snapshot_id, experiment_name, created_at, baseline_reward, payload_json)
                   VALUES (?,?,?,?,?)""",
                (snapshot["snapshot_id"], snapshot["experiment_name"], _now(),
                 float(snapshot["baseline_reward"]), json.dumps(snapshot["payload"])),
            )

    def latest_snapshots(self, experiment_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM policy_snapshots WHERE experiment_name=?
                   ORDER BY created_at DESC LIMIT ?""",
                (experiment_name, limit),
            ).fetchall()
            return [_row_to_dict(r, {"payload_json"}) for r in rows]

    # ------------------------------------------------------------------
    # housekeeping
    # ------------------------------------------------------------------
    def vacuum(self) -> None:
        with self._conn() as conn:
            conn.execute("VACUUM")


def _row_to_dict(row: Optional[sqlite3.Row], json_fields: Optional[set] = None) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    json_fields = json_fields or set()
    for f in json_fields:
        if f in d and isinstance(d[f], str):
            try:
                # Re-key json fields without the _json suffix for convenience
                key = f[:-5] if f.endswith("_json") else f
                d[key] = json.loads(d[f])
                if key != f:
                    del d[f]
            except (json.JSONDecodeError, TypeError):
                pass
    return d
