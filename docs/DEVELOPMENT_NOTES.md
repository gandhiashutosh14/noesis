# Development notes

How NOESIS was built, refined and verified. Where an artifact did not survive, this file says so.

## Original development (before this repository existed)

NOESIS was designed by Ashutosh Gandhi before 2026-09-16. The framework, the two reward
formulations (Cognitive Momentum and the Gated Teacher gate), the strategy book, the SQLite policy
store and the end-to-end test all date from that work. Only the code and its original README
survived from that period.

Two sentences in the original text referred to context that no reader of this repository has: the
README mentioned "the `agent_registry` module debugged in the prior turn", and a docstring in
`rewards/cognitive_momentum.py` cited an image from the design notes. Both were rewritten during
the refinement below.

## Refinement, 2026-09-16

### Planning

1. NOESIS was chosen as the strongest personal agentic project in the author's local work.
2. A confidentiality review found that the shipped example tasks and the built-in text-search
   corpus used domain vocabulary from the author's professional work. Not a secret, but domain bleed a
   knowledgeable reviewer would notice. Replacing them was made a precondition for publishing.
3. The per-project procedure: stage outside the source folder, scrub, add `.gitignore` and MIT
   licence, write a README that passes a 15-second recruiter read, verify every documented
   command by running it, then publish with `gh`.

### Iterations

- Copied the package to a clean staging directory and created a virtual environment
  (`pydantic 2.13.5`, `PyYAML 6.0.3`, `pytest 9.1.1`).
- Ran the untouched end-to-end test first to establish a baseline. It passed:
  8 cycles, 3 judges, 5 strategies, mean reward 0.3417, gate α 3.943 / β 2.057, 1 snapshot,
  0 preference pairs.
- Replaced `examples/task_simple_qa.json` and `examples/task_multi_step.json` with neutral tasks
  (Thompson sampling; precision vs recall), replaced the four domain-specific passages in the
  text-search corpus with passages on Thompson sampling, UCB, precision/recall and the Brier score,
  and changed the knowledge-lookup fixture in `tests/test_e2e.py` to match.
- Rewrote the design-notes docstring sentence to describe the credit-assignment problem on its own
  terms.
- Added a `test_end_to_end()` function so the existing script-style test is collected by pytest.
- Added `pyproject.toml` (setuptools, console script `noesis`, package data for the YAML config and
  example JSONs), `requirements.txt`, `.gitignore`, MIT `LICENSE`, and a new README. The original
  README's mathematical description of the two rewards was kept; the sections tied to an internal
  agent registry were generalised.

### Debugging

- Capturing the `self-improve` output failed once because the summary script wrote to `/tmp`,
  which the Windows Python interpreter does not resolve. Re-ran with a Windows path.
- `show-stats` initialised a `judge_reliability` key and never populated it. Fixed by querying the
  store's per-judge reliability score for the requested task type, using the same
  `judge_reliability_score` call the e2e test already exercised.

### Refinements and verification

Every command in the README was executed from a fresh checkout state (state database deleted
first):

| Command | Result |
|---|---|
| `pip install -e ".[dev]"` | exit 0 |
| `pytest -q` | `1 passed in 1.12s` |
| `noesis.cli.main run-task …47 * 13 + 28…` | final answer 639, reward 0.341, consensus 0.759, agreement 0.962 |
| `noesis.cli.main self-improve --tasks noesis/examples --cycles 3 --snapshot-every 2` | 9 cycles, mean reward 0.3146, 4 snapshots |
| `noesis.cli.main show-stats --task-type arithmetic` | 4 strategy rows, gate (3.710, 2.290) |
| `noesis --help` (console script) | usage printed |

SQLite after the self-improve run: tasks 3, trajectories 9, judge_scores 27, rewards 9,
reflection_notes 9, strategy_stats 8, policy_snapshots 4, gate_params 1, judge_reliability 9,
preference_pairs 0, prompt_versions 0.

A final scan of the staged tree for confidential identifiers returned only benign substrings
(`--task-id` matching an `sk-` pattern) and the author name in the licence.

### What changed and what did not

Changed in the refinement: example fixtures, corpus passages, one test fixture string, one
docstring sentence, one CLI query, packaging and documentation files.

Not changed: the runtime, jury, reward, policy, memory and loop modules. The reward numbers in
this document are outputs of the original code under the mock provider and are reproducible; they
are not claims about task quality.

### Outcome

Published to https://github.com/gandhiashutosh14/noesis as a public repository with a fresh
history. Commits are grouped as: framework code, packaging, documentation. The pre-scrub versions
of the example and corpus files were deliberately never committed, so the history is clean rather
than cleaned.

## Evidence report, 2026-09-16

This increment publishes evaluation evidence from the existing loop rather than building a
separate evaluation framework, and keeps recorded outputs, fixtures and model calls
distinguishable.

- Added `noesis/report/evidence.py` and the `report` CLI subcommand. It reads the store only and
  recomputes per-trajectory step kinds, tool calls and errors, thrash steps (the CMR definition:
  a revise not preceded by a reflect), repeated revisions of one step, judge mean and spread,
  agreement and the contested flag (the jury formulas), and per-judge deviation from the jury
  mean. Where the reward layer stored something recomputable (thrash step ids, step-gate count)
  the two are compared and discrepancies are listed.
- The verdict's trimmed-mean consensus is not persisted, so the report does not claim to
  recompute it; it reports the plain mean of the judges' overall scores and says so.
- Ran the three example tasks for three cycles with the mock provider in an empty directory and
  committed the report and its JSON, with a provenance note naming the revision and the commands.
  The store itself is not committed (`*.db` is ignored).
- Observed in that run and recorded in the README: 3 of 9 tool calls errored, all from the
  multi-step example, where the mock planner passes the calculator an expression the AST
  evaluator refuses; the trajectory completes and is scored anyway.

| Check | Result |
|---|---|
| `pytest -q` | 3 passed (the closed loop; thrash and repeat definitions; report vs store) |
| `self-improve --tasks noesis/examples --cycles 3 --snapshot-every 2` | 9 cycles, mean reward 0.3175 |
| `report --db noesis_state.db ...` | 9 trajectories, 27 judge scores, 9 rewards, 0 discrepancies, 0 contested, 0 thrash, 3 tool errors |

Not claimed: task quality, judge calibration, improvement across cycles. Under the mock provider
every judge score is a deterministic fixture, and the report says so in its own footer.
