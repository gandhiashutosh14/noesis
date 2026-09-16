"""
NOESIS — Prompt Evolution
=========================

When a prompt version is reliably losing in the preference pairs, the
mutator proposes a new version. The new version is registered in the
prompt_versions table with a parent pointer for rollback, marked inactive
by default, and promoted to active only after it has won enough
preference pairs against the parent.

Mutation is LLM-driven (the planner-role model is asked to rewrite the
prompt with a specific failure mode in mind). The mutator pulls the
single most adversarial critique from recent verdicts to condition the
mutation — that critique becomes the "what should change" signal.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from noesis.llm.client import LLMRouter
from noesis.memory.store import MemoryStore


MUTATE_PROMPT_TEMPLATE = """\
You are a prompt engineer. You will mutate an existing prompt to address
the failure mode below.

Current prompt (template_id={template_id}, version={version}):
---
{current}
---

Observed failure pattern, derived from low-scoring trajectories:
{failure_pattern}

Specific critique that recurs from the jury:
{critique}

Rewrite the prompt. Keep its placeholders ({{task}}, {{depth}}, {{tools}})
exactly as they are. Make the changes targeted, not cosmetic. Output ONLY
the new prompt text, no commentary.
"""


@dataclass
class MutationResult:
    template_id: str
    parent_version: str
    new_version: str
    new_body: str


# ---------------------------------------------------------------------------
async def mutate_prompt(
    *,
    template_id: str,
    parent_version: str,
    current_body: str,
    failure_pattern: str,
    critique: str,
    llm: LLMRouter,
    memory: MemoryStore,
) -> Optional[MutationResult]:
    """Produce + register a new prompt version. Inactive by default."""
    prompt = MUTATE_PROMPT_TEMPLATE.format(
        template_id=template_id,
        version=parent_version,
        current=current_body,
        failure_pattern=failure_pattern,
        critique=critique,
    )

    try:
        resp = await llm.complete("planner", prompt)
    except Exception:
        return None

    new_body = (resp.text or "").strip()
    # Sanity check: must still have at least {task} or {trajectory}
    placeholders = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", new_body)
    parent_placeholders = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", current_body)
    if not set(placeholders) >= set(parent_placeholders):
        # The mutator dropped required placeholders — discard
        return None
    if new_body == current_body or len(new_body) < 30:
        return None

    new_version = _bump_version(parent_version)
    memory.upsert_prompt_version(
        template_id=template_id,
        version=new_version,
        role=_role_for(template_id),
        body=new_body,
        parent_version=parent_version,
        active=False,
    )
    return MutationResult(
        template_id=template_id,
        parent_version=parent_version,
        new_version=new_version,
        new_body=new_body,
    )


# ---------------------------------------------------------------------------
def maybe_promote_mutation(
    *,
    memory: MemoryStore,
    template_id: str,
    candidate_version: str,
    parent_version: str,
    min_attempts: int = 4,
    win_threshold: float = 0.05,
) -> bool:
    """Promote candidate_version to active if it has accumulated enough
    evidence of outperforming its parent.

    Evidence == difference in `performance_score` between candidate and
    parent in the prompt_versions table. Requires both to have at least
    `min_attempts` attempts.
    """
    candidate = memory.get_prompt_version(template_id, candidate_version)
    parent = memory.get_prompt_version(template_id, parent_version)
    if not candidate or not parent:
        return False
    if candidate["attempts"] < min_attempts or parent["attempts"] < min_attempts:
        return False
    delta = candidate["performance_score"] - parent["performance_score"]
    if delta < win_threshold:
        return False
    memory.set_active_prompt(template_id, candidate_version)
    return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _bump_version(v: str) -> str:
    """Naive semver bump on the patch component."""
    parts = v.split(".")
    if len(parts) != 3:
        return v + "-mut"
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{major}.{minor}.{patch + 1}"
    except ValueError:
        return v + "-mut"


_ROLE_RE = re.compile(r"^([a-z_]+)\.")


def _role_for(template_id: str) -> str:
    m = _ROLE_RE.match(template_id)
    return m.group(1) if m else "unknown"
