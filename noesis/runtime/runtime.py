"""
NOESIS — Agent Runtime
======================

The execution engine. Reads a task, makes decisions, calls tools and the
LLM, accumulates a trajectory, optionally reflects, synthesizes a final
answer, and emits the AgentState for downstream evaluation.

This is the loop. Other components (judges, rewards, policy update)
operate on the trajectories this loop produces.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from noesis.config.schema import NoesisConfig, ReasoningDepth
from noesis.llm.client import LLMRouter
from noesis.memory.store import MemoryStore
from noesis.runtime.action import AgentAction, REFLECTIVE_ACTIONS, TERMINAL_ACTIONS
from noesis.runtime.state import AgentState, FailureMode, StepKind, TrajectoryStep
from noesis.tools import ToolRegistry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentRuntime:
    """The runtime that takes a task and produces a trajectory + answer.

    Composed, not inherited. Each external service (LLM router, tool
    registry, memory store) is passed in, so tests can substitute mocks
    trivially.
    """

    def __init__(
        self,
        config: NoesisConfig,
        llm_router: LLMRouter,
        tool_registry: ToolRegistry,
        memory_store: MemoryStore,
    ):
        self.config = config
        self.llm = llm_router
        self.tools = tool_registry
        self.memory = memory_store

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        *,
        task_id: str,
        task_type: str,
        task_description: str,
        expected_answer: Optional[str] = None,
        complexity: float = 0.5,
        strategy_id: str = "default",
        prompt_template_id: str = "planner.default",
        prompt_version: Optional[str] = None,
        reasoning_depth: Optional[ReasoningDepth] = None,
    ) -> AgentState:
        depth = reasoning_depth or self.config.default_reasoning_depth
        prompt = self.config.prompt(prompt_template_id, version=prompt_version)

        state = AgentState(
            task_id=task_id,
            task_type=task_type,
            task_description=task_description,
            expected_answer=expected_answer,
            complexity=complexity,
            strategy_id=strategy_id,
            prompt_template_id=prompt.template_id,
            prompt_version=prompt.version,
            reasoning_depth=depth.value if isinstance(depth, ReasoningDepth) else str(depth),
            model_endpoint=self.config.route("planner").primary,
            started_at=_now(),
            token_budget_remaining=8000,
            cost_budget_remaining=0.50,
            latency_budget_remaining_ms=60000,
        )

        # Persist the task descriptor so later analysis can recompute features
        self.memory.upsert_task(
            task_id, task_type, task_description,
            complexity=complexity, expected_answer=expected_answer,
        )

        # Pull reflection memory relevant to this task type — Reflexion-style verbal
        state.retrieved_reflections = self.memory.retrieve_reflections(task_type, limit=3)

        try:
            await self._planning_phase(state, prompt.body)
            await self._execution_phase(state)
            await self._reflect_and_maybe_revise(state)
            await self._synthesize_final(state)
            state.status = "complete" if state.failure_mode == FailureMode.NONE else "error"
        except Exception as e:
            state.status = "error"
            state.failure_mode = FailureMode.LLM_REFUSED  # best-effort categorisation
            state.final_answer = f"Runtime error: {e}"

        state.finished_at = _now()
        return state

    # ------------------------------------------------------------------
    # Phase 1 — Planning
    # ------------------------------------------------------------------
    async def _planning_phase(self, state: AgentState, planner_template: str) -> None:
        # Bias depth from complexity unless the caller overrode it
        depth_str = state.reasoning_depth
        # If complexity > 0.7 and depth is shallow, escalate transparently to "standard"
        if state.complexity >= 0.7 and depth_str == ReasoningDepth.SHALLOW.value:
            state.reasoning_depth = ReasoningDepth.STANDARD.value
            depth_str = state.reasoning_depth

        prompt = planner_template.format(
            task=state.task_description,
            tools=self.tools.descriptions_for_prompt(),
            depth=depth_str,
        )

        t0 = time.perf_counter()
        resp = await self.llm.complete("planner", prompt, json_mode=False)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        plan = _safe_parse_json(resp.text)
        step = TrajectoryStep(
            step_id=f"plan_{uuid4().hex[:6]}",
            kind=StepKind.PLAN,
            description="Initial plan generation",
            started_at=_now(),
            finished_at=_now(),
            latency_ms=latency_ms,
            cost=resp.cost_estimate,
            chosen_action=AgentAction.CHOOSE_PROMPT_STRATEGY.value,
            chosen_model=state.model_endpoint,
            reasoning_depth=depth_str,
            output=plan if plan else {"raw": resp.text},
            confidence=float(plan.get("confidence", 0.6)) if isinstance(plan, dict) else 0.5,
        )
        if plan is None:
            step.failure_mode = FailureMode.PARSING_ERROR
            step.error = "planner did not return valid JSON"
        state.add_step(step)
        state.plan = plan if isinstance(plan, dict) else None

    # ------------------------------------------------------------------
    # Phase 2 — Execution
    # ------------------------------------------------------------------
    async def _execution_phase(self, state: AgentState) -> None:
        if not state.plan or "steps" not in state.plan:
            # Without a parseable plan we still try to synthesize an answer
            # from whatever the planner returned.
            return

        max_steps = min(self.config.max_steps_per_task, len(state.plan["steps"]) + 2)
        executed = 0
        for plan_step in state.plan["steps"]:
            if executed >= max_steps:
                break
            if state.cost_budget_remaining <= 0 or state.latency_budget_remaining_ms <= 0:
                state.failure_mode = FailureMode.BUDGET_EXCEEDED
                break

            kind = plan_step.get("kind", "tool_call")
            if kind == "tool_call":
                await self._execute_tool_step(state, plan_step)
            elif kind == "synthesize":
                # Synthesis-as-step is handled in the synthesize phase; record
                # a trajectory marker so downstream can see the plan honored it.
                state.add_step(TrajectoryStep(
                    step_id=plan_step.get("step_id", f"syn_{uuid4().hex[:6]}"),
                    kind=StepKind.SYNTHESIZE,
                    description=plan_step.get("description", ""),
                    started_at=_now(), finished_at=_now(),
                    latency_ms=0, cost=0.0,
                    chosen_action=AgentAction.SYNTHESIZE_ANSWER.value,
                    confidence=0.6,
                ))
            else:
                # Unknown step kind — record as no-op
                state.add_step(TrajectoryStep(
                    step_id=plan_step.get("step_id", f"noop_{uuid4().hex[:6]}"),
                    kind=StepKind.PLAN,
                    description=f"unrecognised step kind '{kind}'",
                    started_at=_now(), finished_at=_now(),
                    latency_ms=0, cost=0.0,
                    chosen_action="",
                    confidence=0.3,
                ))
            executed += 1

    async def _execute_tool_step(self, state: AgentState, plan_step: Dict[str, Any]) -> None:
        tool_name = plan_step.get("tool")
        if not tool_name:
            return
        try:
            tool = self.tools.get(tool_name)
        except KeyError:
            state.add_step(TrajectoryStep(
                step_id=plan_step.get("step_id", f"tool_{uuid4().hex[:6]}"),
                kind=StepKind.TOOL_CALL,
                description=plan_step.get("description", ""),
                started_at=_now(), finished_at=_now(),
                latency_ms=0, cost=0.0,
                chosen_action=AgentAction.INVOKE_TOOL.value,
                chosen_tool=tool_name,
                error=f"unknown tool '{tool_name}'",
                failure_mode=FailureMode.TOOL_ERROR,
                confidence=0.0,
            ))
            return

        # Build kwargs for the tool. Heuristic: extract numeric expressions for
        # the calculator, otherwise pass the task description as the query.
        kwargs: Dict[str, Any] = {}
        if tool_name == "calculator":
            kwargs["expression"] = _extract_arithmetic(state.task_description) or state.task_description
        elif tool_name == "text_search":
            kwargs["query"] = state.task_description
            kwargs["k"] = 3
        elif tool_name == "code_runner":
            kwargs["code"] = plan_step.get("code", "result = None")
        else:
            kwargs = plan_step.get("input", {}) or {}

        started = _now()
        result = await tool.invoke(**kwargs)
        finished = _now()
        success = bool(result.get("ok", True))
        state.add_step(TrajectoryStep(
            step_id=plan_step.get("step_id", f"tool_{uuid4().hex[:6]}"),
            kind=StepKind.TOOL_CALL,
            description=plan_step.get("description", ""),
            started_at=started, finished_at=finished,
            latency_ms=int(result.get("latency_ms", 0)),
            cost=float(result.get("cost", 0.0)),
            chosen_action=AgentAction.INVOKE_TOOL.value,
            chosen_tool=tool_name,
            output=result,
            error=None if success else result.get("error", "tool failed"),
            failure_mode=FailureMode.NONE if success else FailureMode.TOOL_ERROR,
            confidence=0.8 if success else 0.2,
        ))

    # ------------------------------------------------------------------
    # Phase 3 — Reflection (optional)
    # ------------------------------------------------------------------
    async def _reflect_and_maybe_revise(self, state: AgentState) -> None:
        if state.reasoning_depth == ReasoningDepth.SHALLOW.value:
            return
        if state.reflection_passes >= self.config.max_reflection_passes:
            return

        # Decide whether reflection is warranted. Triggers: any tool error,
        # any step with confidence < 0.5, or budget concerns.
        warrant = (
            any(s.failure_mode == FailureMode.TOOL_ERROR for s in state.steps)
            or any(s.confidence < 0.5 for s in state.steps)
            or state.cost_budget_remaining < 0.05
        )
        if not warrant:
            return

        try:
            reflect_template = self.config.prompt("reflector.default")
        except KeyError:
            return
        interim = self._interim_answer_from_steps(state)
        trajectory_dump = self._dump_trajectory_for_prompt(state)
        prompt = reflect_template.body.format(
            trajectory=trajectory_dump,
            answer=interim,
        )

        t0 = time.perf_counter()
        try:
            resp = await self.llm.complete("planner", prompt, json_mode=False)
        except Exception as e:
            return
        latency_ms = int((time.perf_counter() - t0) * 1000)
        parsed = _safe_parse_json(resp.text) or {}

        reflect_step = TrajectoryStep(
            step_id=f"refl_{uuid4().hex[:6]}",
            kind=StepKind.REFLECT,
            description=parsed.get("reason", "reflection pass"),
            started_at=_now(), finished_at=_now(),
            latency_ms=latency_ms, cost=resp.cost_estimate,
            chosen_action=AgentAction.INVOKE_REFLECTION.value,
            output=parsed,
            confidence=0.6,
        )
        state.add_step(reflect_step)
        state.reflection_passes += 1

        if parsed.get("revise"):
            # Mark a synthetic revise step. We don't re-plan from scratch — we
            # add a directive that the synthesizer can incorporate. Cheaper,
            # avoids loop blowup, still distinguishable in the trajectory.
            state.add_step(TrajectoryStep(
                step_id=f"rev_{uuid4().hex[:6]}",
                kind=StepKind.REVISE,
                description=parsed.get("suggested_action", "revise"),
                started_at=_now(), finished_at=_now(),
                latency_ms=0, cost=0.0,
                chosen_action=AgentAction.REVISE_PLAN.value,
                output=parsed,
                revises_step_id=state.steps[-2].step_id if len(state.steps) > 1 else None,
                confidence=0.55,
            ))

    # ------------------------------------------------------------------
    # Phase 4 — Synthesis
    # ------------------------------------------------------------------
    async def _synthesize_final(self, state: AgentState) -> None:
        try:
            template = self.config.prompt("synthesizer.default")
        except KeyError:
            # Fallback synthesis: stitch tool outputs together verbally
            state.final_answer = self._interim_answer_from_steps(state)
            state.final_confidence = 0.4
            return

        trajectory_dump = self._dump_trajectory_for_prompt(state)
        prompt = template.body.format(
            task=state.task_description,
            trajectory=trajectory_dump,
        )

        t0 = time.perf_counter()
        try:
            resp = await self.llm.complete("synthesizer", prompt, json_mode=False)
        except Exception as e:
            state.final_answer = f"synthesis failed: {e}"
            state.final_confidence = 0.2
            return
        latency_ms = int((time.perf_counter() - t0) * 1000)

        state.final_answer = resp.text.strip()
        # Confidence heuristic: average of step confidences, weighted toward
        # later steps. A trajectory whose late steps look good is more
        # trustworthy than one whose late steps thrash.
        weights = [(i + 1) for i in range(len(state.steps))]
        if weights:
            wc = sum(w * s.confidence for w, s in zip(weights, state.steps))
            ws = sum(weights)
            state.final_confidence = max(0.0, min(1.0, wc / ws)) if ws else 0.5
        else:
            state.final_confidence = 0.5

        state.add_step(TrajectoryStep(
            step_id=f"syn_{uuid4().hex[:6]}",
            kind=StepKind.SYNTHESIZE,
            description="final answer synthesis",
            started_at=_now(), finished_at=_now(),
            latency_ms=latency_ms,
            cost=resp.cost_estimate,
            chosen_action=AgentAction.SYNTHESIZE_ANSWER.value,
            chosen_model=state.model_endpoint,
            output={"text": state.final_answer},
            confidence=state.final_confidence,
        ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dump_trajectory_for_prompt(state: AgentState) -> str:
        lines: List[str] = []
        for i, s in enumerate(state.steps):
            head = f"[{i+1}] {s.kind.value} ({s.chosen_action})"
            if s.chosen_tool:
                head += f" tool={s.chosen_tool}"
            lines.append(head)
            if isinstance(s.output, dict):
                # Compact display of the most informative fields
                if "result" in s.output:
                    lines.append(f"    output: {s.output['result']}")
                elif "passages" in s.output:
                    snippets = [p.get("title", "") for p in s.output.get("passages", [])[:3]]
                    lines.append(f"    output: passages={snippets}")
                elif "revise" in s.output:
                    lines.append(f"    output: revise={s.output['revise']} reason={s.output.get('reason','')}")
                elif "text" in s.output:
                    lines.append(f"    output: {s.output['text'][:160]}")
            elif s.output is not None:
                lines.append(f"    output: {str(s.output)[:160]}")
            if s.error:
                lines.append(f"    error: {s.error}")
        return "\n".join(lines)

    @staticmethod
    def _interim_answer_from_steps(state: AgentState) -> str:
        # Best tool output we have
        for s in reversed(state.steps):
            if isinstance(s.output, dict):
                if "result" in s.output and s.output["result"] is not None:
                    return f"interim: {s.output['result']}"
                if "passages" in s.output and s.output["passages"]:
                    titles = [p.get("title") for p in s.output["passages"][:2]]
                    return f"interim from passages: {titles}"
        return "interim: (no tool output yet)"


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------
def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction. Mocks and real LLMs both sometimes wrap
    JSON in code fences or chatty preamble; this finds the first {...}."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first balanced {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


_ARITHMETIC_RE = re.compile(r"[\d\s\.\+\-\*\/\(\)\,]+")


def _extract_arithmetic(text: str) -> Optional[str]:
    """Pull a likely arithmetic expression out of a natural-language task."""
    candidates = _ARITHMETIC_RE.findall(text)
    candidates = [c.strip() for c in candidates if any(op in c for op in "+-*/")]
    if not candidates:
        return None
    return max(candidates, key=len).strip(" ,")
