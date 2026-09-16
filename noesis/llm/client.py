"""
NOESIS — LLM Client Abstraction
================================

Two concrete clients, both exposing the same async interface:

  * `OpenAIClient` — hits the real OpenAI Chat Completions API (also works
    for Azure OpenAI endpoints via `base_url`).
  * `MockClient` — deterministic, in-process. Required because the rest of
    the system needs to run end-to-end in CI / on a laptop / on a flight
    without an API key. The mock's outputs are structured enough that the
    judges, planners, and synthesizers all behave correctly under it.

The `LLMRouter` picks the right client per role with fallback chains. If
the primary fails (transport, rate limit, key missing), it tries the
fallback list before raising.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from noesis.config.schema import LLMProvider, ModelEndpoint


# ---------------------------------------------------------------------------
# Response envelope — every client returns this shape
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    text: str
    model_id: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str = "stop"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost_estimate(self) -> float:
        """Heuristic cost (USD) — fine-grained per-model pricing lives in routing."""
        # Cheap-rough estimate: $0.10 per 1k prompt + $0.40 per 1k completion
        return (self.prompt_tokens / 1000.0) * 0.10 + (self.completion_tokens / 1000.0) * 0.40


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class BaseLLMClient(ABC):
    def __init__(self, endpoint: ModelEndpoint):
        self.endpoint = endpoint

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        ...

    async def complete_text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Convenience wrapper for single-turn calls."""
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return await self.complete(
            msgs, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )


# ---------------------------------------------------------------------------
# OpenAI client (also covers Azure OpenAI via base_url override)
# ---------------------------------------------------------------------------
class OpenAIClient(BaseLLMClient):
    """Real OpenAI client. Requires `openai>=1.0`. If the library or key is
    missing, surface a clear error rather than crashing on first call."""

    def __init__(self, endpoint: ModelEndpoint):
        super().__init__(endpoint)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI client unavailable: `pip install openai>=1.0`"
            ) from e

        api_key = os.environ.get(endpoint.api_key_env or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OpenAIClient '{endpoint.name}' needs api key in env var "
                f"'{endpoint.api_key_env or 'OPENAI_API_KEY'}'"
            )

        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": endpoint.timeout_seconds}
        if endpoint.base_url:
            kwargs["base_url"] = endpoint.base_url
        self._client = AsyncOpenAI(**kwargs)

    async def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        t0 = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": self.endpoint.model_id,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.endpoint.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.endpoint.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model_id=self.endpoint.model_id,
            provider="openai",
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason or "stop",
            raw={"id": resp.id},
        )


# ---------------------------------------------------------------------------
# Mock client — deterministic, role-aware, structured
# ---------------------------------------------------------------------------
class MockClient(BaseLLMClient):
    """Deterministic in-process LLM.

    Why this exists: the entire NOESIS loop — runtime, jury, rewards, policy
    update — must be testable without network. The mock parses the prompt
    for role hints (`You are a planner`, `You are a judge`, etc.) and emits
    role-appropriate, structurally-valid output.

    The output is *deterministic* given the prompt + a per-instance seed, so
    judge scores, preference pairs, and policy updates are reproducible
    test fixtures.
    """

    def __init__(self, endpoint: ModelEndpoint):
        super().__init__(endpoint)
        self._rng = random.Random(hashlib.sha1(endpoint.name.encode()).hexdigest()[:8])

    async def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        await asyncio.sleep(0.001)  # let event loop yield, like a real client
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        role = self._infer_role(prompt_text)
        body = self._render(role, prompt_text)
        # Approximate token counts — 1 token ~= 4 chars
        ptoks = max(1, len(prompt_text) // 4)
        ctoks = max(1, len(body) // 4)
        return LLMResponse(
            text=body,
            model_id=self.endpoint.model_id,
            provider="mock",
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            latency_ms=2,
            finish_reason="stop",
            raw={"role_inferred": role},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_role(prompt: str) -> str:
        lower = prompt.lower()
        # Specific role markers used by NOESIS prompts
        if "you are a planner" in lower:
            return "planner"
        if "you are a synthesizer" in lower:
            return "synthesizer"
        if "you are a reflector" in lower:
            return "reflector"
        if "you are an evaluator" in lower or "you are a judge" in lower or "score the trajectory" in lower:
            return "judge"
        if "rewrite" in lower or "reformulate" in lower:
            return "rewrite"
        if "mutate" in lower and "prompt" in lower:
            return "mutator"
        return "generic"

    # ------------------------------------------------------------------
    def _render(self, role: str, prompt: str) -> str:
        if role == "planner":
            return self._mock_plan(prompt)
        if role == "synthesizer":
            return self._mock_synthesis(prompt)
        if role == "reflector":
            return self._mock_reflection(prompt)
        if role == "judge":
            return self._mock_judge(prompt)
        if role == "mutator":
            return self._mock_mutation(prompt)
        return self._mock_generic(prompt)

    # ------------------------------------------------------------------
    # Per-role outputs
    # ------------------------------------------------------------------
    def _mock_plan(self, prompt: str) -> str:
        task = _extract_field(prompt, "task")
        tools = _extract_field(prompt, "tools") or "calculator, text_search"
        # Build a plausible JSON plan whose shape downstream parsers expect
        steps: List[Dict[str, Any]] = []
        # Decompose task: if it looks math-y, use calculator; else search
        if re.search(r"\d|sum|product|mean|average|equation|solve", task.lower()):
            steps.append({
                "step_id": "s1",
                "description": f"Compute the numeric component of: {task[:80]}",
                "kind": "tool_call",
                "tool": "calculator",
                "expected_output": "a numeric result",
            })
        else:
            steps.append({
                "step_id": "s1",
                "description": f"Search internal knowledge for relevant context on: {task[:80]}",
                "kind": "tool_call",
                "tool": "text_search",
                "expected_output": "passages relevant to the question",
            })
        steps.append({
            "step_id": "s2",
            "description": "Synthesize the final answer using step 1's output.",
            "kind": "synthesize",
            "expected_output": "a direct natural-language answer",
        })
        plan = {
            "frame": f"Restated request: {task[:200]}",
            "approach": "Decompose into a tool call followed by synthesis. Keep reasoning shallow unless the tool returns conflicting evidence.",
            "steps": steps,
            "confidence": 0.7 + (self._rng.random() * 0.2 - 0.1),
        }
        return json.dumps(plan)

    def _mock_synthesis(self, prompt: str) -> str:
        task = _extract_field(prompt, "task") or "the task"
        trajectory = _extract_field(prompt, "trajectory") or ""
        # Pull any tool outputs out of the trajectory text — the runtime
        # serializes them as "output: <value>" lines.
        nums = re.findall(r"output:\s*([^\n]+)", trajectory)
        observed = nums[-1] if nums else ""
        prefix = f"Based on the executed trajectory for the task '{_short(task, 80)}', "
        if observed:
            return prefix + f"the resolved answer is: {observed.strip()}."
        return prefix + "the synthesized answer is a coherent summary of the steps taken."

    def _mock_reflection(self, prompt: str) -> str:
        answer = _extract_field(prompt, "answer") or ""
        # Reflect: trigger revision rarely so loops terminate
        revise = self._rng.random() < 0.15
        return json.dumps({
            "revise": revise,
            "reason": "Answer appears underspecified." if revise else "Answer is coherent.",
            "suggested_action": "retry_with_alternative_plan" if revise else "accept",
        })

    def _mock_judge(self, prompt: str) -> str:
        # Build a plausible rubric scoring.
        # We bias by features extractable from the prompt — if "tool_call" appears,
        # tool_use_quality scores higher; if the prompt mentions "wrong" or "error",
        # correctness drops.
        rng = random.Random(_hash(prompt))
        base = {dim: 0.55 + rng.random() * 0.35 for dim in [
            "correctness", "faithfulness", "task_completion", "instruction_adherence",
            "tool_use_quality", "efficiency", "robustness", "novelty",
            "safety", "reproducibility",
        ]}
        # Heuristic adjustments
        low = prompt.lower()
        if "tool_call" in low or "tool used" in low:
            base["tool_use_quality"] = min(1.0, base["tool_use_quality"] + 0.10)
        if "error" in low or "failed" in low:
            base["correctness"] = max(0.0, base["correctness"] - 0.25)
            base["robustness"] = max(0.0, base["robustness"] - 0.15)
        # Per-step scores (1 per traj step)
        n_steps = max(1, prompt.lower().count("step_id"))
        step_scores = [
            {"step_id": f"s{i+1}",
             "score": round(0.4 + rng.random() * 0.5, 3),
             "confidence": round(0.5 + rng.random() * 0.4, 3),
             "rationale": "Step executed as planned."}
            for i in range(min(n_steps, 8))
        ]
        return json.dumps({
            "rubric": {k: round(v, 3) for k, v in base.items()},
            "overall_score": round(sum(base.values()) / len(base), 3),
            "confidence": round(0.55 + rng.random() * 0.35, 3),
            "step_scores": step_scores,
            "critique": "Trajectory follows the plan; opportunities to deepen reasoning on step 2.",
        })

    def _mock_mutation(self, prompt: str) -> str:
        # Produce a mutated prompt: append a constraint or reorder a clause
        base = _extract_field(prompt, "current") or ""
        mutations = [
            base + "\n\nBe explicit about uncertainty and surface it inline.",
            base + "\n\nWhen a tool returns conflicting evidence, prefer the most recent source.",
            base + "\n\nIf the task involves arithmetic, double-check via an independent computation.",
            base.replace("You are a planner.", "You are a planner. Be terse.") if "You are a planner." in base else base + "\n\nBe terse.",
        ]
        return mutations[self._rng.randrange(len(mutations))]

    def _mock_generic(self, prompt: str) -> str:
        return "Mock response: " + _short(prompt, 200)


# ---------------------------------------------------------------------------
# Router — picks client per role, with fallback
# ---------------------------------------------------------------------------
class LLMRouter:
    def __init__(self, config: "NoesisConfig"):
        from noesis.config.schema import NoesisConfig  # type: ignore  # noqa
        self.config = config
        self._clients: Dict[str, BaseLLMClient] = {}

    def _get_client(self, endpoint_name: str) -> BaseLLMClient:
        cached = self._clients.get(endpoint_name)
        if cached is not None:
            return cached
        ep = self.config.endpoint(endpoint_name)
        client: BaseLLMClient
        if ep.provider == LLMProvider.OPENAI or ep.provider == LLMProvider.AZURE_OPENAI:
            client = OpenAIClient(ep)
        elif ep.provider == LLMProvider.MOCK:
            client = MockClient(ep)
        elif ep.provider == LLMProvider.ANTHROPIC:
            # Anthropic support is intentionally not bundled in the demo loop;
            # the mock covers offline runs and OpenAI covers cloud runs. To
            # add Anthropic, mirror OpenAIClient against `anthropic.AsyncAnthropic`.
            raise NotImplementedError(
                "Anthropic provider stub: bundle anthropic SDK and implement here."
            )
        else:
            raise ValueError(f"Unknown provider: {ep.provider}")
        self._clients[endpoint_name] = client
        return client

    async def complete(self, role: str, prompt: str, *,
                       system: Optional[str] = None,
                       json_mode: bool = False,
                       temperature: Optional[float] = None) -> LLMResponse:
        """Try the primary endpoint for `role`; on failure, fall back."""
        route = self.config.route(role)
        candidates = [route.primary] + list(route.fallback)
        last_error: Optional[Exception] = None
        for name in candidates:
            try:
                client = self._get_client(name)
                return await client.complete_text(
                    prompt, system=system, json_mode=json_mode, temperature=temperature
                )
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(
            f"All endpoints failed for role '{role}'. Last error: {last_error}"
        )

    async def complete_for_endpoint(
        self, endpoint_name: str, prompt: str, *,
        system: Optional[str] = None, json_mode: bool = False,
    ) -> LLMResponse:
        """Used by the jury — judges are pinned to specific endpoints."""
        client = self._get_client(endpoint_name)
        return await client.complete_text(prompt, system=system, json_mode=json_mode)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _extract_field(prompt: str, name: str) -> str:
    """Extract `{name}:` value from a template-filled prompt. Best-effort."""
    pattern = rf"{re.escape(name)}\s*[:=]\s*(.+?)(?:\n[A-Z][a-z]+:|\Z)"
    m = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _short(text: str, n: int) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 3] + "..."


def _hash(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)
