"""
NOESIS — Configuration schemas
==============================

The control room. Every knob is here, typed, validated, and documented.
Nothing in the rest of the system reads environment variables directly;
everything goes through this layer.

The defaults are biased toward "this will run on a laptop without an API
key" — set OPENAI_API_KEY (or NOESIS_LLM_PROVIDER=mock anyway) and the
mock client kicks in. Real provider keys override automatically.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    MOCK = "mock"           # deterministic in-process LLM for tests + offline


class ReasoningDepth(str, Enum):
    SHALLOW = "shallow"     # one-shot answer, no tool use
    STANDARD = "standard"   # plan + 1-3 tool calls + answer
    DEEP = "deep"           # plan + reflect + retry + multi-tool


class ExplorationStrategy(str, Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"             # upper confidence bound
    THOMPSON = "thompson"   # Thompson sampling over Beta posteriors


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------
class ModelEndpoint(BaseModel):
    """One LLM endpoint."""
    model_config = ConfigDict(extra="forbid")

    name: str                                       # logical name; referenced by other configs
    provider: LLMProvider
    model_id: str                                   # provider-side model name (e.g. "gpt-4o-mini")
    api_key_env: Optional[str] = None               # env var holding the key
    base_url: Optional[str] = None                  # for Azure / self-hosted
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=128, le=128_000)
    timeout_seconds: int = Field(default=60, ge=5, le=600)
    role_tags: List[str] = Field(default_factory=list,
                                 description="Logical roles this endpoint can serve: 'planner','judge','student','teacher'.")


class ModelRouting(BaseModel):
    """Which endpoint to use for which role, with fallback chains."""
    model_config = ConfigDict(extra="forbid")

    primary: str                                    # name of a ModelEndpoint
    fallback: List[str] = Field(default_factory=list,
                                description="Ordered fallback names. Tried on transport failure.")


class JudgeConfig(BaseModel):
    """One judge persona's config."""
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    display_name: str
    endpoint_name: str
    # Each judge can weight rubric dimensions differently in its own response
    rubric_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "correctness": 1.0, "faithfulness": 0.7, "task_completion": 0.9,
            "instruction_adherence": 0.8, "tool_use_quality": 0.6,
            "efficiency": 0.4, "robustness": 0.5, "novelty": 0.3,
            "safety": 1.0, "reproducibility": 0.4,
        }
    )
    persona_prompt: str = ""

    @field_validator("rubric_weights")
    @classmethod
    def _all_positive(cls, v):
        for k, w in v.items():
            if w < 0:
                raise ValueError(f"Rubric weight for '{k}' must be >= 0")
        return v


class JuryConfig(BaseModel):
    """The full jury setup."""
    model_config = ConfigDict(extra="forbid")

    judges: List[JudgeConfig] = Field(min_length=2,
                                       description="At least two judges — single-judge eval is not a jury.")
    disagreement_threshold: float = Field(default=0.25, ge=0.0, le=1.0,
                                          description="Stddev across judges above this triggers 'contested' flag.")
    consensus_method: Literal["mean", "trimmed_mean", "median"] = "trimmed_mean"


class RLConfig(BaseModel):
    """Hyperparameters for the self-improvement loop."""
    model_config = ConfigDict(extra="forbid")

    exploration_strategy: ExplorationStrategy = ExplorationStrategy.THOMPSON
    epsilon: float = Field(default=0.15, ge=0.0, le=1.0)   # for epsilon-greedy
    ucb_c: float = Field(default=1.4, ge=0.0, le=10.0)     # exploration weight for UCB
    preference_margin: float = Field(default=0.10, ge=0.0, le=1.0,
                                     description="Min CMR gap between winner and loser to form a preference pair.")
    regression_guard_threshold: float = Field(default=0.05, ge=0.0, le=1.0,
                                              description="If new policy regresses by more than this, rollback.")
    rollback_window: int = Field(default=10, ge=1)


class RewardWeights(BaseModel):
    """Weights for the components of Cognitive Momentum + Gated Teacher.

    These are *not* learned weights. They're the relative emphasis the system
    places on each component when blending into a scalar. Self-improvement
    happens via policy update, not by changing these — they encode the
    framework's values about what 'good' means.
    """
    model_config = ConfigDict(extra="forbid")

    # Cognitive Momentum components
    correctness: float = 0.30
    momentum: float = 0.20                  # productive movement toward truth
    depth_appropriateness: float = 0.10     # reasoning depth matched task complexity
    productive_recovery: float = 0.08       # reflected, recovered
    calibration: float = 0.07               # confidence matched actual quality
    efficiency: float = 0.05                # cost / latency
    novelty: float = 0.03                   # exploration bonus
    # Gated teacher / step-level
    gated_teacher: float = 0.17             # step-level judge signal after gating

    # Penalties (subtracted, magnitude only — not negative weights)
    thrash_penalty: float = 0.10            # backtracking without reflection
    safety_violation_penalty: float = 1.0   # any safety hit zeros the reward

    @model_validator(mode="after")
    def _weights_sane(self):
        # Positive contributions should approximately sum to 1.0 so the
        # reward is bounded. We allow a small fudge.
        positive_total = (
            self.correctness + self.momentum + self.depth_appropriateness
            + self.productive_recovery + self.calibration + self.efficiency
            + self.novelty + self.gated_teacher
        )
        if not (0.95 <= positive_total <= 1.05):
            raise ValueError(
                f"Positive reward weights should sum to ~1.0, got {positive_total:.3f}"
            )
        return self


class GatedTeacherConfig(BaseModel):
    """SDAR-inspired gate over per-step judge signals.

    The gate is a sigmoid σ(α * (confidence + agreement) - β) where α and β
    are learned online from observed (judge_signal, actual_outcome) pairs.
    """
    model_config = ConfigDict(extra="forbid")

    initial_alpha: float = 4.0
    initial_beta: float = 2.0
    confidence_floor: float = Field(default=0.4, ge=0.0, le=1.0,
                                    description="Signals with confidence below this are zeroed regardless of gate.")
    agreement_floor: float = Field(default=0.5, ge=0.0, le=1.0,
                                   description="Min inter-judge agreement before signal counts at all.")
    negative_attenuation: float = Field(default=0.5, ge=0.0, le=1.0,
                                        description="Soft-down-weight for negative teacher signals (SDAR insight).")
    learning_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class ToolConfig(BaseModel):
    """One tool registration."""
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    import_path: str                            # dotted path to the callable
    callable: str
    cost_estimate: float = 0.1
    latency_estimate_ms: int = 200
    enabled: bool = True
    tags: List[str] = Field(default_factory=list)


class PromptTemplateConfig(BaseModel):
    """A versioned prompt template entry."""
    model_config = ConfigDict(extra="forbid")

    template_id: str
    version: str = "1.0.0"
    role: str                                   # "planner", "reflector", "synthesizer", "judge:<persona_id>"
    body: str
    parent_version: Optional[str] = None        # for rollback
    mutation_count: int = 0
    active: bool = True


class MemoryConfig(BaseModel):
    """Storage layer config."""
    model_config = ConfigDict(extra="forbid")

    sqlite_path: Path = Path("./noesis_state.db")
    enable_vector_recall: bool = False          # we don't ship a vector store; left as opt-in
    trajectory_retention_days: int = 90
    max_reflection_notes_per_task_type: int = 32


class ExperimentConfig(BaseModel):
    """Experiment tracking."""
    model_config = ConfigDict(extra="forbid")

    experiment_name: str = "default"
    seed: int = 7
    log_dir: Path = Path("./noesis_logs")
    record_full_trajectories: bool = True


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------
class NoesisConfig(BaseModel):
    """The root configuration object."""
    model_config = ConfigDict(extra="forbid")

    endpoints: List[ModelEndpoint]
    routing: Dict[str, ModelRouting] = Field(
        ...,
        description="Map of role ('planner','student','synthesizer') -> ModelRouting.",
    )
    jury: JuryConfig
    rl: RLConfig = Field(default_factory=RLConfig)
    rewards: RewardWeights = Field(default_factory=RewardWeights)
    gate: GatedTeacherConfig = Field(default_factory=GatedTeacherConfig)
    tools: List[ToolConfig] = Field(default_factory=list)
    prompts: List[PromptTemplateConfig] = Field(default_factory=list)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    # Runtime guardrails
    max_steps_per_task: int = Field(default=10, ge=1, le=100)
    max_reflection_passes: int = Field(default=2, ge=0, le=10)
    default_reasoning_depth: ReasoningDepth = ReasoningDepth.STANDARD

    # --- validation ---
    @model_validator(mode="after")
    def _wire_consistency(self):
        endpoint_names = {e.name for e in self.endpoints}
        for role, route in self.routing.items():
            if route.primary not in endpoint_names:
                raise ValueError(
                    f"Routing role '{role}' primary='{route.primary}' not in endpoints {sorted(endpoint_names)}"
                )
            for f in route.fallback:
                if f not in endpoint_names:
                    raise ValueError(f"Routing fallback '{f}' for role '{role}' missing from endpoints")
        for j in self.jury.judges:
            if j.endpoint_name not in endpoint_names:
                raise ValueError(f"Judge '{j.persona_id}' references unknown endpoint '{j.endpoint_name}'")
        return self

    # --- convenience getters used by the runtime ---
    def endpoint(self, name: str) -> ModelEndpoint:
        for e in self.endpoints:
            if e.name == name:
                return e
        raise KeyError(f"Endpoint '{name}' not configured")

    def route(self, role: str) -> ModelRouting:
        if role not in self.routing:
            raise KeyError(f"No routing configured for role '{role}'")
        return self.routing[role]

    def prompt(self, template_id: str, version: Optional[str] = None) -> PromptTemplateConfig:
        candidates = [p for p in self.prompts if p.template_id == template_id and p.active]
        if version is not None:
            candidates = [p for p in candidates if p.version == version]
        if not candidates:
            raise KeyError(f"No active prompt template '{template_id}'")
        # Return the latest version (lexicographic on semver-ish; good enough)
        return sorted(candidates, key=lambda p: p.version)[-1]
