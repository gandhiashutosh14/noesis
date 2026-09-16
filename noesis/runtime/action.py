"""
NOESIS — Action Space
=====================

The discrete actions the agent runtime can take at each decision point.
These are first-class so the policy update mechanism can score them
individually (which actions yielded what reward, on which task types).

Action selection happens in `runtime.runtime.AgentRuntime`. The policy
layer biases the selection based on stored strategy stats.
"""
from __future__ import annotations

from enum import Enum


class AgentAction(str, Enum):
    # Plan-level
    CHOOSE_PROMPT_STRATEGY = "choose_prompt_strategy"
    CHOOSE_MODEL = "choose_model"
    CHOOSE_REASONING_DEPTH = "choose_reasoning_depth"

    # Execution-level
    INVOKE_TOOL = "invoke_tool"
    SYNTHESIZE_ANSWER = "synthesize_answer"
    INVOKE_REFLECTION = "invoke_reflection"

    # Control flow
    REVISE_PLAN = "revise_plan"
    RETRY_WITH_ALTERNATIVE = "retry_with_alternative"
    COMPRESS_CONTEXT = "compress_context"
    ESCALATE_TO_STRONGER_MODEL = "escalate_to_stronger_model"
    REQUEST_CLARIFICATION = "request_clarification"
    TERMINATE_AND_ANSWER = "terminate_and_answer"


# Actions that consume reflection passes (counts toward max_reflection_passes)
REFLECTIVE_ACTIONS = {
    AgentAction.INVOKE_REFLECTION,
    AgentAction.REVISE_PLAN,
    AgentAction.RETRY_WITH_ALTERNATIVE,
}

# Terminal actions — once taken, the run ends (modulo error handling)
TERMINAL_ACTIONS = {
    AgentAction.TERMINATE_AND_ANSWER,
    AgentAction.SYNTHESIZE_ANSWER,
}
