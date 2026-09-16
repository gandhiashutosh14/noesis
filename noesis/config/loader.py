"""
NOESIS — Configuration loader
=============================

Reads YAML, applies env-var overrides, validates against the schema, and
returns a NoesisConfig. Validation failures raise ConfigurationError with
the original pydantic message + the file path so the user knows where to
look.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from noesis.config.schema import NoesisConfig

DEFAULT_CONFIG_PATH = Path(__file__).parent / "defaults.yaml"


class ConfigurationError(Exception):
    """Raised when configuration can't be loaded or validated."""


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Allow a few key knobs to be set via env vars.

    Supported:
      NOESIS_LLM_PROVIDER             — forces all endpoints to a single provider
      NOESIS_LLM_MODEL_ID             — forces all endpoints to one model_id
      NOESIS_SQLITE_PATH              — overrides memory.sqlite_path
      NOESIS_EXPERIMENT_NAME          — overrides experiment.experiment_name
      NOESIS_SEED                     — overrides experiment.seed (int)
      NOESIS_MAX_STEPS                — overrides max_steps_per_task (int)
    """
    provider = os.environ.get("NOESIS_LLM_PROVIDER")
    model_id = os.environ.get("NOESIS_LLM_MODEL_ID")
    if provider or model_id:
        for ep in raw.get("endpoints", []):
            if provider:
                ep["provider"] = provider
            if model_id:
                ep["model_id"] = model_id

    sqlite_path = os.environ.get("NOESIS_SQLITE_PATH")
    if sqlite_path:
        raw.setdefault("memory", {})["sqlite_path"] = sqlite_path

    exp_name = os.environ.get("NOESIS_EXPERIMENT_NAME")
    if exp_name:
        raw.setdefault("experiment", {})["experiment_name"] = exp_name

    seed = os.environ.get("NOESIS_SEED")
    if seed:
        raw.setdefault("experiment", {})["seed"] = int(seed)

    max_steps = os.environ.get("NOESIS_MAX_STEPS")
    if max_steps:
        raw["max_steps_per_task"] = int(max_steps)

    return raw


def load_config(path: Optional[Path] = None) -> NoesisConfig:
    """Load NOESIS config from YAML.

    If path is None, loads the bundled defaults.yaml. Env overrides are
    applied before validation.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Config root must be a mapping, got {type(raw).__name__} from {path}"
        )

    raw = _apply_env_overrides(raw)

    try:
        return NoesisConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigurationError(
            f"Configuration validation failed for {path}:\n{e}"
        ) from e
