"""tau2 (tau-bench) — an agentic benchmark. Each attempt is a full agent<->user
simulation of a customer-service task, run via the installed `tau2` package.
Implements the harness's `run_attempt` hook instead of verify().

Exposes: load_tasks, run_attempt, feedback."""

from .benchmark import (
    LLM_CRITIC_MODES,
    VERIFIER,
    load_tasks,
    run_attempt,
    slice_name,
)
from .feedback import feedback

__all__ = [
    "load_tasks", "run_attempt", "feedback", "slice_name",
    "VERIFIER", "LLM_CRITIC_MODES",
]
