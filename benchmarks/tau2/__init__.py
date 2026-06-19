"""tau2 (tau-bench) — an agentic benchmark. Each attempt is a full agent<->user
simulation of a customer-service task, run via the installed `tau2` package.
Implements the harness's `run_attempt` hook instead of verify().

Exposes: load_tasks, run_attempt, feedback."""

from .benchmark import load_tasks, run_attempt
from .feedback import feedback

__all__ = ["load_tasks", "run_attempt", "feedback"]
