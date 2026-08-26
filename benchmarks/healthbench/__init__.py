"""HealthBench — self-contained benchmark. Exposes the three functions the harness
calls: load_tasks, verify, feedback."""

from .benchmark import LLM_CRITIC_MODES, VERIFIER, load_tasks, slice_name, verify
from .feedback import feedback


def resolve_critic_model(mode, *, actor_model, critic_model):
    """Self-blind feedback is self-authored, regardless of a stale config value."""
    return actor_model if mode == "self_blind" else critic_model


__all__ = ["load_tasks", "verify", "feedback", "slice_name", "resolve_critic_model",
           "VERIFIER", "LLM_CRITIC_MODES"]
