"""ARC-AGI-2 — self-contained benchmark (deterministic verifier). Exposes the
three functions the harness calls: load_tasks, verify, feedback."""

from .benchmark import LLM_CRITIC_MODES, VERIFIER, load_tasks, slice_name, verify
from .feedback import feedback

__all__ = ["load_tasks", "verify", "feedback", "slice_name", "VERIFIER", "LLM_CRITIC_MODES"]
