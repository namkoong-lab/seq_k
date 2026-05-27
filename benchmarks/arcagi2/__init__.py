"""ARC-AGI-2 — self-contained benchmark (deterministic verifier). Exposes the
three functions the harness calls: load_tasks, verify, feedback."""

from .benchmark import load_tasks, verify
from .feedback import feedback

__all__ = ["load_tasks", "verify", "feedback"]
