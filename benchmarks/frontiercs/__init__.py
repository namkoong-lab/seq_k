"""FrontierCS (algorithmic track) — official Docker-judge verifier over HTTP,
continuous [0,1] score, template feedback modes (binary / case_summary / raw).
Exposes the three functions the harness calls: load_tasks, verify, feedback."""

from .benchmark import LLM_CRITIC_MODES, VERIFIER, load_tasks, slice_name, verify
from .feedback import feedback

__all__ = ["load_tasks", "verify", "feedback", "slice_name", "VERIFIER", "LLM_CRITIC_MODES"]
