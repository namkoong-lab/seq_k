"""MedXpertQA (Text) — expert-level, multi-option medical MCQ (deliberately hard).
Exposes the three functions the harness calls: load_tasks, verify, feedback."""
from .benchmark import load_tasks, verify
from .feedback import feedback

__all__ = ["load_tasks", "verify", "feedback"]
