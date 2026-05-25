"""CLBench benchmark — self-contained: task loading, verifier, and feedback.

A benchmark exposes exactly three functions; the harness calls these and nothing
else:
    load_tasks() -> list[Task]
    verify(task, attempt, *, judge_model) -> VerifierResult
    feedback(task, attempt, result, mode, *, judge_model) -> str
"""

from .benchmark import load_tasks, verify
from .feedback import feedback

__all__ = ["load_tasks", "verify", "feedback"]
