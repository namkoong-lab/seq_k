"""The five small data types every other module speaks.

Leak-safety convention: the prompt builder (harness.build_prompt) reads only
`Task.prompt`, prior `Attempt.output`, and prior feedback strings. It must never
read `Task.grading` or `VerifierResult.private` — those are for the verifier and
the feedback function only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str            # what the model sees
    grading: dict          # rubric / answer key — read only by verify() and feedback()


@dataclass(frozen=True)
class Attempt:
    index: int             # 0-based attempt number
    output: str            # raw model text


@dataclass(frozen=True)
class VerifierResult:
    success: bool
    score: float                                  # 1.0 / 0.0 for binary; soft scores need no schema change
    raw_eval_output: str                          # PUBLIC diagnostic, safe to show the next retry
    private: dict = field(default_factory=dict)   # DEBUG / feedback-internal (judge JSON, per-rubric verdicts)


@dataclass
class Step:
    attempt_index: int
    prompt: str                    # the EXACT text shown to the model this attempt
    output: str
    result: VerifierResult
    feedback: Optional[str]        # produced after this attempt (None if it passed, was the last, or pass@k)


@dataclass
class Trajectory:
    task_id: str
    metric: str                    # "pass@k" | "seq@k"
    model: str
    feedback_mode: str
    steps: list
    success: bool
    best_score: float
