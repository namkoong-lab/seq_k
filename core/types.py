"""Data types shared across modules.

Leak-safety: build_prompt only ever reads Task.prompt and prior outputs/feedback,
never Task.grading or VerifierResult.private (those are verifier/feedback only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str            # what the model sees
    grading: dict          # answer key; verify()/feedback() only


@dataclass(frozen=True)
class Attempt:
    index: int             # 0-based
    output: str            # raw model text


@dataclass(frozen=True)
class VerifierResult:
    success: bool
    score: float                                  # 1/0 today; float leaves room for soft scores
    raw_eval_output: str                          # safe to show the next attempt
    private: dict = field(default_factory=dict)   # verifier/feedback scratch (judge json, prompts, verdicts)


@dataclass
class Step:
    attempt_index: int
    prompt: str                    # exact text the model saw
    output: str
    result: VerifierResult
    feedback: Optional[str]        # None if passed / last attempt / pass@k
    calls: list = field(default_factory=list)   # judge/critic model calls this attempt: {phase, model, prompt}


@dataclass
class Trajectory:
    task_id: str
    metric: str                    # "pass@k" | "seq@k"
    model: str
    feedback_mode: str
    task_prompt: str               # shared actor context, kept once so the prompt view can show deltas
    steps: list
    success: bool
    best_score: float
