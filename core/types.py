"""Data types shared across modules.

A single attempt has three roles in it:
    actor    — the model being evaluated. Sees Step.prompt, produces Step.output.
    judge    — produces VerifierResult (success/score). Runs on both pass@k and seq@k.
    critic   — produces Step.critic_feedback for the NEXT attempt. seq@k only,
               only on failed non-final attempts. Never affects scoring.

Leak-safety: build_prompt only ever reads Task.prompt and prior outputs/critic
feedback, never Task.grading or VerifierResult.judge_details (those are judge/
critic scratch and must not feed back into the actor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str            # what the actor sees
    grading: dict          # answer key; verify()/feedback() only


@dataclass(frozen=True)
class Attempt:
    index: int             # 0-based
    output: str            # raw actor text


@dataclass(frozen=True)
class VerifierResult:
    """Output of the judge for one attempt."""
    success: bool
    score: float                                        # 1/0 today; float leaves room for soft scores
    raw_eval_output: str                                # judge's PUBLIC diagnostic — safe to show the next attempt
    judge_details: dict = field(default_factory=dict)   # judge's INTERNAL scratch (raw judge output, per-rubric verdicts, etc.)


@dataclass
class Step:
    """One attempt's full record: actor in/out, judge verdict, critic feedback."""
    attempt_index: int
    prompt: str                              # exact text the actor saw
    output: str                              # actor's raw output
    result: VerifierResult                   # judge's verdict
    critic_feedback: Optional[str]           # critic output for the NEXT attempt. None if passed / last attempt / pass@k
    calls: list = field(default_factory=list)  # judge/critic model calls this attempt: {phase, model, prompt}


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
