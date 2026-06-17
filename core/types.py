"""Data types shared across modules.

A single attempt has three independent roles, each in its own JSON section:
    actor    — the model being evaluated. Sees actor.prompt, produces actor.output.
    judge    — produces success/score + a public diagnostic. Runs on every attempt.
    critic   — produces a feedback string for the NEXT attempt's actor. seq@k only,
               only on failed non-final attempts. Never affects scoring.

Each role has its OWN model field (actor.model, judge.model, critic.model). They
default to the same model when not configured separately. The three sections share
NO state in the JSON: judge.details is internal to the judge; critic.feedback is
the critic's output.

Leak-safety: build_prompt only ever reads Task.prompt + prior actor.output + prior
critic.feedback. Never Task.grading or judge.details (those are role-internal).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str            # what the actor sees
    grading: dict          # answer key; verify()/feedback() only


@dataclass(frozen=True)
class Attempt:
    index: int             # 1-based (attempt 1 of K, 2 of K, ...)
    output: str            # raw actor text


@dataclass(frozen=True)
class VerifierResult:
    """Output of the judge for one attempt. The harness packages this into the
    `judge` section of the saved JSON, adding `judge.model` and `judge.calls`."""
    success: bool
    score: float                                # 1/0 today; float leaves room for soft scores
    raw_eval_output: str                        # judge's PUBLIC diagnostic — safe to show the next attempt
    details: dict = field(default_factory=dict) # judge's INTERNAL scratch (per-rubric verdicts, etc.) — never fed to actor


@dataclass
class Step:
    """One attempt's full record. Three sections (actor / judge / critic) are
    independent — each is its own dict in the saved JSON. See core/results.py
    for the schema."""
    attempt_index: int     # 1-based (attempt 1 of K, 2 of K, ...)
    actor: dict            # {model, prompt, output} — model = actor (the one being evaluated)
    judge: dict            # {model, success, score, raw_eval_output, details, calls}
    critic: dict           # {model, feedback, calls}


@dataclass
class Trajectory:
    task_id: str
    metric: str            # "pass@k" | "seq@k"
    model: str             # actor model (shorthand — same as steps[*].actor.model)
    feedback_mode: str
    task_prompt: str       # shared actor context, kept once so the prompt view can show deltas
    steps: list
    success: bool
    best_score: float
