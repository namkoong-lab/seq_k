"""CLBench feedback modes:
    binary    — pass/fail bit only
    raw       — verifier's public diagnostic verbatim
    socratic  — LLM critic, guiding questions
    directive — LLM critic, names what to fix

socratic/directive see the rubric but are told not to leak it (prompts.py).
pass@k never calls this.
"""

from __future__ import annotations

from core import llm

from . import prompts
from .benchmark import build_rubrics_text


def feedback(task, attempt, result, mode, *, critic_model):
    if mode == "binary":
        return ("Your previous answer did not pass rubric grading. "
                "Revise it to satisfy every requirement.")
    if mode == "raw":
        return result.raw_eval_output
    if mode in ("socratic", "directive"):
        return _critic(task, attempt, result, mode, critic_model)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _critic(task, attempt, result, mode, critic_model):
    template = prompts.SOCRATIC if mode == "socratic" else prompts.DIRECTIVE
    details = result.details
    critic_prompt = template.format(
        rubrics_text=build_rubrics_text(task.grading["rubrics"]),
        failed_requirement_count=details.get("failed_requirement_count"),
        requirement_status=details.get("requirement_status"),
        raw_output=attempt.output,
    )
    critic_output = llm.complete(critic_model, critic_prompt, temperature=0.7)
    return critic_output
