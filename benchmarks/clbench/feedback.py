"""Feedback for CLBench — every mode lives here, next to the verifier.

Modes:
    binary    — just the pass/fail bit, no detail.
    raw       — the verifier's public diagnostic verbatim (per-rubric verdicts).
    socratic  — rubric-aware LLM critic that asks guiding questions.
    directive — rubric-aware LLM critic that names what to fix.

The socratic/directive critics read the rubric internally but are instructed (in
prompts.py) not to leak its exact text. pass@k never calls this — it is
feedback-blind.
"""

from __future__ import annotations

from core import llm

from . import prompts
from .benchmark import build_rubrics_text


def feedback(task, attempt, result, mode, *, judge_model):
    if mode == "binary":
        return ("Your previous answer did not pass rubric grading. "
                "Revise it to satisfy every requirement.")
    if mode == "raw":
        return result.raw_eval_output
    if mode in ("socratic", "directive"):
        prompt, text = _critic(task, attempt, result, mode, judge_model)
        result.private["critic_prompt"] = prompt
        return text
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _critic(task, attempt, result, mode, judge_model):
    template = prompts.SOCRATIC if mode == "socratic" else prompts.DIRECTIVE
    private = result.private
    prompt = template.format(
        rubrics_text=build_rubrics_text(task.grading["rubrics"]),
        failed_requirement_count=private.get("failed_requirement_count"),
        requirement_status=private.get("requirement_status"),
        raw_output=attempt.output,
    )
    return prompt, llm.complete(judge_model, prompt, temperature=0.7)
