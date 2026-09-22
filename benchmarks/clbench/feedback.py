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
    if mode == "counts_only":
        # The one recovered mode with no LLM call: the imported runs recorded
        # exactly this string, so it is a template, not a critique.
        d = result.details
        return (f"failed_requirement_count={d.get('failed_requirement_count')}; "
                f"total_requirements={d.get('total_requirements')}")
    if mode in _RECOVERED:
        return _recovered(task, attempt, result, mode, critic_model)
    raise ValueError(f"unknown feedback mode: {mode!r}")


# Recovered from imported runs; see prompts.py for how each was reconstructed.
_RECOVERED = {
    "socratic_no_rubric":       "SOCRATIC_NO_RUBRIC",
    "socratic_blind":           "SOCRATIC_BLIND",
    "directive_no_rubric":      "DIRECTIVE_NO_RUBRIC",
    "compact_eval_output":      "COMPACT_EVAL_OUTPUT",
    "judge_feedback":           "JUDGE_FEEDBACK",
    "judge_feedback_no_points": "JUDGE_FEEDBACK_NO_POINTS",
}


def _recovered(task, attempt, result, mode, critic_model):
    """Render whichever recovered template `mode` names, passing only the fields it
    declares — the blind variants deliberately have no {rubrics_text}, so formatting
    them with it would be a silent leak of the thing they exist to withhold."""
    template = getattr(prompts, _RECOVERED[mode])
    d = result.details
    fields = {
        "rubrics_text": build_rubrics_text(task.grading["rubrics"]),
        "failed_requirement_count": d.get("failed_requirement_count"),
        "requirement_status": d.get("requirement_status"),
        "raw_output": result.raw_eval_output,
        "attempt": attempt.output or "",
    }
    used = {k: v for k, v in fields.items() if "{" + k + "}" in template}
    out = (llm.complete(critic_model, template.format(**used), temperature=0.7) or "").strip()
    return out or result.raw_eval_output


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
