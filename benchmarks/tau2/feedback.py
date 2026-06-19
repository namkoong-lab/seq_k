"""tau2 feedback modes (all built from the parsed SimulationRun, no extra LLM call):

    binary           — pass/fail note only
    raw              — the public per-check failure summary verbatim
    retry_diagnostics — failure summary + a focused retry checklist

All inputs are already leak-safe: `raw_eval_output` and `judge_details` here are
derived from check outcomes (met/match flags, public `info` strings), not gold DB
state. pass@k never calls this.
"""

from __future__ import annotations


def feedback(task, attempt, result, mode, *, judge_model=None):
    if mode == "binary":
        return ("Your previous attempt did not fully satisfy the task. Handle the "
                "next conversation more carefully and complete every required "
                "action and communication.")
    if mode == "raw":
        return result.raw_eval_output or _binary_fallback()
    if mode == "retry_diagnostics":
        return _retry_diagnostics(result.raw_eval_output)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _binary_fallback():
    return "Your previous attempt did not pass the task evaluation."


def _retry_diagnostics(raw_eval_output):
    summary = (raw_eval_output or "").strip() or _binary_fallback()
    return "\n".join([
        "what_went_wrong:",
        summary,
        "",
        "retry_focus:",
        "- Re-read the relevant domain policy before acting; confirm every "
        "precondition with the user.",
        "- Make sure each required write action is performed exactly once with "
        "the correct arguments.",
        "- Explicitly state any information the task requires you to communicate "
        "to the user.",
        "- Do not stop until the customer's request is fully resolved.",
    ]).strip()
