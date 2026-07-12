"""AdvancedIF feedback modes:
    binary   — pass/fail bit only
    raw      — per-question verdicts verbatim (question_N: Yes/No — reason)
    compact  — only the unmet requirements, briefly
    critique — LLM reviewer that sees ONLY the task and the model's answer
               (no rubric, no verifier output); ported from seq_k_eval

binary/raw/compact are derived from the verifier output (no extra LLM call);
critique fires one llm.complete on critic_model. pass@k never calls this.
"""

from __future__ import annotations

from core import llm

from . import prompts


def feedback(task, attempt, result, mode, *, critic_model=None):
    if mode == "binary":
        return ("Your previous response did not satisfy all of the instruction requirements. "
                "Revise it to meet every requirement.")
    if mode == "raw":
        return result.raw_eval_output
    if mode == "compact":
        unmet = [v for v in result.details.get("verdicts", []) if not v["met"]]
        if not unmet:
            return result.raw_eval_output
        items = ["question_{}".format(v["question"]) + (f" ({v['reason']})" if v["reason"] else "")
                 for v in unmet]
        return "Unmet requirements: " + "; ".join(items)
    if mode == "critique":
        return _critique(task, attempt, result, critic_model)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _critique(task, attempt, result, critic_model):
    """Reviewer LLM sees only the task text and the model's answer. Matches
    seq_k_eval: an empty/blank critique degrades to the verifier's raw output."""
    critic_prompt = prompts.CRITIQUE.format(task=task.prompt, answer=attempt.output or "")
    critic_output = (llm.complete(critic_model, critic_prompt, temperature=0.7) or "").strip()
    return critic_output or result.raw_eval_output
