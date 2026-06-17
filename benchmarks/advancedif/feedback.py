"""AdvancedIF feedback modes:
    binary  — pass/fail bit only
    raw     — per-question verdicts verbatim (question_N: Yes/No — reason)
    compact — only the unmet requirements, briefly

All derived from the verifier output (no extra LLM call). pass@k never calls this.
"""

from __future__ import annotations


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
    raise ValueError(f"unknown feedback mode: {mode!r}")
