"""AdvancedIF feedback modes:
    binary   — pass/fail bit only
    raw      — per-question verdicts verbatim (question_N: Yes/No — reason)
    compact  — only the unmet requirements, briefly
    category — only WHICH DIMENSIONS failed (required content / prohibited content /
               formatting / length / tone), not which rubric and not the judge's reason.
               An abstraction of the verdicts, not a per-rubric projection: it probes how
               FINE a localization the actor needs. Categories are precomputed offline by
               scripts/classify_rubrics.py and looked up here (no extra LLM call).
    critique — LLM reviewer that sees ONLY the task and the model's answer
               (no rubric, no verifier output); ported from seq_k_eval

binary/raw/compact/category are derived from the verifier output (no extra LLM call);
critique fires one llm.complete on critic_model. pass@k never calls this.
"""

from __future__ import annotations

from core import llm

from . import prompts

# Display names shown to the actor, in a fixed order for deterministic feedback.
_CATEGORY_ORDER = ["Content", "Constraint", "Format", "Length", "Style"]
_CATEGORY_DISPLAY = {
    "Content": "required content (things you should include but are missing)",
    "Constraint": "prohibited content (things you should not have mentioned)",
    "Format": "formatting / structure",
    "Length": "length / amount",
    "Style": "tone / style",
}


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
    if mode == "category":
        return _category(task, result)
    if mode == "critique":
        return _critique(task, attempt, result, critic_model)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _category(task, result):
    """Name only the DIMENSIONS of the failed requirements — not which rubric, not the
    judge's reason. Categories come from the offline cache attached by load_tasks; a
    missing cache is a hard error (fail-loud, research reproducibility)."""
    rubric_categories = task.grading.get("rubric_categories")
    if rubric_categories is None:
        raise ValueError(
            "category feedback needs precomputed rubric categories; none attached to "
            f"task {task.id}. Run scripts/classify_rubrics.py to build the cache.")
    failed = set()
    for v in result.details.get("verdicts", []):
        if v["met"]:
            continue
        idx = v["question"] - 1                     # verdict question is 1-based
        if 0 <= idx < len(rubric_categories):
            failed.update(rubric_categories[idx])
    areas = [_CATEGORY_DISPLAY[c] for c in _CATEGORY_ORDER if c in failed]
    if not areas:                                   # unmet rubric with no mapped category
        return ("Your previous response did not satisfy all of the instruction "
                "requirements. Revise it to meet every requirement.")
    return ("Your previous response did not satisfy the requirements in these areas: "
            + "; ".join(areas)
            + ". Revise it so it meets every requirement, focusing on these areas.")


def _critique(task, attempt, result, critic_model):
    """Reviewer LLM sees only the task text and the model's answer. Matches
    seq_k_eval: an empty/blank critique degrades to the verifier's raw output."""
    critic_prompt = prompts.CRITIQUE.format(task=task.prompt, answer=attempt.output or "")
    critic_output = (llm.complete(critic_model, critic_prompt, temperature=0.7) or "").strip()
    return critic_output or result.raw_eval_output
