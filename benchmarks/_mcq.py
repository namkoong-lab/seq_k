"""Shared core for multiple-choice benchmarks (MedMCQA, MedQA).

4-option single-best-answer MCQ: the actor picks a letter, we parse it and
exact-match the gold letter. No LLM judge in scoring (deterministic, score
1.0/0.0), following benchmarks/arcagi2. A malformed answer is a real task
failure (score 0 + a format hint), not a swallowed error.

verify/feedback take judge_model to match the harness contract; verify ignores
it. The gold letter and gold option text never appear in raw_eval_output or in
judge feedback (no answer leak).
"""
from __future__ import annotations

import re
import string

from core import llm
from core.types import Task, VerifierResult

LETTERS = string.ascii_uppercase  # "A", "B", "C", "D", ...

ACTOR_INSTRUCTION = (
    "Answer the following multiple-choice question. You may reason briefly, but "
    "end your response with your choice on its own line in the exact form "
    "'Answer: X', where X is one of the option letters."
)

JUDGE_CRITIC = """A student is answering a multiple-choice question and got it wrong. \
Without telling them which option is correct, give one or two sentences of reasoning \
guidance to help them reconsider on their next attempt.

# Question
{question}

# Options
{options_block}

# The student's answer (marked incorrect)
{chosen}

Rules:
- Do NOT state or imply which option is correct. Do NOT name a letter as the answer.
- Point at the concept or distinction they should reconsider.
- Plain text, brief."""


# --------------------------------------------------------------------------- #
# Task construction
# --------------------------------------------------------------------------- #
def make_task(task_id, question, options, gold_index, extra=None):
    options = [str(o) for o in options]
    if not (0 <= gold_index < len(options)):
        raise ValueError(f"gold_index {gold_index} out of range for {len(options)} options")
    grading = {
        "question": str(question),
        "options": options,
        "gold_index": int(gold_index),
        "gold_letter": LETTERS[gold_index],
    }
    if extra:
        grading.update(extra)
    return Task(id=str(task_id), prompt=format_prompt(question, options), grading=grading)


def format_prompt(question, options):
    return (f"{ACTOR_INSTRUCTION}\n\n# Question\n{str(question).strip()}\n\n"
            f"# Options\n{_options_block(options)}")


def _options_block(options):
    return "\n".join(f"{LETTERS[i]}) {str(opt).strip()}" for i, opt in enumerate(options))


# --------------------------------------------------------------------------- #
# Answer parsing
# --------------------------------------------------------------------------- #
def parse_letter(text, options):
    """Extract the chosen option letter from raw actor output, or None.

    Tries, in order: an explicit 'Answer: X' (last one wins), a standalone letter
    (last one wins), then a verbatim option-text match. Only letters within the
    valid range (A .. A+len(options)-1) count.
    """
    s = str(text or "").strip()
    if not s:
        return None
    valid = {LETTERS[i] for i in range(len(options))}

    labelled = re.findall(
        r"(?:answer|final answer|choice|option)\s*(?:is)?\s*[:\-]?\s*\(?([A-Za-z])\)?",
        s, re.IGNORECASE)
    for m in reversed(labelled):
        if m.upper() in valid:
            return m.upper()

    for m in reversed(re.findall(r"\b([A-Za-z])\b", s)):
        if m.upper() in valid:
            return m.upper()

    low = s.lower()
    for i, opt in enumerate(options):
        ot = str(opt).strip().lower()
        if ot and ot in low:
            return LETTERS[i]
    return None


# --------------------------------------------------------------------------- #
# Verifier (deterministic exact letter match)
# --------------------------------------------------------------------------- #
def verify(task, attempt, *, judge_model=None):   # judge_model unused (deterministic)
    options = task.grading["options"]
    gold = task.grading["gold_letter"]
    parsed = parse_letter(attempt.output or "", options)
    if parsed is None:
        hint = (f"Could not read a single-letter answer (A-{LETTERS[len(options) - 1]}) "
                "from your response. End with 'Answer: X'.")
        return VerifierResult(
            success=False, score=0.0, raw_eval_output=hint,
            judge_details={"parsed_letter": None, "gold_letter": gold, "parse_ok": False},
        )
    success = parsed == gold
    raw = "" if success else f"You answered {parsed}. That is incorrect."
    return VerifierResult(
        success=success, score=1.0 if success else 0.0, raw_eval_output=raw,
        judge_details={"parsed_letter": parsed, "gold_letter": gold, "parse_ok": True},
    )


# --------------------------------------------------------------------------- #
# Feedback (binary | raw | judge)
# --------------------------------------------------------------------------- #
def feedback(task, attempt, result, mode, *, judge_model):
    if mode == "binary":
        return ("Your previous answer was incorrect. Reconsider the options and choose "
                "again, ending with 'Answer: X'.")
    if mode == "raw":
        return result.raw_eval_output
    if mode == "judge":
        prompt = JUDGE_CRITIC.format(
            question=task.grading.get("question", ""),
            options_block=_options_block(task.grading["options"]),
            chosen=_chosen_text(task, result),
        )
        return llm.complete(judge_model, prompt, temperature=0.7)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _chosen_text(task, result):
    letter = (result.judge_details or {}).get("parsed_letter")
    if not letter:
        return "(no clear option was selected)"
    idx = LETTERS.index(letter)
    opts = task.grading["options"]
    return f"{letter}) {opts[idx]}" if 0 <= idx < len(opts) else letter
