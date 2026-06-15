"""MedMCQA task loading -> shared MCQ verifier (deterministic letter match).

MedMCQA (openlifescienceai/medmcqa) is 4-option single-best-answer medical MCQ.
The public test split has withheld labels, so we evaluate on `validation`. Rows
whose choice_type is multi-answer are skipped. `cop` is the 0-indexed gold option
(opa=0 .. opd=3). Verification/feedback are delegated to benchmarks._mcq.
"""
from __future__ import annotations

from datasets import load_dataset

from benchmarks import _mcq

DATASET = "openlifescienceai/medmcqa"
OPTION_KEYS = ("opa", "opb", "opc", "opd")

verify = _mcq.verify  # deterministic; judge_model-agnostic


def load_tasks(split="validation", max_rows=None, subject=None):
    ds = load_dataset(DATASET, split=split)
    tasks = []
    for idx, row in enumerate(ds):
        if str(row.get("choice_type", "single")).strip().lower() == "multi":
            continue
        if subject and str(row.get("subject_name", "")).strip() != subject:
            continue
        options = [str(row.get(k, "")).strip() for k in OPTION_KEYS]
        if any(o == "" for o in options):
            continue
        gold = int(row["cop"])
        if not (0 <= gold < len(options)):
            continue
        tasks.append(_mcq.make_task(
            task_id=row.get("id") or f"medmcqa_{split}_{idx:05d}",
            question=str(row.get("question", "")).strip(),
            options=options,
            gold_index=gold,
            extra={"subject_name": str(row.get("subject_name", "")).strip()},
        ))
        if max_rows and len(tasks) >= max_rows:
            break
    if not tasks:
        raise ValueError(f"MedMCQA: no tasks for split={split!r} subject={subject!r}")
    return tasks
