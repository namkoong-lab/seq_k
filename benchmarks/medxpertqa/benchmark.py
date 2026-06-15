"""MedXpertQA (Text) task loading -> shared MCQ verifier.

MedXpertQA (TsinghuaC3I/MedXpertQA, config "Text") is an expert-level, deliberately
hard medical MCQ benchmark (2025) with up to 10 options per question, built to
counter the saturation of MedQA/MedMCQA. Only the `test` split is sizeable (2450
rows; `dev` has 5). Each row has `question` (with the choices appended inline),
an `options` dict (letter -> text), and `label` (the gold letter). The shared MCQ
core handles arbitrary option counts. Verification/feedback delegate to _mcq.
"""
from __future__ import annotations

from datasets import load_dataset

from benchmarks import _mcq

DATASET = "TsinghuaC3I/MedXpertQA"
CONFIG = "Text"

verify = _mcq.verify  # deterministic; judge_model-agnostic


def load_tasks(split="test", max_rows=None, medical_task=None, question_type=None):
    ds = load_dataset(DATASET, CONFIG, split=split)
    tasks = []
    for idx, row in enumerate(ds):
        if medical_task and str(row.get("medical_task", "")).strip() != medical_task:
            continue
        if question_type and str(row.get("question_type", "")).strip() != question_type:
            continue
        opts = row.get("options")
        if not isinstance(opts, dict) or not opts:
            continue
        letters = sorted(opts.keys())                      # 'A','B',...,'J'
        options = [str(opts[L]).strip() for L in letters]
        gold = str(row.get("label", "")).strip().upper()
        if gold not in letters:
            continue
        # The question text appends "Answer Choices: ..." — drop it; _mcq re-renders the options.
        question = str(row.get("question", "")).split("Answer Choices:")[0].strip()
        tasks.append(_mcq.make_task(
            task_id=str(row.get("id") or f"medxpertqa_{split}_{idx:05d}"),
            question=question,
            options=options,
            gold_index=letters.index(gold),
            extra={"medical_task": str(row.get("medical_task", "")).strip(),
                   "question_type": str(row.get("question_type", "")).strip()},
        ))
        if max_rows and len(tasks) >= max_rows:
            break
    if not tasks:
        raise ValueError(f"MedXpertQA: no tasks for split={split!r}")
    return tasks
