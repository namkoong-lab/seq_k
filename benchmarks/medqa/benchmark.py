"""MedQA (USMLE, 4-option) task loading -> shared MCQ verifier.

GBaker/MedQA-USMLE-4-options: each row has `question`, an `options` mapping
A-D -> text, and `answer_idx` (the gold letter). The `test` split is labeled, so
we evaluate on it. Verification/feedback are delegated to benchmarks._mcq.
"""
from __future__ import annotations

import json

from datasets import load_dataset

from benchmarks import _mcq

DATASET = "GBaker/MedQA-USMLE-4-options"
LETTER_ORDER = ("A", "B", "C", "D")

verify = _mcq.verify  # deterministic; judge_model-agnostic


def load_tasks(split="test", max_rows=None):
    ds = load_dataset(DATASET, split=split)
    tasks = []
    for idx, row in enumerate(ds):
        options, gold_letter = _extract(row.get("options"), row.get("answer_idx"))
        tasks.append(_mcq.make_task(
            task_id=f"medqa_{split}_{idx:05d}",
            question=str(row.get("question", "")).strip(),
            options=options,
            gold_index=LETTER_ORDER.index(gold_letter),
        ))
        if max_rows and len(tasks) >= max_rows:
            break
    if not tasks:
        raise ValueError(f"MedQA: no tasks for split={split!r}")
    return tasks


def _extract(options_field, answer_idx):
    if isinstance(options_field, str):
        options_field = json.loads(options_field)
    if not isinstance(options_field, dict):
        raise ValueError(f"MedQA: unexpected options field: {options_field!r}")
    options = [str(options_field[L]).strip() for L in LETTER_ORDER]
    gold = str(answer_idx or "").strip().upper()
    if gold not in LETTER_ORDER:
        raise ValueError(f"MedQA: bad answer_idx {answer_idx!r}")
    return options, gold
