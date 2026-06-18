# MedMCQA + MedQA Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MedMCQA and MedQA as two new 4-option multiple-choice benchmarks to the seq_k harness, sharing one deterministic MCQ core, so pass@K vs seq@K can be run on medical MCQ just like the existing benchmarks.

**Architecture:** A shared, imported-only module `benchmarks/_mcq.py` holds the prompt format, answer parser, deterministic letter-match verifier, and the three feedback modes (`binary` / `raw` / `judge`). Two thin adapter packages (`benchmarks/medmcqa/`, `benchmarks/medqa/`) only load + normalize their HuggingFace dataset and re-export the shared functions via `__init__.py`. No changes to `core/`.

**Tech Stack:** Python 3.12 (via `uv` venv), `datasets` (HF loaders), `litellm` (already used by `core.llm`), `pytest` for the pure-function unit tests. Actor + judge model: `anthropic/claude-haiku-4-5`.

---

## File Structure

- Create: `benchmarks/_mcq.py` — shared MCQ core (prompt, parse, verify, feedback). Imported by both adapters; no `variants/`, never run directly.
- Create: `benchmarks/medmcqa/__init__.py` `benchmark.py` `feedback.py` + `variants/{passk,seqk.binary,seqk.raw,seqk.judge}.yaml`
- Create: `benchmarks/medqa/__init__.py` `benchmark.py` `feedback.py` + `variants/{passk,seqk.binary,seqk.raw,seqk.judge}.yaml`
- Create: `tests/test_mcq.py` — pure unit tests for `parse_letter` + `verify` (no network, no model).
- Modify: `requirements.txt`, `pyproject.toml` — add `datasets>=2.0`.

Note: prompts (actor instruction + judge critic template) live in `_mcq.py`, shared by both benchmarks (DRY) rather than a per-folder `prompts.py`. The `__init__.py` of each adapter still exposes exactly `load_tasks / verify / feedback`, which is all the harness imports.

---

## Task 0: Environment setup

**Files:** none (environment only)

- [ ] **Step 1: Create a 3.12 venv with uv**

Run:
```bash
cd /Users/meng/Documents/seq_k
uv venv --python 3.12 .venv
```
Expected: creates `.venv/` using CPython 3.12.

- [ ] **Step 2: Install deps + pytest into the venv**

Run:
```bash
uv pip install --python .venv -r requirements.txt datasets pytest
```
Expected: installs litellm, huggingface_hub, pyyaml, python-dotenv, datasets, pytest.

- [ ] **Step 3: Confirm imports work**

Run:
```bash
.venv/bin/python -c "import litellm, datasets, yaml, dotenv, pytest; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 4: Confirm the Anthropic key is visible**

Run:
```bash
.venv/bin/python -c "import os; print('ANTHROPIC_API_KEY set:', bool(os.environ.get('ANTHROPIC_API_KEY')))"
```
Expected: `ANTHROPIC_API_KEY set: True`. If `False`, put the key in a `.env` at the repo root (`core/cli.py` calls `load_dotenv()`) before any smoke run — unit tests in Task 1 do not need it.

---

## Task 1: Shared MCQ core (`benchmarks/_mcq.py`) — TDD

**Files:**
- Create: `tests/test_mcq.py`
- Create: `benchmarks/_mcq.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcq.py`:
```python
"""Pure unit tests for the shared MCQ core — no network, no model calls."""
from core.types import Attempt
from benchmarks import _mcq

OPTS = ["aspirin", "penicillin", "insulin", "warfarin"]


def _task(gold=1):
    return _mcq.make_task("t1", "Which drug?", OPTS, gold_index=gold)


def test_parse_answer_colon_last_wins():
    assert _mcq.parse_letter("Maybe B, but Answer: C", OPTS) == "C"


def test_parse_standalone_parenthesised_letter():
    assert _mcq.parse_letter("After reasoning, (D).", OPTS) == "D"


def test_parse_option_text_fallback():
    assert _mcq.parse_letter("It should be penicillin", OPTS) == "B"


def test_parse_garbage_returns_none():
    assert _mcq.parse_letter("I am not sure about this one", OPTS) is None


def test_parse_rejects_out_of_range_letter():
    # E is not a valid option (only A-D); fall through, no option text -> None
    assert _mcq.parse_letter("Answer: E", OPTS) is None


def test_verify_correct():
    r = _mcq.verify(_task(gold=1), Attempt(0, "Answer: B"))
    assert r.success is True and r.score == 1.0
    assert r.raw_eval_output == ""
    assert r.judge_details["parse_ok"] is True


def test_verify_incorrect_does_not_leak_gold():
    r = _mcq.verify(_task(gold=1), Attempt(0, "Answer: A"))
    assert r.success is False and r.score == 0.0
    assert "You answered A" in r.raw_eval_output
    assert "B" not in r.raw_eval_output            # gold letter never shown
    assert r.judge_details["gold_letter"] == "B"   # kept internal only


def test_verify_unparseable_is_failure_with_hint():
    r = _mcq.verify(_task(gold=1), Attempt(0, "no idea honestly"))
    assert r.success is False and r.score == 0.0
    assert r.judge_details["parse_ok"] is False
    assert "Answer:" in r.raw_eval_output


def test_feedback_binary_and_raw_no_llm():
    task = _task(gold=1)
    r = _mcq.verify(task, Attempt(0, "Answer: A"))
    assert "incorrect" in _mcq.feedback(task, Attempt(0, "Answer: A"), r, "binary",
                                        judge_model="x").lower()
    assert _mcq.feedback(task, Attempt(0, "Answer: A"), r, "raw",
                         judge_model="x") == r.raw_eval_output


def test_feedback_unknown_mode_raises():
    task = _task(gold=1)
    r = _mcq.verify(task, Attempt(0, "Answer: A"))
    try:
        _mcq.feedback(task, Attempt(0, "Answer: A"), r, "nope", judge_model="x")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mcq.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks._mcq'`.

- [ ] **Step 3: Implement `benchmarks/_mcq.py`**

Create `benchmarks/_mcq.py`:
```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mcq.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_mcq.py tests/test_mcq.py
git commit -m "feat: shared MCQ core (deterministic letter-match verify + feedback)"
```

---

## Task 2: MedMCQA adapter

**Files:**
- Create: `benchmarks/medmcqa/__init__.py`, `benchmarks/medmcqa/benchmark.py`, `benchmarks/medmcqa/feedback.py`
- Create: `benchmarks/medmcqa/variants/{passk,seqk.binary,seqk.raw,seqk.judge}.yaml`

- [ ] **Step 1: Create `benchmarks/medmcqa/benchmark.py`**

```python
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
```

- [ ] **Step 2: Create `benchmarks/medmcqa/feedback.py`**

```python
"""MedMCQA feedback: binary | raw | judge — all delegated to benchmarks._mcq."""
from __future__ import annotations

from benchmarks._mcq import feedback

__all__ = ["feedback"]
```

- [ ] **Step 3: Create `benchmarks/medmcqa/__init__.py`**

```python
"""MedMCQA — 4-option medical MCQ, deterministic letter-match verifier. Exposes
the three functions the harness calls: load_tasks, verify, feedback."""
from .benchmark import load_tasks, verify
from .feedback import feedback

__all__ = ["load_tasks", "verify", "feedback"]
```

- [ ] **Step 4: Create the four variant YAMLs**

`benchmarks/medmcqa/variants/passk.yaml`:
```yaml
# medmcqa + pass@k baseline (independent, feedback-blind attempts).
metric: pass@k
k: 5
feedback_mode: binary           # ignored under pass@k
model: anthropic/claude-haiku-4-5
judge_model: anthropic/claude-haiku-4-5
temperature: 0.7
max_tasks: 5
out: runs/medmcqa.passk
console_char_limit: 3000
options:
  split: validation
```

`benchmarks/medmcqa/variants/seqk.binary.yaml`:
```yaml
# medmcqa + seq@k with binary feedback (pass/fail bit only).
metric: seq@k
k: 5
feedback_mode: binary           # binary | raw | judge
model: anthropic/claude-haiku-4-5
judge_model: anthropic/claude-haiku-4-5
temperature: 0.7
max_tasks: 5
out: runs/medmcqa.seqk.binary
console_char_limit: 3000
options:
  split: validation
```

`benchmarks/medmcqa/variants/seqk.raw.yaml`:
```yaml
# medmcqa + seq@k with raw feedback (verifier diagnostic: "you answered X, incorrect").
metric: seq@k
k: 5
feedback_mode: raw              # binary | raw | judge
model: anthropic/claude-haiku-4-5
judge_model: anthropic/claude-haiku-4-5
temperature: 0.7
max_tasks: 5
out: runs/medmcqa.seqk.raw
console_char_limit: 3000
options:
  split: validation
```

`benchmarks/medmcqa/variants/seqk.judge.yaml`:
```yaml
# medmcqa + seq@k with LLM-critic feedback (no answer leak: critic never sees gold).
metric: seq@k
k: 5
feedback_mode: judge            # binary | raw | judge
model: anthropic/claude-haiku-4-5
judge_model: anthropic/claude-haiku-4-5
temperature: 0.7
max_tasks: 5
out: runs/medmcqa.seqk.judge
console_char_limit: 3000
options:
  split: validation
```

- [ ] **Step 5: Verify the dataset loads and gold options look right (network)**

Run:
```bash
.venv/bin/python -c "
from benchmarks import medmcqa
ts = medmcqa.load_tasks(split='validation', max_rows=3)
for t in ts:
    print('---', t.id, '| gold', t.grading['gold_letter'])
    print(t.prompt[:600])
    print('GOLD OPTION TEXT:', t.grading['options'][t.grading['gold_index']])
print('loaded', len(ts))
"
```
Expected: 3 tasks print; the GOLD OPTION TEXT is a plausibly-correct answer to each question. **This step confirms `cop` is 0-indexed for this dataset.** If the gold options look systematically wrong (off by one), change `gold = int(row["cop"])` to `gold = int(row["cop"]) - 1` in `benchmark.py` and re-run.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/medmcqa/
git commit -m "feat: add MedMCQA benchmark (validation split, 4-option MCQ)"
```

---

## Task 3: MedQA adapter

**Files:**
- Create: `benchmarks/medqa/__init__.py`, `benchmarks/medqa/benchmark.py`, `benchmarks/medqa/feedback.py`
- Create: `benchmarks/medqa/variants/{passk,seqk.binary,seqk.raw,seqk.judge}.yaml`

- [ ] **Step 1: Create `benchmarks/medqa/benchmark.py`**

```python
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
```

- [ ] **Step 2: Create `benchmarks/medqa/feedback.py`**

```python
"""MedQA feedback: binary | raw | judge — all delegated to benchmarks._mcq."""
from __future__ import annotations

from benchmarks._mcq import feedback

__all__ = ["feedback"]
```

- [ ] **Step 3: Create `benchmarks/medqa/__init__.py`**

```python
"""MedQA (USMLE 4-option) — deterministic letter-match verifier. Exposes the
three functions the harness calls: load_tasks, verify, feedback."""
from .benchmark import load_tasks, verify
from .feedback import feedback

__all__ = ["load_tasks", "verify", "feedback"]
```

- [ ] **Step 4: Create the four variant YAMLs**

`benchmarks/medqa/variants/passk.yaml`:
```yaml
# medqa + pass@k baseline (independent, feedback-blind attempts).
metric: pass@k
k: 5
feedback_mode: binary           # ignored under pass@k
model: anthropic/claude-haiku-4-5
judge_model: anthropic/claude-haiku-4-5
temperature: 0.7
max_tasks: 5
out: runs/medqa.passk
console_char_limit: 3000
options:
  split: test
```

`benchmarks/medqa/variants/seqk.binary.yaml`:
```yaml
# medqa + seq@k with binary feedback (pass/fail bit only).
metric: seq@k
k: 5
feedback_mode: binary           # binary | raw | judge
model: anthropic/claude-haiku-4-5
judge_model: anthropic/claude-haiku-4-5
temperature: 0.7
max_tasks: 5
out: runs/medqa.seqk.binary
console_char_limit: 3000
options:
  split: test
```

`benchmarks/medqa/variants/seqk.raw.yaml`:
```yaml
# medqa + seq@k with raw feedback (verifier diagnostic: "you answered X, incorrect").
metric: seq@k
k: 5
feedback_mode: raw              # binary | raw | judge
model: anthropic/claude-haiku-4-5
judge_model: anthropic/claude-haiku-4-5
temperature: 0.7
max_tasks: 5
out: runs/medqa.seqk.raw
console_char_limit: 3000
options:
  split: test
```

`benchmarks/medqa/variants/seqk.judge.yaml`:
```yaml
# medqa + seq@k with LLM-critic feedback (no answer leak: critic never sees gold).
metric: seq@k
k: 5
feedback_mode: judge            # binary | raw | judge
model: anthropic/claude-haiku-4-5
judge_model: anthropic/claude-haiku-4-5
temperature: 0.7
max_tasks: 5
out: runs/medqa.seqk.judge
console_char_limit: 3000
options:
  split: test
```

- [ ] **Step 5: Verify the dataset loads and gold options look right (network)**

Run:
```bash
.venv/bin/python -c "
from benchmarks import medqa
ts = medqa.load_tasks(split='test', max_rows=3)
for t in ts:
    print('---', t.id, '| gold', t.grading['gold_letter'])
    print('GOLD OPTION TEXT:', t.grading['options'][t.grading['gold_index']])
print('loaded', len(ts))
"
```
Expected: 3 tasks print with a plausibly-correct GOLD OPTION TEXT each.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/medqa/
git commit -m "feat: add MedQA USMLE-4-option benchmark (test split)"
```

---

## Task 4: Dependency manifest

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `datasets` to `requirements.txt`**

Append a line so the file reads:
```
litellm>=1.0
huggingface_hub>=0.20
pyyaml>=6.0
python-dotenv>=1.0
datasets>=2.0
```

- [ ] **Step 2: Add `datasets` to `pyproject.toml` dependencies**

In the `dependencies = [...]` list, add:
```python
    "datasets>=2.0",
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "build: add datasets dependency for MCQ benchmarks"
```

---

## Task 5: End-to-end smoke runs

**Files:** none (produces `runs/` output, not committed)

Prerequisite: `ANTHROPIC_API_KEY` available (Task 0 Step 4). Each run below uses `max_tasks: 5` from the variant; for a faster smoke you may temporarily lower it.

- [ ] **Step 1: MedMCQA pass@k smoke**

Run: `.venv/bin/python -m core run benchmarks/medmcqa/variants/passk.yaml`
Expected: "Loaded N tasks | benchmark=benchmarks.medmcqa ..."; per-task lines printed; `runs/medmcqa.passk/` created with `tasks/`, `summary.json`, `config.json`.

- [ ] **Step 2: MedMCQA seq@k judge smoke (exercises the critic path)**

Run: `.venv/bin/python -m core run benchmarks/medmcqa/variants/seqk.judge.yaml`
Expected: completes; on failed non-final attempts a critic call is made; `runs/medmcqa.seqk.judge/` created.

- [ ] **Step 3: MedMCQA metrics**

Run: `.venv/bin/python -m core metrics runs/medmcqa.passk --k 5`
Expected: prints `pass@1 .. pass@5`.

- [ ] **Step 4: MedQA pass@k + seq@k raw smoke**

Run:
```bash
.venv/bin/python -m core run benchmarks/medqa/variants/passk.yaml
.venv/bin/python -m core run benchmarks/medqa/variants/seqk.raw.yaml
```
Expected: both complete; `runs/medqa.passk/` and `runs/medqa.seqk.raw/` created.

- [ ] **Step 5: MedQA metrics**

Run: `.venv/bin/python -m core metrics runs/medqa.seqk.raw --k 5`
Expected: prints `seq@1 .. seq@5` and a `ΔSeq@K` line.

- [ ] **Step 6: Re-run the unit tests as a final guard**

Run: `.venv/bin/python -m pytest tests/test_mcq.py -q`
Expected: PASS.

---

## Self-Review

**Spec coverage:**
- Shared `_mcq.py` core → Task 1. ✓
- MedMCQA adapter, validation split, single-answer filter, 0-indexed `cop` (verified) → Task 2. ✓
- MedQA adapter, GBaker 4-option, test split → Task 3. ✓
- Deterministic letter-match verifier, no gold leak → Task 1 (`verify`) + tests. ✓
- Feedback modes binary/raw/judge, judge never sees gold/explanation → Task 1 (`feedback`, `JUDGE_CRITIC`). ✓
- Four Anthropic `claude-haiku-4-5` variants per benchmark, k=5, smoke max_tasks → Tasks 2 & 3. ✓
- `datasets` dependency + 3.10+ venv + key → Tasks 0 & 4. ✓
- Verification plan (unit + smoke + metrics) → Tasks 1 & 5. ✓

**Placeholder scan:** none — every code/command step is complete.

**Type/name consistency:** `make_task`, `format_prompt`, `_options_block`, `parse_letter(text, options)`, `verify(task, attempt, *, judge_model=None)`, `feedback(task, attempt, result, mode, *, judge_model)`, `_chosen_text` — used identically across `_mcq.py`, both adapters, and the tests. `grading` keys (`question`, `options`, `gold_index`, `gold_letter`) are written in `make_task` and read in `verify`/`feedback`/smoke steps consistently.

**Known risk carried into execution:** MedMCQA `cop` indexing is verified empirically in Task 2 Step 5 with a documented one-line fix if it is 1-indexed.
