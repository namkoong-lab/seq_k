# MedMCQA + MedQA benchmarks for seq_k

Date: 2026-06-15
Status: approved-pending-review

## Goal

Add two medical multiple-choice benchmarks to the seq_k harness so we can run the
same pass@K vs seq@K comparison the repo already supports for HealthBench and the
other benchmarks:

- **MedMCQA** — Indian medical-entrance MCQ.
- **MedQA** — USMLE-style MCQ.

Both are 4-option, single-correct-answer MCQ with a known gold letter, so they are
**objectively verifiable** and follow the deterministic-verifier pattern already
established by `benchmarks/arcagi2/` (parse the answer, exact-match, no LLM judge
in scoring, score 1.0/0.0).

## Non-goals

- No free-text / open-ended answering judged by rubric (that is HealthBench's
  shape, not MCQ).
- No new harness/core changes. The two adapters plug into the existing
  `load_tasks / verify / feedback` contract (and, for `judge` feedback, the
  existing `judge_model` plumbing). No edits to `core/`.
- No leaderboard submission or held-out test scoring for MedMCQA (its test labels
  are withheld upstream).

## Architecture

```
benchmarks/
  _mcq.py            # shared MCQ core, imported by both adapters (NOT a runnable
                     #   benchmark: no variants/, leading underscore signals this)
  medmcqa/
    __init__.py      #   re-exports load_tasks, verify, feedback
    benchmark.py     #   load_tasks (HF load + normalize) ; verify delegates to _mcq
    feedback.py      #   feedback delegates to _mcq
    prompts.py       #   ACTOR instruction + JUDGE critic template
    variants/        #   passk, seqk.binary, seqk.raw, seqk.judge
  medqa/
    ...same shape...
```

`benchmarks/_mcq.py` is imported by the two packages (`from benchmarks import _mcq`).
The CLI infers the benchmark from `benchmarks/<name>/variants/<x>.yaml` and imports
`benchmarks.<name>`; `_mcq.py` has no `variants/`, so it is never run directly and
does not interfere with inference.

### Shared core: `benchmarks/_mcq.py`

A single, well-bounded module. Each adapter's `benchmark.py` is responsible only
for dataset-specific loading + normalization; everything generic lives here.

- `Choice` / task shape: a normalized MCQ has a question stem, an ordered list of
  4 option strings, and a gold index (0–3). The adapter packs these into
  `Task.grading = {"options": [...], "gold_index": int, "gold_letter": "A".."D"}`
  and builds `Task.prompt` via `format_prompt(...)`.
- `format_prompt(question, options, *, actor_instruction)` — renders the stem,
  lettered options (`A) ...`), and an instruction to answer with a single letter.
- `parse_letter(text)` — extracts the chosen letter A–D from raw actor output, with
  light repair (matches `Answer: X`, `(X)`, a final standalone letter, or the full
  option text). Returns `None` if nothing parseable.
- `verify(task, attempt)` — deterministic. Parses the letter, compares to
  `gold_letter`. `success = (parsed == gold_letter)`; `score = 1.0/0.0`. On an
  unparseable answer: `success=False`, `score=0.0`, with a format-hint diagnostic
  (the ARC-AGI-2 convention: a malformed answer is a real task failure, not a
  swallowed error). The gold letter is **never** placed in `raw_eval_output`.
- `feedback(task, attempt, result, mode, *, judge_model)` — see below.

The `judge_model` kwarg is threaded through `verify`/`feedback` signatures to match
the harness contract even though `verify` ignores it (deterministic).

## Datasets

| Benchmark | HF repo | Split used | Size | Gold field | Options |
|-----------|---------|-----------|------|-----------|---------|
| MedMCQA | `openlifescienceai/medmcqa` | `validation` | 4,183 | `cop` (0–3) | `opa..opd` |
| MedQA | `GBaker/MedQA-USMLE-4-options` | `test` | 1,273 | `answer_idx` (A–D) | `options` dict A–D |

- **MedMCQA**: the public `test` split has withheld labels, so we evaluate on
  `validation`. Keep only single-answer rows (`choice_type == "single"` when
  present). Carry `subject_name` for optional later slicing via `options:`.
- **MedQA**: the GBaker 4-option variant has a labeled `test` split and a clean
  A–D mapping, avoiding the 5-option/biomed-NER complexity of `bigbio/med_qa`.

Both adapters expose `options:` knobs (`split`, `max_rows`, subject/category
filters) consumed by `load_tasks(**options)`, mirroring how arcagi2/clbench take
`data_dir`/`category`.

## Verifier

Deterministic, no scoring LLM call. `verify` lives in `_mcq.py`; each
`benchmark.py` re-exports it. Output:

- `success`: parsed letter == gold letter.
- `score`: 1.0 / 0.0.
- `raw_eval_output` (public, shown to next attempt): e.g.
  `"You answered B. That is incorrect."` or, on parse failure,
  `"Could not read a single-letter answer (A-D) from your response. Reply with just the letter."`
  Never contains the gold letter or gold option text.
- `judge_details` (internal): `{parsed_letter, gold_letter, parse_ok}`.

## Feedback modes

Follows the repo convention that every benchmark has `binary` + `raw` + one richer
mode.

- `binary` — fixed string: previous answer was wrong, reconsider and choose again.
  (No LLM call.)
- `raw` — `result.raw_eval_output` verbatim ("You answered B. That is incorrect.").
  Across a seq@K run the prompt history accumulates these per-attempt lines, giving
  the model natural elimination. (No LLM call.)
- `judge` — an LLM critic (`judge_model`) that sees **only** the question, the
  options, and the attempt's wrong answer. It is told the answer was marked
  incorrect and asked to give a brief reasoning hint toward reconsidering, WITHOUT
  naming the correct option. It never sees the gold label or any dataset
  explanation field, so there is **no leak path** — feedback quality rides on the
  judge model's own medical reasoning. (One LLM call per failed non-final attempt,
  exactly like clbench socratic/healthbench judge.)

## Variants

Four per benchmark, mirroring `benchmarks/healthbench/variants/`:

- `passk.yaml` — `metric: pass@k`, `feedback_mode: binary` (ignored under pass@k).
- `seqk.binary.yaml` — `metric: seq@k`, `feedback_mode: binary`.
- `seqk.raw.yaml` — `metric: seq@k`, `feedback_mode: raw`.
- `seqk.judge.yaml` — `metric: seq@k`, `feedback_mode: judge`.

Common settings:

```yaml
k: 5
model: anthropic/claude-haiku-4-5
judge_model: anthropic/claude-haiku-4-5
temperature: 0.7
max_tasks: 5          # smoke size; bump for real runs
console_char_limit: 3000
out: runs/<bench>.<variant>
```

## Environment / setup

- Create a Python **3.10+** virtualenv (system python is 3.9.6; `pyproject` requires
  ≥3.10). Install `requirements.txt`.
- Set `ANTHROPIC_API_KEY` in the environment or a `.env` at the repo root
  (`core/cli.py` already calls `load_dotenv()`).
- `datasets` is needed to pull MedMCQA/MedQA from HF (HealthBench uses
  `hf_hub_download` on a single JSONL; the MCQ sets are standard HF datasets, so the
  `datasets` library is the clean loader). Add `datasets>=2.0` to requirements.

## Verification plan

1. Unit-level: parsing (`parse_letter` on `"B"`, `"Answer: C"`, `"(d)"`, full option
   text, and garbage → None) and `verify` (correct/incorrect/unparseable) without
   any network or model call.
2. Smoke: `python -m core run benchmarks/medmcqa/variants/passk.yaml` and the three
   seqk variants with `max_tasks: 2`, confirming tasks load, the actor runs, grading
   produces a score, and `runs/<...>/` is written. Repeat for medqa.
3. `python -m core metrics runs/<...> --k 5` prints a pass@/seq@ curve.

## Open questions / risks

- HF dataset field names can drift; `load_tasks` will fail loud (raise) on a missing
  expected field rather than silently mislabel, matching the repo's fail-loud stance.
- `judge` feedback usefulness on MCQ is an empirical question — that is part of what
  the experiment measures.
