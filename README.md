# seq_k

Pass@K vs Seq@K eval, one benchmark at a time.

- **Pass@K** — `k` independent attempts, no feedback. Pass if any does.
- **Seq@K** — up to `k` attempts in sequence. Each attempt also sees an "attempt
  t of K" note (from the first) plus the prior attempts and their feedback. So
  seq@1 ≠ pass@1: the model knows it's in a retry loop.

One run = one metric, set in a YAML. The exact prompt for every attempt is
printed live and saved.

## Layout

```
core/                # engine; benchmark-agnostic
  cli.py  harness.py  llm.py  metrics.py  results.py  types.py
benchmarks/
  clbench/           # one folder per benchmark
    benchmark.py     #   load_tasks + verify
    feedback.py      #   feedback (binary | raw | socratic | directive | …)
    prompts.py       #   judge + critic templates
    variants/        #   one YAML per runnable config
  advancedif/  arcagi2/  healthbench/  researchrubrics/  terminalbench/
```

## Install

```bash
pip install -r requirements.txt
# or  pip install -e .  for a `seq_k` console script
```

Set the provider key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) in env or `.env`.
The model prefix picks the provider: `openai/…`, `anthropic/…`, `gemini/…`,
`deepseek/…`, `dashscope/…`.

## Run

```bash
python -m core run     benchmarks/clbench/variants/seqk.raw.yaml
python -m core metrics runs/clbench.seqk.raw --k 5
python -m core inspect runs/clbench.seqk.raw --task <task_id>
```

## Output

A run writes `runs/<name>/`:

- `full.json` — every trajectory, untruncated.
- `results.json` — summary at the top (`pass@1..@K` or `seq@1..@K`, plus
  `ΔSeq@K` / EGS / LGS for seq runs), then per-task scores + per-rubric verdicts.
- `prompts.md` — review file: shared actor context once, then each attempt's
  injected delta, with judge/critic prompts folded into `<details>`.

Rewritten per task and swapped in atomically, so a crash keeps the finished
tasks. `inspect` and `metrics` take the run folder or its `full.json`.

## Add a variation

Drop a YAML in `benchmarks/<name>/variants/`. Change `metric:` and
`feedback_mode:`; benchmark is inferred from the path. Benchmark-specific knobs
(data paths, category, themes, …) go under `options:`.

## Add a benchmark

Create `benchmarks/<name>/` exposing three functions via `__init__.py`:

```python
def load_tasks(**options) -> list[Task]: ...
def verify(task, attempt, *, judge_model) -> VerifierResult: ...
def feedback(task, attempt, result, mode, *, judge_model) -> str: ...
```

For agentic benchmarks where each attempt runs in an external environment
(Docker, etc.), implement `run_attempt(task, history, t, k, *, seq, model,
judge_model, temperature, options, out) -> (prompt, output, result)` instead —
the harness uses it in place of `llm.complete + verify`.

Add a `variants/` folder.