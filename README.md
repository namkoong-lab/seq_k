# seq_k

A small, readable harness for comparing **Pass@K** and **Seq@K** on a benchmark.

- **Pass@K** — `k` independent, feedback-blind attempts; succeed if any passes.
- **Seq@K** — up to `k` sequential attempts. Every attempt carries a horizon note
  ("attempt t of K") from the first attempt onward, and each later attempt also
  sees the previous attempts and their feedback. Because of the horizon note,
  **seq@1 is not the same as pass@1**: under seq@k the model knows it is in a
  retry loop with feedback coming.

One run does exactly **one metric**, chosen in a config file. Every run prints the
*exact prompt the model saw at each attempt* and saves it to a results file, so
debugging is just reading the output.

## Layout

```
core/             # the engine — benchmark-agnostic
  cli.py          # `python -m core run|inspect|metrics`
  types.py        # Task, Attempt, VerifierResult, Step, Trajectory
  llm.py          # complete(model, prompt, temperature) via LiteLLM, native provider keys
  harness.py      # the run loop (pass@k OR seq@k); prints each attempt live
  results.py      # writes each run folder (full.json/results.json/prompts.txt) + step renderer
  metrics.py      # pass@k / seq@k / ΔSeq@K / EGS / LGS from a results file
benchmarks/
  clbench/        # one self-contained benchmark
    benchmark.py  #   load_tasks() + verify()  (rubric judge)
    feedback.py   #   feedback(...) — binary | raw | socratic | directive
    prompts.py    #   judge + critic prompt text
    variants/     #   one YAML per run = this benchmark + a metric + a feedback choice
      passk.yaml
      seqk.binary.yaml
      seqk.raw.yaml
      seqk.socratic.yaml
```

## Install

```bash
pip install -r requirements.txt
# or: pip install -e .   (gives you a `seq_k` command, equivalent to `python -m core`)
```

Set the API key for whichever provider your config's `model` uses, e.g.
`export OPENAI_API_KEY=...` (or put it in a `.env` file — it's loaded automatically).
The model prefix selects the provider: `openai/…`, `anthropic/…`, `gemini/…`,
`deepseek/…`, `dashscope/…` (Qwen).

## Run

```bash
python -m core run benchmarks/clbench/variants/seqk.raw.yaml
python -m core run benchmarks/clbench/variants/passk.yaml
```

Inspect one saved trajectory step by step, and summarize:

```bash
python -m core inspect runs/clbench.seqk.raw --task <task_id>
python -m core metrics runs/clbench.seqk.raw --k 5
```

## Run output

A run writes one folder, `runs/<name>/`, with three files:

- `full.json` — every trajectory in full (prompts, outputs, grading).
- `results.json` — scores and per-rubric verdicts only, no prompts.
- `prompts.txt` — each agent's exact prompt at every step (`ACTOR` / `JUDGE` /
  `CRITIC`), labelled for leakage review.

`inspect` and `metrics` take the run folder (or its `full.json`).

## Add a variation

A variation is one YAML in a benchmark's `variants/` folder. Copy an existing one
and change `metric:` (`pass@k` | `seq@k`) and `feedback_mode:`
(`binary` | `raw` | `socratic` | `directive`). The benchmark is inferred from the
file's path, so the YAML doesn't even name it. All feedback for a benchmark lives
in that benchmark's `feedback.py`.

## Add a benchmark

Create a new self-contained folder `benchmarks/<name>/` whose `__init__.py`
re-exports three functions:

```python
def load_tasks() -> list[Task]: ...
def verify(task, attempt, *, judge_model) -> VerifierResult: ...
def feedback(task, attempt, result, mode, *, judge_model) -> str: ...
```

Add a `variants/` folder with one YAML per run. Nothing else changes — the CLI
finds the benchmark from the config's path.

## No hidden failures

Nothing here swallows errors or substitutes a placeholder result. A failed model
call, an unparseable judge response, or a missing config key **raises** and stops
the run with a real traceback. Results are written per task, so a crash leaves
every finished task on disk and a visible error for the one that broke.
