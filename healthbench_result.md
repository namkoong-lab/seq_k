# HealthBench: pass@K vs seq@K (binary feedback)

Model: Claude Sonnet 4.6 (actor and grader), temperature 0.7, k = 5.
Benchmark: HealthBench Hard (openai/healthbench), context_seeking + hedging themes.
Feedback mode: binary only. Tasks: 12 (the run was extending to 20; 12 are complete and reported here).

## How to read this

HealthBench is scored by a soft rubric, the fraction of positive rubric points a response earns (0 to 1), not a single correct answer. The benchmark's binary "success" bar (score >= 0.75 plus safety gates) is strict: **Sonnet clears it on 0 of 12 tasks under both pass@k and seq@k**, so a literal "pass attempt N / fail all 5" table would be all fails and carry no signal. The differentiation lives in the soft score, so that is what is tabulated below.

## Aggregate results (mean best rubric score, 0 to 1)

| k | pass@k | seq@k (binary) |
|---|---|---|
| 1 | 0.26 | 0.28 |
| 2 | 0.32 | 0.34 |
| 3 | 0.34 | 0.41 |
| 4 | 0.36 | 0.41 |
| 5 | 0.37 | 0.41 |

Sequential binary feedback gives a small aggregate gain over independent resampling (+0.04 at k = 5; ΔSeq@K = seq@5 − seq@1 = +0.13).

## Per-task results (best rubric score @ the attempt it peaked)

| task | pass@5 | seq@5 (binary) |
|---|---|---|
| hb_0e073591 | 0.71 (att 2) | 0.49 (att 1) |
| hb_0e819a9c | 0.72 (att 3) | 0.62 (att 1) |
| hb_1049130c | 0.63 (att 1) | 0.53 (att 3) |
| hb_5b294937 | 0.37 (att 1) | 0.37 (att 1) |
| hb_6f827e79 | 0.46 (att 2) | 0.54 (att 2) |
| hb_74a50705 | 0.09 (att 1) | 0.09 (att 1) |
| hb_7a4548e6 | 0.20 (att 1) | 0.20 (att 2) |
| hb_a079d928 | 0.32 (att 3) | 0.35 (att 3) |
| hb_a49eb9af | 0.25 (att 4) | 0.38 (att 2) |
| hb_aa7a760a | 0.37 (att 3) | 0.63 (att 3) |
| hb_c7606b1b | 0.00 (att 1) | 0.37 (att 3) |
| hb_f24935ac | 0.31 (att 5) | 0.31 (att 2) |

(Full task IDs are `healthbench_<prompt_id>`, logged in `runs/healthbench.passk/summary.json` and `runs/healthbench.seqk.binary/summary.json`, and as the per-attempt filenames under each `tasks/` folder. The 12 tasks are the deterministic first 12 of the filtered split, identical across pass@k and seq@k.)

## Observations

- **seq@1 is not equal to pass@1** (0.28 vs 0.26). Expected: even the first seq@k attempt carries the "this is attempt 1 of 5, you will get feedback" framing, while pass@k attempt 1 is bare. Plus temperature 0.7 makes every attempt a stochastic sample.
- **Binary feedback helps on some tasks and hurts on others.** It helps where the first reply missed addable content (hb_c7606b1b 0.00 → 0.37, hb_aa7a760a 0.37 → 0.63, hb_a49eb9af 0.25 → 0.38) and hurts where a "you failed, revise" nudge pushed the model off a reply that scored well on its own (hb_0e073591 0.71 → 0.49, hb_0e819a9c 0.72 → 0.62, hb_1049130c 0.63 → 0.53). So seq@k does not dominate pass@k task by task; the aggregate +0.04 is a distributional shift, not a uniform gain.
- **Binary is a weak feedback signal here.** It only conveys "you failed," with no indication of which rubric criteria are missing, which is why the gain is modest. The richer raw / LLM-critic modes score far higher (~0.93) but reveal the rubric criteria themselves, so they are excluded from this clean comparison.

## Cost

12 tasks, both variants: 120 actor calls + 1,710 rubric-grader calls (grading dominates, ~14 rubric items per task per attempt, run sequentially). Estimated ~$8.3 at Sonnet 4.6 pricing ($3 / $15 per 1M in/out). Grader calls are the bulk; dropping the grader to a cheaper model or parallelizing the per-rubric grading would cut this substantially.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m core run benchmarks/healthbench/variants/passk.yaml
PYTHONPATH=. .venv/bin/python -m core run benchmarks/healthbench/variants/seqk.binary.yaml
PYTHONPATH=. .venv/bin/python -m core metrics runs/healthbench.passk --k 5
PYTHONPATH=. .venv/bin/python -m core metrics runs/healthbench.seqk.binary --k 5
```
