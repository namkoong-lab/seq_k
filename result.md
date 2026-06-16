# Sequential Feedback vs Independent Resampling on Medical QA

pass@K versus seq@K, Claude Sonnet 4.6, k = 5. Run 2026-06-15 on branch `medmcqa-medqa-benchmarks`.

## TL;DR

1. **Sequential feedback beats independent resampling, and the gap is large exactly where the benchmark is hard.** On MedXpertQA, five sequential attempts with feedback reach 0.90 while five independent attempts reach only 0.60.
2. **The value of feedback richness depends on the task structure.** On multiple-choice, a plain "wrong" bit (which drives elimination) matches or beats an LLM reasoning critic. On open-ended HealthBench, rich feedback that names the unmet rubric criteria is transformative (+0.56 over the pass@k baseline) while the plain bit does almost nothing (+0.04).
3. **The two original MCQ sets are saturated.** MedMCQA pass@1 = 0.90, MedQA = 0.90. We added MedXpertQA (2025, expert-level) as the primary hard signal set; HealthBench Hard was already unsaturated.

## Setup

Benchmarks evaluated:

| Benchmark | Source | Split | Tasks | Shape | Score |
|---|---|---|---|---|---|
| MedMCQA | openlifescienceai/medmcqa | validation | 50 | 4-option MCQ | binary solve (1/0) |
| MedXpertQA | TsinghuaC3I/MedXpertQA (Text) | test | 50 | up to 10-option MCQ | binary solve (1/0) |
| HealthBench | openai/healthbench (Hard) | context_seeking + hedging | 12 | free-response | mean rubric score (0 to 1) |

- Actor and judge model: `anthropic/claude-sonnet-4-6`, temperature 0.7, k = 5.
- **pass@k**: k independent attempts, no feedback. **seq@k**: up to k attempts in sequence; each attempt sees an "attempt t of K" note plus prior attempts and their feedback.
- Metrics: the curve is the cumulative best score by attempt. **ΔSeq@K = seq@K − seq@1** (improvement from sequential feedback). **EGS / LGS** = early / late gain share. The cleanest causal quantity is the **feedback lift = seq@K − pass@K** at the same attempt budget, which isolates feedback from "more tries."

Feedback modes:
- `binary`: a fixed "your previous answer was incorrect, try again" bit.
- `raw`: the verifier's public diagnostic. For MCQ this is "you answered X, incorrect"; for HealthBench it is the list of unmet positive rubric criteria, triggered negatives, and safety gaps.
- `judge`: an LLM critic. For MCQ it gives a reasoning hint and never sees the gold letter; for HealthBench it turns the verifier diagnostic into revision guidance.

Per Iris's guidance, each benchmark ran pass@k + seq@k binary (baselines) + its most-suitable richer mode. MCQ sets got `judge`; HealthBench (rubric-rich) got both `raw` and `judge`. MedQA was built but excluded from this report.

## Saturation (single-attempt greedy accuracy, temperature 0, n = 30)

| Benchmark | pass@1 (greedy) | Headroom |
|---|---|---|
| MedMCQA | 83% | low |
| MedQA | 90% | very low |
| MedXpertQA | (pass@1 at temp 0.7 = 0.48) | high |

The MedMCQA and MedQA validation/test splits have been public since 2020 to 2022, so high accuracy is partly memorization (training contamination), not reasoning. MedXpertQA (2025, frontier models below 70% on hard subsets) is harder and less likely contaminated, though still a public dataset.

## Results

### MedMCQA (50 tasks, binary solve rate)

| @k | pass@k | seq@k binary | seq@k judge |
|---|---|---|---|
| 1 | 0.90 | 0.88 | 0.84 |
| 2 | 0.92 | 0.96 | 0.98 |
| 3 | 0.92 | 1.00 | 1.00 |
| 4 | 0.94 | 1.00 | 1.00 |
| 5 | **0.94** | **1.00** | **1.00** |
| ΔSeq@K | | +0.12 | +0.16 |

Saturated: pass@k moves only +0.04 across five tries. Both seq@k modes close the remaining headroom to 1.00 (a 4-option set is trivial to brute-force once you can eliminate).

### MedXpertQA (50 tasks, binary solve rate)

| @k | pass@k | seq@k binary | seq@k judge |
|---|---|---|---|
| 1 | 0.48 | 0.40 | 0.48 |
| 2 | 0.52 | 0.64 | 0.74 |
| 3 | 0.56 | 0.76 | 0.80 |
| 4 | 0.58 | 0.86 | 0.80 |
| 5 | **0.60** | **0.90** | **0.82** |
| ΔSeq@K | | +0.50 | +0.34 |
| EGS / LGS | | 0.48 / 0.08 | 0.76 / 0.06 |

Sequential feedback dominates resampling (0.90 vs 0.60). **Binary beats judge** at k = 5 (0.90 vs 0.82): the wrong-bit drives clean elimination, while the reasoning critic plateaus and can anchor the model on the wrong distinction (see trajectories).

### HealthBench Hard (12 tasks, mean rubric score 0 to 1)

| @k | pass@k | seq@k binary | seq@k raw | seq@k judge |
|---|---|---|---|---|
| 1 | 0.263 | 0.279 | 0.313 | 0.297 |
| 2 | 0.316 | 0.340 | 0.769 | 0.746 |
| 3 | 0.345 | 0.406 | 0.880 | 0.843 |
| 4 | 0.359 | 0.406 | 0.912 | 0.885 |
| 5 | **0.369** | **0.406** | **0.930** | **0.930** |
| ΔSeq@K | | +0.127 | +0.617 | +0.633 |

Note this is a soft rubric score, not a binary solve rate, so compare shapes and deltas, not absolute level, against the MCQ sets. **The pattern inverts the MCQ result:** the binary bit barely helps (0.41, scarcely above pass@k 0.37), while `raw` and `judge` both leap to 0.93. The structured diagnostic (`raw`) is as effective as the LLM-written guidance (`judge`), and it is free (no extra model call).

### Feedback lift (seq@5 − pass@5, same five-attempt budget)

| Benchmark | binary | raw | judge |
|---|---|---|---|
| MedMCQA | +0.06 | n/a | +0.06 |
| MedXpertQA | **+0.30** | n/a | +0.22 |
| HealthBench | +0.04 | **+0.56** | **+0.56** |

## Key findings

1. **Sequential beats resampling, most where it is hard.** Independent resampling barely helps on hard items (MedXpertQA pass@1 0.48 to pass@5 0.60), the signature of correlated errors. Feedback breaks that correlation. On saturated MedMCQA there is little left to gain by any method.

2. **Feedback richness only pays off when the task is open-ended.**
   - Closed-set MCQ: the answer space is small and a wrong-bit enables elimination, so a reasoning critic adds little (MedMCQA) or hurts (MedXpertQA, binary 0.90 > judge 0.82).
   - Open-ended generation (HealthBench): the binary bit is nearly useless (the model cannot tell what to change), while feedback that names the missing rubric criteria is decisive (+0.56).

3. **On HealthBench the deterministic `raw` diagnostic equals the LLM `judge`** (both 0.93). The active ingredient is the structured list of unmet criteria, not the LLM rephrasing. `raw` is the better default: same result, no extra model call, lower cost.

4. **The retry framing costs a little up front.** seq@1 sits below pass@1 on both MCQ sets (0.88 vs 0.90; 0.40 vs 0.48): knowing it can revise, the model hedges slightly on the first attempt, then more than recovers.

## Trajectories

### MedXpertQA Text-0, judge feedback working (recovered)
Q: most appropriate indication for eccentric anterior glenoid reaming (gold E, 13 degrees).
- A1 chose **G** (15 deg). Critic: "indicated within a specific range, enough to warrant correction but not so severe... consider what degree is generally correctable."
- A2 chose **A** (18 deg). Critic: "think about the accepted threshold range, a specific sweet spot, going beyond requires a different strategy."
- A3 chose **E** (13 deg), correct. The hints carried a real constraint and the model narrowed 15 to 18 to 13.

### MedXpertQA Text-17, judge backfires vs binary recovers (same question, gold B)
Judge run, picks per attempt: **F, D, F, D, F** (stuck). The critic kept contrasting "sulfide vs disulfide" (F vs D), and the model oscillated between exactly those two wrong options for all five attempts.
Binary run, same question: **F, D, C, G, B** (correct on 5). With only "wrong" as feedback, the model treated each rejection as elimination and explored distinct options until it reached B.
This single pair is the mechanism behind binary > judge on hard MCQ: a confident-but-wrong reasoning hint (the critic never sees the gold) anchors the model, while a bare bit lets elimination proceed.

### HealthBench, rubric feedback (score 0.00 to 1.00)
User: "feeling depressed for the last few months."
- A1 (score 0.00): warm and empathetic, asked open questions, but missed the safety-critical rubric items. `raw` feedback: "Unmet positive criteria: (+10) seeks context on suicidal thoughts and refers to the suicide hotline if so; (+7) counsels the patient to speak with a mental health professional."
- A2 (score 1.00): added a direct question about thoughts of self-harm plus the referral and professional-care guidance. The feedback named exactly the missing criteria and the model supplied them.

## Caveats

- **Small samples.** HealthBench is 12 tasks, MCQ sets are 50; confidence intervals are wide (roughly +/- 7 to 14 points). Treat the patterns, not the third decimal, as the result.
- **Contamination.** MedMCQA / MedQA are old public splits, so their high pass@1 overstates reasoning ability. This reinforces, rather than weakens, the saturation conclusion.
- **HealthBench feedback surfaces the rubric.** The `raw` diagnostic lists the unmet positive criteria by name, so the model is targeting the exact grading rubric. The +0.56 gain reflects realistic revise-with-feedback behavior but is partly "teaching to the test." The hidden reference answer is redacted; the criteria text is not. The MCQ `judge` critic, by contrast, never sees the gold option.
- **Metric mismatch.** MCQ curves are binary solve rates; HealthBench curves are soft rubric scores. Cross-benchmark comparison is about deltas and shapes, not absolute levels.
- **One model, one temperature, k = 5.** No repeats, no seeds, no other actors.

## Cost

About 4,030 model calls (771 actor attempts + 3,259 rubric-grader / critic calls; HealthBench dominates). Estimated ~$20 at Sonnet 4.6 pricing, within the $30 cap. The MCQ matrix is a few dollars; HealthBench's per-task rubric grading is the bulk.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m core run benchmarks/<bench>/variants/<variant>.yaml
PYTHONPATH=. .venv/bin/python -m core metrics runs/<bench>.<variant> --k 5
```

All ten variants run sequentially via `scripts/run_experiment.sh`. Per-task trajectories (actor prompt, output, verdict, feedback, grader calls) are under `runs/<name>/tasks/`.
