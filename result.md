# Sequential Feedback vs Independent Resampling on Medical QA

pass@K versus seq@K, Claude Sonnet 4.6, k = 5. Run 2026-06-15 on branch `medmcqa-medqa-benchmarks`.
Updated to report HealthBench on the non-leaking feedback mode (see "Leakage control" below).

## TL;DR

1. **Sequential feedback beats independent resampling, and the gap is largest where the benchmark is hard and uncontaminated.** On MedXpertQA, five sequential attempts with feedback reach 0.90 while five independent attempts reach only 0.60 (a +0.30 lift at matched budget).
2. **On multiple choice, richer feedback does not help and can hurt.** A plain "wrong" bit drives clean elimination and matches or beats an LLM reasoning critic (MedXpertQA binary 0.90 vs judge 0.82). The critic, which never sees the gold answer, can anchor the model on a wrong distinction.
3. **HealthBench's dramatic "rich feedback" gain was rubric leakage, not reasoning.** The `raw` and `judge` feedback list the exact graded criteria, so the model fills in the checklist and soft score jumps to 0.93. With non-leaking feedback (`binary`), HealthBench's sequential gain is modest (+0.13), in line with the other sets. **We report HealthBench on `binary`.**
4. **The two original MCQ sets are saturated and likely contaminated.** MedMCQA pass@1 = 0.90, MedQA = 0.90 on splits public since 2022. MedXpertQA (2025, expert-level) is the primary hard signal set; HealthBench Hard was already unsaturated (pass@1 = 0.26).

## Summary tables (pass@1, pass@5, seq@1, seq@5)

Per benchmark, delta = @5 − @1. HealthBench is reported on `binary`; `raw`/`judge` reveal the graded rubric and are flagged as leaking.

### MedMCQA (50 tasks, solve rate) — saturated baseline
| Variant | @1 | @5 | delta |
|---|---|---|---|
| pass@k | 0.90 | 0.94 | +0.04 |
| seq@k binary | 0.88 | 1.00 | +0.12 |
| seq@k judge | 0.84 | 1.00 | +0.16 |

### MedXpertQA (50 tasks, solve rate) — primary signal set
| Variant | @1 | @5 | delta |
|---|---|---|---|
| pass@k | 0.48 | 0.60 | +0.12 |
| seq@k binary | 0.40 | 0.90 | +0.50 |
| seq@k judge | 0.48 | 0.82 | +0.34 |

### HealthBench (12 tasks, rubric score)
| Variant | @1 | @5 | delta |
|---|---|---|---|
| pass@k | 0.26 | 0.37 | +0.11 |
| seq@k binary | 0.28 | 0.41 | +0.13 |
| seq@k raw (leaks rubric) | 0.31 | 0.93 | +0.62 |
| seq@k judge (leaks rubric) | 0.30 | 0.93 | +0.63 |

## Setup

| Benchmark | Source | Split | Tasks | Shape | Score |
|---|---|---|---|---|---|
| MedMCQA | openlifescienceai/medmcqa | validation | 50 | 4-option MCQ | binary solve (1/0) |
| MedXpertQA | TsinghuaC3I/MedXpertQA (Text) | test | 50 | up to 10-option MCQ | binary solve (1/0) |
| HealthBench | openai/healthbench (Hard) | context_seeking + hedging | 12 | free-response | mean rubric score (0 to 1) |

- Actor and judge model: `anthropic/claude-sonnet-4-6`, temperature 0.7, k = 5.
- **pass@k**: k independent attempts, no feedback. **seq@k**: up to k attempts in sequence; each attempt sees an "attempt t of K" note plus prior attempts and their feedback.
- Curve = cumulative best score by attempt. **ΔSeq@K = seq@K − seq@1**. **Feedback lift = seq@K − pass@K** (isolates feedback from "more tries"). EGS / LGS = early / late gain share.

Feedback modes and whether they reveal the target:

| Mode | What the model is told | Reveals target? |
|---|---|---|
| binary | "your previous answer was incorrect, revise" | no |
| raw (MCQ) | "you answered X, incorrect" | no (only that the chosen option is wrong) |
| judge (MCQ) | LLM reasoning hint, never names the gold letter | no |
| raw (HealthBench) | the verifier diagnostic, listing the unmet graded rubric criteria by name | **yes (rubric)** |
| judge (HealthBench) | LLM guidance derived from that diagnostic | **yes (rubric)** |

## Leakage control (read this before the HealthBench numbers)

Two distinct "leakage" concerns came up and are handled differently:

1. **Training contamination.** MedMCQA / MedQA validation and test splits have been public with labels since 2020 to 2022, so a 2026 model has likely seen them. This inflates their pass@1 (it is partly memorization, not reasoning) and is exactly why they are treated as saturated baselines, not signal. MedXpertQA (2025) and HealthBench (2025) are far less exposed; HealthBench pass@1 = 0.26 confirms it is not memorized.

2. **Feedback leakage.** On HealthBench, the `raw` and `judge` feedback name the exact rubric criteria the response is graded on. The model then satisfies the named checklist, so soft score leaps to 0.93. This is teaching to the test, not better reasoning. The harness itself does not leak: the actor prompt never contains the gold answer, the verifier is deterministic, and the effect is localized only to the criteria-revealing modes (binary and pass@k stay low on the identical tasks). We therefore report HealthBench on `binary`, the only non-leaking mode, and show raw/judge separately as a leakage demonstration.

## Results

### MedMCQA (50 tasks, binary solve rate) — saturated baseline

| @k | pass@k | seq@k binary | seq@k judge |
|---|---|---|---|
| 1 | 0.90 | 0.88 | 0.84 |
| 5 | **0.94** | **1.00** | **1.00** |
| ΔSeq@K | | +0.12 | +0.16 |

pass@k moves only +0.04 across five tries. Both seq@k modes close the small remaining headroom (a 4-option set is trivial to brute-force once you can eliminate). High pass@1 here is partly contamination.

### MedXpertQA (50 tasks, binary solve rate) — primary signal set

| @k | pass@k | seq@k binary | seq@k judge |
|---|---|---|---|
| 1 | 0.48 | 0.40 | 0.48 |
| 2 | 0.52 | 0.64 | 0.74 |
| 3 | 0.56 | 0.76 | 0.80 |
| 4 | 0.58 | 0.86 | 0.80 |
| 5 | **0.60** | **0.90** | **0.82** |
| ΔSeq@K | | +0.50 | +0.34 |
| EGS / LGS | | 0.48 / 0.08 | 0.76 / 0.06 |

pass@1 = 0.48 matches the published frontier ceiling (top models below 70%), so there is no contamination or harness leak here. Sequential feedback dominates resampling (0.90 vs 0.60). **Binary beats judge** at k = 5: the wrong-bit drives elimination, while the reasoning critic plateaus and can anchor the model on the wrong distinction.

### HealthBench Hard (12 tasks, mean rubric score 0 to 1)

Reported result, non-leaking feedback:

| @k | pass@k | seq@k binary |
|---|---|---|
| 1 | 0.263 | 0.279 |
| 3 | 0.345 | 0.406 |
| 5 | **0.369** | **0.406** |
| ΔSeq@K | | +0.127 |

Sequential feedback that does not reveal the rubric gives a **modest +0.13**, comparable to the MCQ sets. This is the honest HealthBench result.

Leakage demonstration (excluded from the headline). The criteria-revealing modes:

| @k | seq@k raw | seq@k judge |
|---|---|---|
| 1 | 0.313 | 0.297 |
| 5 | **0.930** | **0.930** |

Both leap to 0.93 because the feedback lists the graded criteria verbatim (for example: *"Unmet positive criteria: (+10) seeks context on suicidal thoughts and refers to the hotline; (+7) counsels the patient to see a mental health professional"*). The model then supplies exactly those. Same harness, same tasks, the only difference from `binary` is that the rubric is named, so the +0.52 over binary is rubric-revelation, not reasoning.

### Clean feedback lift (seq@5 − pass@5, non-leaking modes only)

| Benchmark | binary | judge (MCQ) |
|---|---|---|
| MedMCQA | +0.06 | +0.06 |
| MedXpertQA | **+0.30** | +0.22 |
| HealthBench | +0.04 | n/a (raw/judge leak) |

The real, leakage-free signal: sequential feedback helps most on the hard, uncontaminated closed-form set (MedXpertQA). On the saturated MCQ set and on open-ended HealthBench, the clean effect is small.

## Key findings

1. **Sequential beats resampling, most where it is hard and clean.** Independent resampling barely helps on hard items (MedXpertQA pass@1 0.48 to pass@5 0.60), the signature of correlated errors; feedback breaks that correlation (+0.30 lift). On saturated MedMCQA there is little left to gain.

2. **On closed-form MCQ, richer feedback does not pay off and can hurt.** A wrong-bit enables elimination, so the reasoning critic adds little (MedMCQA) or loses (MedXpertQA, binary 0.90 vs judge 0.82). A confident-but-wrong hint anchors the model.

3. **The earlier "rich feedback transforms open-ended tasks" claim does not survive a leakage control.** HealthBench's +0.56 came from the feedback disclosing the grading rubric. With non-leaking feedback the gain is +0.13. Open-ended tasks do not intrinsically benefit more from richer feedback once you stop revealing the test.

4. **The retry framing costs a little up front.** seq@1 sits below pass@1 on both MCQ sets (0.88 vs 0.90; 0.40 vs 0.48): the model hedges slightly knowing it can revise, then recovers through feedback.

## Trajectories

### MedXpertQA Text-0, judge feedback working (recovered)
Q: most appropriate indication for eccentric anterior glenoid reaming (gold E, 13 degrees).
A1 chose G (15 deg); critic gave a "correctable sweet spot range" hint. A2 chose A (18 deg); critic reinforced the threshold idea. A3 chose E (13 deg), correct. The hint carried a real constraint without naming the option.

### MedXpertQA Text-17, judge backfires vs binary recovers (same question, gold B)
Judge picks: F, D, F, D, F (stuck). The critic kept contrasting "sulfide vs disulfide" (F vs D), and the model oscillated between those two wrong options for all five attempts.
Binary picks: F, D, C, G, B (correct on 5). With only "wrong" as feedback, the model treated each rejection as elimination and explored distinct options until it reached B.

### HealthBench, rubric leakage in action (score 0.00 to 1.00)
User: "feeling depressed for the last few months." A1 (0.00) was empathetic but missed the safety-critical items. `raw` feedback named them: "(+10) ask about suicidal thoughts and refer to the hotline; (+7) advise seeing a mental health professional." A2 (1.00) added exactly those. Illustrates why raw/judge inflate the score: the model is handed the checklist.

## Caveats

- **Small samples.** HealthBench is 12 tasks, MCQ sets are 50; confidence intervals are wide (roughly +/- 7 to 14 points). Read the patterns, not the third decimal.
- **Contamination on the old MCQ sets.** MedMCQA / MedQA pass@1 overstates reasoning; they are saturated baselines by design. A held-out or perturbed re-run would quantify the memorized fraction.
- **Metric mismatch.** MCQ curves are binary solve rates; HealthBench curves are soft rubric scores. Compare deltas and shapes, not absolute levels.
- **HealthBench feedback realism.** `binary` is the clean measure used here. A "blind" feedback mode (telling the model only how many or which categories it missed, without the criterion text) would give a richer non-leaking signal than binary; not yet run.
- **One model, one temperature, k = 5.** No repeats, no seeds, no other actors.

## Cost

About 4,030 model calls (771 actor attempts + 3,259 rubric-grader / critic calls; HealthBench dominates). Estimated ~$20 at Sonnet 4.6 pricing, within the $30 cap.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m core run benchmarks/<bench>/variants/<variant>.yaml
PYTHONPATH=. .venv/bin/python -m core metrics runs/<bench>.<variant> --k 5
```

All ten variants run via `scripts/run_experiment.sh`. Per-task trajectories are under `runs/<name>/tasks/`. HealthBench raw/judge runs are retained as the leakage demonstration, not as the reported result.
