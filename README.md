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
# → writes runs/clbench.seqk.raw-2026-06-15T22-38-45Z/

python -m core run     benchmarks/clbench/variants/seqk.raw.yaml \
                       --resume runs/clbench.seqk.raw-2026-06-15T22-38-45Z
# → continues that exact folder (skip-already-done + next-attempt logic)

python -m core metrics runs/clbench.seqk.raw-2026-06-15T22-38-45Z --k 5
python -m core inspect runs/clbench.seqk.raw-2026-06-15T22-38-45Z --task <task_id>
python -m core upload  runs/clbench.seqk.raw-2026-06-15T22-38-45Z    # manual S3 re-sync
```

Every `run` invocation stamps `<out>-<UTC-timestamp>/`, so two people running
the same variant don't stomp each other. Use `--resume <existing-folder>` to
continue a crashed run; `init_run`'s config-mismatch guard refuses to mix
incompatible configs into the same folder.

## S3 sync (default ON)

Every `python -m core run` uploads the finished run dir to `s3://seq-k/<run_name>-<timestamp>/`
at the end. To upload, you need AWS credentials and write access to the bucket
(walked through below).

### First-time AWS setup

```bash
# 1. Install the AWS CLI v2 (one-time, per machine).
brew install awscli                                # macOS; or follow the official docs

# 2. Authenticate.
aws login                                          # browser-based IAM/Identity Center sign-in (CLI v2.30+)
# OR  aws sso login                                # if your org gave you an SSO start URL + you've run `aws configure sso` once
# OR  aws configure                                # if you have static IAM access keys

# 3. Verify you're signed in.
aws sts get-caller-identity                        # should print your Account + ARN

# 4. Verify bucket access.
aws s3 ls s3://seq-k/                              # should list (possibly empty) without error
echo hi > /tmp/seqk-test.txt
aws s3 cp /tmp/seqk-test.txt s3://seq-k/_check.txt && aws s3 rm s3://seq-k/_check.txt
rm /tmp/seqk-test.txt
```

If step 4 errors `AccessDenied`, your IAM principal doesn't have the right
permissions yet — see [Collaborators](#collaborators) below for the policy to
attach.

#### `aws login` vs `aws sso login`

| Use this | When |
|---|---|
| `aws login` | Quick browser-based IAM sign-in (CLI v2.30+, late 2025). No prior `aws configure sso` needed. What we use in this repo. |
| `aws sso login` | Your org uses AWS IAM Identity Center; you've already run `aws configure sso` once with the start URL. |
| `aws configure` | You have static IAM access keys (Access Key ID + Secret Access Key). No browser needed but keys need rotation. |

All three end with the same outcome: working credentials on disk. The auto-sync
in this repo just shells out to `aws s3 sync`, so whichever method gave you
working credentials, it'll Just Work.

Re-auth when your session expires — usually every 8-12 hours.

### Opting out

| Scope | How |
| --- | --- |
| One-off | `python -m core run <config> --no-upload` |
| Per variant | add `s3_sync: false` to the YAML (see `benchmarks/terminalbench/variants/smoke.yaml`) |
| Per machine | `export SEQK_S3_SYNC=0` |

Override the bucket with `SEQK_S3_BUCKET=<name>`. On failure (expired session,
network, AccessDenied) the run exits non-zero; local files are still on disk,
retry with `python -m core upload runs/<name>-<timestamp>`.

## Collaborators

The S3 bucket is shared, so multiple people can run experiments and have their
results land in one place. Two questions to answer for each new collaborator:

**1. Do they have an AWS principal (IAM user or SSO identity) with write access
to `s3://seq-k/`?**

If not, the bucket owner attaches this minimal IAM policy to their user/role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:ListBucket"],                          "Resource": "arn:aws:s3:::seq-k"},
    {"Effect": "Allow", "Action": ["s3:PutObject","s3:GetObject","s3:DeleteObject"], "Resource": "arn:aws:s3:::seq-k/*"}
  ]
}
```

The collaborator then follows the [First-time AWS setup](#first-time-aws-setup)
above. Cross-account access works the same way — bucket owner adds a bucket
policy granting the other account's ARN.

**2. Are they running an experiment that would collide with someone else's?**

Probably not — every `core run` invocation stamps `<out>-<UTC-timestamp>/`, so
two people running the same variant get distinct folders in S3. The only way
to collide is to point `--resume` at the *same* folder name from two machines,
which is opt-in.

## Output

A run writes `runs/<out>-<timestamp>/`:

```
config.json                            frozen run config (benchmark, metric, k, model, options)
summary.json                           derived pass@k / seq@k + per-task scores
tasks/
  <task-slug>_attempt_01.json          one self-contained file per (task, attempt)
  <task-slug>_attempt_02.json
  ...
_harbor_jobs/                          TerminalBench only — Harbor's Docker artifacts
```

Each `tasks/<slug>_attempt_NN.json` has the same shape across every benchmark
(documented at the top of `core/results.py`):

```
prompt           the exact text the actor (model being evaluated) saw
output           the actor's raw response
result           the judge's verdict:
  success / score
  raw_eval_output      judge's PUBLIC diagnostic — safe to show next attempt
  judge_details        judge's INTERNAL scratch (raw judge output, verdicts, etc.)
critic_feedback  feedback for the NEXT attempt (seq@k only; null otherwise)
calls            judge/critic LLM calls made this attempt
```

Per-attempt files mean a crash only loses the in-flight attempt; resume picks
up from the next one. `inspect` and `metrics` both take the run folder.

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