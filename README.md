# seq_k

Pass@K vs Seq@K eval, one benchmark at a time.

- **Pass@K** — `k` independent attempts, no feedback. Pass if any does.
- **Seq@K** — up to `k` attempts in sequence. Each attempt also sees a "This is
  attempt t of K" note, every prior attempt's output, and every prior critic
  feedback. So seq@1 ≠ pass@1: the model knows it's in a retry loop. Optionally
  cap cumulative output tokens per task with `output_budget`, and/or compress
  history with `summarize` (see Variant YAML keys).

One run = one metric, set in a YAML. The exact prompt for every attempt is
printed live and saved.

## Layout

```
core/                # engine; benchmark-agnostic
  cli.py  harness.py  llm.py  metrics.py  results.py  s3sync.py  summarizer.py  types.py
benchmarks/
  clbench/           # one folder per benchmark
    benchmark.py     #   VERIFIER, LLM_CRITIC_MODES, slice_name(), load_tasks, verify
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
`deepseek/…`, `dashscope/…`, `openrouter/…`.

**OpenRouter** (`OPENROUTER_API_KEY`) reaches every vendor through one key — use
`openrouter/<vendor>/<model>`, e.g. `openrouter/anthropic/claude-sonnet-4.5`. It
also reports its actual charge per call, so those runs get exact
`cost_usd` (`cost_source: "reported"`) instead of table-estimated. For agentic
runs (terminalbench) the key must be forwarded into the container explicitly:
`options: {pass_env: [OPENROUTER_API_KEY]}` — the default is `OPENAI_API_KEY`.

## Run

```bash
python -m core run     benchmarks/clbench/variants/seqk.raw.yaml
# → writes runs/clbench-dkr/seqk/anthropic__claude-sonnet-4-6/anthropic__claude-sonnet-4-6/raw/

python -m core run     benchmarks/clbench/variants/seqk.raw.yaml
# → same YAML, same path → AUTO-RESUMES (skips tasks that are already done)

python -m core metrics runs/clbench-dkr/seqk/anthropic__claude-sonnet-4-6/anthropic__claude-sonnet-4-6/raw --k 5
python -m core inspect runs/clbench-dkr/seqk/anthropic__claude-sonnet-4-6/anthropic__claude-sonnet-4-6/raw --task-index 1
python -m core upload  runs/clbench-dkr/seqk/anthropic__claude-sonnet-4-6/anthropic__claude-sonnet-4-6/raw   # manual S3 re-sync
```

The run path is **derived deterministically from the YAML config** — no `out:`
field needed. Re-running the same YAML always resumes that exact path.

### Run-path layout

```
runs/<slice>/<metric>/<agent>/<verifier>/<feedback>/
```

| Slot | What it is | Examples |
|---|---|---|
| `<slice>` | benchmark + dataset variant. `benchmark.slice_name(options)` decides. | `terminalbench`, `clbench-dkr`, `clbench-rsa`, `arcagi2-evaluation` |
| `<metric>` | `passk` or `seqk` | |
| `<agent>` | the actor model id, with `/` → `__` (filesystem-safe). Actor-level switches append suffixes here in a fixed order — `+budget-<N>` (`output_budget`), `+r-<level>` (`reasoning_effort`), `+sum` (`summarize`) — so a switched-on run never collides with its switched-off sibling. | `anthropic__claude-sonnet-4-6`, `…+budget-4000`, `…+r-low`, `…+sum` |
| `<verifier>` | judge model id (if LLM judge) OR fixed string | `anthropic__claude-sonnet-4-6`, `harbor` (terminalbench), `deterministic` (arcagi2) |
| `<feedback>` | template mode name OR critic model id (for LLM critic modes) | `raw`, `binary` or model id for `judge`|

So two runs that differ only on `model` / `judge_model` / `critic_model` /
`feedback_mode` / `metric` / `slice` land at different paths automatically and
never collide.

### k mismatch (extend vs clobber)

Use the `continue: true` flag when resuming from a previous task. If new k > old k, the old attempts will be kept and the experiments will continue at the last recorded k-th attempt. Otherwises, if continue is set to false, the whole run is restarted. 

### Variant YAML — supported keys

```yaml
metric: seq@k                              # passk | seqk
k: 5                                       # max attempts per task
feedback_mode: raw                         # binary | raw | compact | cell_match | retry_diagnostics | socratic | directive | judge | critic (depends on benchmark)
model: anthropic/claude-sonnet-4-6         # the ACTOR — the model being evaluated
judge_model: anthropic/claude-sonnet-4-6   # defaults to `model`. Only used if benchmark's VERIFIER == "llm"
critic_model: anthropic/claude-sonnet-4-6  # defaults to `model` (the actor). Only used if feedback_mode is in LLM_CRITIC_MODES
temperature: 0.7
task_indices: [1, 5, 7]                    # optional: 1-based canonical indices to run (subset). Mutually exclusive with max_tasks.
max_tasks: 5                               # optional: first N tasks. Ignored if task_indices is set.
continue: false                            # see "k mismatch" above. Default false.
s3_sync: true                              # see S3 sync section. Default true.
console_char_limit: 3000                   # how much to truncate when printing live; doesn't affect saved data
reasoning_effort: low                      # optional, ACTOR ONLY (judge/critic never get it). low | medium | high.
                                           #   Opts into the provider's reasoning / extended-thinking mode; litellm
                                           #   translates it per provider and raises if the model has no such mode.
                                           #   Not supported for agentic benchmarks (the actor call lives inside
                                           #   run_attempt) → error. Appends `+r-<level>` to the agent path slot.
output_budget: 4000                        # optional, seq@k ONLY (pass@k + output_budget → error). Cap on
                                           #   cumulative actor output tokens per task. Each attempt is told how
                                           #   many tokens remain; checked AFTER it generates — if its output pushes
                                           #   the task over budget, the attempt fails immediately (judge + critic
                                           #   skipped) and the task ends. Omit = off (attempt schema + paths unchanged).
summarize: true                            # optional, seq@k ONLY. See "Self-summarization" below.
summarizer_model: anthropic/claude-sonnet-4-6   # defaults to `model` — the agent summarizes for itself.
options:                                   # benchmark-specific (data_path, category, themes, …)
  data_path: ~/datasets/AdvancedIF/data.jsonl
```

### Self-summarization (`summarize`, default off)

seq@k prompts grow linearly in `k`: every attempt appends a full prior output plus
its full feedback, so cumulative input tokens over a trajectory grow **quadratically**.
`summarize: true` bounds that.

After each *failed* attempt, the **actor's own model** compresses that attempt's
output **together with the critic feedback it received** into one short summary.
The next attempt sees:

```
<AttemptSummary 1>  …  </AttemptSummary 1>          instead of
<PreviousAttempt 1> … </PreviousAttempt 1> + <Feedback 1> … </Feedback 1>
```

Summaries are a **rolling log**: each attempt is summarized once, independently, and
kept. Nothing is re-summarized or collapsed, so resuming — or extending `k` with
`continue: true` — just appends to the log.

- **Gating** is identical to the critic's: failed attempts only, skipped on success
  and on over-budget attempts, but it *does* run on the final attempt so a later
  k-extension resumes with an unbroken log.
- **Model**: `summarizer_model` defaults to `model`, not down the judge chain — the
  point is that the agent summarizes for *itself*, so what carries forward is what
  the agent chose to remember.
- **Interaction with `output_budget`**: summary tokens **do not** count against the
  budget (it measures answer effort, and summaries are short by construction), but
  they **are** counted in `tokens` / `cost_usd`. Enforced structurally — budget
  accounting filters on `phase == "actor"`, and the summarizer runs under its own phase.
- **Works on every benchmark, including agentic ones.** The harness produces the
  summary uniformly. *Consuming* it differs by contract: standard benchmarks get it
  free through `build_prompt`, while a benchmark that owns its prompt via
  `run_attempt` must read `prior[i]["summarizer"]["summary"]` itself — see
  `terminalbench._retry_context`. A benchmark whose next prompt carries something
  other than `actor.output` (TerminalBench carries the full trajectory) also exposes
  `summarizer_source(output, result)` so the summarizer compresses the right text.
- **Length is the summarizer's choice** — no word cap. Asking for "a summary" is the
  constraint; a hard cap would truncate a complex failure and pad a trivial one. What
  keeps it bounded is `Do not attempt the task again` in the template, which stops it
  turning into a redraft.
- **Interpretation**: feedback reaches the actor *through* the summarizer on these runs,
  so comparing `feedback_mode` values with `summarize: true` measures each mode as
  filtered by self-summarization. Pair with the identical `summarize`-off variant to
  separate the two effects.

## Output

Run folder mirrors S3 exactly **except `config.json` is local-only** (it's a
frozen snapshot of the YAML — useless on a different machine).

```
runs/<slice>/<metric>/<agent>/<verifier>/<feedback>/
├── config.json                LOCAL ONLY (not uploaded to S3)
├── summary.json               aggregate pass@k / seq@k + token totals + last_updated
├── task-1/                    canonical 1-based index — same task → same folder forever
│   ├── task_meta.json         { task_id, prompt, … }  — original identity
│   ├── summary.json           per-task: success, best_score, per-attempt scores, tokens, last_updated
│   ├── attempt-1.json         actor/judge/critic shape (see below)
│   └── attempt-2.json
├── task-2/ …
```

### Per-attempt JSON shape

Identical across every benchmark — three role sections, each independent.

```jsonc
{
  "task_id": "cancel-async-tasks",
  "task_index": 2,                                   // canonical, matches folder name
  "metric": "seq@k",
  "feedback_mode": "raw",
  "attempt_index": 1,                                // 1-based

  // ACTOR — the model being evaluated
  "actor": {
    "model": "anthropic/claude-sonnet-4-6",
    "prompt": "...",                                 // EXACT text the actor saw (full prior trajectories + verifier outputs for seq@k attempt ≥ 2)
    "output": "...",                                 // EXACT response
    "input_tokens": 7422, "cached_tokens": 3116, "thinking_tokens": 0, "output_tokens": 1372,
    "budget": {"total": 4000, "used": 1372, "over": false}  // present ONLY when output_budget is set; "over": true = this attempt busted the budget
  },

  // JUDGE — produces success/score
  "judge": {
    "model": null,                                   // null when verifier isn't LLM (terminalbench: "harbor"; arcagi2: "deterministic") — also null when the attempt went over budget (judge skipped)
    "success": false, "score": 0.0,
    "raw_eval_output": "...",                        // judge's public diagnostic (e.g. full pytest stdout for terminalbench)
    "details": { … },                                // benchmark-specific internal scratch ({"over_budget": true} when the attempt was killed for exceeding output_budget)
    "calls": [                                       // every LLM call the judge made, with provider-reported tokens
      {"model": "...", "prompt": "...", "output": "...",
       "input_tokens": 1234, "cached_tokens": 0, "thinking_tokens": 0, "output_tokens": 56}
    ]
  },

  // CRITIC — produces feedback for next attempt (seq@k failed only)
  "critic": {
    "model": null,                                   // null when feedback_mode is template-only
    "feedback": "...",                               // EXACT string the next attempt's actor.prompt will include (null if pass@k, success, or no critic)
    "calls": [ … ]                                   // [] when no LLM critic
  },

  // SUMMARIZER — compresses (actor.output + critic.feedback) for the next attempt.
  // The WHOLE KEY IS ABSENT unless the run sets `summarize: true`, so summarize-off
  // runs keep the three-section schema above exactly.
  "summarizer": {
    "model": "anthropic/claude-sonnet-4-6",          // defaults to actor's model; null when the summarizer was skipped (success / over budget)
    "summary": "...",                                // EXACT text the next attempt sees as <AttemptSummary N>, replacing the verbatim attempt + feedback
    "calls": [ … ]                                   // one entry, same token schema as judge/critic calls
  }
}
```

### Token counts + cost

Every token-bearing dict — `actor`, each `judge.calls[i]`, each `critic.calls[i]`,
each `summarizer.calls[i]`, and per-model aggregations — uses the same four fields
in the same order:

```jsonc
{"input_tokens": N, "cached_tokens": M, "thinking_tokens": T, "output_tokens": O}
```

| Field | What | Source |
|---|---|---|
| `input_tokens` | total prompt tokens (includes cache reads) | litellm `prompt_tokens` / Harbor `n_input_tokens` |
| `cached_tokens` | prompt tokens served from cache (10× cheaper) | `prompt_tokens_details.cached_tokens` / Harbor `n_cache_tokens` |
| `thinking_tokens` | **subset of `output_tokens`** — reasoning/thinking, for visibility only | OpenAI `reasoning_tokens`, Gemini `thoughts_token_count`, Anthropic 0 (no breakout) |
| `output_tokens` | total completion tokens (already includes thinking) | `completion_tokens` / Harbor `n_output_tokens` |

Per-task and run-level `summary.json` aggregate by **model id** (same model
used by multiple roles → merged) and append `cost_usd`:

```jsonc
"tokens": {
  "anthropic/claude-sonnet-4-6": {
    "input_tokens": 1375958, "cached_tokens": 1218604,
    "thinking_tokens": 0,    "output_tokens":  41715,
    "cost_usd": 1.463368,    "cost_source": "table"
  }
},
"pricing_last_updated": "2026-06-20",
"litellm_version": "1.83.7"
```

**Where cost comes from** (`core/pricing.py`) — first source that can answer wins,
and `cost_source` records which one did:

| `cost_source` | What | Notes |
|---|---|---|
| `reported` | the provider's own charge, summed per call | OpenRouter returns `usage.cost` on every response and we persist the full `raw_response`, so this is **exact**, not an estimate — it accounts for the upstream provider that served the request, BYOK, and `:free` tiers. Used only when *every* call for that model reported a cost. |
| `table` | the hand-maintained `PRICING` dict | public list prices as of `PRICING_LAST_UPDATED`. Edit it to pin a price you disagree with — it outranks litellm. |
| `litellm` | litellm's bundled price map — **`openrouter/*` models only** | OpenRouter spans hundreds of vendor models that would otherwise need hand-entry. Scoped deliberately: litellm's map **moves when the package is upgraded**, so direct-API costs stay on the hand-verified `PRICING` table and can't shift underneath you. `litellm_version` is recorded for the runs this does apply to. Widen `_LITELLM_SCOPE_PREFIX` in `core/pricing.py` to cover more providers. |
| `null` | nobody knew | plus a one-line stderr warning, deduped per session. |

For `table` / `litellm`, the arithmetic is:

```
cost_usd = (input_tokens − cached_tokens) × input_price
         + cached_tokens                  × cached_input_price
         + output_tokens                  × output_price
         (÷ 1,000,000)
```

`thinking_tokens` is **not** added — it's already part of `output_tokens` (billed
at the output rate). It's listed separately for analysis only.

**Modular by design.** Adding a new benchmark requires zero pricing changes: as
long as `verify` / `feedback` / `run_attempt` populates the four token fields
(LLM judges/critics already do via `llm.complete`; agentic benchmarks do via
`result.details.actor_token_usage`), cost aggregation works automatically.

**Adding a new model:** for `openrouter/*`, nothing to do — those runs report their
own exact cost. For a **direct-provider** model, add an entry to `PRICING` and bump
`PRICING_LAST_UPDATED`; until you do, it gets `cost_usd: null` plus a one-line
warning. That's deliberate — direct-API prices are few and stay hand-verified rather
than tracking whatever litellm ships.

## S3 sync (default ON)

Every `python -m core run` uploads the finished run dir to
`s3://seq-k/<slice>/<metric>/…/` at the end. **`config.json` is excluded** (it's
a local-only artifact). To upload, you need AWS credentials and write access to
the bucket.

### First-time AWS setup

```bash
# 1. Install the AWS CLI v2 (one-time, per machine).
brew install awscli                                # macOS

# 2. Authenticate.
aws login                                          # browser-based IAM/Identity Center sign-in (CLI v2.30+)
# OR  aws sso login                                # if your org gave you an SSO start URL + you've run `aws configure sso` once
# OR  aws configure                                # if you have static IAM access keys

# 3. Verify you're signed in.
aws sts get-caller-identity                        # should print your Account + ARN

# 4. Verify bucket access.
aws s3 ls s3://seq-k/                              # should list without error
echo hi > /tmp/seqk-test.txt
aws s3 cp /tmp/seqk-test.txt s3://seq-k/_check.txt && aws s3 rm s3://seq-k/_check.txt
rm /tmp/seqk-test.txt
```

If step 4 errors `AccessDenied`, your IAM principal needs the policy in
[Collaborators](#collaborators).

### `aws login` vs `aws sso login` vs `aws configure`

| Use this | When |
|---|---|
| `aws login` | Quick browser-based IAM sign-in (CLI v2.30+). No prior `aws configure sso` needed. What we use in this repo. |
| `aws sso login` | Your org uses AWS IAM Identity Center; you've already run `aws configure sso` once with the start URL. |
| `aws configure` | You have static IAM access keys. No browser needed but keys need rotation. |

Re-auth when your session expires (usually every 8-12 hours). The harness runs
a pre-flight `aws sts get-caller-identity` at the start of each `run` and a
post-sync `aws s3 ls` verification at the end — so an expired session fails
loud instead of silently dropping uploads.

## Add a variation

Drop a YAML in `benchmarks/<name>/variants/`. Pick `metric`, `feedback_mode`,
`model`, and any benchmark-specific knobs under `options:`. The run path is
derived automatically; you don't write `out:`.

## Add a benchmark

Create `benchmarks/<name>/` exposing these via `__init__.py`:

```python
# Module-level path/role declarations
VERIFIER: str = "llm"                       # "llm" | "deterministic" | "harbor" | …
LLM_CRITIC_MODES: set[str] = {"socratic"}   # feedback modes that invoke an LLM critic (rest are template-only)

def slice_name(options: dict) -> str: ...   # path slot — disambiguates dataset variants
def load_tasks(**options) -> list[Task]:    # Task.canonical_index is 1-based; stable across configs
    ...
def verify(task, attempt, *, judge_model) -> VerifierResult: ...
def feedback(task, attempt, result, mode, *, critic_model) -> str: ...
```

For agentic benchmarks where each attempt runs in an external environment
(Docker, etc.), implement `run_attempt(task, history, t, k, *, seq, model,
judge_model, critic_model, temperature, options, out, prior) -> (prompt, output, result)`
instead — the harness uses it in place of `llm.complete + verify`. `prior` is
the list of raw saved attempt dicts so you can include the full prior
trajectories + verifier outputs in the next attempt's prompt.

Two optional hooks, only relevant if you want `summarize: true` to work well:

```python
# What the summarizer should compress, if the next prompt carries something other
# than actor.output (TerminalBench carries the full trajectory). Default: output.
def summarizer_source(output, result) -> str: ...
# ...and when building your retry context, prefer the summary when it exists:
#   summary = (a.get("summarizer") or {}).get("summary")
```

Add a `variants/` folder.
