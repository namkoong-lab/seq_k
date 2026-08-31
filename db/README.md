# The runs database

Postgres (Neon) mirrors what is on disk so that "which ablation cells are
missing" and "what did the judge cost" are one query instead of a walk over
4.7 GB of JSON.

**It is never required.** The harness runs, resumes, and records results with no
database at all — local manifests are the source of truth. If Neon is
unreachable the run continues, the failed write spools to
`<run>/.db_pending.jsonl`, and `db_sync.py --replay` catches up later. Nothing is
stored only in Postgres.

```
attempt-N.json  ──core/rows.py──▶  run_tasks · attempts · llm_calls  ──views──▶  totals
   (truth)                                   (mirror)                          (derived)
```

---

## Shape

Four tables. Each answers one question at one grain; nothing appears in two of
them.

| table | grain | rows today | answers | size |
|---|---|---:|---|---:|
| `runs` | one experiment | 62 | *what is this experiment?* | 184 kB |
| `run_tasks` | run × task | 1,646 | *which tasks does it cover?* | 296 kB |
| `attempts` | run × task × attempt | 8,435 | *how did each attempt turn out?* | 1.3 MB |
| `llm_calls` | one API call | 187,588 | *what was spent?* | 54 MB |

**56 MB total**, 96% of it `llm_calls` — ResearchRubrics and HealthBench judge
*per rubric criterion*, so one attempt fires ~27 and ~15 judge calls
respectively. Against the 4.7 GB of JSON it describes, the whole database is
about 1%. Neon's free tier (0.5 GB) holds roughly 8× the current corpus; growth
is ~1 MB per ResearchRubrics run.

Plus four views: `run_summary`, `task_summary`, `attempt_summary`,
`run_phase_costs`.

---

## The schema

Verbatim from `db/schema.sql`, which is the authority.

### `runs` — what an experiment IS

Three groups of columns: where it lives, the 15 identity fields (one typed
column each, exactly `core.ids.IDENTITY_FIELDS`), and provenance that sits
*outside* the fingerprint and so can change without forking a run.

```sql
CREATE TABLE runs (
  run_id              uuid PRIMARY KEY,
  fingerprint         text NOT NULL,
  fingerprint_version smallint NOT NULL DEFAULT 2,
  created_at          timestamptz NOT NULL,
  finished_at         timestamptz,
  status              text NOT NULL
    CHECK (status IN ('running','complete','partial','empty','failed','unknown')),
  storage_key         text NOT NULL UNIQUE,   -- <benchmark>/<run_id>; the S3 prefix

  -- identity ---------------------------------------------------------------
  benchmark           text NOT NULL,          -- benchmarks.researchrubrics
  slice_key           text NOT NULL,          -- dataset variant: researchrubrics
  metric              text NOT NULL CHECK (metric IN ('pass@k','seq@k')),
  k                   smallint NOT NULL,      -- attempt budget reached; see note below
  model               text NOT NULL,          -- THE MODEL UNDER TEST
  judge_model         text NOT NULL,          -- grader, or 'harbor'/'deterministic'
  critic_model        text,                   -- NULL = feedback mode uses no LLM critic
  summarizer_model    text,                   -- NULL = context is not a summary mode
  feedback_mode       text NOT NULL,
  context             text NOT NULL,          -- WHAT on retry: na | full | summary
  prompt_variant      text NOT NULL,             -- HOW it is worded (core/prompts.py)
  temperature         real NOT NULL,
  seed                int,                    -- NULL = unseeded
  reasoning_effort    text,
  output_budget       int,

  -- provenance, outside the fingerprint -------------------------------------
  options             jsonb NOT NULL DEFAULT '{}'::jsonb,   -- full benchmark options
  code                jsonb NOT NULL DEFAULT '{}'::jsonb    -- git commit, lib versions
);

CREATE INDEX runs_slice_metric_idx ON runs (slice_key, metric, k);
CREATE INDEX runs_model_idx        ON runs (model);
CREATE INDEX runs_created_idx      ON runs (created_at DESC);
-- ONE RUN PER IDENTITY, full stop. A second run of the same config is a
-- different `seed`, which is part of the fingerprint and so hashes differently.
CREATE UNIQUE INDEX runs_fingerprint_idx
  ON runs (fingerprint, fingerprint_version);
```

Nullability carries meaning. `critic_model` is null when the feedback mode uses
no LLM critic; `seed` is null when unseeded; `summarizer_model` is null outside
summary contexts. Those are facts about the run, not missing data.

**Three columns are deliberately absent**, and should not be added back:

- **`label_path`** — the readable name is a pure function of the columns above
  (`core.results.label_from_fields`), verified reconstructible for all 62 runs.
  Storing it would let it *drift*: a horizon-free run extended from k=5 to k=10
  keeps its old label while `k` grows. Compute it; don't cache it.
- **`s3_uri`** — always `s3://$SEQK_S3_BUCKET/<storage_key>/`.
- **`notes`** — it had no writer anywhere in the codebase. A column with no
  writer is a promise the schema cannot keep; use `code` or `options`.
- **`prompt_digest`** — a hash of the rendered prompt, to detect an un-bumped
  `prompt_variant`. Removed: it is not an experiment axis, it never populated a
  row, and it made *three* overlapping prompt concepts where two suffice.

**Exactly two fields describe the prompt, and they do not overlap:**

| field | question | example |
|---|---|---|
| `context` | which *treatment* — an axis you deliberately vary | `full`, `full-nohorizon`, `summary` |
| `prompt_variant` | which *wording* — the template version | `v1`, `legacy` |

Bump `prompt_variant` when you edit a template. That is the whole discipline;
nothing else in the schema is about prompts.

`code` is jsonb rather than columns because its contents are not fixed: new runs
record `git_commit`, `git_dirty`, `litellm`, `pricing_last_updated`, while the 62
backfilled runs carry only `{"migrated_from": "v2-label-path"}` — they predate
provenance capture, and claiming a commit we cannot verify would be worse than
admitting we don't know.

`k` is **not** always identity — see [Identity](#identity-what-makes-two-runs-the-same-run).
It is the attempt budget the run has been run *to*, and it grows when a
horizon-free run is extended.

### `run_tasks` — which tasks a run covers

```sql
CREATE TABLE run_tasks (
  run_id      uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  task_index  int NOT NULL,        -- canonical index within the slice
  task_id     text NOT NULL,       -- the benchmark's own id
  extra       jsonb NOT NULL DEFAULT '{}'::jsonb,   -- benchmark-specific metrics
  PRIMARY KEY (run_id, task_index)
);
```

Four columns on purpose. `task_index → task_id` is the one thing about a task
that cannot be derived; whether it was solved, in how many attempts, at what
cost all live in `task_summary`.

### `attempts` — how each attempt turned out

```sql
CREATE TABLE attempts (
  run_id         uuid NOT NULL,
  task_index     int NOT NULL,
  attempt_index  int NOT NULL,
  solved         boolean,
  score          real,
  finish_reason  text,
  created_at     timestamptz,
  extra          jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (run_id, task_index, attempt_index),
  FOREIGN KEY (run_id, task_index)
    REFERENCES run_tasks (run_id, task_index) ON DELETE CASCADE
);

CREATE INDEX attempts_run_idx ON attempts (run_id);
```

Outcome only. No tokens, no cost — `attempt_summary` sums those from
`llm_calls`.

### `llm_calls` — what was spent

```sql
CREATE TABLE llm_calls (
  call_id          bigserial PRIMARY KEY,
  run_id           uuid NOT NULL,
  task_index       int NOT NULL,
  attempt_index    int NOT NULL,
  phase            text NOT NULL,        -- actor | judge | critic | summarizer
  call_index       int NOT NULL,         -- ordinal WITHIN (attempt, phase)
  model            text NOT NULL,
  input_tokens     int NOT NULL DEFAULT 0,
  cached_tokens    int NOT NULL DEFAULT 0,
  thinking_tokens  int NOT NULL DEFAULT 0,
  output_tokens    int NOT NULL DEFAULT 0,
  cost_usd         double precision,
  cost_source      text,                 -- reported | rates | unknown
  UNIQUE (run_id, task_index, attempt_index, phase, call_index),
  FOREIGN KEY (run_id, task_index, attempt_index)
    REFERENCES attempts (run_id, task_index, attempt_index) ON DELETE CASCADE
);

CREATE INDEX llm_calls_attempt_idx ON llm_calls (run_id, task_index, attempt_index);
CREATE INDEX llm_calls_phase_idx   ON llm_calls (run_id, phase);
```

`(run_id, task_index, attempt_index, phase, call_index)` is the natural key, but
it is 36 bytes and its btree would cost more than the heap it indexes — hence
the `bigserial` surrogate PK with the natural key demoted to a UNIQUE
constraint, which is still what makes upserts idempotent.

`cost_source` records where the number came from: `reported` is the provider's
own charge (OpenRouter returns `usage.cost`), `rates` is computed from
`core/pricing.py`, `unknown` means neither was available.

### The views

Every derived number lives here and nowhere else.

```sql
CREATE VIEW attempt_summary AS
SELECT a.run_id, a.task_index, a.attempt_index, a.solved, a.score,
       a.finish_reason, a.created_at,
       coalesce(c.n_calls, 0)         AS n_calls,
       coalesce(c.input_tokens, 0)    AS input_tokens,
       coalesce(c.cached_tokens, 0)   AS cached_tokens,
       coalesce(c.thinking_tokens, 0) AS thinking_tokens,
       coalesce(c.output_tokens, 0)   AS output_tokens,
       coalesce(c.cost_usd, 0)        AS cost_usd
FROM attempts a
LEFT JOIN (
  SELECT run_id, task_index, attempt_index, count(*) AS n_calls,
         sum(input_tokens) AS input_tokens, sum(cached_tokens) AS cached_tokens,
         sum(thinking_tokens) AS thinking_tokens, sum(output_tokens) AS output_tokens,
         sum(cost_usd) AS cost_usd
  FROM llm_calls GROUP BY run_id, task_index, attempt_index
) c USING (run_id, task_index, attempt_index);

CREATE VIEW task_summary AS
SELECT t.run_id, t.task_index, t.task_id, t.extra,
       coalesce(a.n_attempts, 0)  AS n_attempts,
       coalesce(a.solved, false)  AS solved,
       a.best_score,
       a.first_solved_attempt,
       -- seq@k stops early on success, so a solved task IS exhausted;
       -- pass@k always draws the full k.
       CASE WHEN r.metric = 'seq@k'
            THEN coalesce(a.solved, false) OR coalesce(a.n_attempts, 0) >= r.k
            ELSE coalesce(a.n_attempts, 0) >= r.k END AS has_all_attempts,
       coalesce(c.cost_usd, 0)    AS cost_usd,
       coalesce(c.n_calls, 0)     AS n_calls
FROM run_tasks t
JOIN runs r USING (run_id)
LEFT JOIN (
  SELECT run_id, task_index, count(*) AS n_attempts,
         bool_or(solved) AS solved, max(score) AS best_score,
         min(attempt_index) FILTER (WHERE solved) AS first_solved_attempt
  FROM attempts GROUP BY run_id, task_index
) a USING (run_id, task_index)
LEFT JOIN (
  SELECT run_id, task_index, sum(cost_usd) AS cost_usd, count(*) AS n_calls
  FROM llm_calls GROUP BY run_id, task_index
) c USING (run_id, task_index);

CREATE VIEW run_summary AS
SELECT r.*,
       coalesce(t.n_tasks, 0)         AS n_tasks,
       coalesce(t.n_solved, 0)        AS n_solved,
       coalesce(t.n_tasks_done, 0)    AS n_tasks_done,
       coalesce(t.n_tasks_partial, 0) AS n_tasks_partial,
       coalesce(t.n_attempts, 0)      AS n_attempts,
       coalesce(t.max_attempt, 0)     AS max_attempt,
       coalesce(c.n_calls, 0)         AS n_calls,
       coalesce(c.input_tokens, 0)    AS input_tokens,
       coalesce(c.output_tokens, 0)   AS output_tokens,
       coalesce(c.cost_usd, 0)        AS cost_usd
FROM runs r
LEFT JOIN (
  SELECT run_id, count(*) AS n_tasks,
         count(*) FILTER (WHERE solved)               AS n_solved,
         count(*) FILTER (WHERE has_all_attempts)     AS n_tasks_done,
         count(*) FILTER (WHERE NOT has_all_attempts) AS n_tasks_partial,
         sum(n_attempts)                              AS n_attempts,
         max(n_attempts)                              AS max_attempt
  FROM task_summary GROUP BY run_id
) t USING (run_id)
LEFT JOIN (
  SELECT run_id, sum(cost_usd) AS cost_usd, count(*) AS n_calls,
         sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens
  FROM llm_calls GROUP BY run_id
) c USING (run_id);

CREATE VIEW run_phase_costs AS
SELECT c.run_id, c.phase, count(*) AS n_calls,
       sum(c.input_tokens) AS input_tokens, sum(c.output_tokens) AS output_tokens,
       sum(c.cost_usd) AS cost_usd
FROM llm_calls c GROUP BY c.run_id, c.phase;
```

---

## A run, end to end

One real run (`researchrubrics`, seq@k, k=10, gpt-5.2, summarised) through all
four tables:

```
runs         1 row
             run_id d4302c9f-…  storage_key researchrubrics/d4302c9f-…
             metric seq@k  k 10  context summary  temperature 0.7  seed (null)

run_tasks    30 rows
             task_index 1 → task_id 6847465956a0f6376a605427   extra {}

attempts     202 rows          (seq@k stops early, so not 30 × 10)
             task 1 attempt 1  solved f  score 0.581  finish_reason stop
             task 1 attempt 2  solved t  score 1.000  finish_reason stop

llm_calls    5,618 rows
             task 1 attempt 1  actor      ×1    161 in    3,731 out   $0.0525
             task 1 attempt 1  judge      ×21   ~3.5k in each          (21 rubric criteria)
             task 1 attempt 1  summarizer ×1  3,677 in      406 out   $0.0121
             task 1 attempt 2  actor      ×1
             task 1 attempt 2  judge      ×21                          (no summarizer: it solved)
```

Rolled back up by the views:

```
task_summary  task 1 | solved t | n_attempts 2 | first_solved_attempt 2
                     | has_all_attempts t          ← solved on 2 of 10: early stop IS completion
                     | cost 0.222

run_summary   n_tasks 30 | n_solved 17 | n_tasks_done 30 | n_attempts 202
              n_calls 5,618 | cost_usd 40.52
```

And the same run on disk, with identical keys in S3:

```
runs/researchrubrics/d4302c9f-2a0a-4050-8194-89f03ad804b9/
    manifest.json      identity, labels, provenance, rollup   ← source of truth
    config.json        frozen config (older readers)
    summary.json
    task-1/ … task-30/
        task_meta.json
        attempt-1.json     ← every DB row is derived from these
        attempt-2.json
        summary.json
```

`attempt-N.json` carries four sections — `actor`, `judge`, `critic`,
`summarizer` — each with `model`, token counts, and the full `raw_response`.
`core/rows.py` walks them: the `actor` section becomes one `llm_calls` row at
`call_index 0`, and each entry in a section's `calls` list becomes another.

**Prompts and transcripts never enter Postgres.** The full prompt, the model's
output, the judge's per-criterion reasoning all stay in the JSON. The database
holds identity, outcomes, and cost — the things you filter and aggregate on.

`manifest.json` denormalises `labels` and `rollup` on purpose: the bucket must
be self-describing, so `db_sync --rebuild` can reconstruct all 187,588 rows from
S3 alone.

---

## Three rules

**1. One source of truth per number.** `llm_calls` is the only place tokens and
cost are stored. Every total is a view.

No table has a cached `cost_usd`, `n_attempts`, or `solved`. This is not
fastidiousness: the harness previously kept two implementations of "total cost"
and they drifted, understating the corpus by **$68** for months
(`index_runs.py` summed only provider-*reported* cost, so every
direct-Anthropic run counted as $0). One definition cannot disagree with itself.

It also makes extension safe. Add a fifth phase and the totals include it the
moment rows exist — with cached columns you would have to find and update the
rollup code, and silence would look like success.

**2. Nothing is a code, everything is a word.** `phase`, `cost_source`, and
`model` are plain text. `WHERE phase = 'judge'` reads as what it is, and no
query needs a join to be legible. There are no lookup tables and no smallint
codes whose meaning lives in a Python dict. `SELECT DISTINCT phase FROM
llm_calls` documents the vocabulary from the data itself. The duplication costs
~23 MB at current volume — the deliberate price of never decoding a result.

**3. Names say what they mean.** `solved` is success. `has_all_attempts` is
exhaustion — a seq@k task that won on attempt 2 of 10 is **both**, because early
stop *is* completion. Counts are `n_*`. Nothing is called `success` at two
grains; nothing is called `attempts` that is not the table.

---

## Identity: what makes two runs the same run

A run's identity is `runs.fingerprint` — a SHA-256 over the 15 fields in
`core.ids.IDENTITY_FIELDS`. Same config → same fingerprint → same directory →
re-running resumes. Different on any field → a different run.

Three fields are normalised before hashing, so a run is never split by something
it did not depend on:

- `judge_model` collapses to the verifier name (`harbor`, `deterministic`) when
  the grader is not an LLM.
- `critic_model` is null unless the feedback mode actually invokes a critic.
- **`k` is identity only when it reaches the prompt.** Under a horizon-free
  context (`*-nohorizon`, `*-noframe`) and for every pass@k run, the prompt never
  mentions the horizon, so seq@5 is literally the 5-attempt prefix of seq@10.
  Both resolve to ONE run, and asking for a larger `k` extends it rather than
  forking.

`core.ids.k_affects_prompt` decides that last one by **rendering the real prompt
at two values of k and comparing**, not by hardcoding a rule about which
contexts embed a counter. A rule would be true of `build_prompt` today and
silently wrong the moment someone adds horizon-aware wording elsewhere. It is
conservative in the right direction: a false *True* costs one duplicate run, a
false *False* silently merges two experiments — so anything unprobeable (agentic
benchmarks build their prompt inside `run_attempt`) is assumed k-dependent
unless it exposes `prompt_probe(k=..., context=..., metric=...)`.

Note that the `k` probe stores **nothing**. It is not prompt metadata — it is a
rule about whether `k` belongs in the identity hash, and it exists to stop you
paying twice for the same attempts. The only prompt fields in the schema are
`context` and `prompt_variant`.

---

## Extending it

### Add a phase (a second critic, a verifier-of-verifier)

Nothing to register anywhere. Add the name to the phase loop in
`core/rows.py:call_rows_for_attempt` — the saved attempt JSON needs a matching
`{"calls": [...]}` section — and it appears in `run_phase_costs` and in every
total automatically, because cost is summed from `llm_calls` and cached nowhere.

### Add benchmark-specific metrics

`run_tasks.extra` and `attempts.extra` are jsonb, ignored by everything generic.
Put ResearchRubrics' per-criterion breakdown or HealthBench's safety flags there
rather than adding columns only one benchmark uses:

```sql
SELECT extra->>'unmet_safety_critical' FROM run_tasks WHERE run_id = '...';
```

### Add an ablation axis (a new identity field)

The expensive one, by design — identity fields are typed columns so the schema
documents the experiment space. Five steps, in order:

1. `core/ids.py` — add the field to `IDENTITY_FIELDS` and to `identity()`.
2. `core/ids.py` — bump `FINGERPRINT_VERSION`.
3. `db/schema.sql` — add the column to `runs`.
4. `core/db.py:upsert_run` — add it to the INSERT.
5. `scripts/runs.py` — add it to `FILTERS` (and `_COL` if the SQL column name
   differs from the CLI flag).

Then `python scripts/storage/db_sync.py --schema && python scripts/storage/db_sync.py --rebuild`.

**Step 2 deserves care.** The version is part of the hash, so a bump changes
*every* fingerprint — finished runs would stop matching and a re-run would redo
completed work. `core.ids.candidates()` emits the fingerprint at each version
newest-first and the registry adopts the first that already exists, which is
what prevents that. The fallback is only sound while a bump strictly **relaxes**
identity (removes a field that provably does not affect results). If your change
alters what a field *means*, an old run is not the same experiment — guard
against backwards adoption in `candidates()`.

If your field does **not** change results — a note, a tag, a cost centre — do
none of this. Put it in `runs.notes` or `runs.code`, which sit outside the
fingerprint precisely so they can change freely.

### Re-run a config

Re-running a config **resumes** it: finished tasks are skipped and only the
short one is redone. That is almost always what you want for a run that merely
died part-way, and it is far cheaper.

To record a SECOND, independent run of the same config, give it a different
`seed`. Seed is part of the fingerprint, so the two hash differently, both stay
live, and their draws can be pooled:

```bash
python -m core run benchmarks/<name>/variants/<x>.yaml    # seed: 1
python -m core run benchmarks/<name>/variants/<x>.seed2.yaml
```

Uniqueness on `(fingerprint, fingerprint_version)` is total — **one run per
identity**. An unseeded re-run of an existing config therefore collides and is
rejected; that rejection is the signal to give it a seed rather than a defect.

If you do not trust what a run produced — a misconfigured judge, a provider
returning garbage — remove it with `scripts/purge_runs.py`, which is
destructive and writes a tombstone to `purged.jsonl`.

### Add a benchmark

Nothing here changes. `benchmark` and `slice_key` are plain text; implement the
benchmark contract (`load_tasks` / `verify` / `feedback`, or `run_attempt` for
agentic ones) and its runs appear. If it is agentic and you want `k` probing to
work, expose `prompt_probe()`.

---

## Schema changes — there are no migrations

`db/schema.sql` is the entire schema and it is idempotent, so applying it is
always safe:

```bash
python scripts/storage/db_sync.py --schema     # add new tables / columns / views
python scripts/storage/db_sync.py --reset      # drop everything and reload from disk
```

There is no migration history because there is nothing to migrate: every row is
derived from `runs/*/manifest.json` and the attempt JSON. For a change SQL
cannot make in place — renaming a column, changing a type, restructuring a
table — edit `schema.sql` and `--reset`. A full reload of the current corpus
takes **about 15 seconds**, because writes are batched per task with
`executemany` rather than one statement per call.

That is what makes the four-table design affordable: you never have to write a
careful `ALTER` against data you cannot reproduce.

### Testing a schema change without touching Neon

There is no staging tier. Use a throwaway local cluster:

```bash
SOCK=$(mktemp -d /tmp/seqkpg.XXXX)      # short path: the socket has a 103-byte limit
PGD=/tmp/seqk-pgdata && rm -rf $PGD
initdb -D $PGD -U seqk --auth=trust
pg_ctl -D $PGD -o "-p 55432 -k $SOCK -c listen_addresses=''" -l /tmp/pg.log start
createdb -h $SOCK -p 55432 -U seqk seqk_test

export DATABASE_URL="postgresql://seqk@/seqk_test?host=$SOCK&port=55432"
python scripts/storage/db_sync.py --schema
python scripts/storage/db_sync.py --rebuild     # loads every run from disk (~15s)

pg_ctl -D $PGD stop                     # when done
```

Then check the views still agree with the manifests. The numbers on disk are the
reference, and this must match `python scripts/runs.py cost`:

```sql
SELECT round(sum(cost_usd)::numeric, 2) FROM run_summary;   -- 483.77
```

---

## The fan-out trap

`runs` has two one-to-many children (`run_tasks`, `llm_calls`). Joining both
directly multiplies rows and silently inflates every SUM:

```sql
-- WRONG: reports $13,849.00
SELECT sum(c.cost_usd) FROM runs r
  LEFT JOIN run_tasks t USING (run_id)
  LEFT JOIN llm_calls c USING (run_id);

-- RIGHT: $483.77 — aggregate each child in its own subquery, then join
```

Measured on the real corpus: **28.6× too high**. Every view above aggregates
each child separately for exactly this reason. If you add a view, do the same,
and check the total against `runs.py cost`.

---

## Queries worth keeping

```sql
-- the phase cost split the papers report
SELECT phase, n_calls, input_tokens, output_tokens, cost_usd
FROM run_phase_costs WHERE run_id = '...';

-- cost by phase across every seq@k run of one benchmark
SELECT c.phase, count(*), sum(c.cost_usd)
FROM llm_calls c JOIN runs r USING (run_id)
WHERE r.slice_key = 'researchrubrics' AND r.metric = 'seq@k'
GROUP BY c.phase ORDER BY 3 DESC;

-- between-seed variance for one config
SELECT seed, n_solved::float / nullif(n_tasks, 0) AS success_rate
FROM run_summary
WHERE slice_key = 'healthbench-default' AND model = 'openrouter/openai/gpt-5.2'
  AND seed IS NOT NULL ORDER BY seed;

-- cumulative-best curve: how many tasks were solved by attempt N
SELECT first_solved_attempt, count(*) FROM task_summary
WHERE run_id = '...' AND solved GROUP BY 1 ORDER BY 1;

-- runs that never finished
SELECT storage_key, label_path, n_tasks_done, n_tasks
FROM run_summary WHERE n_tasks_partial > 0 ORDER BY created_at DESC;
```

---

## Related

- `db/schema.sql` — the schema itself, commented. The authority.
- `scripts/storage/README.md` — day-to-day usage: finding runs, resume semantics,
  Neon setup, S3 layout.
- `core/ids.py` — fingerprint, storage key, the `k` probe.
- `core/rows.py` — the single JSON → rows implementation, shared by the live
  harness, the backfill, and `--rebuild`.
- `core/db.py` — connection, writes (all non-fatal), `apply_schema`, `reset`.
