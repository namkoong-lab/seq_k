-- The whole schema. ONE file, fully idempotent — re-running it is a no-op.
--
-- There is no migration table and no migration history, because the database is
-- a MIRROR: every row is derived from runs/*/manifest.json and the attempt JSON
-- (core/rows.py). For a change SQL cannot express in place, drop and rebuild:
--
--     python scripts/db_sync.py --reset      # drop + recreate + reload from disk
--
-- SIX TABLES: two dimensions, two junctions, one artifact, one ledger.
--
--   runs          what an experiment IS            (identity)
--   tasks         what a problem IS                (one row per slice x task)
--   run_tasks     which tasks a run covers         (junction, M:N)
--   attempts      a GENERATION, as an artifact     (the actor's output)
--   run_attempts  which run claims an attempt,     (junction, M:N)
--                 in what position, and how ITS judge graded it
--   llm_calls     what was spent                   (tokens / cost)
--
-- WHY attempts are not owned by a run. The actor's output for a given
-- generation config is reusable: re-judging a pass@k run with a different judge
-- does not change what the actor wrote, so those attempts should be claimed
-- again rather than paid for again. `attempts.actor_fingerprint` is what makes
-- two generations interchangeable.
--
-- WHY solved/score sit on the junction and not on the attempt. They are JUDGE
-- output. The same actor text graded by two judges has two verdicts, so a
-- verdict belongs to the (run, attempt) pair — never to the artifact.
--
-- Nothing else earns a table. Phase and cost-source are plain text written by a
-- single function (core/rows.py), so a lookup table would add a join and a
-- second thing to keep in sync to protect against a typo that one writer cannot
-- make. `SELECT DISTINCT phase FROM llm_calls` documents the vocabulary from
-- the data itself, and adding a phase costs nothing at all.
--
-- NAMES: `solved` is success. `has_all_attempts` is exhaustion — a seq@k task
-- that won on attempt 2 of 10 is both, because early stop IS completion.
-- Counts are `n_*`. Nothing is called `success` at two grains.

CREATE TABLE IF NOT EXISTS runs (
  run_id              uuid PRIMARY KEY,
  fingerprint         text NOT NULL,
  fingerprint_version smallint NOT NULL DEFAULT 2,
  created_at          timestamptz NOT NULL,
  finished_at         timestamptz,
  status              text NOT NULL
    CHECK (status IN ('running','complete','partial','empty','failed','unknown')),
  storage_key         text NOT NULL UNIQUE,      -- <benchmark>/<run_id>; the S3 prefix

  -- identity: one typed column per field of core.ids.IDENTITY_FIELDS
  benchmark           text NOT NULL,
  slice_key           text NOT NULL,
  metric              text NOT NULL CHECK (metric IN ('pass@k','seq@k')),
  -- The attempt budget this run has been run to. NOT necessarily identity: when
  -- the prompt never mentions the horizon, seq@5 is the 5-attempt prefix of
  -- seq@10, so both resolve to ONE run and this column grows as it is extended.
  k                   smallint NOT NULL,
  model               text NOT NULL,             -- THE MODEL UNDER TEST
  judge_model         text NOT NULL,             -- grader, or 'harbor'/'deterministic'
  critic_model        text,                      -- NULL = feedback mode uses no LLM critic
  summarizer_model    text,                      -- NULL = context is not a summary mode
  feedback_mode       text NOT NULL,
  context             text NOT NULL,             -- WHAT on retry: na | full | summary
  prompt_variant      text NOT NULL,             -- HOW it is worded (core/prompts.py)
  temperature         real NOT NULL,
  seed                int,                       -- NULL = unseeded
  reasoning_effort    text,
  output_budget       int,

  -- provenance; outside the fingerprint, so these can change without forking
  options             jsonb NOT NULL DEFAULT '{}'::jsonb,   -- full benchmark options
  code                jsonb NOT NULL DEFAULT '{}'::jsonb    -- git commit, lib versions
);
-- Deliberately ABSENT, and not to be added back:
--
--   label_path  the readable name is a pure function of the columns above, via
--               core.results.label_from_fields. Storing it would let it DRIFT.
--   s3_uri      always s3://$SEQK_S3_BUCKET/<storage_key>/.
--   notes       a column with no writer is a promise the schema cannot keep.
--
-- EXACTLY TWO fields describe the prompt, and they do not overlap:
--   context         WHAT the agent is given on retry: na | full | summary
--   prompt_variant  HOW it is worded — which template (core/prompts.py)

CREATE INDEX IF NOT EXISTS runs_slice_metric_idx ON runs (slice_key, metric, k);
CREATE INDEX IF NOT EXISTS runs_model_idx        ON runs (model);
CREATE INDEX IF NOT EXISTS runs_created_idx      ON runs (created_at DESC);
-- ONE RUN PER IDENTITY, full stop. Re-running a config RESUMES its existing run
-- rather than making a second one; an independent replicate of the same config is
-- a different `seed`, which is part of the fingerprint and so hashes differently.
-- There is deliberately no way to hold two runs at one identity: an unseeded
-- rerun of an existing config collides here and is rejected, which is the signal
-- to give it a seed.
CREATE UNIQUE INDEX IF NOT EXISTS runs_fingerprint_idx
  ON runs (fingerprint, fingerprint_version);

-- A problem, once. Previously `task_id` was repeated in every run that touched
-- it: 1,646 rows for 120 tasks. The prompt now has a home too.
CREATE TABLE IF NOT EXISTS tasks (
  task_uid    bigserial PRIMARY KEY,
  slice_key   text NOT NULL,        -- the dataset variant it belongs to
  task_index  int  NOT NULL,        -- canonical index WITHIN that slice (task-N/ on disk)
  task_id     text NOT NULL,        -- the benchmark's own id
  prompt      text,
  meta        jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (slice_key, task_id),
  UNIQUE (slice_key, task_index),
  UNIQUE (task_uid, task_index)     -- lets run_tasks pin both with one FK
);

-- Junction: which tasks a run covers.
CREATE TABLE IF NOT EXISTS run_tasks (
  run_id      uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  task_uid    bigint NOT NULL REFERENCES tasks(task_uid),
  extra       jsonb NOT NULL DEFAULT '{}'::jsonb,   -- benchmark-specific metrics
  PRIMARY KEY (run_id, task_uid)
);
CREATE INDEX IF NOT EXISTS run_tasks_task_idx ON run_tasks (task_uid);

-- THE ARTIFACT: one actor generation. Not owned by a run.
--
-- `actor_fingerprint` is what makes two generations interchangeable — a hash of
-- everything that shaped the actor call. For pass@k that excludes the judge
-- entirely (attempts are independent draws). For seq@k attempt 1 it also
-- excludes the judge, since nothing has been graded yet. For seq@k attempt 2+
-- it INCLUDES judge_model/critic_model/feedback_mode, because the prompt
-- carried judge-derived feedback: swap the judge and that attempt is no longer
-- a sample from the same distribution. See core.ids.actor_fingerprint.
--
-- IDENTITY IS THE GENERATION, NOT THE CONFIG. An artifact is unique per
-- (generating run, task, draw) — never per fingerprint. Generation is
-- stochastic: at temperature 0.7 two runs with an identical actor config
-- produce DIFFERENT text, so they are two artifacts, not one. Keying on the
-- fingerprint merged them and silently destroyed 183 real generations.
--
-- `actor_fingerprint` is therefore a LOOKUP KEY, not an identity: "find me an
-- existing generation I am allowed to claim instead of drawing my own". Reuse
-- is a deliberate claim (a SELECT, then a run_attempts row), never an automatic
-- merge on insert.
CREATE TABLE IF NOT EXISTS attempts (
  attempt_id        bigserial PRIMARY KEY,
  task_uid          bigint NOT NULL REFERENCES tasks(task_uid),
  actor_fingerprint text NOT NULL,     -- what this generation is interchangeable WITH
  attempt_index     int NOT NULL,      -- attempt number in the run that GENERATED it
                                       -- (1-based). Equal to run_attempts.attempt_index
                                       -- for a self-generated attempt; the two differ only if
                                       -- a claiming run places it at another position.
  generated_by_run  uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  output_key        text,              -- <storage_key>/task-N/attempt-M.json, in S3 and on disk
  finish_reason     text,
  created_at        timestamptz,
  UNIQUE (generated_by_run, task_uid, attempt_index)
);
CREATE INDEX IF NOT EXISTS attempts_reuse_idx ON attempts (actor_fingerprint, task_uid);
CREATE INDEX IF NOT EXISTS attempts_run_idx   ON attempts (generated_by_run);

-- Junction: which run claims an attempt, where it sits in that run's sequence,
-- and how THAT RUN's judge graded it.
CREATE TABLE IF NOT EXISTS run_attempts (
  run_id        uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  attempt_id    bigint NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
  task_uid      bigint NOT NULL REFERENCES tasks(task_uid),
  attempt_index int NOT NULL,          -- position in THIS run (1-based)
  solved        boolean,               -- this run's judge's verdict
  score         real,
  extra         jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (run_id, attempt_id),
  UNIQUE (run_id, task_uid, attempt_index),
  FOREIGN KEY (run_id, task_uid) REFERENCES run_tasks (run_id, task_uid) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS run_attempts_attempt_idx ON run_attempts (attempt_id);

-- THE ONLY PLACE TOKENS AND COST ARE STORED. ~22 rows per attempt, because
-- ResearchRubrics and HealthBench judge PER RUBRIC CRITERION.
--
-- Keyed by RUN as well as attempt: cost is what a run actually spent. A run
-- that reuses an existing attempt makes no actor call, so it has no actor row
-- here — which is exactly right, it did not pay for one.
CREATE TABLE IF NOT EXISTS llm_calls (
  call_id          bigserial PRIMARY KEY,
  run_id           uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  attempt_id       bigint NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
  phase            text NOT NULL,        -- actor | judge | critic | summarizer
  call_index       int NOT NULL,         -- ordinal WITHIN (run, attempt, phase)
  model            text NOT NULL,
  input_tokens     int NOT NULL DEFAULT 0,
  cached_tokens    int NOT NULL DEFAULT 0,
  thinking_tokens  int NOT NULL DEFAULT 0,
  output_tokens    int NOT NULL DEFAULT 0,
  cost_usd         double precision,
  cost_source      text,                 -- reported | rates | unknown
  UNIQUE (run_id, attempt_id, phase, call_index)
);
CREATE INDEX IF NOT EXISTS llm_calls_attempt_idx ON llm_calls (attempt_id);
CREATE INDEX IF NOT EXISTS llm_calls_phase_idx   ON llm_calls (run_id, phase);

-- ---------------------------------------------------------------------------
-- Views. Every derived number lives here and nowhere else.
--
-- FAN-OUT WARNING: a run has several one-to-many children. Joining two of them
-- directly multiplies rows — measured on the real corpus, the naive join
-- reported $13,849.00 against a true $483.77, 28.6x too high. Each child is
-- aggregated in its OWN subquery and only then joined. Keep it so.
-- ---------------------------------------------------------------------------

-- One row per (run, attempt): the artifact, this run's verdict on it, and what
-- this run spent on it.
CREATE OR REPLACE VIEW attempt_summary AS
SELECT ra.run_id, t.task_index, ra.task_uid, ra.attempt_index, ra.attempt_id,
       ra.solved, ra.score,
       a.actor_fingerprint, a.generated_by_run, a.output_key, a.finish_reason,
       a.created_at,
       (a.generated_by_run IS DISTINCT FROM ra.run_id) AS reused,
       coalesce(c.n_calls, 0)         AS n_calls,
       coalesce(c.input_tokens, 0)    AS input_tokens,
       coalesce(c.cached_tokens, 0)   AS cached_tokens,
       coalesce(c.thinking_tokens, 0) AS thinking_tokens,
       coalesce(c.output_tokens, 0)   AS output_tokens,
       coalesce(c.cost_usd, 0)        AS cost_usd
FROM run_attempts ra
JOIN attempts a ON a.attempt_id = ra.attempt_id
JOIN tasks t ON t.task_uid = ra.task_uid
LEFT JOIN (
  SELECT run_id, attempt_id, count(*) AS n_calls,
         sum(input_tokens) AS input_tokens, sum(cached_tokens) AS cached_tokens,
         sum(thinking_tokens) AS thinking_tokens, sum(output_tokens) AS output_tokens,
         sum(cost_usd) AS cost_usd
  FROM llm_calls GROUP BY run_id, attempt_id
) c ON c.run_id = ra.run_id AND c.attempt_id = ra.attempt_id;

CREATE OR REPLACE VIEW task_summary AS
SELECT rt.run_id, t.task_index, rt.task_uid, t.task_id, rt.extra,
       coalesce(x.n_attempts, 0) AS n_attempts,
       coalesce(x.solved, false) AS solved,
       x.best_score,
       x.first_solved_attempt,
       -- seq@k stops early on success, so a solved task IS exhausted;
       -- pass@k always draws the full k.
       (CASE WHEN r.metric = 'seq@k'
             THEN coalesce(x.solved, false) OR coalesce(x.n_attempts, 0) >= r.k
             ELSE coalesce(x.n_attempts, 0) >= r.k END
        -- An output budget is a second, equally legitimate stopping rule. A run
        -- told to spend at most N actor tokens and stop has FINISHED when it
        -- spends them, even at attempt 2 of 5 — counting that as "partial"
        -- reports a correctly-completed run as a failure.
        OR (r.output_budget IS NOT NULL
            AND coalesce(b.actor_out, 0) >= r.output_budget)) AS has_all_attempts,
       coalesce(c.cost_usd, 0) AS cost_usd,
       coalesce(c.n_calls, 0)  AS n_calls
FROM run_tasks rt
JOIN runs r ON r.run_id = rt.run_id
JOIN tasks t ON t.task_uid = rt.task_uid
LEFT JOIN (
  SELECT run_id, task_uid, count(*) AS n_attempts, bool_or(solved) AS solved,
         max(score) AS best_score,
         min(attempt_index) FILTER (WHERE solved) AS first_solved_attempt
  FROM run_attempts GROUP BY run_id, task_uid
) x ON x.run_id = rt.run_id AND x.task_uid = rt.task_uid
LEFT JOIN (
  SELECT ra.run_id, ra.task_uid, sum(l.cost_usd) AS cost_usd, count(*) AS n_calls
  FROM llm_calls l
  JOIN run_attempts ra ON ra.run_id = l.run_id AND ra.attempt_id = l.attempt_id
  GROUP BY ra.run_id, ra.task_uid
) c ON c.run_id = rt.run_id AND c.task_uid = rt.task_uid
-- Actor output tokens per task, for the output_budget stopping rule above.
-- Same (run_id, task_uid) grain as the joins around it, so it cannot fan out.
LEFT JOIN (
  SELECT ra.run_id, ra.task_uid, sum(l.output_tokens) AS actor_out
  FROM llm_calls l
  JOIN run_attempts ra ON ra.run_id = l.run_id AND ra.attempt_id = l.attempt_id
  WHERE l.phase = 'actor'
  GROUP BY ra.run_id, ra.task_uid
) b ON b.run_id = rt.run_id AND b.task_uid = rt.task_uid;

CREATE OR REPLACE VIEW run_summary AS
SELECT r.*,
       coalesce(t.n_tasks, 0)         AS n_tasks,
       coalesce(t.n_solved, 0)        AS n_solved,
       coalesce(t.n_tasks_done, 0)    AS n_tasks_done,
       coalesce(t.n_tasks_partial, 0) AS n_tasks_partial,
       coalesce(t.n_attempts, 0)      AS n_attempts,
       coalesce(t.max_attempt, 0)     AS max_attempt,
       coalesce(u.n_reused, 0)        AS n_attempts_reused,
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
  SELECT ra.run_id, count(*) AS n_reused
  FROM run_attempts ra JOIN attempts a USING (attempt_id)
  WHERE a.generated_by_run IS DISTINCT FROM ra.run_id
  GROUP BY ra.run_id
) u USING (run_id)
LEFT JOIN (
  SELECT run_id, sum(cost_usd) AS cost_usd, count(*) AS n_calls,
         sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens
  FROM llm_calls GROUP BY run_id
) c USING (run_id);

-- The agent / judge / critic / summarizer split the papers report. A new phase
-- appears here automatically; there is no vocabulary to register it in.
CREATE OR REPLACE VIEW run_phase_costs AS
SELECT c.run_id, c.phase, count(*) AS n_calls,
       sum(c.input_tokens) AS input_tokens, sum(c.output_tokens) AS output_tokens,
       sum(c.cost_usd) AS cost_usd
FROM llm_calls c GROUP BY c.run_id, c.phase;
