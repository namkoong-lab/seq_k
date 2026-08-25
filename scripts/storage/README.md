# Runs: storing them, finding them, adding your own

Everything about getting runs into and out of S3, the local database and Neon.

    S3      the artifacts, and the source of truth
    Postgres  a derived index — rebuildable from the files at any time
    Neon      the shared copy of that index

    scripts/storage/db_sync.py         disk  -> local Postgres
    scripts/storage/sync_from_s3.py    S3    -> local disk + Postgres
    scripts/storage/push_to_neon.sh    local Postgres -> Neon
    scripts/storage/import_hf_configless.py   external bucket -> local

---

# Part 1 — Adding runs

## Run a benchmark

```bash
export AWS_PROFILE=seqk
python -m core run benchmarks/<name>/variants/<yours>.yaml
```

That is all. The harness fingerprints the config (so re-running RESUMES rather
than duplicating), checks S3 auth before spending tokens, writes each task to
the local database as it finishes, and uploads to S3 at the end.

Identity comes from the variant YAML — `metric, k, model, judge_model,
critic_model, feedback_mode, context, prompt_variant, temperature, seed,
reasoning_effort, output_budget, summarizer_model` plus the benchmark's slice.
Change any one and it is a different experiment in its own directory.

`python scripts/runs.py grid` shows what already exists.

## Finish someone else's partial run

```bash
python scripts/runs.py todo                          # what needs work
python scripts/storage/sync_from_s3.py --only <id> --apply
python scripts/runs.py resume <id> > resume.yaml     # derived config — do not edit
python -m core run resume.yaml
```

Check the header says `resuming | run_id=<id>`. If it says `new`, stop — the
config forked instead of continuing.

`runs.py todo` lists early-stopped pass@k runs separately. Those are FINISHED:
their short tasks stopped because they SUCCEEDED. Re-running them adds nothing.

## Pull in runs from elsewhere

Put them under `s3://seqk-data/to-be-organized/<yourname>/` and write an
importer — `scripts/storage/import_hf_configless.py` is the worked example.
The rule every importer follows: **the artifacts outrank the path.** A path is
not a config; guessing from it is how six CL-bench RSA runs got filed as DKR.

## Keep the three layers in step

```bash
python scripts/storage/sync_from_s3.py --apply   # S3 -> disk + local database
python scripts/storage/db_sync.py --rebuild --only <id>   # disk -> local database
python scripts/storage/sync_to_neon.py           # show local-vs-Neon drift
python scripts/storage/sync_to_neon.py --apply   # push everything
python scripts/storage/sync_to_neon.py --only <id> --apply   # push one run
```

Nothing reaches Neon on its own. The harness writes locally during a run
(~5 round trips per task, ~40 min of billed Neon time on a full load), so Neon
moves only when you run the command. `--only` replays one run in seconds;
without it the whole database ships as one `pg_dump | psql`, which REPLACES
rather than merges.

`sync_from_s3.py` only ever ADDS — a local run missing from S3 is left alone,
since it may simply not be uploaded yet. Runs failing validation are downloaded
but not loaded.

## Check your work

```bash
python scripts/validate.py --db     # format + disk vs database
python scripts/validate.py --s3     # + the manifest S3 actually holds (~1 min)
python scripts/runs.py doctor       # unpriced calls, duplicate fingerprints
python scripts/selftest.py          # offline checks, no DB or network
```

`--s3` compares CONTENT. Comparing key names only answers "does a directory of
this name exist on both sides" — that gap once hid 167 stale manifests that
would have failed a rebuild-from-S3 while every check reported 0 missing.

`validate.py` enforces: every manifest field the schema needs (`created_at` is
NOT NULL — one undated run rejects the whole load); `storage_key ==
<slice>/<run_id>`; canonical model names; attempts numbered `1..n` with no gaps;
one `task_id` per task directory; `pass@k` runs carry `context='na'`.

## Remove a run

```bash
python scripts/runs.py replace <id>                          # retire, keep files
python scripts/purge_runs.py <id> --apply --reason "..."     # delete everywhere
```

Prefer `replace`. `purge_runs.py` is the only destructive tool; it always writes
a tombstone to `purged.jsonl` and refuses without `--reason`.

## Gotchas that cost real time

- **`pg_dump` must be >= the server major.** Homebrew puts `postgresql@14` ahead
  of `@15`; the dump aborts *after* `psql --clean` has dropped every Neon table.
  This emptied Neon once. `push_to_neon.sh` now refuses if it can't find a
  matching binary.
- **Neon's pooled endpoint keeps `search_path` between sessions.** After a
  restore an unqualified query says `relation "runs" does not exist` while the
  tables are fine. `SET search_path TO public` first.
- **A failed `aws sso login` leaves the old token in place** and errors
  identically to never having logged in. Check `~/.aws/sso/cache/*.json` mtime.
- **Model names in `config` are canonical** (route stripped) so identity does not
  split on how a model was reached. The route still matters at call time —
  `runs.py resume` recovers it from the run's own attempt files.
- **Disk is the truth; Postgres is derived.** After changing files:
  `db_sync.py --rebuild --only <id>`, or `--reset --rebuild` for everything.
- **`runs/by-label/` is derived** — `runs.py relink` after anything that moves a run.

# Part 2 — Finding and querying runs

Runs are no longer identified by their directory name. A run's identity is a
**fingerprint** — a hash over every field that changes what the run means — and
the directory is just where the registry put it.

    runs/
    ├── .registry.json                        fingerprint -> storage key (a rebuildable cache)
    ├── by-label/<v2 name> -> ../<bench>/...  readable symlinks, regenerated automatically
    └── <benchmark>/<run_id>/
        ├── manifest.json                    identity, labels, provenance, rollup
        ├── config.json                      frozen config (kept for older readers)
        ├── summary.json
        └── task-<i>/attempt-<j>.json

Same layout in S3, same keys.

## Day to day

```bash
python scripts/runs.py ls                                  # everything, newest first
python scripts/runs.py ls --slice researchrubrics --k 10   # filter on any label
python scripts/runs.py show 4e1a2c77                       # one run, by uuid prefix / key / label
python scripts/runs.py phases 4e1a2c77                     # agent/judge/critic/summarizer cost split
python scripts/runs.py cost --by slice,metric              # cost rollup
python scripts/runs.py grid experiments/rr-feedback-channels.grid.yaml
python scripts/runs.py relink                              # regenerate by-label/
```

Every command works **with or without** Postgres. With `DATABASE_URL` set it
queries SQL; without, it reads the manifests on disk and answers the same
questions more slowly. The footer tells you which source answered.

Anywhere a run path is accepted, either form works:

```bash
python -m core inspect runs/researchrubrics/<run_id> --task-index 1
python -m core metrics "runs/by-label/researchrubrics/metric=seqk/k=10/..." --k 10
```

## Resume, and the one way to lose money

Re-running a variant YAML resumes its run, because the same config produces the
same fingerprint. A run starts **fresh** when any identity field differs:
benchmark, slice, metric, k, agent, judge, critic, feedback mode, context,
prompt variant, temperature, seed, reasoning effort, output budget, summarizer
model.

`k` is a special case: it is identity only when it actually reaches the prompt.
Under a horizon-free context (`*-nohorizon`, `*-noframe`) and for every pass@k
run, the prompt never mentions the horizon, so seq@5 is literally the 5-attempt
prefix of seq@10 — both resolve to ONE run, and asking for a larger k extends it
instead of forking. `core.ids.k_affects_prompt` decides by rendering the real
prompt at two values of k and comparing, so the rule cannot go stale when a
template changes. Agentic benchmarks build their own prompt and are assumed
k-dependent unless they expose a `prompt_probe()`.

Two other fields (`output_budget`, `summarizer_model`) are not in the readable
label. Under the old scheme they were policed by a hand-written guard; now they
simply resolve to a different run, and two runs can share a `by-label/` name.
The symlink writer disambiguates with a `~<uuid8>` suffix and never overwrites.

Because a one-field difference is usually a mistake rather than an ablation, the
harness prints a warning before starting a new run when an existing one differs
in exactly one field, naming the field and how much work already exists. Heed
it: that is the difference between resuming $40 of completed attempts and paying
for them twice.

Runs migrated from the old layout carry `prompt=legacy` (they ran under prompt
templates that were never versioned, so claiming `v1` would assert a provenance
we cannot check). The 62 variant YAMLs that produced them are pinned with
`prompt_variant: legacy` so re-running them resumes. **Bump that to `v2` when
you change a prompt template** — that is the whole point of the field.

## Setting up the database (optional)

The harness never needs it. It makes queries fast and makes cross-machine
questions possible.

1. Create a Neon project at https://console.neon.tech (free tier is ~15× the
   current corpus). Name the database `seqk`.
2. Copy the pooled connection string into `.env`:

   ```
   DATABASE_URL=postgresql://USER:PASSWORD@ep-xxx.us-east-2.aws.neon.tech/seqk?sslmode=require
   ```

3. Install the driver and create the schema:

   ```bash
   pip install 'psycopg[binary]'
   python scripts/storage/db_sync.py --schema
   ```

4. Load everything currently on disk:

   ```bash
   python scripts/storage/db_sync.py --rebuild        # idempotent; safe to re-run
   python scripts/storage/db_sync.py --status         # compare DB against disk
   ```

Disable per-machine with `SEQK_DB=0`.

### It is always disposable

The database is a mirror. Manifests on disk are the source of truth, and S3
carries them too, so `--rebuild` reconstructs every row from either. Nothing is
stored only in Postgres.

That is why the harness treats DB failures as non-events: a failed write logs
once, spools to `<run>/.db_pending.jsonl`, and the run continues. Replay later:

```bash
python scripts/storage/db_sync.py --replay
```

`db_sync.py` itself fails loudly when the DB is unreachable — silence is right
when a run's results are at stake and wrong when talking to the DB is the whole
point of the command.

### Useful queries

```sql
-- cost by phase across every seq@k run of one benchmark
SELECT c.phase, count(*), sum(c.cost_usd)
FROM llm_calls c JOIN runs r USING (run_id)
WHERE r.slice_key = 'researchrubrics' AND r.metric = 'seq@k'
GROUP BY c.phase ORDER BY 3 DESC;

-- between-seed variance for one config
SELECT seed, n_solved::float / nullif(n_tasks, 0) AS success_rate
FROM run_summary WHERE slice_key = 'healthbench-default'
  AND model = 'openrouter/openai/gpt-5.2' AND seed IS NOT NULL ORDER BY seed;

-- cumulative-best curve: how many tasks were solved by attempt N
SELECT first_solved_attempt, count(*) FROM task_summary
WHERE run_id = '...' AND solved GROUP BY 1 ORDER BY 1;

-- the phase cost split the papers report
SELECT phase, n_calls, cost_usd FROM run_phase_costs WHERE run_id = '...';
```

## Migration and reversal

`scripts/migrate_to_db.py` moved the 62 pre-existing runs. It is dry-run by
default, journals every move before making it, and never deletes:

```bash
python scripts/migrate_to_db.py            # dry run
python scripts/migrate_to_db.py --apply
python scripts/migrate_to_db.py --verify   # totals vs runs/INDEX.pre-v3.csv
python scripts/migrate_to_db.py --undo     # replay the journal backwards
```

The journal is `runs/migration-journal.jsonl`. Keep it until you are satisfied.

## Rebuilding after damage

```bash
python scripts/runs.py rebuild             # .registry.json + by-label/ from manifests
python scripts/runs.py doctor              # prompt drift, duplicate ids, DB vs disk
python scripts/storage/db_sync.py --rebuild        # Postgres from manifests
python scripts/storage/db_sync.py --reset          # drop the schema and reload (~15s)
```

Losing `.registry.json` or `by-label/` costs a rescan, never data.
