# Runs: storage and lookup

S3 holds the artifacts and is the source of truth. Postgres is a derived index,
rebuildable from the files. Neon is the shared copy of that index.

## Run something

```bash
export AWS_PROFILE=seqk
python -m core run benchmarks/<name>/variants/<yours>.yaml
```

Checks S3 auth before spending tokens, writes to Postgres as tasks finish,
uploads to S3 at the end. Re-running the same YAML **resumes**.

## Finish a partial run

```bash
python scripts/runs.py todo
python scripts/storage/sync_from_s3.py --only <id> --apply
python scripts/runs.py resume <id> > resume.yaml
python -m core run resume.yaml
```

Header must say `resuming | run_id=<id>`. If it says `new`, stop — the config
forked and you're about to pay for finished attempts twice. Early-stopped
pass@k runs are listed separately by `todo` and are FINISHED: their short tasks
stopped because they *succeeded*.

## Storing runs: S3 and Neon

A finished run uploads to S3 by itself. Postgres is written locally as tasks
complete. Neon never moves on its own — push it when you're done:

```bash
python scripts/storage/sync_to_neon.py                     # dry run: what differs
python scripts/storage/sync_to_neon.py --only <id> --apply # push one run, seconds
python -m core upload runs/<benchmark>/<run_id>            # re-upload to S3 by hand
```

Always use `--only`. A bare `--apply` ships the entire database as one
`pg_dump | psql`, which REPLACES Neon rather than merging into it — fine for a
full refresh, destructive if someone else pushed since your last pull.

If you changed files on disk (a repair, a manual edit), rebuild the index for
that run before pushing, since Postgres is derived and won't notice:

```bash
python scripts/storage/db_sync.py --rebuild --only <id>
python scripts/storage/db_sync.py --status                 # compare DB against disk
```

## Runs imported from the HF bucket

Most runs on S3 (372 of 483 in September 2026) were converted from the old
`seq_k_eval` files in `hf://buckets/namkoong-lab/seq-k` by
`scripts/storage/import_hf_configless.py`. The conversion is lossy. The S3 copy
keeps the actor section, `verifier.raw_output` and the feedback text, and drops
`additional_info` (judge cost, the API request), `annotations`,
`feedback_provider.feedback_modes`, `duration_ms`, `task_metadata` and
`dataset`. **The HF files are the only complete record of those runs**: keep the
bucket, and re-derive from it rather than from S3.

What the importer takes, and from where:

| field | source, first that exists |
|---|---|
| `judge.score` | `raw_output.normalized_score` (HealthBench) or `compliance_score` (ResearchRubrics); the `judge_score` annotation; the `correctness` annotation |
| `critic_model`, attempt and run | `feedback_provider.feedback_modes_metadata.<mode>.feedback_llm_model`; the actor |
| judge and critic calls | CL-bench only: `additional_info.metadata.judge_*`, and the tokens and `cost_usd` in the feedback metadata |

Runs imported before 2026-09-22 took the 0/1 `correctness` annotation as the
score and the actor as the critic. The values in the files were right; the
derived rows were not. Re-importing corrects them, but `critic_model` is part of
the fingerprint, so re-importing a run whose feedback writer was overridden
yields a NEW identity. Publish it over the old run_id, not beside it, or the
site's claims lose their runs.

The importer shells out to `hf buckets`; use the venv's `hf`
(`PATH=$PWD/.venv/bin:$PATH`), since older global installs lack that command.

## Check, and remove

```bash
python scripts/validate.py --s3    # format, disk vs DB, vs the manifest S3 holds
python scripts/runs.py doctor      # unpriced calls, duplicate fingerprints
python scripts/runs.py replace <id>                       # retire a run, keep files
python scripts/purge_runs.py <id> --apply --reason "..."  # the only destructive tool
```

`--s3` compares content, not key names — that gap once hid 167 stale manifests
while every check reported 0 missing.

## Identity

A run is a fingerprint over `metric, k, model, judge_model, critic_model,
feedback_mode, context, prompt_variant, temperature, seed, reasoning_effort,
output_budget, summarizer_model` + the slice. The directory name is just where
the registry put it.

```
runs/<benchmark>/<run_id>/{manifest.json, summary.json, task-<i>/attempt-<j>.json}
runs/by-label/  readable symlinks     runs/.registry.json  fingerprint -> key
```

Same keys in S3. `k` is identity only when it reaches the prompt — under
horizon-free contexts and all pass@k, seq@5 is the 5-attempt prefix of seq@10,
so a larger `k` extends rather than forks. Bump `prompt_variant` when you change
a template. The harness warns before starting a run that differs from an
existing one in exactly one field; that's usually a typo, not an ablation.

## Retrieving runs

To pull runs down from S3 onto a fresh machine — this adds only, never deletes,
and loads each run into Postgres as it lands:

```bash
python scripts/storage/sync_from_s3.py                 # dry run: what would come down
python scripts/storage/sync_from_s3.py --apply         # everything
python scripts/storage/sync_from_s3.py --only <id> --apply    # one run
python scripts/storage/sync_from_s3.py --limit 5 --apply      # first N, to test the setup
```

`--only` takes a run id, storage key or label, and repeats. Runs that fail
validation are downloaded but not loaded into the database.

To find a run once you have it:

```bash
python scripts/runs.py ls                              # everything, newest first
python scripts/runs.py ls --slice healthbench --k 10   # filter on any label
python scripts/runs.py show 4e1a2c77                   # identity, fingerprint, progress, cost
python scripts/runs.py phases 4e1a2c77                 # cost split by actor/judge/critic
python scripts/runs.py cost --by slice,metric          # rollup across runs
```

`show` accepts a uuid prefix, a storage key or a `by-label/` name. All of these
work with or without Postgres — without it they read manifests off disk and the
footer says which source answered.

## Database setup (optional)

```bash
pip install 'psycopg[binary]'      # DATABASE_URL=... in .env
python scripts/storage/db_sync.py --schema
python scripts/storage/db_sync.py --rebuild     # idempotent
```

`SEQK_DB=0` disables it. DB failures during a run spool to
`<run>/.db_pending.jsonl`; replay with `--replay`. Losing `.registry.json` or
`by-label/` costs a rescan (`runs.py rebuild`), never data.

