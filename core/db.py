"""Neon Postgres mirror of the run registry.

THE DATABASE IS NEVER ON A RUN'S CRITICAL PATH. Local manifests are the source
of truth (core/registry.py); this module reflects them into SQL so questions
like "which ablation cells are missing" and "what did the judge cost" are one
query instead of a walk over 4.7 GB of JSON.

Consequences of that rule, all enforced here:
  * Every write is wrapped: on failure it logs once and appends the payload to
    `<run>/.db_pending.jsonl` for later replay. A $400 grid must not die
    because Neon cold-started.
  * Every write is idempotent (`ON CONFLICT DO UPDATE`), so replay is safe and
    two racing processes converge.
  * Nothing here is required to exist. No DATABASE_URL, or psycopg not
    installed, and the harness runs exactly as it did before — `enabled()` is
    False and every call returns immediately.

Rebuild from scratch at any time with `python scripts/storage/db_sync.py --rebuild`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
PENDING_NAME = ".db_pending.jsonl"

# Phase and cost-source are stored as plain TEXT. These tuples give Python a
# display order and nothing more — there is no lookup table to keep in sync, and
# `SELECT DISTINCT phase FROM llm_calls` documents the vocabulary from the data.
PHASES = ("actor", "judge", "critic", "summarizer")
# EXACTLY the vocabulary core.pricing.cost_for returns, plus "unknown" for the
# null case. It used to say "rates", a word pricing.py has never produced, and
# core/rows.py filters on this tuple — so every table- and litellm-priced call
# was stored as cost_source='unknown' while summary.json (which does not filter)
# recorded the true source. 22k calls disagreed with their own run summary.
COST_SOURCES = ("reported", "table", "litellm", "unknown")

_conn = None
_warned = False
# Admin scripts set this. The harness must never fail because of the DB, but a
# command whose entire purpose is to load the DB must not report success while
# silently dropping a third of the rows — which is exactly what happened when a
# transient Neon error hit mid-rebuild and _warn_once muted every one after the
# first.
STRICT = False


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
_dotenv_loaded = False


def dsn():
    """Connection string from the environment, or from .env.

    core/cli.py loads .env, but the db scripts are entry points too and used to
    see nothing — so `python scripts/storage/db_sync.py --status` failed with
    "DATABASE_URL is not set" while the key sat in .env the whole time. Load it
    here, where the need actually is, once per process.
    """
    global _dotenv_loaded
    if not _dotenv_loaded:
        _dotenv_loaded = True
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        except Exception:                          # noqa: BLE001 - env is optional
            pass
    return os.environ.get("DATABASE_URL") or os.environ.get("SEQK_DATABASE_URL")


def enabled():
    """True when a mirror is configured AND reachable-in-principle."""
    if os.environ.get("SEQK_DB", "1") in ("0", "false", "False"):
        return False
    if not dsn():
        return False
    try:
        import psycopg  # noqa: F401
    except ImportError:
        _warn_once("DATABASE_URL is set but psycopg is not installed "
                   "(`pip install 'psycopg[binary]'`); running without the DB mirror")
        return False
    return True


def connect(*, required=False):
    """Cached connection. Returns None when disabled or unreachable, unless
    `required` (used by the admin scripts, which SHOULD fail loudly)."""
    global _conn
    if _conn is not None and not getattr(_conn, "closed", False):
        return _conn
    if not enabled():
        if required:
            raise RuntimeError(
                "No database configured. Set DATABASE_URL to your Neon connection "
                "string and `pip install 'psycopg[binary]'`."
            )
        return None
    import psycopg
    try:
        _conn = psycopg.connect(dsn(), autocommit=True, connect_timeout=10)
        # Never inherit search_path from the session we are handed. Neon's pooled
        # endpoint reuses server sessions across clients, so a `SET search_path`
        # from ANOTHER client leaks in — a pg_dump restore sets it to '' and every
        # later query then fails with "relation runs does not exist" even though
        # the tables are right there. Pin it explicitly on every connection.
        with _conn.cursor() as cur:
            cur.execute("SET search_path TO public")
    except Exception as exc:                       # noqa: BLE001 - any driver error
        if required:
            raise
        _warn_once(f"database unreachable ({exc.__class__.__name__}: {exc}); "
                   f"continuing with local manifests only")
        return None
    return _conn


class using:
    """Temporarily point every db.* call at another DSN.

        with db.using(os.environ["SEQK_NEON_URL"]):
            db.upsert_run(manifest, run_path=path)

    The module keeps ONE cached connection, so swapping means closing it and
    restoring it afterwards — otherwise a mirror-to-Neon would leave every later
    local write going to Neon instead.
    """

    def __init__(self, target_dsn):
        self._dsn = target_dsn
        self._saved_conn = None
        self._saved_env = None

    def __enter__(self):
        global _conn
        self._saved_conn, _conn = _conn, None
        self._saved_env = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = self._dsn
        return self

    def __exit__(self, *exc):
        global _conn
        try:
            if _conn is not None and not getattr(_conn, "closed", False):
                _conn.close()
        except Exception:                          # noqa: BLE001
            pass
        _conn = self._saved_conn
        if self._saved_env is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = self._saved_env
        return False


def close():
    global _conn
    if _conn is not None and not getattr(_conn, "closed", False):
        _conn.close()
    _conn = None


def apply_schema(*, required=True):
    """Apply db/schema.sql. Idempotent, so this is safe to run any time.

    There is no migration history because there is nothing to migrate: every row
    is derived from the manifests and attempt JSON on disk. A change SQL cannot
    make in place is handled by `reset()` — drop and reload — which is why the
    schema is one file rather than a numbered sequence.
    """
    conn = connect(required=required)
    if conn is None:
        return False
    with conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    return True


def reset(*, required=True):
    """Drop every table (cascading to the views) and recreate the schema.

    Safe by construction: the database holds nothing that is not reproducible
    from runs/*/manifest.json plus the attempt files. Callers follow this with a
    rebuild.
    """
    conn = connect(required=required)
    if conn is None:
        return False
    with conn.cursor() as cur:
        # Discover the tables rather than listing them. A hard-coded list went
        # stale the moment the schema grew: `tasks` and `run_attempts` were not
        # dropped, their rows survived the "reset", and every subsequent insert
        # collided with them — 616 UniqueViolations reported as a clean reset.
        cur.execute("""SELECT tablename FROM pg_tables WHERE schemaname = 'public'""")
        for (t,) in cur.fetchall():
            cur.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    return True


# --------------------------------------------------------------------------- #
# Writes — all non-fatal
# --------------------------------------------------------------------------- #
def upsert_run(manifest, *, run_path=None):
    """Mirror a run's identity. Safe to call repeatedly.

    Conflict target is `run_id`, the row's actual identity — NOT the fingerprint.
    A fingerprint-targeted upsert would fail to match on re-insert and blow up on
    the primary key instead. The unique index in schema.sql enforces one run per
    fingerprint; violating it raises, which is what we want, because two runs
    claiming one identity is a bug rather than a state to merge.

    Racing processes are handled upstream: core/registry.py resolves the
    fingerprint under a file lock, so both come away with the same run_id and
    therefore the same row here.
    """
    cfg = manifest.get("config", {})
    return _guard(run_path, "upsert_run", {"run_id": manifest.get("run_id")}, lambda cur: cur.execute(
        """
        INSERT INTO runs (run_id, fingerprint, fingerprint_version, created_at, finished_at,
                          status, storage_key, benchmark, slice_key, metric, k,
                          model, judge_model, critic_model, feedback_mode, context,
                          prompt_variant, temperature, seed, reasoning_effort, output_budget,
                          summarizer_model, options, code)
        VALUES (%(run_id)s, %(fingerprint)s, %(fpv)s, %(created_at)s, %(finished_at)s,
                %(status)s, %(storage_key)s, %(benchmark)s, %(slice_key)s,
                %(metric)s, %(k)s, %(model)s, %(judge_model)s, %(critic_model)s,
                %(feedback_mode)s, %(context)s, %(prompt_variant)s, %(temperature)s,
                %(seed)s, %(reasoning_effort)s, %(output_budget)s, %(summarizer_model)s,
                %(options)s, %(code)s)
        ON CONFLICT (run_id) DO UPDATE SET
            -- storage_key MUST be here: anything that re-keys a run changes it,
            -- and leaving it out lets Neon point at stale S3 prefixes while
            -- attempts.output_key had already moved on.
            storage_key = EXCLUDED.storage_key,
            finished_at = COALESCE(EXCLUDED.finished_at, runs.finished_at),
            -- k grows when a horizon-free run is extended; never shrink it
            k = GREATEST(runs.k, EXCLUDED.k),
            status = EXCLUDED.status,
            code = EXCLUDED.code
        """,
        {
            "run_id": manifest["run_id"], "fingerprint": manifest["fingerprint"],
            "fpv": manifest.get("fingerprint_version", 1),
            "created_at": manifest["created_at"], "finished_at": manifest.get("finished_at"),
            "status": manifest.get("status", "running"),
            "storage_key": manifest["storage_key"],
            "benchmark": cfg.get("benchmark"), "slice_key": cfg.get("slice_key"),
            "metric": cfg.get("metric"),
            # runs.k is the attempt BUDGET this run has reached, which is not
            # always identity: under a horizon-free prompt seq@5 and seq@10 are
            # one run, and this column is the ceiling it has been run to.
            # k_target first: `config.k` is None whenever k is not identity.
            "k": manifest.get("k_target") or cfg.get("k")
                 or _require_k(manifest),
            "model": cfg.get("model"),
            "judge_model": cfg.get("judge_model"), "critic_model": cfg.get("critic_model"),
            "feedback_mode": cfg.get("feedback_mode"), "context": cfg.get("context"),
            "prompt_variant": cfg.get("prompt_variant"), "temperature": cfg.get("temperature"),
            "seed": cfg.get("seed"), "reasoning_effort": cfg.get("reasoning_effort"),
            "output_budget": cfg.get("output_budget"),
            "summarizer_model": cfg.get("summarizer_model"),
            "options": json.dumps(manifest.get("options") or {}),
            "code": json.dumps(manifest.get("code") or {}),
        }))


_TASK_SQL = """
    INSERT INTO tasks (slice_key, task_index, task_id, prompt, meta)
    VALUES (%(slice_key)s, %(task_index)s, %(task_id)s, %(prompt)s, %(meta)s)
    ON CONFLICT (slice_key, task_id) DO UPDATE SET
        task_index = EXCLUDED.task_index,
        prompt = COALESCE(EXCLUDED.prompt, tasks.prompt),
        meta = CASE WHEN EXCLUDED.meta = '{}'::jsonb THEN tasks.meta ELSE EXCLUDED.meta END
    RETURNING task_uid
"""
_RUN_TASK_SQL = """
    INSERT INTO run_tasks (run_id, task_uid, extra) VALUES (%s, %s, %s)
    ON CONFLICT (run_id, task_uid) DO UPDATE SET extra = EXCLUDED.extra
"""
# One row per GENERATION. Conflict target is (generated_by_run, task_uid,
# attempt_index) — NOT the fingerprint. Two runs with an identical actor config
# still drew different text at temperature 0.7, so they are two artifacts.
# Reuse is decided BEFORE generating (find_reusable below), never by merging
# here.
#
# ALL of a task's attempts go in ONE statement. A per-attempt INSERT..RETURNING
# cannot be batched, and at ~40ms of Neon round trip each it turned a 15-second
# rebuild into a 20-minute one. DO UPDATE (not DO NOTHING) matters here too:
# only an UPDATE branch returns a row, and we need every attempt_id back,
# including the ones that already existed — those are precisely the reuses.
_ATTEMPT_COLS = ("task_uid", "actor_fingerprint", "attempt_index", "generated_by_run",
                 "output_key", "finish_reason", "created_at")
_ATTEMPT_SQL = """
    INSERT INTO attempts (task_uid, actor_fingerprint, attempt_index, generated_by_run,
                          output_key, finish_reason, created_at)
    VALUES {values}
    ON CONFLICT (generated_by_run, task_uid, attempt_index) DO UPDATE SET
        actor_fingerprint = EXCLUDED.actor_fingerprint,
        output_key = COALESCE(EXCLUDED.output_key, attempts.output_key),
        finish_reason = COALESCE(EXCLUDED.finish_reason, attempts.finish_reason)
    RETURNING attempt_id, actor_fingerprint, attempt_index
"""
_CLAIM_SQL = """
    INSERT INTO run_attempts (run_id, attempt_id, task_uid, attempt_index, solved, score, extra)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (run_id, attempt_id) DO UPDATE SET
        attempt_index = EXCLUDED.attempt_index, solved = EXCLUDED.solved,
        score = EXCLUDED.score, extra = EXCLUDED.extra
"""
_CALL_SQL = """
    INSERT INTO llm_calls (run_id, attempt_id, phase, call_index, model,
                           input_tokens, cached_tokens, thinking_tokens, output_tokens,
                           cost_usd, cost_source)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (run_id, attempt_id, phase, call_index) DO UPDATE SET
        model = EXCLUDED.model, input_tokens = EXCLUDED.input_tokens,
        cached_tokens = EXCLUDED.cached_tokens, thinking_tokens = EXCLUDED.thinking_tokens,
        output_tokens = EXCLUDED.output_tokens, cost_usd = EXCLUDED.cost_usd,
        cost_source = EXCLUDED.cost_source
"""


def record_task(run_id, task, attempts, claims, calls, *, run_path=None):
    """Mirror one task across all five child tables.

    `attempts` are artifacts, `claims` are this run's claims on them (parallel
    lists), `calls` is a list-of-lists parallel to both. Batched per TASK: a
    single ResearchRubrics task carries ~270 LLM calls, and one round trip each
    would put minutes of Neon latency into every run.
    """
    def _do(cur):
        cur.execute(_TASK_SQL, {**task, "meta": json.dumps(task.get("meta") or {})})
        task_uid = cur.fetchone()[0]
        cur.execute(_RUN_TASK_SQL, (run_id, task_uid, json.dumps(task.get("extra") or {})))
        if not attempts:
            return
        params, tmpl = [], ",".join(["(" + ",".join(["%s"] * len(_ATTEMPT_COLS)) + ")"]
                                    * len(attempts))
        for art in attempts:
            params.extend([task_uid] + [art[c] for c in _ATTEMPT_COLS[1:]])
        cur.execute(_ATTEMPT_SQL.format(values=tmpl), params)
        ids_by_key = {draw: aid for aid, _fp, draw in cur.fetchall()}

        claim_rows, call_rows = [], []
        for art, claim, per_attempt in zip(attempts, claims, calls):
            attempt_id = ids_by_key[art["attempt_index"]]
            claim_rows.append((run_id, attempt_id, task_uid, claim["attempt_index"],
                               claim["solved"], claim["score"],
                               json.dumps(claim.get("extra") or {})))
            for c in per_attempt:
                call_rows.append((run_id, attempt_id, c["phase"], c["call_index"], c["model"],
                                  c["input_tokens"], c["cached_tokens"], c["thinking_tokens"],
                                  c["output_tokens"], c["cost_usd"], c["cost_source"]))
        cur.executemany(_CLAIM_SQL, claim_rows)
        if call_rows:
            cur.executemany(_CALL_SQL, call_rows)
    return _guard(run_path, "record_task",
                  {"run_id": run_id, "task_id": task.get("task_id")}, _do)


def find_reusable(task_uid, actor_fingerprint, *, limit, exclude_run=None):
    """Existing generations this config is allowed to claim instead of drawing.

    The reuse mechanism, and it is a QUERY rather than a merge: two independent
    draws from the same distribution are distinct artifacts, so the caller picks
    which ones to claim. Ordered oldest first so a claim set is deterministic.
    """
    return query(
        """SELECT attempt_id, attempt_index, generated_by_run, output_key
           FROM attempts
           WHERE task_uid = %(task_uid)s AND actor_fingerprint = %(fp)s
             AND (%(exclude)s::uuid IS NULL OR generated_by_run <> %(exclude)s::uuid)
           ORDER BY attempt_id LIMIT %(limit)s""",
        {"task_uid": task_uid, "fp": actor_fingerprint, "limit": limit,
         "exclude": exclude_run}, required=False)


def finish_run(run_id, *, status, finished_at=None, run_path=None, **_ignored):
    """Stamp the terminal status. Deliberately writes NO totals.

    Counts and cost are views over llm_calls (see db/schema.sql), so there is no
    cached number here to fall out of date when a run is resumed, extended, or
    killed mid-flight. `**_ignored` swallows a legacy `rollup=` kwarg."""
    return _guard(run_path, "finish_run", {"run_id": run_id, "status": status}, lambda cur: cur.execute(
        """
        UPDATE runs SET status = %(status)s,
               finished_at = COALESCE(%(finished_at)s, finished_at)
        WHERE run_id = %(run_id)s
        """,
        {"run_id": run_id, "status": status, "finished_at": finished_at}))


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def query(sql, params=None, *, required=True):
    """Run a SELECT, returning list[dict].

    `params` is passed through AS IS, None included. It used to be `params or
    ()`, and an empty tuple is not the same as None: it puts psycopg into
    placeholder-parsing mode, so every literal `%` in the SQL became a bad
    placeholder and any LIKE pattern raised ProgrammingError through this
    helper while working fine in raw psycopg.
    """
    conn = connect(required=required)
    if conn is None:
        return []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def execute(sql, params=None, *, required=True):
    """Run a writing statement, returning the number of rows affected.

    `query()` is SELECT-only. This is its counterpart, and it COMMITS: the
    connection is autocommit, so a DELETE here is final. Callers that need
    several statements to succeed or fail together should open their own
    transaction rather than chaining calls to this.
    """
    conn = connect(required=required)
    if conn is None:
        return 0
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
def _guard(run_path, op, payload, fn):
    """Execute `fn(cursor)` ATOMICALLY; on failure log once and spool for replay.

    The transaction matters as much as the guard. The connection is autocommit,
    so without it a write that fails half way leaves the rows it already made
    behind — and a retry then collides with its own debris (observed as a
    UniqueViolation on run_attempts during a rebuild retry). All-or-nothing per
    task means a retry always starts from a clean slate.
    """
    conn = connect()
    if conn is None:
        _spool(run_path, op, payload)
        return False
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                fn(cur)
        return True
    except Exception as exc:                       # noqa: BLE001 - the DB may never break a run
        if STRICT:
            raise
        _warn_once(f"db {op} failed ({exc.__class__.__name__}: {exc}); "
                   f"spooling to {PENDING_NAME} and continuing")
        _spool(run_path, op, payload)
        return False


def _spool(run_path, op, payload):
    """Append a failed write to <run>/.db_pending.jsonl for scripts/storage/db_sync.py.

    Best-effort by construction: if even this fails, the manifests and attempt
    JSON on disk still hold everything needed to rebuild the DB from scratch.
    """
    if not run_path:
        return
    try:
        with open(Path(run_path) / PENDING_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps({"op": op, "payload": payload}, default=str) + "\n")
    except OSError:
        pass


def _warn_once(msg):
    global _warned
    if not _warned:
        print(f"! {msg}")
        _warned = True
