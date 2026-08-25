"""Reconcile Postgres with what is on disk.

The database is a mirror of the manifests and attempt files, so it is always
disposable: drop it, re-run this, and every number comes back. Three jobs:

    python scripts/storage/db_sync.py --schema         # apply db/schema.sql (idempotent)
    python scripts/storage/db_sync.py --rebuild        # (re)load every run from disk
    python scripts/storage/db_sync.py --reset          # drop everything, then rebuild
    python scripts/storage/db_sync.py --replay         # push writes that failed live

`--rebuild` is idempotent — every statement is an upsert keyed on natural
identity — so running it twice changes nothing, and running it against a
half-populated database completes it.

There is no migration history. The database holds nothing that is not derived
from the manifests and attempt files, so a change SQL cannot make in place is
just `--reset`: drop the tables and reload. That is why db/ is one schema file
rather than a numbered sequence.

This script, unlike the harness, FAILS LOUDLY when the database is unreachable.
That asymmetry is deliberate: silence is right when a run's results are at
stake, and wrong when the whole point of the command is to talk to the DB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core import db, registry, rows  # noqa: E402


def do_schema():
    db.apply_schema(required=True)
    print("schema applied (idempotent)")


def do_reset(runs_root):
    db.reset(required=True)
    print("dropped and recreated every table; reloading from disk")
    do_rebuild(runs_root)


def _supersede_order(manifests):
    """Insert order satisfying `runs.superseded_by -> runs.run_id`.

    A retired run points at the one that replaced it, so it cannot be inserted
    first. On-disk manifest order is arbitrary, which made a rebuild fail or not
    depending on filesystem iteration order. Emit a run only once its target is
    present; anything left over (a dangling or circular pointer) goes last, so a
    bad pointer surfaces as its own FK error rather than silently reordering.
    """
    pending = list(manifests)
    emitted, out = set(), []
    while pending:
        # Filter on run_id, NOT on object identity: a comprehension that
        # unpacks and rebuilds `(path, manifest)` tuples produces new objects,
        # so an id()-based residue check matches nothing and this loops forever.
        ready = [x for x in pending
                 if not x[1].get("superseded_by") or x[1]["superseded_by"] in emitted]
        if not ready:
            out.extend(pending)                    # dangling/cyclic — let the FK complain
            break
        ready_ids = {x[1]["run_id"] for x in ready}
        emitted |= ready_ids
        out.extend(ready)
        pending = [x for x in pending if x[1]["run_id"] not in ready_ids]
    return out


def do_rebuild(runs_root, *, only=None):
    db.connect(required=True)
    db.apply_schema(required=True)
    # Loud: a partial load that reports success is worse than a failure.
    db.STRICT = True
    n_runs = n_tasks = 0
    total = 0.0
    failed = []
    for path, m in _supersede_order(registry.iter_manifests(runs_root)):
        if only and only not in (m["run_id"], m["storage_key"]) and only not in (m.get("label_path") or ""):
            continue
        cfg = m.get("config", {})
        # `config.k` is None when k is not part of identity (horizon-free
        # variant); k_target is the budget the run was actually run to.
        k = m.get("k_target") or cfg.get("k")
        seq = cfg.get("metric") == "seq@k"
        if not db.upsert_run(m, run_path=path):
            print(f"  ! FAILED to upsert {m['storage_key']}")
            failed.append(m["storage_key"])
            continue
        for idx in rows.iter_task_indices(path):
            meta = _task_meta(path, idx)
            for attempt in range(3):               # transient Neon errors are common
                try:
                    if rows.mirror_task(m["run_id"], path, idx, ident=cfg, k=k, seq=seq,
                                        task_id=meta.get("task_id"), prompt=meta.get("prompt"),
                                        storage_key=m["storage_key"]):
                        n_tasks += 1
                    break
                except Exception as exc:           # noqa: BLE001
                    if attempt == 2:
                        print(f"  ! FAILED task-{idx} of {m['storage_key']}: "
                              f"{exc.__class__.__name__}: {exc}")
                        failed.append(f"{m['storage_key']} task-{idx}")
                        break
                    db.close()                     # force a fresh connection
        roll = m.get("rollup") or rows.run_rollup(path, k=k, seq=seq)
        db.finish_run(m["run_id"], status=m.get("status", "unknown"),
                      finished_at=m.get("finished_at"), run_path=path)
        n_runs += 1
        total += roll.get("cost_usd", 0.0)
        print(f"  {m['storage_key']}  {m.get('label_path','')[:60]}  "
              f"{roll.get('attempts_total',0)} attempts  ${roll.get('cost_usd',0):.2f}")
    print(f"\nloaded {n_runs} runs / {n_tasks} tasks   ${total:.2f}")
    if failed:
        # A silently-partial mirror is worse than none: it looks authoritative.
        print(f"\n*** {len(failed)} ITEM(S) DID NOT LOAD ***")
        for k in failed:
            print(f"    {k}")
    _report(runs_root)
    if failed:
        sys.exit(1)


def do_replay(runs_root):
    """Push spooled writes from <run>/.db_pending.jsonl, then clear the spool."""
    db.connect(required=True)
    replayed = failed = 0
    for path, m in registry.iter_manifests(runs_root):
        spool = Path(path) / db.PENDING_NAME
        if not spool.exists():
            continue
        cfg = m.get("config", {})
        k = m.get("k_target") or cfg.get("k")
        seq = cfg.get("metric") == "seq@k"
        # Rather than replaying each spooled payload verbatim (which may predate
        # a schema change), re-derive the run's rows from disk. Same result,
        # always current, and idempotent.
        ok = db.upsert_run(m, run_path=path)
        for idx in rows.iter_task_indices(path):
            meta = _task_meta(path, idx)
            ok = rows.mirror_task(m["run_id"], path, idx, ident=cfg, k=k, seq=seq,
                                  task_id=meta.get("task_id"), prompt=meta.get("prompt"),
                                  storage_key=m["storage_key"]) and ok
        if ok:
            spool.rename(spool.with_suffix(".jsonl.done"))
            replayed += 1
        else:
            failed += 1
    print(f"replayed {replayed} run(s); {failed} still pending")


def _task_meta(run_path, task_index):
    """task-N/task_meta.json — the task's id and prompt, written once per run."""
    f = Path(run_path) / f"task-{task_index}" / "task_meta.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _smoke():
    """Execute one of every query shape the CLI issues.

    The offline selftest stubs the provider but cannot check SQL, so a column
    renamed in schema.sql can pass every test and still break `runs.py phases`
    against a real database — which is exactly what happened once, when the
    `phases` lookup table went away and an `ORDER BY ordinal` outlived it. This
    makes a schema/CLI disagreement fail here instead of in front of a user.
    """
    checks = [
        ("run_summary",     "SELECT * FROM run_summary LIMIT 1", None),
        ("task_summary",    "SELECT * FROM task_summary LIMIT 1", None),
        ("attempt_summary", "SELECT * FROM attempt_summary LIMIT 1", None),
        ("run_phase_costs", "SELECT phase, n_calls, input_tokens, output_tokens, cost_usd "
                            "FROM run_phase_costs LIMIT 1", None),
        ("runs.py ls",      "SELECT * FROM run_summary WHERE slice_key = %s "
                            "ORDER BY created_at DESC LIMIT 1", ("researchrubrics",)),
    ]
    bad = 0
    for name, sql, params in checks:
        try:
            db.query(sql, params)
            print(f"  ok    {name}")
        except Exception as exc:                       # noqa: BLE001
            bad += 1
            print(f"  FAIL  {name}: {exc}")
    return bad


def _report(runs_root):
    # run_summary, not runs: cost is a view over llm_calls, never a column.
    got = db.query("SELECT count(*) n, coalesce(sum(cost_usd),0) c FROM run_summary")
    calls = db.query("SELECT count(*) n FROM llm_calls")
    att = db.query("SELECT count(*) n FROM attempts")
    disk = sum(1 for _p, _m in registry.iter_manifests(runs_root))
    print(f"\ndatabase: {got[0]['n']} runs, {att[0]['n']} attempts, {calls[0]['n']} llm_calls, "
          f"${float(got[0]['c']):.2f}")
    print(f"disk:     {disk} runs")
    if got[0]["n"] != disk:
        print("  ! counts differ — re-run with --rebuild")
    print("\nquery smoke test:")
    if _smoke():
        print("  ! some queries failed — schema.sql and the CLI disagree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--reset", action="store_true", help="drop every table, then rebuild")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--only", help="limit --rebuild to one run_id / storage key / label")
    ap.add_argument("--status", action="store_true", help="compare DB against disk")
    args = ap.parse_args()

    if not any([args.schema, args.rebuild, args.reset, args.replay, args.status]):
        ap.error("pick one of --schema / --rebuild / --reset / --replay / --status")
    if not db.dsn():
        sys.exit("DATABASE_URL is not set. Put your Neon connection string in .env:\n"
                 "  DATABASE_URL=postgresql://USER:PASS@HOST.neon.tech/seqk?sslmode=require")
    if args.schema:
        do_schema()
    if args.reset:
        do_reset(args.runs_root)
    if args.rebuild and not args.reset:
        do_rebuild(args.runs_root, only=args.only)
    if args.replay:
        do_replay(args.runs_root)
    if args.status and not args.rebuild:
        db.connect(required=True)
        _report(args.runs_root)


if __name__ == "__main__":
    main()
