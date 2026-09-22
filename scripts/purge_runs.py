"""Permanently remove runs from disk, S3 and BOTH databases.

    python scripts/purge_runs.py <run_id> ...                    # dry run
    python scripts/purge_runs.py --from-file ids.txt --apply --reason "..."

The only destructive tool here. Purge is for runs that should not exist at all:
empty shells, imports broken beyond repair, runs whose attempts were almost
entirely rejected by QA. To record a SECOND run of one config, give it a
different `seed` — that is a different fingerprint, so both coexist.

Appends a tombstone to `purged.jsonl` (identity, counts, source, reason) so
a missing cell stays explainable. Deletes DB first, then S3, then local disk —
an interruption leaves the files, which are the only irreplaceable part.

BOTH databases, and that word is load bearing. Everything else here writes to
DATABASE_URL, the local working copy; Neon is the copy other people query
(core/neonsync.py). Purging only the local one leaves the run alive in Neon
indefinitely, because nothing else ever deletes from Neon — `sync_to_neon.py`
without `--only` is a pg_dump replace, so it corrects this by accident at best
and only if someone happens to run it. A purge that does not reach the mirror is
not a purge: `runs.py` and every dashboard pointed at Neon keep returning the run
as live. Twelve purged runs sat in Neon exactly this way, showing up as partial
runs that no longer existed on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db, ids, registry, s3sync  # noqa: E402

LEDGER = Path("purged.jsonl")


def resolve(prefixes, runs_root):
    """Match id prefixes to manifests. Ambiguity or a miss is fatal."""
    all_m = {m["run_id"]: (p, m) for p, m in registry.iter_manifests(runs_root)}
    out, bad = [], []
    for pre in prefixes:
        hits = [rid for rid in all_m if rid.startswith(pre)]
        if len(hits) != 1:
            bad.append((pre, len(hits)))
            continue
        out.append(all_m[hits[0]])
    if bad:
        for pre, n in bad:
            print(f"  {pre}: {'no such run' if not n else f'{n} runs match — be more specific'}")
        sys.exit("refusing to purge: could not resolve every id")
    return out


def describe(m):
    c = m["config"]
    code = m.get("code") or {}
    return {
        "run_id": m["run_id"], "storage_key": m["storage_key"],
        "slice_key": c.get("slice_key"), "model": c.get("model"),
        "metric": c.get("metric"), "k": m.get("k_target") or c.get("k"),
        "feedback_mode": c.get("feedback_mode"), "judge_model": c.get("judge_model"),
        "context": c.get("context"), "prompt_variant": c.get("prompt_variant"),
        "seed": c.get("seed"), "fingerprint": m.get("fingerprint"),
        "imported_from": code.get("imported_from"),
        "deprecated_skipped": code.get("deprecated_skipped"),
    }


def _check_claims(items, *, apply):
    """Refuse to purge a run whose generations another LIVE run claims.

    `attempts.generated_by_run` is a cross-run foreign key with ON DELETE
    CASCADE: a re-judge claims the source's generations rather than re-paying for
    them (core/rejudge.py), and the artifact row stays attributed to the source.
    Delete the source and that row goes with it, taking the claiming run's
    attempts out of the database — while its files sit untouched on disk. The
    claiming run then fails every rebuild with a ForeignKeyViolation and quietly
    loads zero tasks.

    This is not hypothetical: purging one run did exactly that to another, and it
    surfaced only because a rebuild happened to be run afterwards.

    Purging anyway is legitimate once the claiming runs are told — so the fix is
    offered rather than imposed: mark each orphaned claim `source_purged` and
    core/rows.py re-attributes the artifact to the run that still holds the file.
    """
    ids_ = {m["run_id"] for _p, m in items}
    blocked = {}
    for _p, m in items:
        rows_ = db.query("""SELECT DISTINCT ra.run_id::text rid, r.storage_key
                            FROM attempts a
                            JOIN run_attempts ra ON ra.attempt_id = a.attempt_id
                            JOIN runs r ON r.run_id = ra.run_id
                            WHERE a.generated_by_run = %s AND ra.run_id <> %s""",
                         (m["run_id"], m["run_id"]), required=False) or []
        others = [r for r in rows_ if r["rid"] not in ids_]
        if others:
            blocked[m["storage_key"]] = others
    if not blocked:
        return
    print("!! REFUSING: other runs CLAIM generations from these runs.\n")
    for key, others in blocked.items():
        print(f"  {key} is the source for:")
        for o in others:
            print(f"      {o['storage_key']}")
    print("\n  Purging would cascade-delete those shared artifact rows, so the claiming runs\n"
          "  would vanish from the database while their files stayed on disk, and every\n"
          "  later rebuild would fail on them with a ForeignKeyViolation.\n")
    print("  To purge anyway, first orphan the claims so they stop pointing at a dead run:\n"
          "      python scripts/purge_runs.py --orphan-claims <source_run_id> --apply\n"
          "  That stamps `reused_from.source_purged` on each claiming attempt file; the claim\n"
          "  and its cost accounting survive, and the artifact is re-attributed to the run\n"
          "  that still holds it.")
    sys.exit(2)


def orphan_claims(source_prefix, runs_root, *, apply):
    """Mark every claim on `source_prefix`'s generations as orphaned, on disk."""
    n_runs = n_att = 0
    for path, m in registry.iter_manifests(runs_root):
        hits = []
        for f in sorted(Path(path).glob("task-*/attempt-*.json")):
            try:
                a = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            ru = a.get("reused_from") or {}
            if not str(ru.get("run_id", "")).startswith(source_prefix):
                continue
            if ru.get("source_purged"):
                continue
            hits.append((f, a))
        if not hits:
            continue
        n_runs += 1
        print(f"  {m['storage_key']}: {len(hits)} claimed attempt(s)")
        if not apply:
            continue
        for f, a in hits:
            a["reused_from"]["source_purged"] = True
            a["reused_from"]["source_purged_at"] = ids.iso(ids.utc_now())
            f.write_text(json.dumps(a, indent=1, ensure_ascii=False), encoding="utf-8")
            n_att += 1
    print(f"\n{'orphaned' if apply else 'would orphan'} {n_att or '?'} attempt(s) across {n_runs} run(s)"
          + ("" if apply else " — re-run with --apply"))
    return n_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="*")
    ap.add_argument("--from-file")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--reason", default="", help="recorded in the tombstone ledger")
    ap.add_argument("--keep-s3", action="store_true", help="leave the S3 objects in place")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--orphan-claims", metavar="RUN_ID",
                    help="mark claims on this (already purged) run's generations as orphaned, "
                         "so they stop referencing a row that no longer exists")
    args = ap.parse_args()

    if args.orphan_claims:
        orphan_claims(args.orphan_claims, args.runs_root, apply=args.apply)
        return

    prefixes = list(args.run_ids)
    if args.from_file:
        prefixes += [l.split("#")[0].strip()
                     for l in Path(args.from_file).read_text().splitlines() if l.split("#")[0].strip()]
    if not prefixes:
        sys.exit("no run ids given")
    if args.apply and not args.reason:
        sys.exit("--apply requires --reason: a purge with no recorded reason is unexplainable later")

    items = resolve(prefixes, args.runs_root)
    _check_claims(items, apply=args.apply)
    print(f"{len(items)} run(s) to purge\n")
    counts = {}
    for path, m in items:
        rid = m["run_id"]
        row = db.query("""SELECT n_tasks, n_attempts, round(cost_usd::numeric,2) cost
                          FROM run_summary WHERE run_id=%s""", (rid,))
        row = row[0] if row else {"n_tasks": "?", "n_attempts": "?", "cost": "?"}
        counts[rid] = row
        d = describe(m)
        print(f"  {m['storage_key']}")
        print(f"      {d['slice_key']} · {d['model']} · {d['metric']} k={d['k']} · "
              f"{d['feedback_mode']} · judge={d['judge_model']}")
        print(f"      {row['n_tasks']} tasks · {row['n_attempts']} attempts · ${row['cost']}"
              + (f" · {d['deprecated_skipped']} attempts were QA-rejected"
                 if d.get('deprecated_skipped') else ""))
    if not args.apply:
        print("\nDRY RUN — nothing removed. Re-run with --apply --reason \"…\".")
        return

    s3sync._load_env_once()
    bucket = s3sync._bucket()
    purged_ids = []
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    for path, m in items:
        rid, key = m["run_id"], m["storage_key"]
        # 1. database — ON DELETE CASCADE clears run_tasks, attempts, run_attempts, llm_calls
        db.execute("DELETE FROM runs WHERE run_id=%s", (rid,))
        # 2. S3
        if not args.keep_s3:
            subprocess.run(["aws", "s3", "rm", f"s3://{bucket}/{key}/", "--recursive",
                            "--only-show-errors"], check=False)
        # 3. local files, last: the only part that cannot be rebuilt
        shutil.rmtree(path, ignore_errors=True)
        rec = describe(m)
        rec.update({"purged_at": ids.iso(ids.utc_now()), "reason": args.reason,
                    "counts": {k: str(v) for k, v in counts[rid].items()}})
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  purged {key}")
        purged_ids.append(rid)
        done += 1
    # tasks is a shared dimension; drop rows no live run references any more
    orphans = db.execute("""DELETE FROM tasks WHERE task_uid NOT IN
                            (SELECT task_uid FROM run_tasks)""")
    registry.rebuild(args.runs_root, relink=True)
    print(f"\npurged {done} run(s); {orphans if orphans is not None else 0} orphan task row(s) "
          f"removed; tombstones appended to {LEDGER}")
    _purge_from_neon(purged_ids)


def _purge_from_neon(run_ids):
    """Delete the same rows from the Neon mirror. LOUD on failure.

    The harness treats a Neon failure as harmless because the run's data is safe
    on disk regardless. The opposite is true here: the data is already gone
    everywhere else, so a Neon write that silently fails leaves the mirror
    serving a run that no longer exists, and nothing downstream can tell. So this
    reports what it did, and says plainly what to run if it could not.
    """
    if not run_ids:
        return
    from core import neonsync
    if not neonsync.enabled():
        print("! Neon is not configured (SEQK_NEON_URL unset, or SEQK_NEON_SYNC=0).\n"
              "  These runs are still live in any Neon mirror you have. Purge them there with:\n"
              f"      DELETE FROM runs WHERE run_id IN ({', '.join(repr(r) for r in run_ids)});")
        return
    url = os.environ["SEQK_NEON_URL"]
    try:
        with db.using(url):
            for rid in run_ids:
                db.execute("DELETE FROM runs WHERE run_id=%s", (rid,))
            db.execute("""DELETE FROM tasks WHERE task_uid NOT IN
                          (SELECT task_uid FROM run_tasks)""")
            left = db.query("SELECT count(*) c FROM runs WHERE run_id = ANY(%s)",
                            ([str(r) for r in run_ids],))
        still = left[0]["c"] if left else 0
        if still:
            raise RuntimeError(f"{still} of {len(run_ids)} rows survived the delete")
        print(f"→ neon: purged {len(run_ids)} run(s) from the mirror too")
    except Exception as exc:                                   # noqa: BLE001
        print(f"\n!! NEON PURGE FAILED ({type(exc).__name__}: {exc}).\n"
              f"   The runs are gone from disk and from the local database, but they are STILL LIVE\n"
              f"   in Neon — anything querying the mirror will keep returning them. Fix it with:\n"
              f"       python scripts/storage/sync_to_neon.py --apply\n"
              f"   or delete them directly:\n"
              f"       DELETE FROM runs WHERE run_id IN ({', '.join(repr(str(r)) for r in run_ids)});",
              file=sys.stderr)


if __name__ == "__main__":
    main()
