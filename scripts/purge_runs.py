"""Permanently remove runs from disk, S3 and the database.

    python scripts/purge_runs.py <run_id> ...                    # dry run
    python scripts/purge_runs.py --from-file ids.txt --apply --reason "..."

The only destructive tool here. Use `runs.py replace` instead when a better run
supersedes an old one — that keeps the files. Purge is for runs that should not
exist at all: empty shells, imports broken beyond repair, runs whose attempts
were almost entirely rejected by QA.

Appends a tombstone to `purged.jsonl` (identity, counts, source, reason) so
a missing cell stays explainable. Deletes DB first, then S3, then local disk —
an interruption leaves the files, which are the only irreplaceable part.
"""

from __future__ import annotations

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="*")
    ap.add_argument("--from-file")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--reason", default="", help="recorded in the tombstone ledger")
    ap.add_argument("--keep-s3", action="store_true", help="leave the S3 objects in place")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    prefixes = list(args.run_ids)
    if args.from_file:
        prefixes += [l.split("#")[0].strip()
                     for l in Path(args.from_file).read_text().splitlines() if l.split("#")[0].strip()]
    if not prefixes:
        sys.exit("no run ids given")
    if args.apply and not args.reason:
        sys.exit("--apply requires --reason: a purge with no recorded reason is unexplainable later")

    items = resolve(prefixes, args.runs_root)
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
        done += 1
    # tasks is a shared dimension; drop rows no live run references any more
    orphans = db.execute("""DELETE FROM tasks WHERE task_uid NOT IN
                            (SELECT task_uid FROM run_tasks)""")
    registry.rebuild(args.runs_root, relink=True)
    print(f"\npurged {done} run(s); {orphans if orphans is not None else 0} orphan task row(s) "
          f"removed; tombstones appended to {LEDGER}")


if __name__ == "__main__":
    main()
