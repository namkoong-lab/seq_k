"""Bring Neon in line with the local database.

    python scripts/storage/sync_to_neon.py                  # show the drift
    python scripts/storage/sync_to_neon.py --apply          # push everything
    python scripts/storage/sync_to_neon.py --only <run_id> --apply   # one run

Run it after finishing runs. Nothing pushes to Neon on its own: the harness
writes locally during a run (~5 round trips per task, which against Neon cost
~40 minutes of billed time on a full load), so Neon moves only when you say so.

`--only` replays one run from disk — seconds. Without it, the whole local
database is shipped in a single pg_dump | psql, which is a REPLACE, not a merge.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core import db, registry, rows  # noqa: E402

COUNTS = {"runs": "SELECT count(*) FROM runs",
          "run_tasks": "SELECT count(*) FROM run_tasks",
          "attempts": "SELECT count(*) FROM attempts",
          "llm_calls": "SELECT count(*) FROM llm_calls"}


def neon_url():
    db.dsn()                                   # loads .env
    url = os.environ.get("SEQK_NEON_URL")
    if not url:
        sys.exit("SEQK_NEON_URL is not set in .env")
    return url


def _scalar(sql):
    """First column of the first row. One round trip — the previous form called
    db.query twice per count (once for the value, once to read the column name),
    which doubled the billed Neon time of a plain drift check."""
    row = db.query(sql)[0]
    return row[next(iter(row))]


def counts(url=None):
    if url is None:
        return {k: _scalar(v) for k, v in COUNTS.items()}
    with db.using(url):
        return {k: _scalar(v) for k, v in COUNTS.items()}


def push_one(run_id_prefix, url, runs_root):
    hits = [(p, m) for p, m in registry.iter_manifests(runs_root)
            if m["run_id"].startswith(run_id_prefix)]
    if len(hits) != 1:
        sys.exit(f"{run_id_prefix!r} matched {len(hits)} runs; be more specific")
    path, m = hits[0]
    cfg = m["config"]
    k = m.get("k_target") or cfg.get("k")
    seq = cfg.get("metric") == "seq@k"
    import json
    with db.using(url):
        if not db.upsert_run(m, run_path=None):
            sys.exit("Neon rejected the run row")
        n = 0
        for idx in rows.iter_task_indices(path):
            mp = Path(path) / f"task-{idx}" / "task_meta.json"
            meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
            rows.mirror_task(m["run_id"], path, idx, ident=cfg, k=k, seq=seq,
                             task_id=meta.get("task_id"), prompt=meta.get("prompt"),
                             storage_key=m["storage_key"])
            n += 1
        db.finish_run(m["run_id"], status=m.get("status", "unknown"),
                      finished_at=m.get("finished_at"), run_path=None)
    print(f"pushed {m['storage_key']} ({n} tasks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--only", help="one run id (or unique prefix) instead of everything")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    url = neon_url()

    loc, rem = counts(), counts(url)
    print(f"{'':12}{'local':>12}{'neon':>12}{'drift':>10}")
    for key in COUNTS:
        d = loc[key] - rem[key]
        print(f"  {key:10}{loc[key]:>12}{rem[key]:>12}{d:>+10}" if d else
              f"  {key:10}{loc[key]:>12}{rem[key]:>12}{'—':>10}")
    if not args.apply:
        print("\nDRY RUN — re-run with --apply"
              + (f" (or --only {args.only} --apply)" if args.only else ""))
        return

    if args.only:
        push_one(args.only, url, args.runs_root)
    else:
        here = Path(__file__).resolve().parent
        subprocess.run(["bash", str(here / "push_to_neon.sh"), "--apply"], check=True)

    rem = counts(url)
    print(f"\nneon now: " + " · ".join(f"{k} {rem[k]}" for k in COUNTS))
    if all(loc[k] == rem[k] for k in COUNTS):
        print("local and Neon match.")
if __name__ == "__main__":
    main()
