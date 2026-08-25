"""Download runs that are in S3 but not local, validate them, load into Postgres.

    python scripts/storage/sync_from_s3.py                 # dry run
    python scripts/storage/sync_from_s3.py --apply
    python scripts/storage/sync_from_s3.py --apply --push  # …then push to Neon
    python scripts/storage/sync_from_s3.py --only <id> --apply    # one run

`core/s3sync.py` only uploads, so this is how a collaborator's run reaches the
shared index. Only ever ADDS: a local run missing from S3 is left alone, since
it may simply not be uploaded yet. Runs that fail validation are downloaded but
not loaded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))          # validate.py lives one level up

from core import db, registry, rows, s3sync  # noqa: E402
from validate import Report, check_run  # noqa: E402


def s3_run_keys(bucket):
    """<slice>/<run_id> for every run in the canonical area of the bucket."""
    r = subprocess.run(["aws", "s3", "ls", f"s3://{bucket}/", "--recursive"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"aws s3 ls failed:\n{r.stderr.strip()}")
    keys = set()
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        p = line.split()[-1]
        if p.startswith("to-be-organized/"):
            continue                     # staging area, imported by its own scripts
        parts = p.split("/")
        if len(parts) >= 3:
            keys.add(f"{parts[0]}/{parts[1]}")
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--push", action="store_true", help="also run push_to_neon.sh --apply")
    ap.add_argument("--limit", type=int, help="stop after N runs (useful for a first try)")
    ap.add_argument("--only", action="append", default=[],
                    help="pull just this run (id, or <slice>/<run_id>). Repeatable. "
                         "Use this to pick up ONE partial run instead of the whole bucket.")
    args = ap.parse_args()

    s3sync._load_env_once()
    bucket = s3sync._bucket()
    local = {m["storage_key"] for _p, m in registry.iter_manifests(args.runs_root)}
    # `local` is what we already have; --only overrides it below so an explicitly
    # named run is re-fetched (that is how you pick up someone else's progress on
    # a run you also hold).
    remote = s3_run_keys(bucket)

    new = sorted(remote - local)
    gone = sorted(local - remote)
    if args.only:
        want = set()
        for sel in args.only:
            hits = [k for k in remote if k == sel or k.endswith("/" + sel) or sel in k]
            if len(hits) != 1:
                sys.exit(f"--only {sel!r} matched {len(hits)} runs in S3; be more specific")
            want.add(hits[0])
        already = want & local
        new = sorted(want)
        gone = []
        refreshed = len(already)
    if args.limit:
        new = new[:args.limit]

    print(f"S3 s3://{bucket}: {len(remote)} run(s) · local: {len(local)}")
    if args.only:
        # --only names runs explicitly, so "new" is the wrong word for one you
        # already hold — it is a refresh, and saying otherwise made the tally
        # contradict the note above it.
        fresh = len(new) - refreshed
        print(f"  to download         : {len(new)}"
              + (f"  ({refreshed} refresh, {fresh} new)" if refreshed else ""))
    else:
        print(f"  new in S3, not here : {len(new)}")
        print(f"  here, not in S3     : {len(gone)}  (not touched — upload with: "
              f"python -m core upload runs/<slice>/<run_id>)")
    for k in new[:20]:
        print(f"     + {k}")
    if len(new) > 20:
        print(f"     … and {len(new)-20} more")
    if not args.apply:
        print("\nDRY RUN — nothing downloaded. Re-run with --apply.")
        return
    if not new:
        print("\nnothing to do.")
        return

    root = Path(args.runs_root)
    loaded, skipped = 0, []
    for key in new:
        dest = root / key
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["aws", "s3", "sync", f"s3://{bucket}/{key}/", str(dest),
                            "--only-show-errors"], capture_output=True, text=True)
        if r.returncode != 0:
            skipped.append((key, f"download failed: {r.stderr.strip()[:80]}"))
            continue
        mp = dest / "manifest.json"
        if not mp.exists():
            skipped.append((key, "no manifest.json — not a run directory"))
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))

        rep = Report()
        check_run(str(dest), m, rep)
        if rep.fail:
            skipped.append((key, f"failed validation: {rep.fail[0][1]}"))
            continue

        cfg = m["config"]
        k = m.get("k_target") or cfg.get("k")
        seq = cfg.get("metric") == "seq@k"
        if not db.upsert_run(m, run_path=str(dest)):
            skipped.append((key, "database rejected the run row"))
            continue
        for idx in rows.iter_task_indices(str(dest)):
            meta_p = dest / f"task-{idx}" / "task_meta.json"
            meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
            rows.mirror_task(m["run_id"], str(dest), idx, ident=cfg, k=k, seq=seq,
                             task_id=meta.get("task_id"), prompt=meta.get("prompt"),
                             storage_key=m["storage_key"])
        db.finish_run(m["run_id"], status=m.get("status", "unknown"),
                      finished_at=m.get("finished_at"), run_path=str(dest))
        print(f"  loaded {key}")
        loaded += 1

    registry.rebuild(args.runs_root, relink=True)
    print(f"\nloaded {loaded} run(s)")
    if skipped:
        print(f"SKIPPED {len(skipped)} — downloaded but NOT loaded:")
        for k, why in skipped:
            print(f"  {k}\n      {why}")
    if args.push and loaded:
        subprocess.run(["bash", "scripts/storage/push_to_neon.sh", "--apply"], check=False)


if __name__ == "__main__":
    main()
