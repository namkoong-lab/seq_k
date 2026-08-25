"""Mirror one finished run into Neon, at the end of the run.

WHY NOT JUST POINT DATABASE_URL AT NEON. The harness writes each task as it
finishes — roughly five round trips per task. Against Neon that put ~40 minutes
of connected time on a full corpus load and billed CPU for all of it, which is
why the working database is local. But then Neon only moved when someone
remembered to run push_to_neon.sh, and it silently fell behind.

So: local during the run (fast), one Neon write at the end (cheap). A 30-task
run is ~30 batched statements over a single connection — seconds, not minutes.

Failure is NEVER fatal. The run's artifacts are already on disk and in S3, and
the local database already has it; Neon is a mirror. A network blip must not
lose a finished run, so this warns and returns.

Opt out with SEQK_NEON_SYNC=0, or by leaving SEQK_NEON_URL unset.
"""

from __future__ import annotations

import os
import sys

from core import db


def enabled():
    if os.environ.get("SEQK_NEON_SYNC", "").strip() in ("0", "false", "no"):
        return False
    db.dsn()                       # ensures .env is loaded
    return bool(os.environ.get("SEQK_NEON_URL"))


def push_run(run_id, run_path, manifest, *, ident, k, seq):
    """Replay one run's rows into Neon. Returns True on success."""
    if not enabled():
        return False
    from core import rows
    url = os.environ["SEQK_NEON_URL"]
    try:
        with db.using(url):
            if not db.upsert_run(manifest, run_path=None):   # run_path=None: never spool
                raise RuntimeError("upsert_run rejected the run row")
            n = 0
            for idx in rows.iter_task_indices(run_path):
                meta = _task_meta(run_path, idx)
                rows.mirror_task(run_id, run_path, idx, ident=ident, k=k, seq=seq,
                                 task_id=meta.get("task_id"), prompt=meta.get("prompt"),
                                 storage_key=manifest["storage_key"])
                n += 1
            db.finish_run(run_id, status=manifest.get("status", "unknown"),
                          finished_at=manifest.get("finished_at"), run_path=None)
        print(f"→ neon: mirrored {n} task(s) for {manifest['storage_key']}")
        return True
    except Exception as exc:                                  # noqa: BLE001
        print(f"⚠ neon mirror failed ({type(exc).__name__}: {exc}).\n"
              f"   The run is safe — it is on disk, in S3 and in the local database.\n"
              f"   Catch Neon up with: bash scripts/storage/push_to_neon.sh --apply",
              file=sys.stderr)
        return False


def _task_meta(run_path, idx):
    import json
    from pathlib import Path
    p = Path(run_path) / f"task-{idx}" / "task_meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
