"""Check every run on disk against the format the rest of the system assumes.

    python scripts/validate.py                 # everything
    python scripts/validate.py --slice mbpppro
    python scripts/validate.py --db            # also check disk agrees with Postgres
    python scripts/validate.py --s3            # also check S3 holds the same manifests
    python scripts/validate.py --strict        # exit 1 on any WARN as well as any FAIL

FAIL  breaks queries or refuses to load.  WARN  unusual but legal.

--s3 compares CONTENT, not names. Comparing the set of storage keys on both
sides only answers "does a directory with this name exist in both places?" — it
cannot see that the file inside differs. That gap hid 167 stale manifests for
days: `created_at` was backfilled on disk and never re-uploaded, so S3 held
manifests that would fail the NOT NULL constraint and reject an entire load,
while every key-set check reported 0 missing and 0 orphans.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import ids, registry  # noqa: E402

MANIFEST_REQUIRED = ("run_id", "fingerprint", "fingerprint_version", "created_at",
                     "status", "storage_key", "config")
CONFIG_REQUIRED = ("benchmark", "slice_key", "metric", "k", "model", "judge_model",
                   "critic_model", "feedback_mode", "context", "prompt_variant",
                   "temperature", "seed", "reasoning_effort", "output_budget",
                   "summarizer_model")
ATTEMPT_REQUIRED = ("task_id", "task_index", "metric", "feedback_mode",
                    "attempt_index", "actor", "judge")
ACTOR_REQUIRED = ("model", "output")
METRICS = ("seq@k", "pass@k")
CONTEXTS = ("na", "full", "summary")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class Report:
    def __init__(self):
        self.fail, self.warn, self.runs, self.attempts = [], [], 0, 0

    def f(self, key, msg):
        self.fail.append((key, msg))

    def w(self, key, msg):
        self.warn.append((key, msg))


def check_run(path, m, rep, deep=True):
    key = m.get("storage_key") or path
    rep.runs += 1

    for f in MANIFEST_REQUIRED:
        if m.get(f) in (None, ""):
            rep.f(key, f"manifest missing `{f}`"
                       + (" (runs.created_at is NOT NULL — this rejects the whole load)"
                          if f == "created_at" else ""))
    cfg = m.get("config") or {}
    for f in CONFIG_REQUIRED:
        if f not in cfg:
            rep.f(key, f"config missing `{f}`")

    if cfg.get("metric") not in METRICS:
        rep.f(key, f"metric {cfg.get('metric')!r} is not one of {METRICS}")
    if cfg.get("metric") == "pass@k":
        if cfg.get("context") is not None:
            rep.f(key, f"pass@k must have an EMPTY context, got {cfg.get('context')!r}")
    elif cfg.get("context") not in ("full", "summary"):
        rep.f(key, f"seq@k context {cfg.get('context')!r} is not one of ('full','summary')")
    if cfg.get("metric") == "pass@k" and cfg.get("feedback_mode") is not None:
        rep.f(key, f"pass@k must have an EMPTY feedback_mode, got "
                   f"{cfg.get('feedback_mode')!r} — it never calls feedback()")
    if cfg.get("metric") == "seq@k" and not cfg.get("feedback_mode"):
        rep.f(key, "seq@k must name a feedback_mode")
    # A critic is recorded only where one can run: an LLM-critic mode on seq@k.
    # Changing feedback_mode without recomputing this is how a renamed run ended
    # up claiming a critic mode with no critic model.
    if cfg.get("metric") == "pass@k" and cfg.get("critic_model") is not None:
        rep.f(key, "pass@k must have critic_model null — the critic never runs")
    if not m.get("k_target") and not cfg.get("k"):
        rep.f(key, "neither k_target nor config.k is set; runs.k is NOT NULL")

    rid = m.get("run_id") or ""
    if not _UUID.match(rid):
        rep.f(key, f"run_id {rid!r} is not a uuid")
    want_key = ids.storage_key(cfg.get("slice_key") or "", rid)
    if m.get("storage_key") != want_key:
        rep.f(key, f"storage_key is {m.get('storage_key')!r}, should be {want_key!r}")
    if os.path.basename(os.path.normpath(path)) != rid:
        rep.f(key, "directory name does not match run_id")

    for f in ("model", "judge_model", "critic_model", "summarizer_model"):
        v = cfg.get(f)
        if v and ids.canonical_model(v) != v:
            rep.f(key, f"{f} {v!r} is not canonical — should be "
                       f"{ids.canonical_model(v)!r} (see core.ids.canonical_model)")

    if not deep:
        return

    tdirs = sorted(glob.glob(os.path.join(path, "task-*")),
                   key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    if not tdirs:
        rep.f(key, "no task directories at all")
    seen_idx, seen_ids = set(), set()
    for td in tdirs:
        name = os.path.basename(td)
        mt = re.fullmatch(r"task-(\d+)", name)
        if not mt:
            rep.f(key, f"{name}: directory name must be task-<int>")
            continue
        idx = int(mt.group(1))
        if idx in seen_idx:
            rep.f(key, f"{name}: duplicate task index")
        seen_idx.add(idx)

        meta_p = os.path.join(td, "task_meta.json")
        if not os.path.exists(meta_p):
            rep.f(key, f"{name}: no task_meta.json")
        else:
            try:
                meta = json.loads(Path(meta_p).read_text(encoding="utf-8"))
            except ValueError:
                rep.f(key, f"{name}: task_meta.json is not valid JSON")
                meta = {}
            if meta.get("task_index") != idx:
                rep.f(key, f"{name}: task_meta.task_index={meta.get('task_index')} "
                           f"disagrees with the directory")
            tid = meta.get("task_id")
            if not tid:
                rep.f(key, f"{name}: task_meta has no task_id")
            elif tid in seen_ids:
                rep.f(key, f"{name}: task_id {tid} already used by another task in this run "
                           "(attempts of one task were split across task dirs)")
            else:
                seen_ids.add(tid)

        nums = sorted(int(re.search(r"attempt-(\d+)", f).group(1))
                      for f in glob.glob(os.path.join(td, "attempt-*.json")))
        if not nums:
            rep.f(key, f"{name}: no attempt files")
        elif nums != list(range(1, len(nums) + 1)):
            missing = [i for i in range(1, max(nums) + 1) if i not in nums]
            rep.f(key, f"{name}: attempts must be 1..n with no gaps; missing {missing}")
        kt = m.get("k_target") or cfg.get("k") or 0
        if kt and len(nums) > kt and not (m.get("code") or {}).get("k_overrun_accepted"):
            # WARN, not FAIL: nothing downstream breaks — a metric at k just reads
            # the first k attempts. Rewriting k to match would misreport the config
            # the run was launched with, so flag it and let a human decide which
            # number is the truth.
            #
            # The two imported ARC-AGI-2 runs that trip this did NOT declare
            # max_rounds=3, contrary to what this comment used to claim: their
            # import_notes are null, and every max_rounds path in
            # import_hf_configless.py records a note. Their k came from the highest
            # seq@N key in trajectories_metrics.json. So the summary and the
            # artifacts disagree and neither is established as the config — which is
            # exactly why this stays a WARN.
            #
            # Once a human HAS decided, `code.k_overrun_accepted` records the
            # decision on the run and silences this — the point of the warning is to
            # get a ruling, not to re-ask for one already given.
            rep.w(key, f"{name}: {len(nums)} attempts exceeds the declared k={kt} "
                       "(upstream overrun, or k is recorded wrong)")

        for af in sorted(glob.glob(os.path.join(td, "attempt-*.json"))):
            rep.attempts += 1
            try:
                a = json.loads(Path(af).read_text(encoding="utf-8"))
            except ValueError:
                rep.f(key, f"{name}/{os.path.basename(af)}: not valid JSON")
                continue
            for f in ATTEMPT_REQUIRED:
                if f not in a:
                    rep.f(key, f"{name}/{os.path.basename(af)}: missing `{f}`")
            for f in ACTOR_REQUIRED:
                if f not in (a.get("actor") or {}):
                    rep.f(key, f"{name}/{os.path.basename(af)}: actor missing `{f}`")
            want = int(re.search(r"attempt-(\d+)", af).group(1))
            if a.get("attempt_index") != want:
                rep.f(key, f"{name}/{os.path.basename(af)}: attempt_index="
                           f"{a.get('attempt_index')} disagrees with the filename")
            if a.get("reconstructed") and (a.get("actor") or {}).get("input_tokens") is not None:
                rep.w(key, f"{name}/{os.path.basename(af)}: reconstructed attempt has token "
                           "counts — those cannot be known")
            # A CLAIMED attempt (core/rejudge.py): the actor output belongs to
            # another run and this one paid only for the judge. The provenance is
            # load-bearing, not decorative — core/rows.py reads it to attribute the
            # artifact to its source and to suppress the actor cost row, so an
            # incomplete block would silently re-bill a re-judge for generations it
            # never made, and a rebuild would relabel it as freshly generated.
            ru = a.get("reused_from")
            if ru is not None:
                if not isinstance(ru, dict) or not ru.get("run_id") or not ru.get("output_key"):
                    rep.f(key, f"{name}/{os.path.basename(af)}: reused_from must carry "
                               "`run_id` and `output_key` naming the generating run")
                elif ru.get("run_id") == m.get("run_id"):
                    rep.f(key, f"{name}/{os.path.basename(af)}: reused_from names THIS run "
                               "— a run cannot claim its own generation as reused")
                if (a.get("judge") or {}).get("model") is None:
                    rep.w(key, f"{name}/{os.path.basename(af)}: claimed attempt has no judge "
                               "model — a re-judge exists to record a new verdict")


def check_s3(storage_keys, rep, *, workers=12):
    """Compare each run's manifest on disk with the copy published to S3.

    Only the manifest: it is small, it carries the run's identity, and it is what
    gates a load. Checking all 76k attempt files would cost far more and catch
    far less — a stale manifest is what breaks a rebuild-from-S3, because
    `runs.created_at` is NOT NULL and one undated run rejects the whole load.

    Anything that rewrites a manifest without re-uploading it produces this
    divergence: a restamp, a re-key, a timestamp backfill. The
    local side stays right and the published side silently rots.
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    from core import s3sync
    s3sync._load_env_once()
    bucket = s3sync._bucket()

    # An expired SSO token makes EVERY fetch fail. Reporting that as 447 runs
    # "not in S3" is worse than useless — it looks like catastrophic data loss
    # and sends you hunting for a problem that is a login. Tell the two apart.
    _AUTH = ("expired", "credential", "sso", "AccessDenied", "InvalidAccessKeyId",
             "Unable to locate credentials", "TokenRefreshRequired")

    def fetch(key):
        r = subprocess.run(["aws", "s3", "cp", f"s3://{bucket}/{key}/manifest.json", "-"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            err = (r.stderr or "")
            if any(t.lower() in err.lower() for t in _AUTH):
                return key, None, ("__ABORT__", err.strip().splitlines()[-1][:120])
            return key, None, "not in S3 — this run has never been uploaded, or was uploaded elsewhere"
        try:
            return key, json.loads(r.stdout), None
        except ValueError:
            return key, None, "manifest.json in S3 is not valid JSON"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fetch, storage_keys))

    aborts = [(k, e[1]) for k, _r, e in results if isinstance(e, tuple) and e[0] == "__ABORT__"]
    if aborts:
        print(f"S3 CHECK ABORTED — could not authenticate to s3://{bucket}\n"
              f"  {aborts[0][1]}\n"
              f"  ({len(aborts)} of {len(results)} fetches failed this way.)\n"
              f"  This says nothing about whether the runs are in S3. Re-auth and retry:\n"
              f"      aws sso login --profile seqk\n", file=sys.stderr)
        sys.exit(2)

    for key, remote, err in results:
        if err:
            rep.f(key, err)
            continue
        local_p = Path("runs") / key / "manifest.json"
        try:
            local = json.loads(local_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue                      # the on-disk checks already reported this
        if not remote.get("created_at"):
            rep.f(key, "S3 manifest has no `created_at` — a rebuild from S3 would fail the "
                       "NOT NULL constraint and reject EVERY run, not just this one. "
                       "Re-upload it: python -m core upload runs/" + key)
        if remote.get("fingerprint") != local.get("fingerprint"):
            rep.f(key, f"S3 manifest is a DIFFERENT run: fingerprint "
                       f"{str(remote.get('fingerprint'))[:22]}… in S3 vs "
                       f"{str(local.get('fingerprint'))[:22]}… locally")
        if remote.get("storage_key") != local.get("storage_key"):
            rep.f(key, f"S3 manifest says storage_key={remote.get('storage_key')!r}, "
                       f"local says {local.get('storage_key')!r}")
        for field in ("status", "finished_at", "created_at"):
            if remote.get(field) != local.get(field):
                rep.w(key, f"S3 manifest `{field}` is {remote.get(field)!r}, "
                           f"local is {local.get(field)!r} — one side is behind")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--slice", help="only runs in this slice")
    ap.add_argument("--shallow", action="store_true", help="manifest checks only")
    ap.add_argument("--db", action="store_true", help="also compare disk against Postgres")
    ap.add_argument("--s3", action="store_true",
                    help="also compare each run's manifest against the copy in S3")
    ap.add_argument("--s3-workers", type=int, default=12,
                    help="parallel S3 fetches for --s3 (default 12)")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    args = ap.parse_args()

    rep = Report()
    keys = set()
    for path, m in registry.iter_manifests(args.runs_root):
        if args.slice and (m.get("config") or {}).get("slice_key") != args.slice:
            continue
        keys.add(m.get("storage_key"))
        check_run(path, m, rep, deep=not args.shallow)

    if args.s3:
        check_s3(sorted(keys), rep, workers=args.s3_workers)

    if args.db:
        from core import db
        # Honour --slice on BOTH sides. Comparing the whole database against a
        # filtered disk scan reports every run outside the slice as a stale row
        # — 387 false failures the first time this ran.
        if args.slice:
            dbk = {r["storage_key"] for r in db.query(
                "SELECT storage_key FROM runs WHERE slice_key = %s", (args.slice,))}
        else:
            dbk = {r["storage_key"] for r in db.query("SELECT storage_key FROM runs")}
        for k in sorted(keys - dbk):
            rep.f(k, "on disk but not in the database — run db_sync.py --rebuild")
        for k in sorted(dbk - keys):
            rep.f(k, "in the database but not on disk — stale row, run db_sync.py --reset")

    print(f"checked {rep.runs} run(s), {rep.attempts} attempt file(s)\n")
    for lvl, items in (("FAIL", rep.fail), ("WARN", rep.warn)):
        if not items:
            continue
        print(f"{lvl} ({len(items)}):")
        shown = items[:60]
        # Pad to the widest key, never truncate. A storage_key is up to 56 chars
        # (`healthbench-default/<uuid>`); clipping it to 52 printed something that
        # still LOOKED like a uuid but resolved to nothing, so the id in the
        # warning could not be pasted into any of the commands it suggests.
        width = max((len(k) for k, _ in shown), default=0)
        for k, msg in shown:
            print(f"  {k:{width}} {msg}")
        if len(items) > 60:
            print(f"  … and {len(items)-60} more")
        print()
    if not rep.fail and not rep.warn:
        print("ALL CLEAN — every run matches the expected format.")
    sys.exit(1 if rep.fail or (args.strict and rep.warn) else 0)


if __name__ == "__main__":
    main()
