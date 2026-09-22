"""Local run registry: fingerprint -> storage key. The authority for resume.

The database is a mirror, never a gate. A grid that has already spent hundreds
of dollars must not die because Neon cold-started or the wifi dropped, so the
only thing on a run's critical path is this file plus the manifests it indexes:

    <runs_root>/.registry.json      fingerprint -> {run_id, storage_key, created_at}
    <runs_root>/<key>/manifest.json the same facts, per run, self-describing

The index is a CACHE. Every fact in it is duplicated in a manifest inside the
run directory, so `rebuild()` reconstructs it from the filesystem alone (and
`scripts/storage/db_sync.py --from-s3` from the bucket alone). Losing it costs a rescan,
never data.

Concurrency: scripts/run_grid.py launches many runs at once, and two processes
computing the same fingerprint would otherwise both mint a UUID and fork the
experiment in half. Every read-modify-write here happens under an exclusive
flock on <runs_root>/.registry.lock.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from pathlib import Path

from core import ids, results

INDEX_NAME = ".registry.json"
LOCK_NAME = ".registry.lock"
MANIFEST_NAME = "manifest.json"
BY_LABEL = "by-label"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def resolve(runs_root, candidates, *, options, code=None,
            k_target=None, continue_run=False, scope=None):
    """Find or create the run. Returns (run_path, manifest, created).

    `candidates` is core.ids.candidates(...): [(version, fingerprint, identity)]
    newest first. Each is tried in order and the first that already exists wins,
    so a FINGERPRINT_VERSION bump never orphans a finished run. A new run is
    always created at the NEWEST version.

    Idempotent: calling twice with the same config returns the same directory,
    which is what makes `python -m core run <same yaml>` resume rather than
    start over.
    """
    version, fp, ident = candidates[0]
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)

    with _lock(root):
        index = _read_index(root)
        for v, cand_fp, _cand_ident in candidates:
            entry = index.get(_key(cand_fp, v))
            if entry is None:
                continue
            run_path = root / entry["storage_key"]
            manifest = _read_manifest(run_path)
            if manifest is None:
                # Indexed but the directory is gone (moved or deleted by hand).
                print(f"! registry: {entry['storage_key']} is indexed but missing on disk; "
                      f"starting a fresh run for this config")
                continue
            if v != version:
                print(f"! resuming a run recorded under fingerprint v{v} "
                      f"(current is v{version}) — identity only relaxed since, so this "
                      f"is the same experiment")
            changed = False
            if k_target is not None and k_target > (manifest.get("k_target") or 0):
                # Horizon-free run being extended to a larger k: same identity,
                # more attempts. Record the new ceiling.
                manifest["k_target"] = k_target
                changed = True
            if scope is not None:
                # WIDEST scope ever asked for, not the latest. A resume over a
                # subset (`task_indices: [3]`, to retry one stuck task) must not
                # rewrite a 30-task run's scope down to 1 and declare it finished.
                #
                # The FIRST time scope is recorded on a run that predates the
                # field, there is no stored scope to widen against, and a narrow
                # resume would write itself down as the whole intent — a 300-task
                # run topped up on six tasks would read `tasks_requested: 6`.
                # Seed from the task directories already on disk instead: they are
                # the only evidence left of what the run has covered.
                prior = manifest.get("scope") or _scope_from_disk(run_path)
                merged = _widen_scope(prior, scope)
                if merged != manifest.get("scope"):
                    manifest["scope"] = merged
                    changed = True
            if changed:
                manifest = write_manifest(str(run_path), manifest)
            resumed = manifest
            break
        else:
            resumed = None

        if resumed is not None:
            # k=05 and k=10 can share a run (horizon-free) but have different
            # readable names, so link THIS invocation's label too. Both names
            # then resolve to the one directory. Derived from the identity being
            # asked for, not from the resumed manifest, which carries its own.
            # Unconditional, and linking is idempotent. A conditional here is
            # wrong twice over: after a k-extension the manifest's own label has
            # already been rewritten to the new k, so "differs from stored" is
            # false exactly when the new link is needed.
            _, asked_label = _derived_labels(ident, k_target)
            _link_label(root, {**resumed, "label_path": asked_label})
            return str(root / resumed["storage_key"]), resumed, False

        _warn_near_misses(root, ident)
        run_id = ids.new_run_id()
        created_at = ids.utc_now()
        storage_key = ids.storage_key(ident["slice_key"], run_id)
        run_path = root / storage_key
        run_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "fingerprint": fp,
            "fingerprint_version": version,
            "created_at": ids.iso(created_at),
            "finished_at": None,
            "status": "running",
            "storage_key": storage_key,
            "config": dict(ident),
            "options": dict(options or {}),
            "k_target": k_target,
            "scope": dict(scope) if scope else None,
            "code": dict(code or {}),
            "rollup": {},
        }
        manifest = write_manifest(str(run_path), manifest)
        index[_key(fp, version)] = {"run_id": run_id, "storage_key": storage_key,
                                    "created_at": manifest["created_at"]}
        _write_index(root, index)

    _link_label(root, manifest)
    return str(run_path), manifest, True


def exists(runs_root, candidates):
    """Whether any candidate fingerprint already has a run on disk. NO side effects.

    Deliberately separate from resolve(), which creates. The auto-seed rule in
    core/harness.py has to know "would this be a NEW run?" BEFORE deciding to
    stamp a seed on it — seed is part of the fingerprint, so seeding a config
    that already has an unseeded run would fork it instead of resuming, and pay
    for its finished attempts a second time.
    """
    root = Path(runs_root)
    index = _read_index(root)
    for v, fp, _ident in candidates:
        entry = index.get(_key(fp, v))
        if entry is not None and _read_manifest(root / entry["storage_key"]) is not None:
            return True
    return False


def _scope_from_disk(run_path):
    """A scope record inferred from the task directories that exist.

    Used ONLY to seed runs made before `scope` was recorded, and it is a floor,
    not a claim: it says what the run demonstrably covered, never what it was
    asked to cover. A run abandoned at task 3 of 30 still infers 3 — the
    information to say otherwise was never written down, and inventing it would
    be worse than admitting the gap.
    """
    import re
    try:
        idx = sorted(int(m.group(1)) for m in
                     (re.fullmatch(r"task-(\d+)", e) for e in os.listdir(run_path)) if m)
    except OSError:
        return None
    if not idx:
        return None
    return {"max_tasks": None, "task_indices": None,
            "tasks_requested": len(idx), "indices_requested": idx,
            "inferred_from_disk": True}


def _widen_scope(old, new):
    """Union of two scope records — what the run has EVER been asked to cover.

    Scope answers "was this run finished?", so it has to accumulate. A resume
    narrowed to one task is still a resume of the whole run; taking the latest
    scope verbatim would shrink `tasks_requested` to 1 and turn an abandoned
    30-task run into a complete 1-task one, which is the exact confusion the
    field exists to end.
    """
    if not old:
        return dict(new)
    idx = sorted(set(old.get("indices_requested") or []) |
                 set(new.get("indices_requested") or []))
    return {
        # Explicit selectors describe ONE invocation, so the merged record keeps
        # them only while both invocations agreed; otherwise the resolved index
        # list below is the honest answer and these would just contradict it.
        "max_tasks": old.get("max_tasks") if old.get("max_tasks") == new.get("max_tasks") else None,
        "task_indices": (old.get("task_indices")
                         if old.get("task_indices") == new.get("task_indices") else None),
        "tasks_requested": max(len(idx), old.get("tasks_requested") or 0,
                               new.get("tasks_requested") or 0),
        "indices_requested": idx,
        # Sticky: once any part of a scope was inferred rather than declared, the
        # whole record is a floor and must keep saying so.
        **({"inferred_from_disk": True} if old.get("inferred_from_disk") else {}),
    }


def _derived_labels(cfg, k_target):
    """(labels, label_path) — both PURE FUNCTIONS of the identity columns."""
    k = k_target or cfg.get("k")
    labels = {"slice": cfg["slice_key"], "metric": cfg["metric"], "k": k,
              "agent": cfg["model"], "judge": cfg["judge_model"],
              "fb": cfg["feedback_mode"], "critic": cfg["critic_model"],
              "context": cfg["context"], "prompt": cfg["prompt_variant"],
              "temp": cfg["temperature"], "seed": cfg["seed"],
              "reason": cfg["reasoning_effort"]}
    label_path = results.label_from_fields(
        slice_key=cfg["slice_key"], metric=cfg["metric"], k=k, model=cfg["model"],
        judge_model=cfg["judge_model"], critic_model=cfg["critic_model"],
        feedback_mode=cfg["feedback_mode"], context=cfg["context"],
        prompt_variant=cfg["prompt_variant"], temperature=cfg["temperature"],
        seed=cfg["seed"], reasoning_effort=cfg["reasoning_effort"])
    return labels, label_path


def write_manifest(run_path, manifest):
    """Write manifest.json atomically. Called on create and on every update.

    `labels` and `label_path` are RECOMPUTED here rather than taken from the
    caller. Both are pure functions of `config` (plus k_target), so every writer
    that changed identity — the importer, core/rejudge.py, a k-extension in
    resolve(), each migration script — had to remember to refresh them, and none
    did: all 439 label_paths and 166 labels blocks had drifted, six of them
    naming the wrong benchmark slice. Deriving on write is the only way they
    cannot go stale, and it is why db/schema.sql refuses to store label_path as
    a column at all.

    Returns the manifest AS WRITTEN, derived fields included — callers need the
    real thing (to link `by-label/`, to mirror to Neon), not the copy they
    passed in.
    """
    m = dict(manifest)
    if m.get("config"):
        m["labels"], m["label_path"] = _derived_labels(m["config"], m.get("k_target"))
    _atomic_write(Path(run_path) / MANIFEST_NAME, _dumps(m))
    return m


def read_manifest(run_path):
    return _read_manifest(Path(run_path))


def update_manifest(run_path, **fields):
    """Merge `fields` into the manifest on disk. Returns the new manifest.

    Read-modify-write under an exclusive lock on <run>/.manifest.lock. One run
    can be worked on by several processes at once — shards of one config over
    disjoint `task_indices` resolve to the same run — and without the lock the
    last writer wins the whole file, silently dropping whatever the others had
    just merged in (observed: a shard's scope entry vanishing under another's)."""
    import fcntl
    with open(Path(run_path) / ".manifest.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        m = _read_manifest(Path(run_path))
        if m is None:
            return None
        m.update(fields)
        return write_manifest(run_path, m)


def register_existing(runs_root, run_path, manifest):
    """Add an already-materialised run to the index (used by the migration)."""
    root = Path(runs_root)
    with _lock(root):
        index = _read_index(root)
        index[_key(manifest["fingerprint"], manifest.get("fingerprint_version"))] = {
            "run_id": manifest["run_id"],
            "storage_key": manifest["storage_key"],
            "created_at": manifest["created_at"],
        }
        _write_index(root, index)
    _link_label(root, manifest)


def iter_manifests(runs_root):
    """Every manifest under runs_root, cheaply: walk only the date levels."""
    root = Path(runs_root)
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        # never descend into run internals or the symlink view
        dirnames[:] = [d for d in dirnames
                       if not d.startswith("task-") and d != BY_LABEL and d != "_harbor_jobs"]
        if MANIFEST_NAME in filenames:
            dirnames[:] = []
            m = _read_manifest(Path(dirpath))
            if m is not None:
                yield dirpath, m


def rebuild(runs_root, *, relink=False):
    """Rebuild .registry.json from the manifests on disk. Never deletes runs."""
    root = Path(runs_root)
    index, dupes = {}, []
    for dirpath, m in iter_manifests(root):
        key = _key(m["fingerprint"], m.get("fingerprint_version"))
        if key in index and index[key]["storage_key"] != m["storage_key"]:
            dupes.append((key, index[key]["storage_key"], m["storage_key"]))
            # keep the earliest, so rebuild is deterministic regardless of walk order
            if m["created_at"] >= index[key]["created_at"]:
                continue
        index[key] = {"run_id": m["run_id"], "storage_key": m["storage_key"],
                      "created_at": m["created_at"]}
        if relink:
            _link_label(root, m)
    with _lock(root):
        _write_index(root, index)
    return index, dupes


def relink_all(runs_root):
    """Regenerate the whole by-label/ symlink view.

    Prunes first. Anything that changes a run's storage key leaves every old
    symlink dangling while adding new ones beside them — one such pass left 128 links
    of which 71 pointed at paths that no longer existed. Rebuilding from scratch
    is the only way to be sure the view matches the manifests.
    """
    root = Path(runs_root)
    view = root / BY_LABEL
    if view.exists():
        shutil.rmtree(view, ignore_errors=True)
    n = 0
    for _dirpath, m in iter_manifests(root):
        if _link_label(root, m):
            n += 1
    return n


def _warn_near_misses(root, ident):
    """Before starting a NEW run, warn if an existing one differs in exactly one
    field.

    The expensive mistake this catches: runs migrated from the old layout are
    tagged `prompt=legacy`, while a fresh run defaults to `prompt=v1`. Re-running
    an old YAML is then a different fingerprint, so instead of resuming ~$40 of
    completed work it silently starts over. One differing field is almost always
    a typo or a defaulting surprise; two or more is a deliberate ablation.

    Only runs on creation (rare), so the manifest scan costs nothing in the
    steady state.
    """
    near = []
    for _dirpath, m in iter_manifests(root):
        other = m.get("config") or {}
        if set(other) != set(ident):
            continue
        diff = [f for f in ident if other.get(f) != ident[f]]
        if len(diff) == 1:
            f = diff[0]
            near.append((f, other.get(f), m.get("label_path") or m["storage_key"],
                         (m.get("rollup") or {}).get("attempts_total", 0)))
    for field, val, where, attempts in near[:3]:
        print(f"! starting a NEW run, but an existing one differs only in {field}: "
              f"{val!r} vs {ident[field]!r}")
        print(f"    existing: {where}  ({attempts} attempts already done)")
        print(f"    if you meant to resume it, set {field}={val!r} in the config.")


# --------------------------------------------------------------------------- #
# by-label/ — the human-readable view
# --------------------------------------------------------------------------- #
def _link_label(root, manifest):
    """Symlink <runs_root>/by-label/<v2 path> -> the uuid directory.

    Derived state, regenerable, gitignored. It exists so `ls`, `grep -r` and tab
    completion keep working after the move to opaque keys. Created on every run
    init so it can never go stale by forgetting to run a command.
    """
    rel = manifest.get("label_path")
    if not rel:
        return False
    link = root / BY_LABEL / rel
    target = root / manifest["storage_key"]
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and os.path.realpath(link) == os.path.realpath(target):
            return False
        if link.is_symlink() or link.exists():
            # Two runs can share a label: `output_budget` and `summarizer_model`
            # are in the fingerprint but not in the v2 name. Disambiguate rather
            # than clobber — silently pointing one experiment's label at
            # another's data is far worse than an ugly suffix.
            link = link.with_name(link.name + "~" + manifest["run_id"].replace("-", "")[:8])
            if link.is_symlink():
                if os.path.realpath(link) == os.path.realpath(target):
                    return False
                link.unlink()
            elif link.exists():
                return False
        os.symlink(os.path.relpath(target, link.parent), link)
        return True
    except OSError as exc:
        # A broken symlink view must never fail a run.
        print(f"! registry: could not link {link}: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _key(fingerprint, version=None):
    return f"{fingerprint}@v{ids.FINGERPRINT_VERSION if version is None else version}"


def _read_index(root):
    path = root / INDEX_NAME
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print(f"! registry: {path} unreadable ({exc}); rebuilding from manifests")
        return {k: v for k, v in
                ((_key(m["fingerprint"], m.get("fingerprint_version")),
                  {"run_id": m["run_id"], "storage_key": m["storage_key"],
                   "created_at": m["created_at"]})
                 for _d, m in iter_manifests(root))}
    return doc.get("runs", {})


def _write_index(root, index):
    _atomic_write(root / INDEX_NAME, _dumps({"version": 1, "runs": index}))


def _read_manifest(run_path):
    path = Path(run_path) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _dumps(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n"


class _lock:
    """Exclusive advisory lock over the registry index."""

    def __init__(self, root):
        self.path = Path(root) / LOCK_NAME
        self.fd = None

    def __enter__(self):
        try:
            import fcntl
        except ImportError:                      # pragma: no cover - non-POSIX
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EROFS, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
            self.fd = None                       # unlockable fs: proceed unlocked
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None
        return False
