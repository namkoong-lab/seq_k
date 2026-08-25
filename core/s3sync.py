"""End-of-run upload of a results folder to S3.

Default: every `python -m core run ...` syncs `runs/<name>/` to
`s3://<bucket>/<name>/` at the end. Three opt-out paths:

    --no-upload         CLI flag on `core run`              (one-off)
    s3_sync: false      key in a variant YAML               (per-variant)
    SEQK_S3_SYNC=0      environment variable                (per-machine)

The bucket name is read from `SEQK_S3_BUCKET`, defaulting to `seq-k`. We shell
out to the `aws` CLI so the user's existing SSO / credential config Just Works.

Failures (missing CLI, expired creds, network, AccessDenied) are loud: this
module RAISES and the run fails. The local files on disk are still the source
of truth — retry with `python -m core upload runs/<name>`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_BUCKET = "seq-k"

# Local bookkeeping, never uploaded: a spool of failed DB writes and the
# registry index, both of which are machine-local and regenerable.
_NOT_UPLOADED = {".db_pending.jsonl", ".registry.json", ".registry.lock"}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def upload_run(out, *, s3_sync=None, runs_root="runs"):
    """Sync `out/` to `s3://<bucket>/<storage key>/`. No-op if disabled.

    Everything is uploaded, manifest.json included. That is deliberate and load
    bearing: the manifest carries the run's identity, labels and provenance, so
    the bucket is self-describing and the whole database can be rebuilt from S3
    alone (`scripts/storage/db_sync.py --rebuild --from-s3`). Only the local bookkeeping
    files — .db_pending.jsonl and the registry index — stay behind."""
    if not _enabled(s3_sync):
        print(f"→ s3 sync skipped (disabled) for {out}")
        return

    out_path = Path(out)
    if not out_path.is_dir():
        raise FileNotFoundError(f"run dir not found: {out_path}")

    bucket = _bucket()
    prefix = _s3_prefix(out_path, runs_root)
    _require_aws_cli()
    # No secret scrub here any more: Harbor's container scratch (the only
    # artifacts that can carry unredacted fixture secrets) lives in
    # `harbor_jobs/<storage_key>/`, OUTSIDE runs/, and is never uploaded. This
    # used to scrub `<run>/_harbor_jobs`, a path that has not existed since the
    # split — a no-op that read like a safeguard. If you ever DO choose to upload
    # Harbor's container scratch, scrub it first: those artifacts can carry
    # unredacted fixture secrets.

    target = f"s3://{bucket}/{prefix}/"
    _warn_if_someone_else_advanced(out_path, bucket, prefix)
    print(f"→ syncing {out_path}/ to {target}")
    _aws_s3_sync(out_path, target)
    _verify_upload(out_path, bucket, prefix)
    print(f"→ done. {target}")


def _warn_if_someone_else_advanced(out_path, bucket, prefix):
    """Warn if S3 holds attempt files this machine does not.

    Two people can pick up the same partial run. `aws s3 sync` is additive so
    different tasks merge cleanly, but on the SAME task the second upload
    overwrites the first. Never blocks — refusing would strand local work too.
    """
    import subprocess
    r = subprocess.run(["aws", "s3", "ls", f"s3://{bucket}/{prefix}/", "--recursive"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return                                    # first upload of this run, or no listing
    remote = {line.split()[-1].split(f"{prefix}/", 1)[-1]
              for line in r.stdout.splitlines() if line.strip()}
    local = {str(p.relative_to(out_path)) for p in Path(out_path).rglob("*") if p.is_file()}
    ahead = sorted(f for f in remote - local if "/attempt-" in f)
    if not ahead:
        return
    print(f"\n⚠ S3 already has {len(ahead)} attempt file(s) this machine does not:")
    for f in ahead[:8]:
        print(f"    {f}")
    if len(ahead) > 8:
        print(f"    … and {len(ahead)-8} more")
    print("  Someone else has advanced this run since you pulled it. Uploading now")
    print("  keeps their work (sync only adds) UNLESS you both ran the same task,")
    print("  in which case yours overwrites theirs. To be safe, stop, run")
    print(f"  `python scripts/storage/sync_from_s3.py --only {prefix.split('/')[-1]} --apply`,")
    print("  then re-run — already-done tasks are skipped, so it costs nothing.\n")


def check_auth_or_die(*, s3_sync=None):
    """Pre-flight: confirm AWS credentials work BEFORE the run starts, so a
    multi-hour run doesn't end with a silent S3 sync failure on an expired
    session. No-op when s3 sync is disabled.

    Why this matters: `aws s3 sync` returns exit code 0 even when an expired
    session refuses every operation — it logs the error to stderr but doesn't
    fail. So at upload time the sync looks successful when nothing landed. The
    pre-flight catches the common case (session bad before run starts);
    _verify_upload catches the rest (session expired mid-run)."""
    if not _enabled(s3_sync):
        return
    _require_aws_cli()
    completed = subprocess.run(
        ["aws", "sts", "get-caller-identity"], capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"AWS auth pre-flight failed (exit {completed.returncode}):\n"
            f"{(completed.stderr or completed.stdout).strip()}\n\n"
            "Run `aws login` (or `aws sso login`) and try again. To skip the "
            "upload entirely, pass --no-upload or set SEQK_S3_SYNC=0."
        )


def is_disabled_by_env():
    """True iff SEQK_S3_SYNC=0 — useful for CLI to short-circuit early."""
    return os.environ.get("SEQK_S3_SYNC") == "0"


# --------------------------------------------------------------------------- #
# Internals — each does one thing and is independently testable
# --------------------------------------------------------------------------- #
def _enabled(s3_sync):
    """Resolve the opt-out chain: explicit param > env var > default-on."""
    if s3_sync is False:
        return False
    if is_disabled_by_env():
        return False
    return True


_dotenv_loaded = False


def _load_env_once():
    """Read .env if present.

    core/cli.py loads it, but the standalone scripts are entry points too and
    used to see nothing — `_bucket()` then silently fell back to the default
    name and pointed operations at a bucket that is not yours. Same fix, and
    same reason, as core.db.dsn().
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except Exception:                              # noqa: BLE001 - env is optional
        pass


def _bucket():
    _load_env_once()
    return os.environ.get("SEQK_S3_BUCKET") or DEFAULT_BUCKET


def _require_aws_cli():
    if shutil.which("aws") is not None:
        return
    raise RuntimeError(
        "aws CLI not found on PATH. Install it "
        "(https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) "
        "or disable S3 sync (set s3_sync: false in the variant, or "
        "SEQK_S3_SYNC=0 in your shell)."
    )


def _s3_prefix(out_path, runs_root):
    """The S3 key prefix for a local run path.

    Prefers the manifest's `storage_key`, so local and remote agree on one key
    even if the directory was moved by hand. Falls back to the path relative to
    runs_root for anything without a manifest."""
    from core import registry
    manifest = registry.read_manifest(out_path)
    if manifest and manifest.get("storage_key"):
        return str(manifest["storage_key"]).strip("/")
    try:
        return str(out_path.resolve().relative_to(Path(runs_root).resolve()))
    except ValueError:
        # out_path isn't under runs_root — fall back to basename.
        return out_path.name


def _aws_s3_sync(local_dir, target_uri):
    """`aws s3 sync` without --delete. Only local bookkeeping is excluded."""
    cmd = ["aws", "s3", "sync", f"{local_dir}/", target_uri, "--no-progress",
           "--exclude", ".db_pending.jsonl", "--exclude", ".registry.json",
           "--exclude", ".registry.lock"]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"aws s3 sync failed (exit {completed.returncode}):\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}\n\n"
            f"Retry with: python -m core upload {local_dir}"
        )


def _verify_upload(local_dir, bucket, prefix):
    """Confirm files actually landed in S3 — `aws s3 sync` can exit 0 on an
    expired session that silently refused everything. We recursively list the
    target prefix and compare file counts. Local bookkeeping files are excluded
    from both sides since we don't upload them."""
    expected = sum(1 for p in Path(local_dir).rglob("*")
                   if p.is_file() and p.name not in _NOT_UPLOADED)
    target = f"s3://{bucket}/{prefix}/"
    completed = subprocess.run(
        ["aws", "s3", "ls", "--recursive", target], capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"post-sync verification failed (aws s3 ls exit {completed.returncode}):\n"
            f"{completed.stderr.strip()}\n\n"
            f"Retry with: python -m core upload {local_dir}"
        )
    landed = len([ln for ln in completed.stdout.splitlines() if ln.strip()])
    if landed < expected:
        raise RuntimeError(
            f"S3 sync silently dropped files: {expected} local, {landed} on S3.\n"
            f"Likely an expired session mid-sync.\n"
            f"Run `aws login` and retry with: python -m core upload {local_dir}"
        )
