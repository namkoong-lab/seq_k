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


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def upload_run(out, *, s3_sync=None, runs_root="runs"):
    """Sync `out/` to `s3://<bucket>/<out relative to runs_root>/`. config.json
    is excluded — it's a local-only artifact. No-op if disabled."""
    if not _enabled(s3_sync):
        print(f"→ s3 sync skipped (disabled) for {out}")
        return

    out_path = Path(out)
    if not out_path.is_dir():
        raise FileNotFoundError(f"run dir not found: {out_path}")

    bucket = _bucket()
    prefix = _s3_prefix(out_path, runs_root)
    _require_aws_cli()
    _scrub_harbor_secrets(out_path / "_harbor_jobs")

    target = f"s3://{bucket}/{prefix}/"
    print(f"→ syncing {out_path}/ to {target}  (config.json excluded)")
    _aws_s3_sync(out_path, target)
    _verify_upload(out_path, bucket, prefix)
    print(f"→ done. {target}")


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


def _bucket():
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


def _scrub_harbor_secrets(harbor_dir):
    """Best-effort scrub of Harbor artifacts before they leave the machine.

    Delegates to scripts/scrub_secrets.py so the regex list lives in one place.
    A scrub failure raises — we never upload unscrubbed artifacts.
    """
    if not harbor_dir.is_dir():
        return
    scrubber = Path(__file__).resolve().parent.parent / "scripts" / "scrub_secrets.py"
    if not scrubber.is_file():
        raise RuntimeError(f"scrub_secrets.py not found at {scrubber}")
    print(f"→ scrubbing secrets in {harbor_dir}/")
    completed = subprocess.run(
        [sys.executable, str(scrubber), str(harbor_dir)],
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"secret scrub failed (exit {completed.returncode}):\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _s3_prefix(out_path, runs_root):
    """Return the S3 key prefix for a local run path. Mirrors the path under
    runs_root (so `runs/<a>/<b>/<c>` → `<a>/<b>/<c>`)."""
    try:
        return str(out_path.resolve().relative_to(Path(runs_root).resolve()))
    except ValueError:
        # out_path isn't under runs_root — fall back to basename.
        return out_path.name


def _aws_s3_sync(local_dir, target_uri):
    """`aws s3 sync` without --delete. config.json is excluded (local-only)."""
    cmd = ["aws", "s3", "sync", f"{local_dir}/", target_uri, "--no-progress",
           "--exclude", "config.json"]
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
    target prefix and compare file counts. config.json is excluded from both
    sides since we don't upload it."""
    expected = sum(1 for p in Path(local_dir).rglob("*") if p.is_file() and p.name != "config.json")
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
            f"S3 sync silently dropped files: {expected} local (ex-config.json), {landed} on S3.\n"
            f"Likely an expired session mid-sync.\n"
            f"Run `aws login` and retry with: python -m core upload {local_dir}"
        )
