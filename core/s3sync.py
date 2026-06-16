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
def upload_run(out, *, s3_sync=None):
    """Sync `out/` to `s3://<bucket>/<basename of out>/`. No-op if disabled."""
    if not _enabled(s3_sync):
        print(f"→ s3 sync skipped (disabled) for {out}")
        return

    out_path = Path(out)
    if not out_path.is_dir():
        raise FileNotFoundError(f"run dir not found: {out_path}")

    bucket = _bucket()
    prefix = out_path.name
    _require_aws_cli()
    _scrub_harbor_secrets(out_path / "_harbor_jobs")

    target = f"s3://{bucket}/{prefix}/"
    print(f"→ syncing {out_path}/ to {target}")
    _aws_s3_sync(out_path, target)
    print(f"→ done. {target}")


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


def _aws_s3_sync(local_dir, target_uri):
    """`aws s3 sync` without --delete. Raises with a retry hint on failure."""
    cmd = ["aws", "s3", "sync", f"{local_dir}/", target_uri, "--no-progress"]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"aws s3 sync failed (exit {completed.returncode}):\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}\n\n"
            f"Retry with: python -m core upload {local_dir}"
        )
