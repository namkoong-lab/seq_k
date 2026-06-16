#!/usr/bin/env bash
# Upload a single run's Harbor artifacts to S3 after scrubbing secrets.
#
# DEPRECATED: prefer `python -m core upload <run-dir>`, which syncs the whole
# run dir (config.json, summary.json, tasks/, _harbor_jobs/) and shares the
# secret-scrubbing + bucket-resolution code with the auto-sync on `core run`.
# This script remains for now as a one-line bash fallback.
#
# Usage:  scripts/upload_run.sh runs/terminalbench.comparison.all.seqk.raw
#
# Reads the bucket from $SEQK_S3_BUCKET (export it once, e.g. in ~/.zshrc).

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <run-dir>  (e.g. runs/terminalbench.comparison.all.seqk.raw)" >&2
    exit 2
fi
RUN="$1"
BUCKET="${SEQK_S3_BUCKET:-}"
if [ -z "$BUCKET" ]; then
    echo "error: SEQK_S3_BUCKET not set. Run: export SEQK_S3_BUCKET=seqk-runs-<your-id>" >&2
    exit 2
fi
if [ ! -d "$RUN/_harbor_jobs" ]; then
    echo "no _harbor_jobs/ under $RUN — nothing to upload" >&2
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${RUN#runs/}"

echo "→ scrubbing secrets in $RUN/_harbor_jobs/"
python "$REPO_ROOT/scripts/scrub_secrets.py" "$RUN/_harbor_jobs/"

echo "→ syncing to s3://$BUCKET/$PREFIX/_harbor_jobs/"
aws s3 sync "$RUN/_harbor_jobs/" "s3://$BUCKET/$PREFIX/_harbor_jobs/" --no-progress

echo "→ done. share with:"
echo "    aws s3 presign s3://$BUCKET/$PREFIX/_harbor_jobs/<trial>/agent/trajectory.json --expires-in 604800"
