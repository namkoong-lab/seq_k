#!/usr/bin/env bash
# One-shot transfer of the LOCAL database to Neon.
#
# Neon bills CPU time, and the incremental writer costs ~5 round trips per task
# — a full corpus load ran ~40 minutes of connected time. Doing the work against
# a local Postgres and shipping the finished result in one dump is the same data
# for a fraction of that.
#
#   scripts/push_to_neon.sh            # dry run: show sizes, change nothing
#   scripts/push_to_neon.sh --apply
set -euo pipefail
cd "$(dirname "$0")/../.."
LOCAL=$(grep '^DATABASE_URL=' .env | cut -d= -f2-)
NEON=$(grep '^SEQK_NEON_URL=' .env | cut -d= -f2-)
[ -n "$NEON" ] || { echo "SEQK_NEON_URL not set in .env"; exit 1; }
# Refuse unless DATABASE_URL is a LOCAL unix socket. This is the guard that stops
# a mis-set env from dumping Neon back over itself. It matches on host=/ (a socket
# path) rather than a specific directory name — the local cluster used to live in a
# mktemp dir, which vanished on reboot and took the working database with it.
case "$LOCAL" in *"host=/"*) ;; *) echo "DATABASE_URL is not a local socket — refusing"; exit 1;; esac

echo "local : $(psql "$LOCAL" -tAc 'SELECT count(*) FROM runs') runs, \
$(psql "$LOCAL" -tAc 'SELECT count(*) FROM llm_calls') llm_calls"
echo "neon  : $(psql "$NEON" -tAc 'SET search_path TO public; SELECT count(*) FROM runs' 2>/dev/null || echo '?') runs"

if [ "${1:-}" != "--apply" ]; then echo; echo "DRY RUN — pass --apply to transfer."; exit 0; fi

# --clean drops the objects first so this is a replace, not a merge. Safe because
# the database is derived: `db_sync.py --reset` rebuilds it from disk either way.
#
# NOTE ON THE POOLED ENDPOINT: pg_dump's restore issues
# `set_config('search_path','',false)`. On Neon's `-pooler` host that server
# session goes back into the pool still carrying the empty search_path, and the
# next client to borrow it sees "relation runs does not exist" while the tables
# are present. core/db.py now pins `SET search_path TO public` per connection, so
# the app is immune; plain psql is not. Prefer the DIRECT (non-pooler) endpoint
# for the restore if you have it.
# pg_dump REFUSES to dump a server newer than itself, and Homebrew leaves an
# older postgresql@14 ahead of @15 on PATH. When that happened the dump aborted
# — but psql had already run `--clean` and DROPPED every table on Neon, leaving
# it empty with no restore behind it. So: pick a pg_dump at least as new as the
# server, and fail loudly if there isn't one, BEFORE anything is dropped.
SERVER_MAJOR=$(psql "$LOCAL" -tAc 'SHOW server_version' | cut -d. -f1)
PGDUMP=""
for c in "/opt/homebrew/opt/postgresql@${SERVER_MAJOR}/bin/pg_dump" \
         /opt/homebrew/opt/postgresql@1[5-9]/bin/pg_dump \
         /usr/local/opt/postgresql@1[5-9]/bin/pg_dump pg_dump; do
  command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
  v=$("$c" --version | grep -oE '[0-9]+' | head -1)
  if [ "${v:-0}" -ge "$SERVER_MAJOR" ]; then PGDUMP="$c"; break; fi
done
[ -n "$PGDUMP" ] || { echo "no pg_dump >= server major $SERVER_MAJOR found — refusing (a failed dump after --clean would leave Neon EMPTY)"; exit 1; }
echo "using $PGDUMP ($($PGDUMP --version)) against server major $SERVER_MAJOR"

echo "dumping local and restoring into Neon (one connection, one pass)…"
"$PGDUMP" --no-owner --no-privileges --clean --if-exists "$LOCAL" \
  | psql "$NEON" -v ON_ERROR_STOP=1 --quiet
# Every post-restore query MUST set search_path first. pg_dump emits
# `set_config('search_path','',false)`, and on a POOLED Neon connection that
# setting outlives the restore session and poisons the next one — so an
# unqualified count here reports "relation runs does not exist" on a restore
# that in fact succeeded. That false alarm has been chased twice; don't drop it.
echo "neon now: $(psql "$NEON" -tAc 'SET search_path TO public; SELECT count(*) FROM runs') runs, \
$(psql "$NEON" -tAc 'SET search_path TO public; SELECT count(*) FROM llm_calls') llm_calls"
