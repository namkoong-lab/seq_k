"""In-place secret scrubber for Harbor trial artifacts.

Walks a directory and redacts secret-shaped tokens in text files using the same
regex set the terminalbench benchmark uses for its JSON output
(benchmarks/terminalbench/benchmark.py:_SECRET_PATTERNS).

Run before uploading _harbor_jobs/ to S3.

    python scripts/scrub_secrets.py runs/<name>/_harbor_jobs/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SECRET_PATTERNS = [re.compile(p) for p in (
    r"sk-[A-Za-z0-9_-]{8,}",
    r"or-[A-Za-z0-9_-]{8,}",
    r"ghp_[A-Za-z0-9_]{8,}",
    r"github_pat_[A-Za-z0-9_]{8,}",
    r"hf_[A-Za-z0-9_]{8,}",
    r"AKIA[0-9A-Z]{16}",
    r"ASIA[0-9A-Z]{16}",
)]

SCRUB_SUFFIXES = (".pane", ".log", ".cast", ".txt", ".json", ".j2")
REPLACEMENT = "[redacted_secret]"


def scrub_file(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  skip {path}: {exc}", file=sys.stderr)
        return 0
    hits = 0
    for pat in SECRET_PATTERNS:
        text, n = pat.subn(REPLACEMENT, text)
        hits += n
    if hits:
        path.write_text(text, encoding="utf-8")
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="directory to scrub recursively")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        sys.exit(f"not found: {root}")

    files_scanned = files_redacted = total_hits = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCRUB_SUFFIXES:
            continue
        files_scanned += 1
        hits = scrub_file(path)
        if hits:
            files_redacted += 1
            total_hits += hits
            print(f"  redacted {hits:3d}  {path.relative_to(root)}")

    print(f"\nscanned {files_scanned} files, redacted {total_hits} secrets in {files_redacted} files")


if __name__ == "__main__":
    main()
