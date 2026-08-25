"""Bounded parallel map for per-rubric judge calls.

ResearchRubrics and HealthBench grade one LLM call per rubric criterion, ~25 per
attempt. `pmap` runs them concurrently.

Two constraints callers depend on:
  - input order is preserved; verdicts are zipped positionally against rubrics.
  - every thread in one pmap must share a phase — `core.llm._phase` is a global,
    so call pmap inside a single `with llm.phase(...)` block.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

# Per-call-site cap. Override with SEQK_JUDGE_WORKERS when running many configs
# concurrently, so total in-flight requests stay under the provider's rate limit:
# roughly (concurrent runs) x (this value) requests at once.
DEFAULT_WORKERS = int(os.environ.get("SEQK_JUDGE_WORKERS", "12"))


def pmap(fn, items, *, workers=None):
    """[fn(x) for x in items], evaluated concurrently, order preserved.

    Falls back to a serial comprehension for 0/1 items so trivial cases don't
    pay thread-pool setup. Exceptions propagate exactly as the serial version
    would — the first failure raised wins, and the benchmark stays fail-loud.
    """
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    n = max(1, min(workers or DEFAULT_WORKERS, len(items)))
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(fn, items))
