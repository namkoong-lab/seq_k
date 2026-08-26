"""Launch many variant YAMLs concurrently, supervised, with bounded request load.

Three things stack here:
  1. WITHIN an attempt — core/parallel.py fans out the per-rubric judge calls.
  2. ACROSS runs — this script runs N config files as separate processes.
  3. OVER TIME — a run that dies is relaunched automatically (see below).

Peak load is what matters, so keep this under the provider's rate limit:

    peak in-flight requests ~= (concurrent runs) x (SEQK_JUDGE_WORKERS + 1)

WHY SUPERVISION: long grids die from transient provider faults (504s that outlive
their retries, upstream capacity blips) hours in, one run at a time. Because the
harness derives a deterministic path per config and resumes it, relaunching a
dead run is nearly free — completed tasks are skipped and only the in-flight
attempt is redone. So a died run is retried up to --max-restarts times.

The guard against an infinite loop is PROGRESS, not just a counter: if a restart
produces no new attempt files, the failure is structural (bad config, missing
data, a real bug) and retrying forever would hide it, so we stop and report.

Logs are APPENDED across restarts, never truncated — the error that killed a run
must survive the relaunch that follows it.

    python scripts/run_grid.py benchmarks/researchrubrics/variants/sensitivity/*.yaml \
        --concurrency 14 --judge-workers 6 --max-restarts 8
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Running this as `python scripts/run_grid.py` puts scripts/ on sys.path, NOT the
# repo root — so `from core import ...` below fails and _attempt_count silently
# falls back to a grid-wide count, which makes every run look like it shares one
# progress number. Put the repo root first so the per-run path resolution works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


_NOTE_SUFFIX = {"no_horizon": "-nohorizon", "none": "-noframe"}


def _variant(c):
    """prompt_variant for a config, folding in a legacy `attempt_note`.

    The retry framing used to be its own field; it is a template concern now, so
    it rides on the variant name (core/prompts.py)."""
    base = c.get("prompt_variant") or "v1"
    note = c.get("attempt_note")
    suffix = _NOTE_SUFFIX.get(note, "")
    return base + suffix if suffix and not base.endswith(suffix) else base


def _attempt_count(cfg):
    """Attempt files belonging to THIS config's run — the progress signal.

    Must be per-run, not per-runs_root: when many configs share a runs_root, a
    grid-wide count makes a wedged run look alive because its neighbours are
    still working (and vice versa). We resolve the config's own run directory by
    FINGERPRINT and count only inside it.

    (This previously called results.build_run_path with a signature that stopped
    existing at the v2 rename, so it always threw and always fell back to the
    grid-wide count — the exact failure mode the docstring warns about.)
    """
    try:
        c = yaml.safe_load(open(cfg, encoding="utf-8")) or {}
    except Exception:
        return 0
    root = c.get("runs_root", "runs")
    try:
        import importlib
        from core import harness, ids, registry, results
        bench = c.get("benchmark")
        if not bench:
            parts = Path(cfg).resolve().parts
            i = len(parts) - 1 - parts[::-1].index("benchmarks")
            bench = ".".join(parts[i:parts.index("variants", i)])
        else:
            bench = "benchmarks." + bench
        mod = importlib.import_module(bench)
        model = c["model"]
        # attempt_note was folded into prompt_variant; honour old configs.
        context = c.get("context") or results.context_from_legacy(
            c["metric"], c.get("summarize", False))
        prompt_variant = _variant(c)
        critic_model = harness.resolve_critic_model(
            mod, c["feedback_mode"], actor_model=model,
            critic_model=c.get("critic_model"),
        )
        ident = ids.identity(
            benchmark_module=mod, options=c.get("options") or {}, metric=c["metric"],
            k=c["k"], model=model, judge_model=c.get("judge_model") or model,
            critic_model=critic_model, feedback_mode=c["feedback_mode"],
            context=context, prompt_variant=prompt_variant,
            temperature=c.get("temperature", 0.7), seed=c.get("seed"),
            reasoning_effort=c.get("reasoning_effort"), output_budget=c.get("output_budget"),
            summarizer_model=c.get("summarizer_model") or model)
        fp = ids.fingerprint(ident)
        for path, m in registry.iter_manifests(root):
            if m.get("fingerprint") == fp:
                return len(glob.glob(os.path.join(path, "**", "attempt-*.json"), recursive=True))
        return 0        # not started yet: a real, correct zero
    except Exception:
        # Fall back to the grid-wide count rather than reporting 0, which would
        # look like "no progress" and trigger a spurious give-up.
        return len(glob.glob(os.path.join(root, "**", "attempt-*.json"), recursive=True))


def _run_one(cfg, log_dir, env, max_restarts, backoff):
    name = Path(cfg).name
    log = Path(log_dir) / (Path(cfg).stem + ".log")
    t0 = time.time()
    for i in range(max_restarts + 1):
        before = _attempt_count(cfg)
        with open(log, "a", encoding="utf-8") as fh:   # append: keep prior failures
            fh.write(f"\n{'='*70}\n=== launch {i + 1} at {time.strftime('%H:%M:%S')}\n{'='*70}\n")
            fh.flush()
            proc = subprocess.run([sys.executable, "-m", "core", "run", str(cfg)],
                                  stdout=fh, stderr=subprocess.STDOUT, env=env)
        if proc.returncode == 0:
            print(f"  [done] {name}  after {i + 1} launch(es), {(time.time()-t0)/60:.0f} min", flush=True)
            return cfg, 0, i
        if i == max_restarts:
            break
        after = _attempt_count(cfg)
        if after <= before:
            print(f"  [giving up] {name}: launch {i + 1} made no progress "
                  f"({before} attempts before and after) — likely structural, see {log}", flush=True)
            return cfg, proc.returncode, i
        print(f"  [restart {i + 1}/{max_restarts}] {name} died (rc={proc.returncode}) "
              f"after +{after - before} attempts; relaunching in {backoff}s", flush=True)
        time.sleep(backoff)
    print(f"  [FAILED] {name}: exhausted {max_restarts} restarts -> {log}", flush=True)
    return cfg, proc.returncode, max_restarts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--concurrency", type=int, default=7)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2,
                    help="litellm num_retries WITHIN a run (rate limits / 5xx)")
    ap.add_argument("--max-restarts", type=int, default=8,
                    help="relaunches of a whole run that died; resume makes these cheap")
    ap.add_argument("--restart-backoff", type=int, default=20)
    ap.add_argument("--log-dir", default="runs/_grid_logs")
    args = ap.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ,
               SEQK_JUDGE_WORKERS=str(args.judge_workers),
               SEQK_NUM_RETRIES=str(args.retries))

    print(f"launching {len(args.configs)} runs | concurrency={args.concurrency} "
          f"| judge_workers={args.judge_workers} | retries={args.retries} "
          f"| max_restarts={args.max_restarts}")
    print(f"peak in-flight requests ~= {args.concurrency * (args.judge_workers + 1)}\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(
            lambda c: _run_one(c, args.log_dir, env, args.max_restarts, args.restart_backoff),
            args.configs))

    failed = [c for c, rc, _ in results if rc != 0]
    restarted = [(c, n) for c, rc, n in results if n]
    print(f"\nwall clock: {(time.time()-t0)/60:.1f} min")
    print(f"{len(results)-len(failed)}/{len(results)} succeeded")
    if restarted:
        print("recovered after restarts: " + ", ".join(f"{Path(c).stem}x{n}" for c, n in restarted))
    if failed:
        print("FAILED: " + ", ".join(Path(c).stem for c in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
