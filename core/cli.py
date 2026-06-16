"""CLI: run | inspect | metrics | upload.

    python -m core run     benchmarks/clbench/variants/seqk.raw.yaml
    python -m core inspect runs/clbench.seqk.raw-<timestamp> --task <id>
    python -m core metrics runs/clbench.seqk.raw-<timestamp> --k 5
    python -m core upload  runs/clbench.seqk.raw-<timestamp>

The benchmark comes from the config's path, or an explicit `benchmark:` key.

Every `run` invocation stamps `runs/<name>-<UTC-timestamp>/` so collaborators
don't stomp each other. To continue a crashed run, pass `--resume <existing-folder>`
and the harness reuses that folder verbatim (init_run's config-mismatch guard
refuses to mix incompatible configs).

`run` also uploads the finished run to S3 by default — `--no-upload` skips it,
the variant YAML can set `s3_sync: false`, or set `SEQK_S3_SYNC=0` once per
machine. See core/s3sync.py for the full opt-out chain.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import yaml
from dotenv import load_dotenv

from core import harness, metrics, results, s3sync


def _benchmark_module(config_path, cfg):
    """benchmarks.* module for a config: explicit `benchmark:` key wins, else
    inferred from benchmarks/<name>/variants/<x>.yaml."""
    name = cfg.pop("benchmark", None)
    if name:
        return "benchmarks." + name
    parts = Path(config_path).resolve().parts
    if "benchmarks" in parts and "variants" in parts:
        i = len(parts) - 1 - parts[::-1].index("benchmarks")   # last 'benchmarks'
        j = parts.index("variants", i)
        return ".".join(parts[i:j])
    raise ValueError(
        f"cannot determine benchmark for {config_path!r}: put it under "
        "benchmarks/<name>/variants/ or add a 'benchmark:' key to the config")


def _run(config_path, no_upload, resume):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    benchmark = importlib.import_module(_benchmark_module(config_path, cfg))
    if no_upload:
        cfg["s3_sync"] = False
    if resume:
        cfg["resume"] = resume
    harness.run(benchmark, **cfg)


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(prog="seq_k")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a benchmark from a config file")
    p_run.add_argument("config")
    p_run.add_argument("--no-upload", action="store_true",
                       help="skip the end-of-run S3 sync (default: upload to s3://seq-k/<name>/)")
    p_run.add_argument("--resume", metavar="RUN_DIR",
                       help="continue an existing run folder instead of stamping a fresh "
                            "`<out>-<timestamp>/` (e.g. --resume runs/terminalbench.passk-2026-06-15T22-38-45Z)")

    p_inspect = sub.add_parser("inspect", help="print a saved trajectory step by step")
    p_inspect.add_argument("results_file")
    p_inspect.add_argument("--task", required=True)

    p_metrics = sub.add_parser("metrics", help="print pass@k / seq@k from a results file")
    p_metrics.add_argument("results_file")
    p_metrics.add_argument("--k", type=int, required=True)

    p_upload = sub.add_parser("upload", help="sync an existing run dir to S3 manually")
    p_upload.add_argument("run_dir")

    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args.config, no_upload=args.no_upload, resume=args.resume)
    elif args.command == "inspect":
        results.inspect(args.results_file, args.task)
    elif args.command == "metrics":
        metrics.summarize(args.results_file, args.k)
    elif args.command == "upload":
        s3sync.upload_run(args.run_dir)


if __name__ == "__main__":
    main()
