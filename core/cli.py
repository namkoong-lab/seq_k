"""CLI: run | rejudge | inspect | metrics | upload.

    python -m core run     benchmarks/clbench/variants/seqk.raw.yaml
    python -m core rejudge <run> --judge openrouter/openai/gpt-5.4
    python -m core inspect runs/researchrubrics/<run_id> --task-index 1
    python -m core metrics runs/by-label/researchrubrics/metric=seqk/k=10/... --k 10
    python -m core upload  runs/researchrubrics/<run_id>

The benchmark comes from the config's path, or an explicit `benchmark:` key.

A run's directory is resolved by IDENTITY FINGERPRINT (core/ids.py) via
core/registry.py, so running the same YAML again RESUMES the same run and
changing any result-affecting field starts a new one. `runs/by-label/` holds
readable symlinks to the same directories — either form works everywhere a run
path is accepted. Find runs with `python scripts/runs.py ls`.

`run` uploads the finished run to S3 by default — `--no-upload` skips it,
the variant YAML can set `s3_sync: false`, or set `SEQK_S3_SYNC=0` once per
machine. See core/s3sync.py for the full opt-out chain.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import yaml
from dotenv import load_dotenv

from core import harness, metrics, rejudge, results, s3sync


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


def _run(config_path, no_upload):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    benchmark = importlib.import_module(_benchmark_module(config_path, cfg))
    if no_upload:
        cfg["s3_sync"] = False
    # Allow either `continue` or `continue_run` in the YAML (continue is a
    # Python keyword so we can't accept it as a kwarg directly).
    if "continue" in cfg:
        cfg["continue_run"] = cfg.pop("continue")
    harness.run(benchmark, **cfg)


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(prog="seq_k")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a benchmark from a config file")
    p_run.add_argument("config")
    p_run.add_argument("--no-upload", action="store_true",
                       help="skip the end-of-run S3 sync (default: upload to s3://seq-k/<path>)")

    p_rejudge = sub.add_parser(
        "rejudge", help="re-grade an existing run's generations with a different judge")
    p_rejudge.add_argument("run", help="run id prefix, <slice>/<run_id>, or a by-label path")
    p_rejudge.add_argument("--judge", required=True, help="the new judge model")
    p_rejudge.add_argument("--apply", action="store_true",
                           help="actually run the judge (default: dry run)")
    p_rejudge.add_argument("--runs-root", default="runs")
    p_rejudge.add_argument("--no-upload", action="store_true")

    p_inspect = sub.add_parser("inspect", help="print a saved task's trajectory step by step")
    p_inspect.add_argument("run_path", help="full run directory path")
    g = p_inspect.add_mutually_exclusive_group(required=True)
    g.add_argument("--task-index", type=int, help="canonical 1-based task index")
    g.add_argument("--task-id", help="benchmark-native task id")

    p_metrics = sub.add_parser("metrics", help="print pass@k / seq@k for a run")
    p_metrics.add_argument("run_path")
    p_metrics.add_argument("--k", type=int, required=True)

    p_upload = sub.add_parser("upload", help="sync an existing run dir to S3 manually")
    p_upload.add_argument("run_path")

    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args.config, no_upload=args.no_upload)
    elif args.command == "rejudge":
        rejudge.rejudge(args.run, judge_model=args.judge, runs_root=args.runs_root,
                        apply=args.apply, s3_sync=False if args.no_upload else None)
    elif args.command == "inspect":
        results.inspect(args.run_path, task_index=args.task_index, task_id=args.task_id)
    elif args.command == "metrics":
        metrics.summarize(args.run_path, args.k)
    elif args.command == "upload":
        s3sync.upload_run(args.run_path)


if __name__ == "__main__":
    main()
