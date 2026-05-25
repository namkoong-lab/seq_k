"""seq_k command line: run | inspect | metrics.

    python -m core run     benchmarks/clbench/variants/seqk.raw.yaml
    python -m core inspect runs/clbench.seqk.raw --task <id>
    python -m core metrics runs/clbench.seqk.raw --k 5

No registry: a run config lives inside its benchmark folder
(benchmarks/<name>/variants/<x>.yaml), and the benchmark is inferred from that
path. An explicit `benchmark:` key in the YAML overrides the inference. A bad
name raises ImportError; a missing or misspelled config key raises from run(...).
Nothing is silently defaulted.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import yaml
from dotenv import load_dotenv

from core import harness, metrics, results


def _benchmark_module(config_path, cfg):
    """Resolve which benchmarks.* module a config belongs to.

    Explicit wins: a `benchmark:` key names it (e.g. `clbench`). Otherwise infer
    from the path: benchmarks/<name>/variants/<x>.yaml -> benchmarks.<name>.
    """
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


def _run(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    benchmark = importlib.import_module(_benchmark_module(config_path, cfg))
    harness.run(benchmark, **cfg)


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(prog="seq_k")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a benchmark from a config file")
    p_run.add_argument("config")

    p_inspect = sub.add_parser("inspect", help="print a saved trajectory step by step")
    p_inspect.add_argument("results_file")
    p_inspect.add_argument("--task", required=True)

    p_metrics = sub.add_parser("metrics", help="print pass@k / seq@k from a results file")
    p_metrics.add_argument("results_file")
    p_metrics.add_argument("--k", type=int, required=True)

    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args.config)
    elif args.command == "inspect":
        results.inspect(args.results_file, args.task)
    elif args.command == "metrics":
        metrics.summarize(args.results_file, args.k)


if __name__ == "__main__":
    main()
