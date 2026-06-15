"""Measure how saturated MedMCQA / MedQA are for the actor model.

Greedy (temp 0) single-attempt accuracy = pass@1 ceiling. The gap below 100%
is the headroom where seq@k vs pass@k can show any signal. Cheap: ~N calls/bench.
"""
from dotenv import load_dotenv
load_dotenv()

from concurrent.futures import ThreadPoolExecutor

from core import llm
from core.types import Attempt
import benchmarks.medmcqa as medmcqa
import benchmarks.medqa as medqa

MODEL = "anthropic/claude-sonnet-4-6"
N = 30


def score_one(bench, task):
    out = llm.complete(MODEL, task.prompt, 0.0)
    return bench.verify(task, Attempt(0, out)).success


def run(name, bench, options):
    tasks = bench.load_tasks(max_rows=N, **options)
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda t: score_one(bench, t), tasks))
    acc = sum(results) / len(results)
    print(f"{name:<10} n={len(results):<4} pass@1(greedy) = {acc:6.1%}   "
          f"failed = {len(results) - sum(results)}/{len(results)}")
    return acc


if __name__ == "__main__":
    print("=== single-attempt accuracy (Sonnet 4.6, temp 0) ===")
    run("MedMCQA", medmcqa, {"split": "validation"})
    run("MedQA", medqa, {"split": "test"})
