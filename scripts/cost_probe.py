"""Throwaway cost probe: run a few tasks and measure real Sonnet 4.6 token spend.

Wraps litellm.completion in-process (no source changes) to accumulate prompt/
completion tokens per harness.run, then extrapolates to a full 40-task run.
Pricing: Sonnet 4.6 = $3/1M input, $15/1M output.
"""
from dotenv import load_dotenv
load_dotenv()

import litellm
from core import harness
import benchmarks.medmcqa as medmcqa
import benchmarks.medqa as medqa

IN_PER_M, OUT_PER_M = 3.0, 15.0
_orig = litellm.completion
_acc = {"calls": 0, "in": 0, "out": 0}


def _wrapped(*a, **k):
    r = _orig(*a, **k)
    u = getattr(r, "usage", None)
    if u is not None:
        _acc["calls"] += 1
        _acc["in"] += int(getattr(u, "prompt_tokens", 0) or 0)
        _acc["out"] += int(getattr(u, "completion_tokens", 0) or 0)
    return r


litellm.completion = _wrapped
MODEL = "anthropic/claude-sonnet-4-6"


def probe(name, bench, metric, feedback_mode, n, options):
    _acc.update(calls=0, **{"in": 0, "out": 0})
    harness.run(bench, metric=metric, k=5, feedback_mode=feedback_mode, model=MODEL,
                judge_model=MODEL, temperature=0.7, max_tasks=n,
                out=f"runs/_probe_{name}", options=options, console_char_limit=300)
    cost = _acc["in"] / 1e6 * IN_PER_M + _acc["out"] / 1e6 * OUT_PER_M
    per_task = cost / n
    return {"name": name, "n": n, "calls": _acc["calls"], "in": _acc["in"],
            "out": _acc["out"], "cost": cost, "per_task": per_task}


def main():
    rows = []
    rows.append(probe("medmcqa_passk", medmcqa, "pass@k", "binary", 3, {"split": "validation"}))
    rows.append(probe("medmcqa_judge", medmcqa, "seq@k", "judge", 3, {"split": "validation"}))
    rows.append(probe("medqa_judge", medqa, "seq@k", "judge", 3, {"split": "test"}))

    print("\n" + "=" * 78)
    print(f"{'config':<18}{'tasks':>6}{'llm_calls':>10}{'in_tok':>9}{'out_tok':>9}"
          f"{'cost$':>8}{'$/task':>8}{'40-task$':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['name']:<18}{r['n']:>6}{r['calls']:>10}{r['in']:>9}{r['out']:>9}"
              f"{r['cost']:>8.4f}{r['per_task']:>8.4f}{r['per_task']*40:>9.2f}")
    print("=" * 78)
    passk = next(r for r in rows if r["name"] == "medmcqa_passk")["per_task"]
    mj = next(r for r in rows if r["name"] == "medmcqa_judge")["per_task"]
    qj = next(r for r in rows if r["name"] == "medqa_judge")["per_task"]
    # Full experiment estimate: per benchmark, passk + 3 seq variants (binary/raw ~ passk
    # cost since no judge calls; judge is the dear one). 40 tasks each.
    mm = (passk * 3 + mj) * 40
    qq = (passk * 3 + qj) * 40  # use medmcqa passk/task as proxy for medqa non-judge
    print(f"Rough full-experiment estimate (40 tasks, k=5, 4 variants/benchmark):")
    print(f"  MedMCQA 4 variants ~ ${mm:.2f}   MedQA 4 variants ~ ${qq:.2f}   TOTAL ~ ${mm+qq:.2f}")


if __name__ == "__main__":
    main()
