"""Run HealthBench pass@k + seq@k binary on 20 tasks, with live cost tracking.

Wraps litellm.completion to accumulate token usage so we get per-variant API cost
(Sonnet 4.6: $3/1M in, $15/1M out). Resumable: the 12 finished tasks are skipped.
"""
from dotenv import load_dotenv
load_dotenv()

import litellm
from core import harness
import benchmarks.healthbench as hb

IN_PER_M, OUT_PER_M = 3.0, 15.0
acc = {"in": 0, "out": 0, "calls": 0}
_orig = litellm.completion


def _wrap(*a, **k):
    r = _orig(*a, **k)
    u = getattr(r, "usage", None)
    if u is not None:
        acc["calls"] += 1
        acc["in"] += int(getattr(u, "prompt_tokens", 0) or 0)
        acc["out"] += int(getattr(u, "completion_tokens", 0) or 0)
    return r


litellm.completion = _wrap
MODEL = "anthropic/claude-sonnet-4-6"


def run(metric, out):
    acc.update({"in": 0, "out": 0, "calls": 0})
    harness.run(hb, metric=metric, k=5, feedback_mode="binary", model=MODEL,
                judge_model=MODEL, temperature=0.7, max_tasks=20, out=out,
                console_char_limit=2000)
    cost = acc["in"] / 1e6 * IN_PER_M + acc["out"] / 1e6 * OUT_PER_M
    print(f"\n##COST {out}: calls={acc['calls']} in_tok={acc['in']} "
          f"out_tok={acc['out']} cost=${cost:.2f} (this run only; excludes the 12 cached tasks)")


if __name__ == "__main__":
    run("pass@k", "runs/healthbench.passk")
    run("seq@k", "runs/healthbench.seqk.binary")
    print("\n##DONE healthbench binary 20-task run")
