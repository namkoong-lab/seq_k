"""Pass@k / Seq@k and the trajectory-shape stats, computed from a results file.

Reading-only: never re-runs models. One results file holds one metric (pass@k or
seq@k), so a comparison is just summarizing two files.
"""

from __future__ import annotations

from core import results


def _cumulative_best_by_attempt(traj, k):
    """Best score achieved by attempt index t (0..k-1), carried forward."""
    best, curve = 0.0, []
    steps = traj["steps"]
    for t in range(k):
        if t < len(steps):
            best = max(best, steps[t]["result"]["score"])
        curve.append(best)
    return curve


def summarize(out, k):
    trajs = results.load(out)
    if not trajs:
        raise ValueError(f"no trajectories in {out}")

    metric = trajs[0]["metric"]                       # one metric per file
    label = "seq" if metric == "seq@k" else "pass"
    curves = [_cumulative_best_by_attempt(t, k) for t in trajs]
    at = [sum(c[t] for c in curves) / len(curves) for t in range(k)]   # metric@(t+1)

    print(f"{out}: {len(trajs)} tasks | metric={metric}")
    for t in range(k):
        print(f"  {label}@{t + 1} = {at[t]:.3f}")

    if metric == "seq@k" and k >= 2:
        delta = at[k - 1] - at[0]                     # ΔSeq@K = Seq@K - Seq@1
        if delta > 0:
            egs = (at[1] - at[0]) / delta             # early gain share
            lgs = (at[k - 1] - at[k - 2]) / delta     # late gain share
            print(f"  ΔSeq@K = {delta:.3f}  EGS = {egs:.2f}  LGS = {lgs:.2f}")
        else:
            print(f"  ΔSeq@K = {delta:.3f}  (no gain over seq@1)")
    return at
