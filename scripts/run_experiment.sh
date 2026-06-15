#!/usr/bin/env bash
# Run the medical seq@k vs pass@k experiment matrix sequentially.
# Selected variants per Iris's guidance: pass@k + seq@k binary (baseline) + the
# most-suitable richer feedback mode; HealthBench (rubric-rich) gets raw + judge.
# Runs are resumable (harness skips finished tasks), so re-running continues.
set -u
cd /Users/meng/Documents/seq_k
export PYTHONPATH=.
PY=".venv/bin/python"

VARIANTS=(
  benchmarks/medmcqa/variants/passk.yaml
  benchmarks/medmcqa/variants/seqk.binary.yaml
  benchmarks/medmcqa/variants/seqk.judge.yaml
  benchmarks/medxpertqa/variants/passk.yaml
  benchmarks/medxpertqa/variants/seqk.binary.yaml
  benchmarks/medxpertqa/variants/seqk.judge.yaml
  benchmarks/healthbench/variants/passk.yaml
  benchmarks/healthbench/variants/seqk.binary.yaml
  benchmarks/healthbench/variants/seqk.raw.yaml
  benchmarks/healthbench/variants/seqk.judge.yaml
)

for y in "${VARIANTS[@]}"; do
  out="runs/$(basename "$(dirname "$(dirname "$y")")").$(basename "$y" .yaml)"
  echo ""
  echo "===================================================================="
  echo "RUN $y  (start $(date +%H:%M:%S))"
  echo "===================================================================="
  $PY -m core run "$y" || echo "!!!! FAILED $y"
done

echo ""
echo "===== ALL RUNS COMPLETE ($(date +%H:%M:%S)) — metrics summary ====="
for d in runs/medmcqa.* runs/medxpertqa.* runs/healthbench.*; do
  [ -d "$d" ] || continue
  echo "--- $d ---"
  $PY -m core metrics "$d" --k 5 || true
done
