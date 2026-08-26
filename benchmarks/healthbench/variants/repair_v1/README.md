# HealthBench repair v1

Uniform clean replication for canonical HealthBench indices 1–30. These runs
intentionally do not resume or overwrite legacy `*_v0` artifacts. All variants
use protocol `healthbench-repair-v1`, GPT-5.4 grading at threshold 0.75, k=5,
temperature 0.7, actor reasoning effort `low`, and replicate label 42.

Pass variants produce exactly five independent draws per task. Judge and
self-blind variants stop on first success; both use the actor as feedback writer,
while only judge feedback receives verifier output.
