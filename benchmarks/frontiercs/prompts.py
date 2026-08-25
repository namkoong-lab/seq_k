"""FrontierCS actor prompt pieces.

Diverged from seq_k_eval's "Return only code without markdown" wording, which
reasoning-style actors ignore: they think in prose, scatter fragment blocks
along the way, and sometimes trail off after a `// Full solution below`
placeholder without ever writing the program — or spend the whole output
budget reasoning and get truncated mid-program. Hence naming the one code
block explicitly and pushing hard for brevity outside it. (The migrated
run_old data is on a different score metric, so it is not a prompt-comparable
baseline.)
"""

BASE_PROMPT = (
    "Write a single-file self-contained C++17 solution. Reply with the complete "
    "program in a single ```cpp code block. Do not explain your work; if you must "
    "reason first, keep it to a few lines."
)
