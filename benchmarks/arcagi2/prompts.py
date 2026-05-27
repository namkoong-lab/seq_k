"""Static prompt text for ARC-AGI-2. The full actor prompt is assembled
procedurally in benchmark.format_problem() from the task's train/test grids.
"""

PUZZLE_INTRO = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Your goal is to identify the "
    "pattern or transformation in the training examples that maps the input to the output, then "
    "apply that pattern to the test input to give a final output."
)

OUTPUT_INSTRUCTION = (
    'Your final response must be only a JSON object with key "test" that maps to a list of '
    "output grids, one per test input in order. Each output grid should use the same "
    "list-of-lists format as the training outputs, with integer cells from 0 to 9. Do not "
    "include markdown or explanation."
)
