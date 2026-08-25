"""Actor prompt for HumanEval Pro."""

ACTOR_PROMPT = """You are solving a Python code generation task.

There are two related functions:
1. Base function: solve the simpler base problem.
2. Main function: solve the harder problem by using or building on the base function.

Write complete Python code that defines all required functions.
Do not import unnecessary packages.
Do not include explanations outside the code.

Base problem:
```python
{base_problem}
```

Main problem:
```python
{main_problem}
```

Return only the Python code."""
