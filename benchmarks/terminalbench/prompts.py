"""Prompt scaffolding for TerminalBench.

Harbor supplies the authoritative task instruction inside the sandbox via the
`{{ instruction }}` placeholder in the agent prompt template. For seq@k we append
the prior attempts' feedback below it as supplementary retry context.
"""

BASE_NOTE = (
    "Harbor provides the authoritative Terminal-Bench task instruction inside the sandbox. "
    "Verify success with terminal commands before stopping, and trust observed terminal "
    "output over self-assessment."
)


def _escape_jinja(text):
    out = str(text or "")
    for a, b in (("{{", "{ {"), ("}}", "} }"), ("{%", "{ %"), ("%}", "% }"),
                 ("{#", "{ #"), ("#}", "# }")):
        out = out.replace(a, b)
    return out


def build_prompt_template(retry_context):
    """Build the Harbor agent prompt template (a Jinja file Harbor renders)."""
    supplemental = _escape_jinja(retry_context).strip()
    if not supplemental:
        return "{{ instruction }}\n"
    return (
        "{{ instruction }}\n\n"
        "Additional seq@k retry context from earlier Harbor attempts (treat the Harbor task "
        "instruction above as authoritative if anything conflicts):\n\n"
        f"{supplemental}\n"
    )
