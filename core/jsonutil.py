"""Tolerant JSON parsing for LLM-emitted payloads.

Rubric judges are asked for JSON and mostly comply, but they quote user text back
inside string fields — regexes, Windows paths, LaTeX — and emit sequences like
`\\d`, `\\(`, `\\Users`. Those are ILLEGAL in JSON (only \\" \\\\ \\/ \\b \\f \\n \\r \\t
\\uXXXX are valid), so json.loads raises `Invalid \\escape` on output that is
otherwise perfectly well-formed and semantically fine.

That mattered here: a single such response killed an entire multi-hour grid run,
because the benchmark parsers are deliberately fail-loud.

`loads_lenient` tries a strict parse first and only on failure escapes the stray
backslashes and retries. Strictly-valid JSON therefore takes the identical path
it always did — this can only rescue input that would otherwise have raised, and
never changes how a valid payload is interpreted.
"""

from __future__ import annotations

import json
import re

# A backslash NOT starting one of JSON's legal escapes.
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})')


def loads_lenient(text):
    """json.loads, retried once with stray backslashes escaped.

    Raises the ORIGINAL JSONDecodeError if the repaired text still won't parse,
    so genuinely malformed judge output stays as loud as before.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as first:
        try:
            return json.loads(_BAD_ESCAPE.sub(r"\\\\", text))
        except json.JSONDecodeError:
            raise first
