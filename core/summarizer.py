"""Self-summarization of prior attempts — seq@k context control.

When a run sets `summarize: true`, the ACTOR's own model compresses each failed
attempt into a short summary covering BOTH halves of what that attempt
contributes to the next prompt: the actor's own output AND the critic feedback
it received. The next attempt then carries one <AttemptSummary i> block in place
of the verbatim <PreviousAttempt i> + <Feedback i> pair.

Effect on prompt growth: per-attempt context goes from (full output + full
feedback) to one bounded summary, so cumulative input tokens over a trajectory
grow far more slowly in k. The history is a rolling LOG of per-attempt summaries
— each attempt is summarized once, independently, and kept. Nothing is re-summarized
or collapsed, so a resumed or k-extended run just appends.

Why the actor's own model: the agent summarizes for itself, so what survives into
the next attempt is what the agent chose to remember about its own failure. That's
a treatment worth measuring, not only a compression trick.

Leak safety (same contract as harness.build_prompt): the summarizer sees only
task.prompt, the actor's own output, and the critic feedback. Never task.grading,
never judge.details.

A benchmark can override the template with a module-level `SUMMARIZER_PROMPT`
(e.g. to preserve ARC grid structure); it must accept the same format fields.
"""

from __future__ import annotations

from core import llm

DEFAULT_MAX_WORDS = 200

# Written in second person: this is the actor talking to its own future self.
# Temperature is 0 at the call site — compression, not generation.
SUMMARY_PROMPT = """\
You are reviewing one of your own failed attempts at the task below, so that a \
later attempt of yours can learn from it without re-reading the whole thing.

<Task>
{task}
</Task>

<YourAttempt>
{output}
</YourAttempt>
{feedback_block}
Write a summary of at most {max_words} words, addressed to your future self, that \
preserves everything needed to do better next time:
- the approach you took and the key choices you made
- what the feedback said was wrong, missing, or unsatisfied — keep the specifics \
(names, numbers, quoted requirements); do not generalize them away
- anything you should not repeat

Write only the summary. Do not attempt the task again.
"""

_FEEDBACK_BLOCK = """
<FeedbackYouReceived>
{feedback}
</FeedbackYouReceived>
"""


def summarize(model, *, task_prompt, output, feedback,
              max_words=DEFAULT_MAX_WORDS, template=None):
    """One summarization call on `model` (the actor's own model by default).

    Returns the summary text, or None if the model returned nothing — callers
    treat None as "no summary", which makes render_history fall back to the
    verbatim attempt rather than silently dropping it from the next prompt.
    """
    prompt = (template or SUMMARY_PROMPT).format(
        task=task_prompt,
        output=output or "",
        feedback_block=_FEEDBACK_BLOCK.format(feedback=feedback) if feedback else "",
        max_words=max_words,
    )
    return (llm.complete(model, prompt, temperature=0.0) or "").strip() or None


def render_history(history):
    """Prior-attempt blocks for the next actor prompt.

    `history` entries are dicts {attempt, feedback, summary}. When a summary
    exists it REPLACES both the verbatim output and the feedback — that pair is
    exactly what it was built from, so keeping either alongside would defeat the
    compression. Entries without a summary render verbatim, which keeps a
    partially-summarized history (summarizer skipped, or a run resumed across a
    config change) correct rather than lossy.
    """
    parts = []
    for i, entry in enumerate(history, 1):
        if entry.get("summary"):
            parts.append(f"<AttemptSummary {i}>\n{entry['summary']}\n</AttemptSummary {i}>")
            continue
        parts.append(f"<PreviousAttempt {i}>\n{entry['attempt'].output}\n</PreviousAttempt {i}>")
        if entry.get("feedback"):
            parts.append(f"<Feedback {i}>\n{entry['feedback']}\n</Feedback {i}>")
    return parts
