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

# Second person throughout: the actor is writing to its own future self, and the
# prompt says so plainly rather than dressing it up as a summarization task.
# Temperature is 0 at the call site — compression, not generation.
#
# Deliberately unprescriptive. No length cap and no checklist of what to include:
# "a summary" is its own length constraint, and WHAT to carry forward is the
# model's judgement — that judgement is part of what a self-summarization run is
# measuring. Telling it the summary REPLACES the attempt and feedback is the one
# piece of context it can't infer and genuinely needs. "Do not attempt the task
# again" is the only real guardrail: without it the summarizer starts drafting a
# better answer, which would leak an unscored extra attempt into the next prompt.
SUMMARY_PROMPT = """\
You are reviewing one of your own failed attempts at the task below.

<Task>
{task}
</Task>

<YourAttempt>
{output}
</YourAttempt>
{feedback_block}
Write a summary for your future self. A later attempt of yours will see this \
summary in place of the attempt and feedback above, so include whatever you think \
is important to carry forward.

Write only the summary. Do not attempt the task again.
"""

_FEEDBACK_BLOCK = """
<FeedbackYouReceived>
{feedback}
</FeedbackYouReceived>
"""


def summarize(model, *, task_prompt, output, feedback, template=None):
    """One summarization call on `model` (the actor's own model by default).

    Returns the summary text, or None if the model returned nothing — callers
    treat None as "no summary", which makes render_history fall back to the
    verbatim attempt rather than silently dropping it from the next prompt.
    """
    prompt = (template or SUMMARY_PROMPT).format(
        task=task_prompt,
        output=output or "",
        feedback_block=_FEEDBACK_BLOCK.format(feedback=feedback) if feedback else "",
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
