"""Human escalation: stop and ask, rather than guess.

This node does almost nothing, and that's the point. It marks the run as awaiting
a human and leaves a specific question in state. LangGraph's checkpointer holds
everything else, so answering resumes the run exactly where it paused instead of
starting over.

The value is in what it *prevents*: a low-confidence run that would otherwise
have invented a plausible answer and presented it with the same polish as a good
one.
"""

from __future__ import annotations

from ..state.schema import RunState, RunStatus
from .common import log

NODE = "human_escalation"

_FALLBACK = (
    "I'm not confident enough in this run to continue without input. "
    "Could you clarify what you'd like me to predict, and from which column?"
)


def human_escalation_agent(state: RunState) -> RunState:
    """Pause the run with a specific question for the user."""
    question = state.get("human_question") or _FALLBACK

    return RunState(
        status=RunStatus.AWAITING_HUMAN,
        needs_human=True,
        human_question=question,
        current_stage=NODE,
        messages=[
            log(
                NODE,
                f"Paused for human input (confidence {state.get('confidence', 0):.0%}): "
                f"{question.splitlines()[0]}",
                "warning",
            )
        ],
    )


def resume_after_human(state: RunState) -> RunState:
    """Fold a human's answer back into the run and clear the escalation flag.

    The answer is appended to the goal so every downstream prompt sees it: the
    Supervisor re-reads the goal on resume, and a clarification that only lived
    in `human_answer` would be invisible to it.
    """
    answer = (state.get("human_answer") or "").strip()
    if not answer:
        return RunState(current_stage=NODE)

    return RunState(
        status=RunStatus.RUNNING,
        user_goal=f"{state['user_goal']}\n\nUser clarification: {answer}",
        needs_human=False,
        human_question=None,
        confidence=0.75,
        current_stage="supervisor",
        messages=[log(NODE, f"Human answered: {answer}")],
    )
