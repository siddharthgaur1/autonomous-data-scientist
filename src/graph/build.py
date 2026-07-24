"""Wire the agents into the run graph.

Shape: the Supervisor plans, then a linear pipeline does the work, then the
Reviewer decides whether it ships. The interesting edges are the ones that aren't
linear — the Reviewer can send a stage back, and any node can bail to a human.

Every node is wrapped so its output lands in SQLite before the graph moves on.
That's what makes a run auditable: the transition log is written by the graph
itself, not by each agent remembering to do it.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from ..agents.cleaning import cleaning_agent
from ..agents.eda import eda_agent
from ..agents.evaluation import evaluation_agent
from ..agents.features import features_agent
from ..agents.human import human_escalation_agent
from ..agents.model_selection import model_selection_agent
from ..agents.narrative import narrative_agent
from ..agents.report import report_agent
from ..agents.reviewer import reviewer_agent
from ..agents.supervisor import supervisor_agent
from ..agents.tuning import tuning_agent
from ..persistence.store import RunStore
from ..state.schema import AgentMessage, RunState, RunStatus
from ..tools.llm import CostCapExceeded

#: The linear spine, in order. The Reviewer's `retry_stage` names one of these.
PIPELINE = [
    "cleaning",
    "eda",
    "features",
    "model_selection",
    "tuning",
    "evaluation",
    "narrative",
    "report",
]

_AGENTS: dict[str, Callable[[RunState], RunState]] = {
    "supervisor": supervisor_agent,
    "cleaning": cleaning_agent,
    "eda": eda_agent,
    "features": features_agent,
    "model_selection": model_selection_agent,
    "tuning": tuning_agent,
    "evaluation": evaluation_agent,
    "narrative": narrative_agent,
    "report": report_agent,
    "reviewer": reviewer_agent,
    "human_escalation": human_escalation_agent,
}


def _merged(state: RunState, update: RunState) -> RunState:
    """Approximate what the graph will hold after this update, for persistence.

    The additive fields have to be merged by hand — a node returns only what it
    appended, so a naive `{**state, **update}` would truncate the log to the last
    node's messages.
    """
    merged = {**state, **update}
    for field in ("messages", "artifacts"):
        merged[field] = list(state.get(field, [])) + list(update.get(field, []))
    return merged  # type: ignore[return-value]


def _instrument(
    name: str, fn: Callable[[RunState], RunState], store: RunStore | None
) -> Callable[[RunState], RunState]:
    """Wrap a node with persistence and a failure net.

    An agent raising is a normal event here — a bad API key, a cost cap, a model
    returning something unparseable. The graph turns that into a failed run with
    a readable message instead of a stack trace in the API's logs.
    """

    def node(state: RunState) -> RunState:
        try:
            update = fn(state)
        except CostCapExceeded as exc:
            update = RunState(
                status=RunStatus.FAILED,
                error=str(exc),
                messages=[AgentMessage(agent=name, content=str(exc), level="error")],
            )
        except Exception as exc:  # noqa: BLE001 - the boundary where runs fail safely
            update = RunState(
                status=RunStatus.FAILED,
                error=f"{name} failed: {type(exc).__name__}: {exc}",
                messages=[
                    AgentMessage(
                        agent=name,
                        content=f"Unrecoverable error: {type(exc).__name__}: {exc}",
                        level="error",
                    )
                ],
            )

        if store is not None:
            try:
                store.record_transition(_merged(state, update), stage=name)
            except Exception:  # noqa: BLE001, S110 - the audit log must never kill the run
                pass
        return update

    return node


def _after_supervisor(state: RunState) -> str:
    if state.get("status") == RunStatus.FAILED or state.get("error"):
        return "fail"
    if state.get("needs_human"):
        return "human_escalation"
    return "cleaning"


def _continue_or_stop(next_stage: str) -> Callable[[RunState], str]:
    """A linear step that still respects failure."""

    def route(state: RunState) -> str:
        if state.get("status") == RunStatus.FAILED or state.get("error"):
            return "fail"
        return next_stage

    return route


def _after_reviewer(state: RunState) -> str:
    """Approve ends the run; revise loops back; escalate asks a human."""
    if state.get("status") == RunStatus.FAILED:
        return "fail"
    verdict = state.get("reviewer_verdict")
    if verdict == "revise":
        # A revise verdict must never fall through to "done" — that would ship a
        # run the reviewer explicitly rejected. An unusable retry_stage means a
        # human decides.
        if state.get("retry_stage") in PIPELINE:
            return state["retry_stage"]
        return "human_escalation"
    if verdict == "escalate" or state.get("needs_human"):
        return "human_escalation"
    return "done"


def _finalise(state: RunState) -> RunState:
    """Mark a run completed. Reached only when the Reviewer approved."""
    return RunState(
        status=RunStatus.COMPLETED,
        current_stage="done",
        messages=[
            AgentMessage(
                agent="graph",
                content=f"Run completed. Confidence {state.get('confidence', 0):.0%}.",
            )
        ],
    )


def _fail(state: RunState) -> RunState:
    """Terminal node for unrecoverable errors."""
    return RunState(
        status=RunStatus.FAILED,
        current_stage="failed",
        messages=[
            AgentMessage(
                agent="graph",
                content=f"Run ended without a result: {state.get('error')}",
                level="error",
            )
        ],
    )


def build_graph(store: RunStore | None = None, checkpointer=None):
    """Compile the run graph. `store` is optional so tests can skip persistence."""
    builder = StateGraph(RunState)

    for name, fn in _AGENTS.items():
        builder.add_node(name, _instrument(name, fn, store))
    builder.add_node("done", _instrument("done", _finalise, store))
    builder.add_node("fail", _instrument("fail", _fail, store))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _after_supervisor,
        {"cleaning": "cleaning", "human_escalation": "human_escalation", "fail": "fail"},
    )

    # The linear spine: each stage flows to the next, or bails out on error.
    for current, following in zip(PIPELINE, PIPELINE[1:] + ["reviewer"]):
        builder.add_conditional_edges(
            current,
            _continue_or_stop(following),
            {following: following, "fail": "fail"},
        )

    builder.add_conditional_edges(
        "reviewer",
        _after_reviewer,
        {
            **{stage: stage for stage in PIPELINE},
            "human_escalation": "human_escalation",
            "done": "done",
            "fail": "fail",
        },
    )

    # Escalation is a pause, not a stage: the run stops here and the checkpointer
    # holds the state until a human answers and the API resumes the thread.
    builder.add_edge("human_escalation", END)
    builder.add_edge("done", END)
    builder.add_edge("fail", END)

    return builder.compile(checkpointer=checkpointer)
