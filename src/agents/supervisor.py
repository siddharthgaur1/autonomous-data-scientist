"""Supervisor: infer the task, pick the target, produce the plan.

Everything downstream keys off `task_type` and `target_column`, so this is the
one node whose mistake is unrecoverable — a regression run against a churn flag
wastes the entire pipeline. Hence the explicit confidence score: when the goal is
vague or the target is a guess, the Supervisor says so and the graph escalates
rather than pressing on.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from ..config import get_settings
from ..state.schema import RunState, TaskType
from ..tools.llm import call_structured
from .common import describe_frame, log

NODE = "supervisor"

_SYSTEM = """You are the supervising data scientist on an automated ML pipeline.

Given a user's goal and a dataset schema, decide:
- task_type: regression (predict a number), classification (predict a category),
  timeseries (predict a value ordered over time, where past predicts future),
  or unknown if the goal cannot be mapped to the data.
- target_column: the column to predict. It MUST be one of the given columns.
- time_column: the column defining order, for timeseries only, else null.
- plan: 3-8 short ordered steps describing how you'll approach this dataset
  specifically. Reference real column names.
- confidence: 0-1. Be honest. Below 0.6 a human is asked to intervene.
  Score low when the goal is ambiguous, the target is a guess among several
  plausible columns, or the data doesn't support the goal.
- reasoning: one paragraph on why this target and task type.
"""


class SupervisorDecision(BaseModel):
    """Structured output from the supervisor's first look at the data."""

    task_type: TaskType
    target_column: str
    time_column: str | None = None
    plan: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


def supervisor_agent(state: RunState) -> RunState:
    """Infer task type and target, then set the plan."""
    df = pd.read_csv(state["raw_df_path"])

    decision = call_structured(
        state,
        NODE,
        _SYSTEM,
        f"Goal: {state['user_goal']}\n\nDataset:\n{describe_frame(df)}",
        SupervisorDecision,
    )

    messages = [
        log(NODE, f"Task inferred as {decision.task_type.value}: {decision.reasoning}")
    ]
    confidence = decision.confidence

    # The model can name a column that doesn't exist. Trust the frame, not the LLM.
    target = decision.target_column
    if target not in df.columns:
        messages.append(
            log(
                NODE,
                f"Proposed target '{target}' is not a column in the dataset.",
                "error",
            )
        )
        confidence = min(confidence, 0.2)

    if decision.task_type is TaskType.UNKNOWN:
        confidence = min(confidence, 0.3)

    needs_human = confidence < get_settings().confidence_threshold
    question = None
    if needs_human:
        question = (
            f"I read '{state['user_goal']}' as a {decision.task_type.value} problem "
            f"predicting '{target}', but I'm only {confidence:.0%} confident. "
            f"{decision.reasoning}\n\n"
            f"Available columns: {', '.join(df.columns)}.\n"
            "Which column should I predict, and is that the right kind of task?"
        )

    return RunState(
        task_type=decision.task_type,
        target_column=target,
        time_column=decision.time_column,
        plan=decision.plan,
        confidence=confidence,
        needs_human=needs_human,
        human_question=question,
        current_stage=NODE,
        messages=messages,
    )
