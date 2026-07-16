"""The single state object that flows through the whole graph.

LangGraph wants a TypedDict at the top level so it can merge partial updates from
each node. The nested payloads are Pydantic models because they cross the LLM
boundary and benefit from validation. `messages` and `artifacts` use additive
reducers, so a node returns only what it appended, not the whole list.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """What kind of ML problem the Supervisor inferred from the goal."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIMESERIES = "timeseries"
    UNKNOWN = "unknown"


class RunStatus(str, Enum):
    """Terminal and in-flight states for a run."""

    RUNNING = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentMessage(BaseModel):
    """One entry in the agent log. Appended by every node, never overwritten."""

    agent: str
    content: str
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: Literal["info", "warning", "error"] = "info"


class Artifact(BaseModel):
    """A file a node produced, relative to the run directory."""

    kind: Literal["plot", "report", "model", "data", "model_card"]
    path: str
    label: str = ""


class Transformation(BaseModel):
    """One cleaning step, recorded so the Q&A agent can explain it later."""

    column: str
    action: str
    reason: str
    rows_affected: int = 0


class EDAFindings(BaseModel):
    """What the EDA agent learned. Claims in the narrative must cite these."""

    n_rows: int = 0
    n_cols: int = 0
    target: str | None = None
    summary: str = ""
    correlations: dict[str, float] = Field(default_factory=dict)
    anomalies: list[str] = Field(default_factory=list)


class EngineeredFeature(BaseModel):
    """A proposed feature and the reasoning behind it."""

    name: str
    reasoning: str
    kept: bool = True
    drop_reason: str = ""


class CandidateModel(BaseModel):
    """A baseline model trained during selection."""

    name: str
    estimator: str
    baseline_score: float
    metric: str
    notes: str = ""


class TuningResults(BaseModel):
    """Optuna study output for the chosen model."""

    search_space: dict[str, Any] = Field(default_factory=dict)
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_score: float = 0.0
    baseline_score: float = 0.0
    n_trials: int = 0

    @property
    def improvement(self) -> float:
        """Absolute gain of the tuned model over its own baseline."""
        return self.best_score - self.baseline_score


class EvalMetrics(BaseModel):
    """Holdout metrics plus a business-readable reading of them."""

    metrics: dict[str, float] = Field(default_factory=dict)
    split: str = ""
    interpretation: str = ""


class TokenUsage(BaseModel):
    """Per-node LLM spend, accumulated to enforce the per-run cost cap."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class RunState(TypedDict, total=False):
    """State passed between every node in the graph.

    `total=False` because nodes fill fields in progressively; the graph is
    responsible for ordering, not the type system.
    """

    run_id: str
    user_goal: str
    status: RunStatus
    task_type: TaskType
    plan: list[str]
    current_stage: str

    raw_df_path: str
    clean_df_path: str | None
    target_column: str | None
    time_column: str | None
    transformations: list[Transformation]

    eda_findings: EDAFindings | None
    engineered_features: list[EngineeredFeature]
    candidate_models: list[CandidateModel]
    chosen_model: str | None
    tuning_results: TuningResults | None
    eval_metrics: EvalMetrics | None
    narrative: str

    confidence: float
    needs_human: bool
    human_question: str | None
    human_answer: str | None

    reviewer_verdict: str | None
    retry_stage: str | None
    retry_counts: dict[str, int]
    error: str | None

    messages: Annotated[list[AgentMessage], operator.add]
    artifacts: Annotated[list[Artifact], operator.add]
    token_usage: dict[str, TokenUsage]


def new_run_state(run_id: str, user_goal: str, raw_df_path: str) -> RunState:
    """Build the initial state for a fresh run."""
    return RunState(
        run_id=run_id,
        user_goal=user_goal,
        status=RunStatus.RUNNING,
        task_type=TaskType.UNKNOWN,
        plan=[],
        current_stage="supervisor",
        raw_df_path=raw_df_path,
        clean_df_path=None,
        target_column=None,
        time_column=None,
        transformations=[],
        eda_findings=None,
        engineered_features=[],
        candidate_models=[],
        chosen_model=None,
        tuning_results=None,
        eval_metrics=None,
        narrative="",
        confidence=1.0,
        needs_human=False,
        human_question=None,
        human_answer=None,
        reviewer_verdict=None,
        retry_stage=None,
        retry_counts={},
        error=None,
        messages=[],
        artifacts=[],
        token_usage={},
    )
