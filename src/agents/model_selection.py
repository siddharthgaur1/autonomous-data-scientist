"""Model selection: train a shortlist, compare, explain the pick.

Candidates are scored by cross-validation **on the training split only**. The
holdout is not touched here — if it were, the final metrics would be reporting on
data that already influenced which model got chosen, and the number would be
optimistic for reasons nobody would notice.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..state.schema import CandidateModel, RunState, TaskType
from ..tools import ml
from ..tools.llm import call_structured
from .common import load_active_df, log

NODE = "model_selection"

_SYSTEM = """You are a data scientist choosing a model.

You are given cross-validated scores for several candidates on a real dataset.
Pick the one to take forward and explain the shortlist to a stakeholder.

Prefer the best score, but say so when a simpler model is within noise of a
complex one — an interpretable model that scores 0.002 lower is usually the
better business choice, and you should recommend it when that is the case.

- chosen: must be exactly one of the candidate names given.
- rationale: 2-4 sentences a non-specialist can follow.
"""


class SelectionDecision(BaseModel):
    """Which candidate to take forward, and why."""

    chosen: str
    rationale: str


def model_selection_agent(state: RunState) -> RunState:
    """Cross-validate the candidate shortlist and choose one."""
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    df = load_active_df(state)
    target = state["target_column"]
    task = TaskType(state["task_type"])

    if task is TaskType.UNKNOWN:
        return RunState(
            current_stage=NODE,
            error="Cannot select a model for an unknown task type.",
            messages=[log(NODE, "Task type is unknown; nothing to train.", "error")],
        )

    df = df.dropna(subset=[target])
    split = ml.make_split(df, target, task, time_column=state.get("time_column"))
    metric = ml.primary_metric(task)
    scoring = "roc_auc" if metric == "roc_auc" else "r2"

    # Time order must survive cross-validation too, not just the final split.
    cv = TimeSeriesSplit(n_splits=3) if task is TaskType.TIMESERIES else 5

    candidates: list[CandidateModel] = []
    messages = []

    for name, estimator in ml.candidate_estimators(task).items():
        pipeline = ml.make_pipeline(split.X_train, estimator)
        try:
            scores = cross_val_score(
                pipeline, split.X_train, split.y_train, cv=cv, scoring=scoring,
                error_score="raise",
            )
        except Exception as exc:  # noqa: BLE001 - one bad candidate shouldn't end the run
            messages.append(log(NODE, f"{name} failed to fit: {exc}", "warning"))
            continue

        candidates.append(
            CandidateModel(
                name=name,
                estimator=type(estimator).__name__,
                baseline_score=round(float(scores.mean()), 4),
                metric=metric,
                notes=f"cross-validated over {len(scores)} folds, "
                      f"sd {scores.std():.4f}",
            )
        )

    if not candidates:
        return RunState(
            current_stage=NODE,
            error="Every candidate model failed to fit.",
            messages=[*messages, log(NODE, "No candidate could be trained.", "error")],
        )

    candidates.sort(key=lambda c: c.baseline_score, reverse=True)
    table = "\n".join(
        f"  {c.name} ({c.estimator}): {c.metric}={c.baseline_score} [{c.notes}]"
        for c in candidates
    )

    decision = call_structured(
        state,
        NODE,
        _SYSTEM,
        f"Goal: {state['user_goal']}\nTask: {task.value}\nTarget: {target}\n"
        f"Split: {split.description}\n\nCandidates:\n{table}",
        SelectionDecision,
    )

    chosen = decision.chosen
    if chosen not in {c.name for c in candidates}:
        chosen = candidates[0].name
        messages.append(
            log(NODE, f"Model named a candidate that wasn't on the list; "
                      f"falling back to the best scorer, {chosen}.", "warning")
        )

    messages.append(log(NODE, f"Chose {chosen}. {decision.rationale}"))

    return RunState(
        candidate_models=candidates,
        chosen_model=chosen,
        current_stage=NODE,
        messages=messages,
    )
