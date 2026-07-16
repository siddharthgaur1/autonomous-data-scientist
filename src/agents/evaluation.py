"""Evaluation: fit the final model, open the holdout once, interpret the result.

This is the only node that touches the holdout, and it does so exactly once. It
also exports the model (Phase 6), because the thing that gets scored and the
thing that gets shipped must be the same fitted object — refitting later "just to
save it" is how a .pkl ends up not matching its own metrics.

If tuning underperformed its baseline, the baseline configuration ships. Reporting
a tuned model that's worse than where it started would be honest about the number
and dishonest about the choice.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..state.schema import Artifact, EvalMetrics, RunState, TaskType
from ..tools import ml, plotting
from ..tools.codegen import run_dir_for
from ..tools.llm import call_structured
from ..tools.model_io import save_model
from .common import load_active_df, log, relative_to_run

NODE = "evaluation"

_SYSTEM = """You are explaining a model's holdout performance to a business audience.

You are given real metrics from held-out data the model never saw. Write an
interpretation that:
- says plainly how good this is, in the language of the user's goal
- names the practical limitation a decision-maker should know about
- does NOT invent numbers; reference only the metrics given

For classification: accuracy alone is misleading on imbalanced data — say so if
the balance warrants it. For regression: put the error in the target's own units.
"""


class Interpretation(BaseModel):
    """A business-readable reading of the holdout metrics."""

    interpretation: str


def evaluation_agent(state: RunState) -> RunState:
    """Fit on train, score on holdout, export the fitted pipeline."""
    df = load_active_df(state).dropna(subset=[state["target_column"]])
    target = state["target_column"]
    task = TaskType(state["task_type"])
    chosen = state["chosen_model"]
    run_dir = run_dir_for(state)

    split = ml.make_split(df, target, task, time_column=state.get("time_column"))

    tuning = state.get("tuning_results")
    params = {}
    if tuning and tuning.best_params and tuning.best_score >= tuning.baseline_score:
        params = tuning.best_params

    base = ml.candidate_estimators(task)[chosen]
    estimator = type(base)(**{**base.get_params(), **params})
    pipeline = ml.make_pipeline(split.X_train, estimator)
    pipeline.fit(split.X_train, split.y_train)

    metrics = ml.evaluate(pipeline, split.X_test, split.y_test, task)
    metrics = {k: round(v, 4) for k, v in metrics.items()}

    interpretation = call_structured(
        state,
        NODE,
        _SYSTEM,
        f"Goal: {state['user_goal']}\nTask: {task.value}\nTarget: {target}\n"
        f"Model: {chosen} with params {params or 'defaults'}\n"
        f"Split: {split.description}\n"
        f"Holdout size: {len(split.X_test)} rows\n"
        f"Holdout metrics: {metrics}\n"
        f"Target distribution: {split.y_train.describe().to_dict()}",
        Interpretation,
    )

    artifacts: list[Artifact] = []
    importances = ml.feature_importances(pipeline)
    if importances:
        plotting.register_template()
        path = plotting.save_fig(
            plotting.feature_importance(importances), run_dir, "feature_importance"
        )
        artifacts.append(
            Artifact(kind="plot", path=relative_to_run(path, run_dir),
                     label="Feature importance")
        )

    if state.get("candidate_models"):
        plotting.register_template()
        names = [c.name for c in state["candidate_models"]]
        scores = [c.baseline_score for c in state["candidate_models"]]
        path = plotting.save_fig(
            plotting.model_comparison(names, scores, ml.primary_metric(task)),
            run_dir,
            "model_comparison",
        )
        artifacts.append(
            Artifact(kind="plot", path=relative_to_run(path, run_dir),
                     label="Model comparison")
        )

    model_path, card_path = save_model(
        pipeline,
        run_dir,
        task_type=task.value,
        target=target,
        features=list(split.X_train.columns),
        metrics=metrics,
        training_data_path=state["raw_df_path"],
        best_params=params,
    )
    artifacts += [
        Artifact(kind="model", path=relative_to_run(model_path, run_dir),
                 label="Fitted pipeline"),
        Artifact(kind="model_card", path=relative_to_run(card_path, run_dir),
                 label="Model card"),
    ]

    headline = ml.primary_metric(task)
    messages = [
        log(NODE, f"Holdout {headline}={metrics.get(headline)} on "
                  f"{len(split.X_test)} unseen rows. {split.description}"),
        log(NODE, f"Exported model.pkl and model_card.json."),
    ]

    return RunState(
        eval_metrics=EvalMetrics(
            metrics=metrics,
            split=split.description,
            interpretation=interpretation.interpretation,
        ),
        current_stage=NODE,
        messages=messages,
        artifacts=artifacts,
    )
