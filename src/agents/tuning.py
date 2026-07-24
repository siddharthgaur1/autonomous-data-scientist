"""Tuning: an Optuna study over the chosen model's hyperparameters.

Search spaces are hand-written per estimator rather than LLM-generated. A model
inventing `n_estimators=-4` costs a whole trial to discover, and the spaces below
are the boring, well-understood ones anyway.

Like selection, this only ever sees the training split; the holdout stays sealed
until the Evaluation agent opens it exactly once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna

from ..state.schema import RunState, TaskType, TuningResults
from ..tools import ml
from .common import load_active_df, log

NODE = "tuning"

N_TRIALS = 25

#: Documented search spaces, keyed by the candidate name used in selection.
_SPACES: dict[str, dict[str, Any]] = {
    "RandomForest": {
        "n_estimators": ("int", 100, 600),
        "max_depth": ("int", 3, 20),
        "min_samples_leaf": ("int", 1, 12),
        "max_features": ("categorical", ["sqrt", "log2", None]),
    },
    "GradientBoosting": {
        "n_estimators": ("int", 60, 400),
        "learning_rate": ("float_log", 0.01, 0.3),
        "max_depth": ("int", 2, 7),
        "subsample": ("float", 0.6, 1.0),
    },
    "DecisionTree": {
        "max_depth": ("int", 2, 18),
        "min_samples_leaf": ("int", 1, 20),
        "min_samples_split": ("int", 2, 20),
    },
    "Ridge": {"alpha": ("float_log", 0.001, 100.0)},
    "LogisticRegression": {
        "C": ("float_log", 0.001, 100.0),
        "max_iter": ("categorical", [1000]),
    },
    "LinearRegression": {},
}


def _suggest(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    """Turn the declarative space above into Optuna suggestions."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def _objective(
    space: dict[str, Any],
    estimator_factory: Callable[[dict[str, Any]], Any],
    X,
    y,
    cv,
    scoring: str,
) -> Callable[[optuna.Trial], float]:
    from sklearn.model_selection import cross_val_score

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, space)
        pipeline = ml.make_pipeline(X, estimator_factory(params))
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        return float(scores.mean())

    return objective


def tuning_agent(state: RunState) -> RunState:
    """Run the study and record the improvement over the model's own baseline."""
    from sklearn.model_selection import TimeSeriesSplit

    chosen = state.get("chosen_model")
    task = TaskType(state["task_type"])
    target = state["target_column"]

    baseline = next(
        (c.baseline_score for c in state.get("candidate_models", []) if c.name == chosen),
        0.0,
    )
    space = _SPACES.get(chosen or "", {})

    if not space:
        messages = [
            log(NODE, f"{chosen} has no hyperparameters worth tuning; "
                      "keeping the baseline configuration.")
        ]
        return RunState(
            tuning_results=TuningResults(
                search_space={}, best_params={}, best_score=baseline,
                baseline_score=baseline, n_trials=0,
            ),
            current_stage=NODE,
            messages=messages,
        )

    df = load_active_df(state).dropna(subset=[target])
    split = ml.make_split(df, target, task, time_column=state.get("time_column"))
    metric = ml.primary_metric(task)
    scoring = "roc_auc" if metric == "roc_auc" else "r2"
    cv = TimeSeriesSplit(n_splits=3) if task is TaskType.TIMESERIES else 5

    base_estimator = ml.candidate_estimators(task)[chosen]

    def factory(params: dict[str, Any]) -> Any:
        estimator = type(base_estimator)(**{**base_estimator.get_params(), **params})
        return estimator

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        _objective(space, factory, split.X_train, split.y_train, cv, scoring),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best_score = round(float(study.best_value), 4)
    results = TuningResults(
        search_space={k: list(v) for k, v in space.items()},
        best_params=study.best_params,
        best_score=best_score,
        baseline_score=baseline,
        n_trials=len(study.trials),
    )

    delta = best_score - baseline
    direction = "improved" if delta >= 0 else "did not improve"
    messages = [
        log(
            NODE,
            f"Optuna ran {results.n_trials} trials on {chosen}; {metric} {direction} "
            f"{baseline:.4f} -> {best_score:.4f} ({delta:+.4f}). "
            f"Best params: {study.best_params}",
        )
    ]
    if delta < 0:
        messages.append(
            log(NODE, "Tuning underperformed the baseline; the baseline "
                      "configuration will be used instead.", "warning")
        )

    return RunState(tuning_results=results, current_stage=NODE, messages=messages)
