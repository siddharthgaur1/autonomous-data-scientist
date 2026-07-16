"""Model building, splitting and scoring.

Deliberately *not* LLM-generated. The agents above this layer decide what to
model and why; the fitting itself is deterministic code for two reasons: a fitted
estimator cannot cross the sandbox's JSON boundary, and the split logic is where
leakage creeps in — that belongs in reviewed code, not in a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..state.schema import TaskType


@dataclass
class Split:
    """A train/holdout split plus a description of how it was made."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    description: str


def is_classification(task: TaskType) -> bool:
    return task is TaskType.CLASSIFICATION


def primary_metric(task: TaskType) -> str:
    """The metric used to rank candidates and to drive the Optuna study."""
    return "roc_auc" if is_classification(task) else "r2"


def make_split(
    df: pd.DataFrame,
    target: str,
    task: TaskType,
    test_size: float = 0.2,
    time_column: str | None = None,
) -> Split:
    """Split into train and holdout, respecting time order for timeseries.

    A random split on a timeseries lets the model learn from the future and
    score beautifully on data it has effectively already seen. For
    TaskType.TIMESERIES the split is strictly time-ordered instead.
    """
    X = df.drop(columns=[target])
    y = df[target]

    if task is TaskType.TIMESERIES:
        if time_column and time_column in X.columns:
            order = pd.to_datetime(X[time_column], errors="coerce").argsort()
            X, y = X.iloc[order], y.iloc[order]
        cut = int(len(X) * (1 - test_size))
        desc = (
            f"Time-ordered split: first {cut} rows train, "
            f"last {len(X) - cut} rows holdout (no shuffling)."
        )
        return Split(X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:], desc)

    from sklearn.model_selection import train_test_split

    stratify = y if is_classification(task) and y.nunique() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    kind = "Stratified random" if stratify is not None else "Random"
    desc = f"{kind} {int((1 - test_size) * 100)}/{int(test_size * 100)} split, seed 42."
    return Split(X_tr, X_te, y_tr, y_te, desc)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot low-cardinality categoricals.

    Fitted inside the pipeline on the training fold only, so imputation
    statistics never see the holdout.
    """
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [
        c
        for c in X.select_dtypes(include=["object", "category", "bool"]).columns
        if X[c].nunique() <= 30
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def candidate_estimators(task: TaskType) -> dict[str, Any]:
    """The shortlist of estimators for a task type.

    Kept to a linear baseline, a tree, and two ensembles: enough spread to show
    whether the problem needs non-linearity, without a 40-model bake-off.
    """
    if is_classification(task):
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=6),
            "RandomForest": RandomForestClassifier(random_state=42, n_estimators=200),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
        }
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "DecisionTree": DecisionTreeRegressor(random_state=42, max_depth=6),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=200),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }


def make_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    """Wrap an estimator with preprocessing so the whole thing is one fitted object."""
    return Pipeline([("prep", build_preprocessor(X)), ("model", estimator)])


def score(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, task: TaskType) -> float:
    """Score on the primary metric for the task."""
    return evaluate(pipeline, X, y, task)[primary_metric(task)]


def evaluate(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, task: TaskType
) -> dict[str, float]:
    """Full metric set for a fitted pipeline on held-out data."""
    preds = pipeline.predict(X)

    if is_classification(task):
        avg = "binary" if pd.Series(y).nunique() <= 2 else "macro"
        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, average=avg, zero_division=0)),
            "recall": float(recall_score(y, preds, average=avg, zero_division=0)),
            "f1": float(f1_score(y, preds, average=avg, zero_division=0)),
        }
        metrics["roc_auc"] = _safe_auc(pipeline, X, y)
        return metrics

    resid = np.asarray(y) - np.asarray(preds)
    return {
        "r2": float(r2_score(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
    }


def _safe_auc(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """ROC AUC, or 0.0 where it isn't defined (single class, no probabilities)."""
    if not hasattr(pipeline, "predict_proba") or pd.Series(y).nunique() < 2:
        return 0.0
    try:
        proba = pipeline.predict_proba(X)
        if proba.shape[1] == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return 0.0


def feature_names(pipeline: Pipeline) -> list[str]:
    """Column names after preprocessing, for aligning importances."""
    try:
        return list(pipeline.named_steps["prep"].get_feature_names_out())
    except (AttributeError, KeyError, ValueError):
        return []


def feature_importances(pipeline: Pipeline) -> dict[str, float]:
    """Importances or |coefficients| from the fitted model, keyed by feature name.

    This is what the follow-up Q&A reads, so it comes from the real fitted
    estimator rather than from anything the LLM remembers writing.
    """
    model = pipeline.named_steps.get("model")
    names = feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return {}

    if not names or len(names) != len(values):
        names = [f"f{i}" for i in range(len(values))]

    pairs = sorted(zip(names, values), key=lambda kv: kv[1], reverse=True)
    return {name: float(value) for name, value in pairs}
