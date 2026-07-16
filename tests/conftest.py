"""Shared fixtures. No test in this suite may touch the OpenAI API.

`_isolate` is autouse and points every path-producing setting at a tmpdir, so a
test run can't write into the developer's real `runs/` or read their real `.env`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import get_settings
from src.persistence.store import RunStore
from src.state.schema import (
    CandidateModel,
    EDAFindings,
    EvalMetrics,
    RunState,
    TaskType,
    TuningResults,
    new_run_state,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point config at a throwaway directory and a fake key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MAX_RUN_COST_USD", "1.0")
    monkeypatch.setenv("SANDBOX_TIMEOUT_S", "60")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def store(settings) -> RunStore:
    return RunStore(settings.db_path)


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """A 30-row classification fixture with a signal, a mess, and a gap.

    Small enough to run the whole graph in a test, real enough that sklearn
    doesn't complain and the cleaning agent has something to fix.
    """
    rng = np.random.default_rng(7)
    n = 30
    tenure = rng.integers(1, 60, n)
    charges = rng.uniform(20, 120, n).round(2)
    logit = -2.0 + 0.05 * charges - 0.04 * tenure
    churn = (1 / (1 + np.exp(-logit)) > rng.uniform(0, 1, n)).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": [f"{c:,.2f}" for c in charges],  # messy: text numbers
            "plan": rng.choice(["basic", "BASIC", "premium"], n),  # messy: casing
            "churned": churn,
        }
    )
    df.loc[[3, 11], "tenure_months"] = np.nan  # missing values

    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_df(tiny_csv: Path) -> pd.DataFrame:
    return pd.read_csv(tiny_csv)


@pytest.fixture
def clean_csv(tmp_path: Path) -> Path:
    """An already-clean version, for agents downstream of cleaning."""
    rng = np.random.default_rng(7)
    n = 30
    charges = rng.uniform(20, 120, n).round(2)
    tenure = rng.integers(1, 60, n)
    logit = -2.0 + 0.05 * charges - 0.04 * tenure
    churn = (1 / (1 + np.exp(-logit)) > rng.uniform(0, 1, n)).astype(int)
    # Guarantee both classes exist; stratified splits need it and n=30 is small.
    churn[:8] = 0
    churn[8:16] = 1

    path = tmp_path / "clean.csv"
    pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": charges,
            "plan": rng.choice(["basic", "premium"], n),
            "churned": churn,
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def base_state(tiny_csv: Path) -> RunState:
    """A run state part-way through the pipeline."""
    state = new_run_state("testrun0001", "predict churn", str(tiny_csv))
    state["task_type"] = TaskType.CLASSIFICATION
    state["target_column"] = "churned"
    return state


@pytest.fixture
def completed_state(base_state: RunState, clean_csv: Path) -> RunState:
    """A run with everything filled in, for reviewer/narrative/QA tests."""
    state = dict(base_state)
    state["clean_df_path"] = str(clean_csv)
    state["chosen_model"] = "RandomForest"
    state["eda_findings"] = EDAFindings(
        n_rows=30, n_cols=4, target="churned",
        summary="Charges correlate with churn.",
        correlations={"monthly_charges": 0.42},
    )
    state["candidate_models"] = [
        CandidateModel(name="RandomForest", estimator="RandomForestClassifier",
                       baseline_score=0.72, metric="roc_auc", notes="5 folds"),
        CandidateModel(name="LogisticRegression", estimator="LogisticRegression",
                       baseline_score=0.68, metric="roc_auc", notes="5 folds"),
    ]
    state["tuning_results"] = TuningResults(
        search_space={"n_estimators": ["int", 100, 600]},
        best_params={"n_estimators": 200}, best_score=0.75,
        baseline_score=0.72, n_trials=25,
    )
    state["eval_metrics"] = EvalMetrics(
        metrics={"roc_auc": 0.74, "accuracy": 0.70},
        split="Stratified random 80/20 split, seed 42.",
        interpretation="Better than guessing, not production-ready.",
    )
    state["narrative"] = "The model reaches 0.74 roc_auc on held-out customers."
    return state  # type: ignore[return-value]
