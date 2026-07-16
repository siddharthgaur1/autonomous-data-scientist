"""EDA: distributions, correlations, target relationship, anomalies.

The figures are built by `tools.plotting`, not by generated code. They go into a
slide deck a human presents, and a model improvising chart styling produces a
different-looking deck on every run. The LLM's job here is reading the computed
statistics, not drawing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..state.schema import Artifact, EDAFindings, RunState
from ..tools import plotting
from ..tools.codegen import run_dir_for
from ..tools.llm import call_structured
from .common import describe_frame, load_active_df, log, relative_to_run

NODE = "eda"

_SYSTEM = """You are a data scientist interpreting exploratory statistics.

You are given a dataset schema and computed statistics. Write findings that a
business stakeholder can act on. Only state things the numbers support — you are
being audited, and every claim is checked against the statistics you were given.

- summary: 3-5 sentences on what this data looks like and what predicts the target.
- anomalies: concrete oddities worth a human's attention (imbalance, suspicious
  correlations near 1.0 that suggest leakage, implausible ranges, sparse columns).
  Empty list if genuinely nothing stands out.
"""


class EDAInterpretation(BaseModel):
    """The LLM's reading of the computed statistics."""

    summary: str
    anomalies: list[str] = Field(default_factory=list)


def _target_correlations(df: pd.DataFrame, target: str) -> dict[str, float]:
    """Correlation of each numeric feature with the target, strongest first."""
    numeric = df.select_dtypes(include=np.number)
    if target not in numeric.columns or numeric.shape[1] < 2:
        return {}
    corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    corr = corr.dropna().sort_values(key=abs, ascending=False)
    return {k: round(float(v), 4) for k, v in corr.items()}


def eda_agent(state: RunState) -> RunState:
    """Compute statistics, emit plots, and interpret the result."""
    df = load_active_df(state)
    run_dir = run_dir_for(state)
    target = state.get("target_column")
    plotting.register_template()

    artifacts: list[Artifact] = []
    messages = []

    raw_df = pd.read_csv(state["raw_df_path"])
    fig = plotting.missingness(raw_df)
    if fig is not None:
        path = plotting.save_fig(fig, run_dir, "missingness")
        artifacts.append(
            Artifact(kind="plot", path=relative_to_run(path, run_dir),
                     label="Missing values before cleaning")
        )

    if target and target in df.columns:
        path = plotting.save_fig(
            plotting.target_distribution(df, target), run_dir, "target_distribution"
        )
        artifacts.append(
            Artifact(kind="plot", path=relative_to_run(path, run_dir),
                     label=f"Distribution of {target}")
        )

    fig = plotting.correlation_heatmap(df)
    if fig is not None:
        path = plotting.save_fig(fig, run_dir, "correlations")
        artifacts.append(
            Artifact(kind="plot", path=relative_to_run(path, run_dir),
                     label="Correlation between numeric features")
        )

    correlations = _target_correlations(df, target) if target else {}
    for feature in list(correlations)[:2]:
        path = plotting.save_fig(
            plotting.target_relationship(df, feature, target),
            run_dir,
            f"relationship_{feature}",
        )
        artifacts.append(
            Artifact(kind="plot", path=relative_to_run(path, run_dir),
                     label=f"{target} vs {feature}")
        )

    stats_block = (
        f"{describe_frame(df)}\n\n"
        f"Numeric summary:\n{df.describe().round(3).to_string()}\n\n"
        f"Correlation with target '{target}':\n"
        + ("\n".join(f"  {k}: {v}" for k, v in correlations.items()) or "  (none)")
    )

    interpretation = call_structured(
        state,
        NODE,
        _SYSTEM,
        f"Goal: {state['user_goal']}\nTarget: {target}\n\n{stats_block}",
        EDAInterpretation,
    )

    findings = EDAFindings(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        target=target,
        summary=interpretation.summary,
        correlations=correlations,
        anomalies=interpretation.anomalies,
    )

    messages.append(
        log(NODE, f"Explored {findings.n_rows} rows x {findings.n_cols} columns; "
                  f"produced {len(artifacts)} figures.")
    )
    for anomaly in interpretation.anomalies:
        messages.append(log(NODE, f"Anomaly: {anomaly}", "warning"))

    return RunState(
        eda_findings=findings,
        current_stage=NODE,
        messages=messages,
        artifacts=artifacts,
    )
