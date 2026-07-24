"""Narrative: the insight summary.

The one rule that matters: every number in the narrative must come from state.
An LLM asked to "summarise the results" will happily round 0.7412 to "about 75%"
and then invent a lift figure to go with it. So the prompt gets an explicit fact
sheet and forbids anything outside it, and a check afterwards flags numbers that
appear in the text but not in the facts.
"""

from __future__ import annotations

import re

from ..state.schema import RunState
from ..tools.llm import call_llm
from .common import log

NODE = "narrative"

_SYSTEM = """You are writing the findings section of a data science report.

You are given a FACT SHEET. Every quantitative claim you make must come from it.
Do not invent, round beyond one decimal place of what you were given, or
extrapolate to numbers that aren't there. If you want to say something the facts
don't support, don't say it.

Write 4-6 short paragraphs in plain business English:
1. What was asked and what the data looked like.
2. What the exploration revealed (cite specific correlations or distributions).
3. What model was chosen and why.
4. How well it performs on unseen data, and what that means practically.
5. What to do next, and what you would not trust this model for.

No markdown headers, no bullet lists. Prose. Name real columns and real numbers.
"""


def _fact_sheet(state: RunState) -> str:
    """Everything the narrative is allowed to cite, and nothing else."""
    lines = [
        f"Goal: {state['user_goal']}",
        f"Task type: {state.get('task_type')}",
        f"Target column: {state.get('target_column')}",
    ]

    eda = state.get("eda_findings")
    if eda:
        lines += [
            "",
            f"Dataset after cleaning: {eda.n_rows} rows x {eda.n_cols} columns",
            f"EDA summary: {eda.summary}",
        ]
        if eda.correlations:
            top = list(eda.correlations.items())[:6]
            lines.append(
                "Correlations with target: "
                + ", ".join(f"{k}={v}" for k, v in top)
            )
        for anomaly in eda.anomalies:
            lines.append(f"Anomaly noted: {anomaly}")

    transformations = state.get("transformations", [])
    if transformations:
        lines += ["", "Cleaning transformations:"]
        lines += [f"  - {t.column}: {t.action} ({t.reason})" for t in transformations[:12]]

    kept = [f for f in state.get("engineered_features", []) if f.kept]
    dropped = [f for f in state.get("engineered_features", []) if not f.kept]
    if kept:
        lines += ["", "Features created:"]
        lines += [f"  - {f.name}: {f.reasoning}" for f in kept[:12]]
    if dropped:
        lines += ["", "Features dropped:"]
        lines += [f"  - {f.name}: {f.drop_reason}" for f in dropped[:12]]

    candidates = state.get("candidate_models", [])
    if candidates:
        lines += ["", "Candidate models (cross-validated on training data):"]
        lines += [
            f"  - {c.name}: {c.metric}={c.baseline_score} ({c.notes})"
            for c in candidates
        ]

    lines.append(f"Chosen model: {state.get('chosen_model')}")

    tuning = state.get("tuning_results")
    if tuning:
        lines += [
            (f"Tuning: {tuning.n_trials} Optuna trials, "
            f"baseline {tuning.baseline_score} -> best {tuning.best_score} "
            f"({tuning.improvement:+.4f})"),
            f"Best params: {tuning.best_params}",
        ]

    evaluation = state.get("eval_metrics")
    if evaluation:
        lines += [
            "",
            f"Holdout split: {evaluation.split}",
            f"Holdout metrics: {evaluation.metrics}",
            f"Interpretation: {evaluation.interpretation}",
        ]

    return "\n".join(lines)


def _uncited_numbers(text: str, facts: str) -> list[str]:
    """Numbers in the narrative that don't appear in the fact sheet.

    Deliberately loose: years, small integers and percentages derived from the
    facts are ignored, because the goal is catching an invented "£2.3M uplift",
    not policing every digit.
    """
    fact_numbers = set(re.findall(r"\d+\.?\d*", facts))
    suspicious = []
    for token in re.findall(r"\d+\.\d{2,}", text):
        if token not in fact_numbers and token.rstrip("0").rstrip(".") not in fact_numbers:
            suspicious.append(token)
    return suspicious


def narrative_agent(state: RunState) -> RunState:
    """Write the narrative from the fact sheet, then check it for invention."""
    facts = _fact_sheet(state)
    text = call_llm(state, NODE, _SYSTEM, f"FACT SHEET:\n{facts}", temperature=0.3)

    messages = [log(NODE, f"Wrote narrative ({len(text.split())} words).")]

    uncited = _uncited_numbers(text, facts)
    if uncited:
        messages.append(
            log(
                NODE,
                "Narrative contains numbers not present in the fact sheet: "
                f"{', '.join(uncited[:5])}. Reviewer should check these.",
                "warning",
            )
        )

    return RunState(narrative=text, current_stage=NODE, messages=messages)
