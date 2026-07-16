"""Reviewer: validate the run before anything reaches the user.

Deterministic checks run first and the LLM never sees a veto it can overrule: a
missing .pptx is a fact, not an opinion, and asking a model to confirm it wastes
a call and invites agreement-shaped nonsense. The LLM is used for the judgement
call the checks can't make — whether the story hangs together.

A suspiciously perfect score is treated as a red flag rather than a success. On
real data, r2 = 0.999 means leakage far more often than it means brilliance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..state.schema import RunState, TaskType
from ..tools.codegen import run_dir_for
from ..tools.llm import call_structured
from .common import log

NODE = "reviewer"

#: Above this on the primary metric, assume leakage until proven otherwise.
TOO_GOOD = 0.999

_SYSTEM = """You are a senior reviewer signing off an automated analysis before a
stakeholder sees it. You are the last line of defence and you are accountable for
what ships.

Judge whether this run is trustworthy. Look for:
- metrics that are implausible for the data described (too good = leakage)
- a narrative claiming things the metrics don't support
- features that encode the target
- a holdout split that doesn't respect time order on a timeseries problem
- missing deliverables

verdict: "approve" if it can ship, "revise" if a specific stage should re-run,
"escalate" if a human must decide.
retry_stage: for "revise", exactly one of: cleaning, features, model_selection,
tuning, evaluation, narrative, report. Otherwise null.
concerns: specific, concrete problems. Empty if none.
confidence: 0-1, how much you trust this run's output.
"""


class ReviewVerdict(BaseModel):
    """The reviewer's decision."""

    verdict: str = Field(description="approve | revise | escalate")
    retry_stage: str | None = None
    concerns: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


VALID_STAGES = {
    "cleaning", "features", "model_selection", "tuning", "evaluation",
    "narrative", "report",
}


def _hard_checks(state: RunState) -> list[str]:
    """Facts about the run that don't need an opinion."""
    problems: list[str] = []
    run_dir = run_dir_for(state)

    evaluation = state.get("eval_metrics")
    if not evaluation or not evaluation.metrics:
        problems.append("No holdout metrics were produced.")
    else:
        task = TaskType(state["task_type"])
        primary = "roc_auc" if task is TaskType.CLASSIFICATION else "r2"
        score = evaluation.metrics.get(primary)
        if score is not None:
            if score > TOO_GOOD:
                problems.append(
                    f"{primary}={score} is implausibly perfect — this is almost "
                    "certainly target leakage, not a good model."
                )
            if task is TaskType.CLASSIFICATION and score < 0.5:
                problems.append(
                    f"roc_auc={score} is at or below random guessing."
                )
            if task is not TaskType.CLASSIFICATION and score < 0:
                problems.append(
                    f"r2={score} is worse than predicting the mean."
                )

    if not state.get("narrative", "").strip():
        problems.append("The narrative is empty.")

    kinds = {a.kind for a in state.get("artifacts", [])}
    if "report" not in kinds or not (run_dir / "report.pptx").exists():
        problems.append("report.pptx was not produced.")
    if "model" not in kinds or not (run_dir / "model.pkl").exists():
        problems.append("model.pkl was not exported.")
    if "plot" not in kinds:
        problems.append("No EDA plots were produced.")

    if state.get("error"):
        problems.append(f"A stage reported an error: {state['error']}")

    return problems


def _summarise(state: RunState, hard_problems: list[str]) -> str:
    """The dossier the reviewer judges. Everything relevant, nothing invented."""
    lines = [
        f"Goal: {state['user_goal']}",
        f"Task: {state.get('task_type')}",
        f"Target: {state.get('target_column')}",
    ]

    eda = state.get("eda_findings")
    if eda:
        lines.append(f"Dataset: {eda.n_rows} rows x {eda.n_cols} columns")
        if eda.anomalies:
            lines.append(f"EDA anomalies: {eda.anomalies}")

    evaluation = state.get("eval_metrics")
    lines += [
        f"Chosen model: {state.get('chosen_model')}",
        f"Holdout split: {evaluation.split if evaluation else 'none'}",
        f"Holdout metrics: {evaluation.metrics if evaluation else 'none'}",
    ]

    tuning = state.get("tuning_results")
    if tuning:
        lines.append(
            f"Tuning: {tuning.baseline_score} -> {tuning.best_score} "
            f"over {tuning.n_trials} trials"
        )

    features = state.get("engineered_features", [])
    lines += [
        f"Features kept: {[f.name for f in features if f.kept]}",
        f"Features dropped: {[(f.name, f.drop_reason) for f in features if not f.kept]}",
        f"Automatic checks flagged: {hard_problems or 'nothing'}",
        "",
        f"Narrative:\n{state.get('narrative', '')[:3000]}",
    ]
    return "\n".join(lines)


def reviewer_agent(state: RunState) -> RunState:
    """Run the checks, get a verdict, and decide where the run goes next."""
    hard_problems = _hard_checks(state)
    retry_counts = dict(state.get("retry_counts", {}))

    verdict = call_structured(state, NODE, _SYSTEM, _summarise(state, hard_problems),
                              ReviewVerdict)

    concerns = hard_problems + [c for c in verdict.concerns if c not in hard_problems]
    confidence = min(verdict.confidence, 0.3 if hard_problems else 1.0)

    decision = verdict.verdict.lower().strip()
    if hard_problems and decision == "approve":
        # The model approved something the checks already rejected. The checks win.
        decision = "revise"

    stage = verdict.retry_stage if verdict.retry_stage in VALID_STAGES else None
    if decision == "revise" and not stage:
        stage = "cleaning" if any("leakage" in c.lower() for c in concerns) else "features"

    messages = [log(NODE, f"Verdict: {decision}. {verdict.reasoning}")]
    for concern in concerns:
        messages.append(log(NODE, f"Concern: {concern}", "warning"))

    # A stage that has already been retried twice will not fix itself on a third
    # pass — hand it to a human instead of looping.
    if decision == "revise" and stage:
        if retry_counts.get(stage, 0) >= 2:
            messages.append(
                log(NODE, f"Stage '{stage}' has already been retried "
                          f"{retry_counts[stage]} times; escalating instead.",
                    "warning")
            )
            decision, stage = "escalate", None
        else:
            retry_counts[stage] = retry_counts.get(stage, 0) + 1
            messages.append(log(NODE, f"Sending the run back to '{stage}'."))

    needs_human = decision == "escalate"
    question = None
    if needs_human:
        question = (
            "I'm not confident enough to sign this run off.\n\n"
            + "\n".join(f"- {c}" for c in concerns)
            + f"\n\n{verdict.reasoning}\n\nHow would you like me to proceed?"
        )

    return RunState(
        reviewer_verdict=decision,
        retry_stage=stage if decision == "revise" else None,
        retry_counts=retry_counts,
        confidence=confidence,
        needs_human=needs_human,
        human_question=question,
        current_stage=NODE,
        messages=messages,
    )
