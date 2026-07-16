"""Follow-up Q&A over a completed run.

The point of this graph is that it does *not* guess. "Why did you drop feature X?"
is answered from the transformation log the Cleaning agent wrote; "what's the most
important predictor?" is answered from `feature_importances_` on the pickled
estimator that actually shipped. If the answer isn't in the run, the agent says so
rather than reconstructing something plausible.

Two nodes: gather the evidence, then answer from it. Separate from the run graph
because it loads a finished run by id and has no pipeline state of its own.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from ..persistence.store import RunStore
from ..state.schema import RunState, TokenUsage
from ..tools import ml
from ..tools.llm import call_llm
from ..tools.model_io import load_card, load_model

_SYSTEM = """You are the data scientist who built this model, answering a question
about your own work.

You are given the complete record of the run: what you cleaned and why, what
features you created and dropped, which models you compared, how you tuned, what
the holdout metrics were, and the real feature importances read from the fitted
model file.

Rules:
- Answer ONLY from the record. It is the truth about what you built.
- If the record doesn't contain the answer, say so plainly and say what you would
  need to check. Never reconstruct a plausible-sounding answer.
- Quote the specific number, transformation, or feature name you're relying on.
- Be direct and brief. The user is looking at the report and wants the reason.
"""


class QAState(TypedDict, total=False):
    """State for the Q&A graph — a question, the evidence, and an answer."""

    run_id: str
    question: str
    evidence: str
    answer: str
    token_usage: Annotated[dict[str, TokenUsage], lambda a, b: {**a, **b}]


def _importances_block(run_state: RunState) -> str:
    """Real importances from the pickled pipeline, not from memory."""
    run_dir = get_settings().run_dir(run_state["run_id"])
    try:
        pipeline = load_model(run_dir)
        card = load_card(run_dir)
    except (FileNotFoundError, OSError) as exc:
        return f"(model file could not be loaded: {exc})"

    importances = ml.feature_importances(pipeline)
    if not importances:
        return (
            f"Estimator {card.get('estimator')} does not expose feature importances "
            "or coefficients."
        )

    top = list(importances.items())[:15]
    lines = [f"Estimator: {card.get('estimator')} (from model.pkl)"]
    lines += [f"  {name}: {value:.4f}" for name, value in top]
    return "\n".join(lines)


def gather_evidence(state: QAState) -> QAState:
    """Assemble everything known about the run into one evidence block."""
    store = RunStore(get_settings().db_path)
    run = store.get_run(state["run_id"])
    if run is None:
        return QAState(evidence="", answer=f"No run found with id {state['run_id']}.")

    parts = [
        f"Goal: {run['user_goal']}",
        f"Task type: {run.get('task_type')}",
        f"Target column: {run.get('target_column')}",
        f"Status: {run.get('status')}",
    ]

    transformations = run.get("transformations", [])
    if transformations:
        parts += ["", "Cleaning transformations (what was done and why):"]
        parts += [
            f"  - {t.column}: {t.action} — {t.reason} ({t.rows_affected} rows)"
            for t in transformations
        ]

    features = run.get("engineered_features", [])
    kept = [f for f in features if f.kept]
    dropped = [f for f in features if not f.kept]
    if kept:
        parts += ["", "Features created and kept:"]
        parts += [f"  - {f.name}: {f.reasoning}" for f in kept]
    if dropped:
        parts += ["", "Features dropped:"]
        parts += [f"  - {f.name}: {f.drop_reason or f.reasoning}" for f in dropped]

    candidates = run.get("candidate_models", [])
    if candidates:
        parts += ["", "Models compared (cross-validated on training data):"]
        parts += [
            f"  - {c.name}: {c.metric}={c.baseline_score} ({c.notes})"
            for c in candidates
        ]
    parts.append(f"Model chosen: {run.get('chosen_model')}")

    tuning = run.get("tuning_results")
    if tuning:
        parts += [
            "",
            f"Tuning: {tuning.n_trials} Optuna trials over {tuning.search_space}",
            f"Best params: {tuning.best_params}",
            f"Score: {tuning.baseline_score} -> {tuning.best_score}",
        ]

    evaluation = run.get("eval_metrics")
    if evaluation:
        parts += [
            "",
            f"Holdout split: {evaluation.split}",
            f"Holdout metrics: {evaluation.metrics}",
            f"Interpretation: {evaluation.interpretation}",
        ]

    parts += ["", "Feature importances read from the fitted model file:",
              _importances_block(run)]

    if run.get("narrative"):
        parts += ["", "Narrative that was written:", run["narrative"]]

    return QAState(evidence="\n".join(parts))


def answer_question(state: QAState) -> QAState:
    """Answer strictly from the gathered evidence."""
    if state.get("answer"):
        return QAState()  # gather_evidence already resolved it (unknown run)
    if not state.get("evidence"):
        return QAState(answer="I have no record of that run, so I can't answer.")

    # A throwaway RunState carries the cost accounting for this one-off call.
    billing: RunState = {"run_id": state["run_id"], "token_usage": {}}  # type: ignore[typeddict-item]
    answer = call_llm(
        billing,
        "qa",
        _SYSTEM,
        f"RUN RECORD:\n{state['evidence']}\n\nQUESTION: {state['question']}",
        temperature=0.1,
    )
    return QAState(answer=answer, token_usage=billing.get("token_usage", {}))


def build_qa_graph():
    """Compile the Q&A graph: gather evidence, then answer from it."""
    builder = StateGraph(QAState)
    builder.add_node("gather_evidence", gather_evidence)
    builder.add_node("answer_question", answer_question)
    builder.add_edge(START, "gather_evidence")
    builder.add_edge("gather_evidence", "answer_question")
    builder.add_edge("answer_question", END)
    return builder.compile()


def ask(run_id: str, question: str) -> str:
    """Ask a question about a completed run."""
    result = build_qa_graph().invoke(QAState(run_id=run_id, question=question))
    return result.get("answer", "")
