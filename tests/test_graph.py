"""Full-graph tests: a 30-row fixture must reach a completed state, mocked end to end.

Every LLM call and every sandbox codegen call is faked. What's under test is the
orchestration — routing, retries, escalation, persistence — not the model's
prose. The sklearn training inside is real.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents.evaluation import Interpretation
from src.agents.model_selection import SelectionDecision
from src.agents.reviewer import ReviewVerdict
from src.agents.supervisor import SupervisorDecision
from src.graph.build import build_graph
from src.state.schema import RunStatus, TaskType, new_run_state
from src.tools.sandbox import ExecResult


@pytest.fixture
def wired(monkeypatch, settings, clean_csv):
    """Patch every LLM and codegen boundary with a fixed, sane response."""
    run_dir_holder = {}

    def fake_cleaning(state, node, task, context=""):
        run_dir = settings.run_dir(state["run_id"])
        run_dir_holder["dir"] = run_dir
        pd.read_csv(clean_csv).to_csv(run_dir / "clean.csv", index=False)
        return (
            ExecResult(
                ok=True,
                result={
                    "transformations": [
                        {"column": "monthly_charges", "action": "parsed text to float",
                         "reason": "stored as text with separators", "rows_affected": 30}
                    ],
                    "rows_before": 30,
                    "rows_after": 30,
                },
            ),
            "# cleaning code",
            [],
        )

    def fake_features(state, node, task, context=""):
        run_dir = settings.run_dir(state["run_id"])
        df = pd.read_csv(run_dir / "clean.csv")
        df["charges_per_month"] = df["monthly_charges"] / (df["tenure_months"] + 1)
        df.to_csv(run_dir / "featured.csv", index=False)
        return (
            ExecResult(
                ok=True,
                result={
                    "features": [
                        {"name": "charges_per_month", "reasoning": "spend intensity",
                         "kept": True, "drop_reason": ""}
                    ]
                },
            ),
            "# feature code",
            [],
        )

    monkeypatch.setattr("src.agents.cleaning.generate_and_run", fake_cleaning)
    monkeypatch.setattr("src.agents.features.generate_and_run", fake_features)

    monkeypatch.setattr(
        "src.agents.supervisor.call_structured",
        lambda *a, **k: SupervisorDecision(
            task_type=TaskType.CLASSIFICATION, target_column="churned",
            plan=["clean", "explore", "model"], confidence=0.9, reasoning="binary flag",
        ),
    )
    monkeypatch.setattr(
        "src.agents.eda.call_structured",
        lambda *a, **k: __import__(
            "src.agents.eda", fromlist=["EDAInterpretation"]
        ).EDAInterpretation(summary="Charges track churn.", anomalies=[]),
    )
    monkeypatch.setattr(
        "src.agents.model_selection.call_structured",
        lambda *a, **k: SelectionDecision(chosen="RandomForest", rationale="best auc"),
    )
    monkeypatch.setattr(
        "src.agents.evaluation.call_structured",
        lambda *a, **k: Interpretation(interpretation="Modest but usable."),
    )
    monkeypatch.setattr(
        "src.agents.narrative.call_llm",
        lambda *a, **k: "The model separates churners better than chance.",
    )
    monkeypatch.setattr("src.agents.tuning.N_TRIALS", 3)
    return run_dir_holder


def _approve(monkeypatch):
    monkeypatch.setattr(
        "src.agents.reviewer.call_structured",
        lambda *a, **k: ReviewVerdict(verdict="approve", confidence=0.8, reasoning="ok"),
    )


class TestFullRun:
    def test_reaches_completed_with_every_deliverable(
        self, wired, monkeypatch, store, settings, tiny_csv
    ):
        _approve(monkeypatch)
        graph = build_graph(store=store, checkpointer=None)

        state = new_run_state("fullrun0001", "predict churn", str(tiny_csv))
        store.create_run(state)
        final = graph.invoke(state, {"configurable": {"thread_id": "fullrun0001"}})

        assert final["status"] is RunStatus.COMPLETED, final.get("error")
        assert final["task_type"] is TaskType.CLASSIFICATION
        assert final["chosen_model"]
        assert final["eval_metrics"].metrics
        assert final["narrative"]

        run_dir = settings.run_dir("fullrun0001")
        assert (run_dir / "clean.csv").exists()
        assert (run_dir / "report.pptx").exists()
        assert (run_dir / "model.pkl").exists()
        assert (run_dir / "model_card.json").exists()

        kinds = {a.kind for a in final["artifacts"]}
        assert {"plot", "report", "model", "model_card"} <= kinds

    def test_every_transition_is_persisted(self, wired, monkeypatch, store, tiny_csv):
        _approve(monkeypatch)
        graph = build_graph(store=store, checkpointer=None)

        state = new_run_state("fullrun0002", "predict churn", str(tiny_csv))
        store.create_run(state)
        graph.invoke(state, {"configurable": {"thread_id": "fullrun0002"}})

        stages = [stage for _, stage, _ in store.history("fullrun0002")]
        for expected in ["supervisor", "cleaning", "eda", "model_selection",
                         "evaluation", "report", "reviewer", "done"]:
            assert expected in stages, f"{expected} was not persisted"

    def test_run_is_replayable(self, wired, monkeypatch, store, tiny_csv):
        _approve(monkeypatch)
        graph = build_graph(store=store, checkpointer=None)

        state = new_run_state("fullrun0003", "predict churn", str(tiny_csv))
        store.create_run(state)
        graph.invoke(state, {"configurable": {"thread_id": "fullrun0003"}})

        replay = store.replay("fullrun0003")
        assert len(replay) > 5
        # The log tells the story of the run: the target appears and persists.
        assert replay[0].get("target_column") is None
        assert replay[-1]["target_column"] == "churned"


class TestEscalation:
    def test_low_confidence_stops_instead_of_fabricating(
        self, wired, monkeypatch, store, tiny_csv
    ):
        """Acceptance criterion 4: a low-confidence run must not invent a result."""
        monkeypatch.setattr(
            "src.agents.supervisor.call_structured",
            lambda *a, **k: SupervisorDecision(
                task_type=TaskType.UNKNOWN, target_column="churned",
                confidence=0.2, reasoning="I cannot tell what is being asked",
            ),
        )
        graph = build_graph(store=store, checkpointer=None)
        state = new_run_state("escalate001", "do something clever", str(tiny_csv))
        store.create_run(state)
        final = graph.invoke(state, {"configurable": {"thread_id": "escalate001"}})

        assert final["status"] is RunStatus.AWAITING_HUMAN
        assert final["needs_human"] is True
        assert final["human_question"]
        assert not final.get("narrative"), "a paused run must not produce a narrative"
        assert not final.get("eval_metrics"), "a paused run must not produce metrics"

    def test_reviewer_escalation_pauses_the_run(
        self, wired, monkeypatch, store, tiny_csv
    ):
        monkeypatch.setattr(
            "src.agents.reviewer.call_structured",
            lambda *a, **k: ReviewVerdict(
                verdict="escalate", confidence=0.2,
                concerns=["metrics look wrong"], reasoning="not comfortable",
            ),
        )
        graph = build_graph(store=store, checkpointer=None)
        state = new_run_state("escalate002", "predict churn", str(tiny_csv))
        store.create_run(state)
        final = graph.invoke(state, {"configurable": {"thread_id": "escalate002"}})

        assert final["status"] is RunStatus.AWAITING_HUMAN
        assert "metrics look wrong" in final["human_question"]


class TestFailure:
    def test_an_unrecoverable_error_ends_with_a_clear_message(
        self, wired, monkeypatch, store, tiny_csv
    ):
        def boom(*a, **k):
            raise RuntimeError("OpenAI is down")

        monkeypatch.setattr("src.agents.supervisor.call_structured", boom)
        graph = build_graph(store=store, checkpointer=None)
        state = new_run_state("failrun0001", "predict churn", str(tiny_csv))
        store.create_run(state)
        final = graph.invoke(state, {"configurable": {"thread_id": "failrun0001"}})

        assert final["status"] is RunStatus.FAILED
        assert "OpenAI is down" in final["error"]

    def test_cost_cap_ends_the_run(self, wired, monkeypatch, store, tiny_csv):
        from src.tools.llm import CostCapExceeded

        def broke(*a, **k):
            raise CostCapExceeded("Run has spent $1.00 of its $1.00 LLM budget.")

        monkeypatch.setattr("src.agents.supervisor.call_structured", broke)
        graph = build_graph(store=store, checkpointer=None)
        state = new_run_state("failrun0002", "predict churn", str(tiny_csv))
        store.create_run(state)
        final = graph.invoke(state, {"configurable": {"thread_id": "failrun0002"}})

        assert final["status"] is RunStatus.FAILED
        assert "budget" in final["error"]
