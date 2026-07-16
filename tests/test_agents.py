"""Each agent against a fixed fake LLM response and a tiny DataFrame.

Agents import `call_llm`/`call_structured` into their own namespace, so the patch
target is the agent module, not `src.tools.llm`. Patching the source module would
leave the already-bound name untouched and quietly hit the real API.
"""

from __future__ import annotations

import pytest

from src.agents.evaluation import evaluation_agent
from src.agents.features import features_agent
from src.agents.model_selection import model_selection_agent
from src.agents.narrative import narrative_agent
from src.agents.report import report_agent
from src.agents.reviewer import ReviewVerdict, reviewer_agent
from src.agents.supervisor import SupervisorDecision, supervisor_agent
from src.agents.tuning import tuning_agent
from src.state.schema import RunState, TaskType


def fake_structured(response):
    """A stand-in for call_structured that always returns `response`."""
    return lambda state, node, system, user, schema, **kwargs: response


def fake_text(response):
    return lambda state, node, system, user, **kwargs: response


class TestSupervisor:
    def test_sets_task_target_and_plan(self, base_state, monkeypatch):
        monkeypatch.setattr(
            "src.agents.supervisor.call_structured",
            fake_structured(
                SupervisorDecision(
                    task_type=TaskType.CLASSIFICATION,
                    target_column="churned",
                    plan=["clean", "model"],
                    confidence=0.9,
                    reasoning="churned is a binary flag",
                )
            ),
        )
        update = supervisor_agent(base_state)
        assert update["task_type"] is TaskType.CLASSIFICATION
        assert update["target_column"] == "churned"
        assert update["needs_human"] is False
        assert update["plan"] == ["clean", "model"]

    def test_low_confidence_escalates_with_a_question(self, base_state, monkeypatch):
        monkeypatch.setattr(
            "src.agents.supervisor.call_structured",
            fake_structured(
                SupervisorDecision(
                    task_type=TaskType.CLASSIFICATION,
                    target_column="churned",
                    confidence=0.3,
                    reasoning="the goal is ambiguous",
                )
            ),
        )
        update = supervisor_agent(base_state)
        assert update["needs_human"] is True
        assert "churned" in update["human_question"]

    def test_hallucinated_target_column_crushes_confidence(self, base_state, monkeypatch):
        """The model named a column that isn't in the file. Trust the file."""
        monkeypatch.setattr(
            "src.agents.supervisor.call_structured",
            fake_structured(
                SupervisorDecision(
                    task_type=TaskType.CLASSIFICATION,
                    target_column="does_not_exist",
                    confidence=0.95,
                    reasoning="confidently wrong",
                )
            ),
        )
        update = supervisor_agent(base_state)
        assert update["confidence"] <= 0.2
        assert update["needs_human"] is True


class TestFeatures:
    def test_leaky_feature_is_dropped_even_when_the_llm_kept_it(
        self, base_state, clean_csv, monkeypatch, settings
    ):
        """The LLM proudly builds a copy of the target. The backstop must catch it."""
        import pandas as pd

        run_dir = settings.run_dir(base_state["run_id"])
        df = pd.read_csv(clean_csv)
        df["churn_copy"] = df["churned"]  # perfectly correlated with the target
        df.to_csv(run_dir / "featured.csv", index=False)

        state = dict(base_state)
        state["clean_df_path"] = str(clean_csv)

        def fake_generate(state, node, task, context=""):
            from src.tools.sandbox import ExecResult

            return (
                ExecResult(
                    ok=True,
                    result={
                        "features": [
                            {"name": "churn_copy", "reasoning": "very predictive!",
                             "kept": True, "drop_reason": ""}
                        ]
                    },
                ),
                "# code",
                [],
            )

        monkeypatch.setattr("src.agents.features.generate_and_run", fake_generate)
        update = features_agent(state)  # type: ignore[arg-type]

        dropped = [f for f in update["engineered_features"] if not f.kept]
        assert any(f.name == "churn_copy" for f in dropped), "leakage was not caught"
        assert "encodes the answer" in next(
            f.drop_reason for f in dropped if f.name == "churn_copy"
        )
        assert "churn_copy" not in pd.read_csv(run_dir / "featured.csv").columns

    def test_failure_continues_on_cleaned_data(self, base_state, clean_csv, monkeypatch):
        """Feature engineering is an enhancement — its failure isn't fatal."""
        state = dict(base_state)
        state["clean_df_path"] = str(clean_csv)

        def fake_generate(state, node, task, context=""):
            from src.tools.sandbox import ExecResult

            return ExecResult(ok=False, traceback="boom"), "# code", []

        monkeypatch.setattr("src.agents.features.generate_and_run", fake_generate)
        update = features_agent(state)  # type: ignore[arg-type]
        assert "error" not in update
        assert update["messages"][-1].level == "warning"


class TestModelSelection:
    def test_trains_candidates_and_records_scores(
        self, base_state, clean_csv, monkeypatch
    ):
        from src.agents.model_selection import SelectionDecision

        state = dict(base_state)
        state["clean_df_path"] = str(clean_csv)
        monkeypatch.setattr(
            "src.agents.model_selection.call_structured",
            fake_structured(
                SelectionDecision(chosen="RandomForest", rationale="best score")
            ),
        )
        update = model_selection_agent(state)  # type: ignore[arg-type]
        assert len(update["candidate_models"]) >= 3
        assert update["chosen_model"] == "RandomForest"
        assert all(c.metric == "roc_auc" for c in update["candidate_models"])

    def test_falls_back_when_the_llm_picks_a_nonexistent_model(
        self, base_state, clean_csv, monkeypatch
    ):
        from src.agents.model_selection import SelectionDecision

        state = dict(base_state)
        state["clean_df_path"] = str(clean_csv)
        monkeypatch.setattr(
            "src.agents.model_selection.call_structured",
            fake_structured(SelectionDecision(chosen="XGBoostDeluxe", rationale="?")),
        )
        update = model_selection_agent(state)  # type: ignore[arg-type]
        assert update["chosen_model"] in {c.name for c in update["candidate_models"]}


class TestTuning:
    def test_records_improvement_over_baseline(self, completed_state, monkeypatch):
        state = dict(completed_state)
        state["chosen_model"] = "DecisionTree"
        state["candidate_models"][0].name = "DecisionTree"

        monkeypatch.setattr("src.agents.tuning.N_TRIALS", 3)
        update = tuning_agent(state)  # type: ignore[arg-type]

        results = update["tuning_results"]
        assert results.n_trials == 3
        assert results.search_space
        assert results.best_params

    def test_model_without_a_search_space_is_skipped(self, completed_state):
        state = dict(completed_state)
        state["chosen_model"] = "LinearRegression"
        update = tuning_agent(state)  # type: ignore[arg-type]
        assert update["tuning_results"].n_trials == 0


class TestEvaluation:
    def test_scores_holdout_and_exports_the_model(
        self, completed_state, monkeypatch, settings
    ):
        from src.agents.evaluation import Interpretation

        monkeypatch.setattr(
            "src.agents.evaluation.call_structured",
            fake_structured(Interpretation(interpretation="It is okay.")),
        )
        update = evaluation_agent(completed_state)

        assert "roc_auc" in update["eval_metrics"].metrics
        run_dir = settings.run_dir(completed_state["run_id"])
        assert (run_dir / "model.pkl").exists()
        assert (run_dir / "model_card.json").exists()
        assert {"model", "model_card"} <= {a.kind for a in update["artifacts"]}

    def test_baseline_ships_when_tuning_made_things_worse(
        self, completed_state, monkeypatch, settings
    ):
        from src.agents.evaluation import Interpretation
        from src.tools.model_io import load_card

        state = dict(completed_state)
        state["tuning_results"].best_score = 0.5  # worse than the 0.72 baseline
        state["tuning_results"].best_params = {"n_estimators": 999}

        monkeypatch.setattr(
            "src.agents.evaluation.call_structured",
            fake_structured(Interpretation(interpretation="ok")),
        )
        evaluation_agent(state)  # type: ignore[arg-type]

        card = load_card(settings.run_dir(state["run_id"]))
        assert card["best_params"] == {}, "a worse tuned model was shipped"


class TestNarrative:
    def test_writes_from_the_fact_sheet(self, completed_state, monkeypatch):
        monkeypatch.setattr(
            "src.agents.narrative.call_llm",
            fake_text("The model reaches 0.74 roc_auc on unseen customers."),
        )
        update = narrative_agent(completed_state)
        assert "0.74" in update["narrative"]
        assert not any(m.level == "warning" for m in update["messages"])

    def test_invented_numbers_are_flagged(self, completed_state, monkeypatch):
        monkeypatch.setattr(
            "src.agents.narrative.call_llm",
            fake_text("Churn will fall by 37.9182 percent, saving 12.4471 million."),
        )
        update = narrative_agent(completed_state)
        assert any(
            "not present in the fact sheet" in m.content for m in update["messages"]
        )


class TestReport:
    def test_builds_a_real_pptx(self, completed_state, settings):
        update = report_agent(completed_state)
        path = settings.run_dir(completed_state["run_id"]) / "report.pptx"
        assert path.exists() and path.stat().st_size > 10_000

        from pptx import Presentation

        assert len(Presentation(path).slides) >= 4
        assert any(a.kind == "report" for a in update["artifacts"])


class TestReviewer:
    def test_approves_a_sound_run(self, completed_state, monkeypatch, settings):
        run_dir = settings.run_dir(completed_state["run_id"])
        (run_dir / "report.pptx").write_bytes(b"x")
        (run_dir / "model.pkl").write_bytes(b"x")
        state = dict(completed_state)
        state["artifacts"] = [
            _artifact("report", "report.pptx"),
            _artifact("model", "model.pkl"),
            _artifact("plot", "plots/x.png"),
        ]

        monkeypatch.setattr(
            "src.agents.reviewer.call_structured",
            fake_structured(
                ReviewVerdict(verdict="approve", confidence=0.85, reasoning="fine")
            ),
        )
        update = reviewer_agent(state)  # type: ignore[arg-type]
        assert update["reviewer_verdict"] == "approve"
        assert update["needs_human"] is False

    def test_perfect_score_is_overridden_to_revise(
        self, completed_state, monkeypatch, settings
    ):
        """The LLM says ship it; the leakage check says no. The check wins."""
        run_dir = settings.run_dir(completed_state["run_id"])
        (run_dir / "report.pptx").write_bytes(b"x")
        (run_dir / "model.pkl").write_bytes(b"x")

        state = dict(completed_state)
        state["eval_metrics"].metrics = {"roc_auc": 1.0}
        state["artifacts"] = [
            _artifact("report", "report.pptx"),
            _artifact("model", "model.pkl"),
            _artifact("plot", "plots/x.png"),
        ]

        monkeypatch.setattr(
            "src.agents.reviewer.call_structured",
            fake_structured(
                ReviewVerdict(verdict="approve", confidence=0.99, reasoning="perfect!")
            ),
        )
        update = reviewer_agent(state)  # type: ignore[arg-type]
        assert update["reviewer_verdict"] == "revise"
        assert any("leakage" in m.content for m in update["messages"])

    def test_third_retry_escalates_instead_of_looping(
        self, completed_state, monkeypatch
    ):
        state = dict(completed_state)
        state["retry_counts"] = {"features": 2}
        state["narrative"] = ""  # guarantees a hard problem

        monkeypatch.setattr(
            "src.agents.reviewer.call_structured",
            fake_structured(
                ReviewVerdict(
                    verdict="revise", retry_stage="features", confidence=0.4,
                    reasoning="still broken",
                )
            ),
        )
        update = reviewer_agent(state)  # type: ignore[arg-type]
        assert update["reviewer_verdict"] == "escalate"
        assert update["needs_human"] is True


def _artifact(kind, path):
    from src.state.schema import Artifact

    return Artifact(kind=kind, path=path, label=path)


class TestHumanEscalation:
    def test_pauses_with_the_question(self, base_state):
        from src.agents.human import human_escalation_agent
        from src.state.schema import RunStatus

        state = dict(base_state)
        state["human_question"] = "Which column is the target?"
        update = human_escalation_agent(state)  # type: ignore[arg-type]

        assert update["status"] is RunStatus.AWAITING_HUMAN
        assert update["human_question"] == "Which column is the target?"

    def test_answer_folds_into_the_goal(self, base_state):
        from src.agents.human import resume_after_human

        state = dict(base_state)
        state["human_answer"] = "Predict the churned column."
        update = resume_after_human(state)  # type: ignore[arg-type]

        assert "Predict the churned column." in update["user_goal"]
        assert update["needs_human"] is False
        assert update["current_stage"] == "supervisor"
