"""State round-trips through SQLite, and a run's history is a real audit trail."""

from __future__ import annotations

import pytest

from src.persistence.store import RunStore
from src.state.schema import (
    AgentMessage,
    Artifact,
    CandidateModel,
    EDAFindings,
    EvalMetrics,
    RunStatus,
    TaskType,
    TokenUsage,
    Transformation,
    TuningResults,
    new_run_state,
)
from src.state.serde import state_from_json, state_to_json


class TestSerde:
    def test_typed_payloads_survive_the_round_trip(self, tiny_csv):
        state = new_run_state("r1", "predict churn", str(tiny_csv))
        state["task_type"] = TaskType.CLASSIFICATION
        state["status"] = RunStatus.AWAITING_HUMAN
        state["eda_findings"] = EDAFindings(n_rows=30, n_cols=5, target="churned",
                                            summary="s", correlations={"a": 0.4})
        state["transformations"] = [
            Transformation(column="x", action="imputed", reason="nulls", rows_affected=2)
        ]
        state["candidate_models"] = [
            CandidateModel(name="RF", estimator="RandomForestClassifier",
                           baseline_score=0.7, metric="roc_auc")
        ]
        state["tuning_results"] = TuningResults(best_params={"n": 1}, best_score=0.75,
                                                baseline_score=0.7, n_trials=10)
        state["eval_metrics"] = EvalMetrics(metrics={"roc_auc": 0.74}, split="80/20")
        state["messages"] = [AgentMessage(agent="eda", content="hello")]
        state["artifacts"] = [Artifact(kind="plot", path="plots/a.png", label="A")]
        state["token_usage"] = {"eda": TokenUsage(prompt_tokens=10, cost_usd=0.001)}

        back = state_from_json(state_to_json(state))

        assert back["task_type"] is TaskType.CLASSIFICATION
        assert back["status"] is RunStatus.AWAITING_HUMAN
        assert isinstance(back["eda_findings"], EDAFindings)
        assert back["eda_findings"].correlations == {"a": 0.4}
        assert isinstance(back["transformations"][0], Transformation)
        assert isinstance(back["candidate_models"][0], CandidateModel)
        assert isinstance(back["tuning_results"], TuningResults)
        assert isinstance(back["eval_metrics"], EvalMetrics)
        assert isinstance(back["messages"][0], AgentMessage)
        assert isinstance(back["artifacts"][0], Artifact)
        assert isinstance(back["token_usage"]["eda"], TokenUsage)

    def test_tuning_improvement_survives(self, tiny_csv):
        state = new_run_state("r2", "goal", str(tiny_csv))
        state["tuning_results"] = TuningResults(best_score=0.8, baseline_score=0.7)
        back = state_from_json(state_to_json(state))
        assert back["tuning_results"].improvement == pytest.approx(0.1)


class TestStore:
    def test_create_and_get(self, store: RunStore, tiny_csv):
        state = new_run_state("run1", "predict churn", str(tiny_csv))
        store.create_run(state)

        loaded = store.get_run("run1")
        assert loaded is not None
        assert loaded["user_goal"] == "predict churn"
        assert loaded["status"] is RunStatus.RUNNING

    def test_unknown_run_is_none(self, store: RunStore):
        assert store.get_run("nope") is None

    def test_transitions_are_ordered_and_complete(self, store: RunStore, tiny_csv):
        state = new_run_state("run2", "goal", str(tiny_csv))
        store.create_run(state)

        for stage in ["supervisor", "cleaning", "eda"]:
            state["current_stage"] = stage
            store.record_transition(state, stage=stage)

        history = store.history("run2")
        assert [s for _, s, _ in history] == ["created", "supervisor", "cleaning", "eda"]
        assert [seq for seq, _, _ in history] == [0, 1, 2, 3]

    def test_replay_returns_each_state_in_order(self, store: RunStore, tiny_csv):
        state = new_run_state("run3", "goal", str(tiny_csv))
        store.create_run(state)
        state["task_type"] = TaskType.REGRESSION
        store.record_transition(state, stage="supervisor")
        state["chosen_model"] = "Ridge"
        store.record_transition(state, stage="model_selection")

        replay = store.replay("run3")
        assert len(replay) == 3
        assert replay[0]["task_type"] is TaskType.UNKNOWN
        assert replay[1]["task_type"] is TaskType.REGRESSION
        assert replay[2]["chosen_model"] == "Ridge"

    def test_replay_can_stop_before_a_stage(self, store: RunStore, tiny_csv):
        state = new_run_state("run4", "goal", str(tiny_csv))
        store.create_run(state)
        state["chosen_model"] = "Ridge"
        store.record_transition(state, stage="model_selection")

        assert len(store.replay("run4", upto_seq=0)) == 1
        assert store.replay("run4", upto_seq=0)[0]["chosen_model"] is None

    def test_latest_snapshot_tracks_status(self, store: RunStore, tiny_csv):
        state = new_run_state("run5", "goal", str(tiny_csv))
        store.create_run(state)
        state["status"] = RunStatus.COMPLETED
        store.record_transition(state, stage="done")

        assert store.get_run("run5")["status"] is RunStatus.COMPLETED
        assert store.list_runs()[0]["status"] == "completed"

    def test_list_runs_is_newest_first(self, store: RunStore, tiny_csv):
        for i in range(3):
            store.create_run(new_run_state(f"r{i}", f"goal {i}", str(tiny_csv)))
        runs = store.list_runs()
        assert len(runs) == 3
        assert {r["run_id"] for r in runs} == {"r0", "r1", "r2"}
