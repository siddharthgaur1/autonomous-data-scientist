"""API surface tests. The graph itself is stubbed — routes and guards are the subject."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.state.schema import RunStatus, new_run_state


@pytest.fixture
def client(monkeypatch, settings, store):
    from src.api import app as api_module

    monkeypatch.setattr(api_module, "_store", store)
    # Never let a test start a real run thread.
    monkeypatch.setattr(api_module, "execute_run", lambda state, store=None: state)
    return TestClient(api_module.app)


class TestHealth:
    def test_reports_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "database": True}


class TestCreateRun:
    def test_accepts_a_csv_and_returns_a_real_run_id(self, client, tiny_csv, store):
        response = client.post(
            "/runs",
            files={"file": ("data.csv", tiny_csv.read_bytes(), "text/csv")},
            data={"goal": "predict churn"},
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        assert run_id != "pending"
        # The id must be pollable the instant it's handed back.
        assert store.get_run(run_id) is not None
        assert client.get(f"/runs/{run_id}").status_code == 200

    def test_rejects_a_non_csv(self, client):
        response = client.post(
            "/runs",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"goal": "predict churn"},
        )
        assert response.status_code == 400

    def test_rejects_an_empty_goal(self, client, tiny_csv):
        response = client.post(
            "/runs",
            files={"file": ("data.csv", tiny_csv.read_bytes(), "text/csv")},
            data={"goal": "   "},
        )
        assert response.status_code == 400


class TestGetRun:
    def test_unknown_run_is_404(self, client):
        assert client.get("/runs/nope").status_code == 404

    def test_returns_the_public_shape(self, client, store, tiny_csv):
        store.create_run(new_run_state("api1", "predict churn", str(tiny_csv)))
        body = client.get("/runs/api1").json()
        assert body["run_id"] == "api1"
        assert body["status"] == "running"
        assert "narrative" in body and "artifacts" in body

    def test_history_is_exposed(self, client, store, tiny_csv):
        store.create_run(new_run_state("api2", "goal", str(tiny_csv)))
        body = client.get("/runs/api2/history").json()
        assert body[0]["stage"] == "created"


class TestArtifacts:
    def test_serves_a_file_from_the_run_dir(self, client, store, settings, tiny_csv):
        store.create_run(new_run_state("api3", "goal", str(tiny_csv)))
        (settings.run_dir("api3") / "model_card.json").write_text('{"ok": true}')

        response = client.get("/runs/api3/artifacts/model_card.json")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_path_traversal_is_refused(self, client, store, tiny_csv, tmp_path):
        store.create_run(new_run_state("api4", "goal", str(tiny_csv)))
        secret = tmp_path / "secret.txt"
        secret.write_text("do not serve me")

        response = client.get("/runs/api4/artifacts/../../secret.txt")
        assert response.status_code == 404
        assert "do not serve me" not in response.text


class TestAsk:
    def test_answers_via_the_qa_graph(self, client, monkeypatch, store, tiny_csv):
        from src.api import app as api_module

        store.create_run(new_run_state("api5", "predict churn", str(tiny_csv)))
        monkeypatch.setattr(api_module.qa, "ask", lambda rid, q: f"answer to {q}")

        response = client.post("/runs/api5/ask", json={"question": "which feature?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "answer to which feature?"

    def test_empty_question_is_rejected(self, client, store, tiny_csv):
        store.create_run(new_run_state("api6", "goal", str(tiny_csv)))
        assert client.post("/runs/api6/ask", json={"question": " "}).status_code == 400


class TestAnswerEscalation:
    def test_conflict_when_the_run_is_not_waiting(self, client, store, tiny_csv):
        store.create_run(new_run_state("api7", "goal", str(tiny_csv)))
        response = client.post("/runs/api7/answer", json={"answer": "the churned column"})
        assert response.status_code == 409

    def test_resumes_a_paused_run(self, client, monkeypatch, store, tiny_csv):
        from src.api import app as api_module

        state = new_run_state("api8", "goal", str(tiny_csv))
        store.create_run(state)
        state["status"] = RunStatus.AWAITING_HUMAN
        store.record_transition(state, stage="human_escalation")

        seen = {}
        monkeypatch.setattr(
            api_module, "resume_run",
            lambda rid, answer, store=None: seen.update(rid=rid, answer=answer),
        )
        response = client.post("/runs/api8/answer", json={"answer": "churned"})
        assert response.status_code == 200
        assert response.json()["status"] == "resuming"
