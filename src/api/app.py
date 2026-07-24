"""FastAPI surface for the agent.

Runs execute in a background thread rather than blocking the request: a full run
is minutes of LLM calls and Optuna trials, and no sensible client holds a socket
open for that. `POST /runs` returns immediately with a run_id; the dashboard polls
`GET /runs/{id}`, which reads the same SQLite log the graph writes as it goes.

`ponytail: threads, not a task queue.` A single API process is the documented
ceiling — a run dies with the process and there's no cross-worker fan-out. Swap in
Celery or arq if you need durability or more than one worker.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import get_settings
from ..graph import qa
from ..graph.runner import execute_run, prepare_run, resume_run
from ..persistence.store import RunStore
from ..state.schema import RunState, RunStatus

app = FastAPI(
    title="Autonomous Data Scientist",
    description="Upload a CSV and a goal; get back a model, a deck, and answers.",
    version="1.0.0",
)

_store: RunStore | None = None
_lock = threading.Lock()


def store() -> RunStore:
    """The run store, created once on first use."""
    global _store
    with _lock:
        if _store is None:
            _store = RunStore(get_settings().db_path)
    return _store


class AskRequest(BaseModel):
    """A follow-up question about a completed run."""

    question: str


class AnswerRequest(BaseModel):
    """A human's answer to an escalated run's question."""

    answer: str


class RunCreated(BaseModel):
    """Acknowledgement that a run has been accepted."""

    run_id: str
    status: str


def _public(state: RunState) -> dict:
    """Shape a run for the API: progress, findings, artifacts — no raw frames."""
    eda = state.get("eda_findings")
    tuning = state.get("tuning_results")
    evaluation = state.get("eval_metrics")

    return {
        "run_id": state["run_id"],
        "user_goal": state["user_goal"],
        "status": RunStatus(state["status"]).value,
        "current_stage": state.get("current_stage"),
        "task_type": state.get("task_type"),
        "target_column": state.get("target_column"),
        "plan": state.get("plan", []),
        "confidence": state.get("confidence"),
        "needs_human": state.get("needs_human", False),
        "human_question": state.get("human_question"),
        "error": state.get("error"),
        "narrative": state.get("narrative", ""),
        "eda_findings": eda.model_dump() if eda else None,
        "transformations": [t.model_dump() for t in state.get("transformations", [])],
        "engineered_features": [
            f.model_dump() for f in state.get("engineered_features", [])
        ],
        "candidate_models": [c.model_dump() for c in state.get("candidate_models", [])],
        "chosen_model": state.get("chosen_model"),
        "tuning_results": tuning.model_dump() if tuning else None,
        "eval_metrics": evaluation.model_dump() if evaluation else None,
        "artifacts": [a.model_dump() for a in state.get("artifacts", [])],
        "messages": [m.model_dump(mode="json") for m in state.get("messages", [])],
        "token_usage": {k: v.model_dump() for k, v in state.get("token_usage", {}).items()},
        "reviewer_verdict": state.get("reviewer_verdict"),
    }


def _load(run_id: str) -> RunState:
    state = store().get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}")
    return state


@app.get("/health")
def health() -> dict:
    """Liveness plus a real check that the run store is reachable."""
    try:
        store().list_runs(limit=1)
        db_ok = True
    except Exception:  # noqa: BLE001 - health must report, not raise
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}


@app.post("/runs", response_model=RunCreated, status_code=202)
def create_run(
    goal: str = Form(..., description="Plain-English goal, e.g. 'predict churn'"),
    file: UploadFile = File(..., description="The CSV to analyse"),  # noqa: B008 - FastAPI's DI pattern requires call-in-default
) -> RunCreated:
    """Accept a CSV and a goal, and start a run in the background."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    if not goal.strip():
        raise HTTPException(status_code=400, detail="A goal is required.")

    tmp = Path(tempfile.mkdtemp()) / Path(file.filename).name
    with tmp.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # Registered synchronously so the run_id we hand back is immediately pollable;
    # only the graph itself runs in the background.
    state = prepare_run(goal, tmp, store=store())
    threading.Thread(
        target=execute_run, args=(state,), kwargs={"store": store()}, daemon=True
    ).start()

    return RunCreated(run_id=state["run_id"], status="accepted")


@app.get("/runs")
def list_runs(limit: int = 50) -> list[dict]:
    """Recent runs, newest first."""
    return store().list_runs(limit=limit)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Status, findings and artifacts for one run."""
    return _public(_load(run_id))


@app.get("/runs/{run_id}/history")
def get_history(run_id: str) -> list[dict]:
    """The audit trail: every state transition, in order."""
    _load(run_id)
    return [
        {"seq": seq, "stage": stage, "at": at}
        for seq, stage, at in store().history(run_id)
    ]


@app.get("/runs/{run_id}/artifacts/{path:path}")
def get_artifact(run_id: str, path: str) -> FileResponse:
    """Download an artifact (plot, .pptx, .pkl) from a run."""
    _load(run_id)
    run_dir = get_settings().run_dir(run_id).resolve()
    target = (run_dir / path).resolve()

    # Never serve outside the run directory, whatever the client asks for.
    if not target.is_relative_to(run_dir) or not target.is_file():
        raise HTTPException(status_code=404, detail="No such artifact.")
    return FileResponse(target, filename=target.name)


@app.post("/runs/{run_id}/ask")
def ask_question(run_id: str, request: AskRequest) -> dict:
    """Answer a follow-up question from the run's record and its fitted model."""
    _load(run_id)
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A question is required.")
    return {"run_id": run_id, "question": request.question,
            "answer": qa.ask(run_id, request.question)}


@app.post("/runs/{run_id}/answer")
def answer_escalation(run_id: str, request: AnswerRequest) -> dict:
    """Supply the human input a paused run asked for, and resume it."""
    state = _load(run_id)
    if state.get("status") != RunStatus.AWAITING_HUMAN:
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is not waiting for input.",
        )

    threading.Thread(
        target=resume_run,
        args=(run_id, request.answer),
        kwargs={"store": store()},
        daemon=True,
    ).start()
    return {"run_id": run_id, "status": "resuming"}
