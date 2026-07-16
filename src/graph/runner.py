"""Start and resume runs.

The checkpointer is opened per invocation rather than held open for the process's
lifetime: a run is a single `invoke`, and a long-lived Redis connection parked in
a module global is a reconnect bug waiting for the first network blip.

Redis being unavailable degrades the run rather than blocking it — SQLite still
has the full transition log, so you lose mid-run resume, not the audit trail.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..config import get_settings
from ..persistence.store import RunStore
from ..state.schema import RunState, RunStatus, new_run_state
from .build import build_graph


def new_run_id() -> str:
    """Short, readable, and unique enough for a run directory name."""
    return uuid.uuid4().hex[:12]


@contextmanager
def _checkpointer() -> Iterator[object | None]:
    """Yield a Redis checkpointer, or None if Redis isn't reachable."""
    try:
        from langgraph.checkpoint.redis import RedisSaver
    except ImportError:
        yield None
        return

    try:
        with RedisSaver.from_conn_string(get_settings().redis_url) as saver:
            saver.setup()
            yield saver
    except Exception:  # noqa: BLE001 - Redis is a convenience here, not a dependency
        yield None


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": 50}


def prepare_run(
    user_goal: str, csv_path: str | Path, store: RunStore | None = None
) -> RunState:
    """Register a run and stage its data, without executing it.

    Split from `execute_run` so the API can hand back a real run_id immediately
    and let the work happen in the background — a caller that polls the moment it
    gets a 202 finds a real run, not a 404.
    """
    settings = get_settings()
    store = store or RunStore(settings.db_path)

    run_id = new_run_id()
    run_dir = settings.run_dir(run_id)

    # Copy the input in, so the run directory is self-contained and the data hash
    # in the model card refers to something that still exists next month.
    local_csv = run_dir / "raw.csv"
    local_csv.write_bytes(Path(csv_path).read_bytes())

    state = new_run_state(run_id, user_goal, str(local_csv))
    store.create_run(state)
    return state


def execute_run(state: RunState, store: RunStore | None = None) -> RunState:
    """Run the graph over a prepared run to completion (or to a human pause)."""
    store = store or RunStore(get_settings().db_path)
    run_id = state["run_id"]

    with _checkpointer() as saver:
        graph = build_graph(store=store, checkpointer=saver)
        final = graph.invoke(state, _config(run_id))

    store.record_transition(final, stage="final")
    return final


def start_run(
    user_goal: str, csv_path: str | Path, store: RunStore | None = None
) -> RunState:
    """Prepare and execute a run, blocking until it finishes."""
    store = store or RunStore(get_settings().db_path)
    return execute_run(prepare_run(user_goal, csv_path, store), store)


def resume_run(run_id: str, human_answer: str, store: RunStore | None = None) -> RunState:
    """Answer a paused run's question and let it continue.

    Without a checkpointer the graph can't resume mid-thread, so the run restarts
    from the Supervisor with the clarification folded into the goal. Slower, but
    it still reaches an answer rather than dead-ending on a Redis outage.
    """
    settings = get_settings()
    store = store or RunStore(settings.db_path)

    state = store.get_run(run_id)
    if state is None:
        raise KeyError(f"Unknown run: {run_id}")
    if state.get("status") != RunStatus.AWAITING_HUMAN:
        raise ValueError(f"Run {run_id} is not waiting for input (status: {state['status']}).")

    from ..agents.human import resume_after_human

    state["human_answer"] = human_answer
    state.update(resume_after_human(state))  # type: ignore[arg-type]
    store.record_transition(state, stage="human_answered")

    with _checkpointer() as saver:
        graph = build_graph(store=store, checkpointer=saver)
        final = graph.invoke(state, _config(run_id))

    store.record_transition(final, stage="final")
    return final
