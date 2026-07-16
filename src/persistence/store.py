"""SQLite run history.

Two tables: `runs` holds the latest snapshot for cheap status reads, and
`transitions` is the append-only audit log — one row per node that touched the
state. Replaying a run means folding the transitions in `seq` order, which is
what makes a finished run auditable rather than just archived.

stdlib sqlite3 rather than an ORM: the schema is two tables and never joins.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..state.schema import RunState, RunStatus
from ..state.serde import state_from_json, state_to_json

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    user_goal  TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    stage      TEXT NOT NULL,
    at         TEXT NOT NULL,
    state_json TEXT NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_transitions_run ON transitions (run_id, seq);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Persists run snapshots and the transition log to SQLite."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_run(self, state: RunState) -> None:
        """Insert a brand-new run and its seq-0 transition."""
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (run_id, user_goal, status, created_at, updated_at,"
                " state_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    state["run_id"],
                    state["user_goal"],
                    RunStatus(state["status"]).value,
                    now,
                    now,
                    state_to_json(state),
                ),
            )
        self.record_transition(state, stage="created")

    def record_transition(self, state: RunState, stage: str) -> None:
        """Append one audit-log row and refresh the run's latest snapshot."""
        run_id = state["run_id"]
        now = _now()
        payload = state_to_json(state)
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM transitions"
                " WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            c.execute(
                "INSERT INTO transitions (run_id, seq, stage, at, state_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, row["next"], stage, now, payload),
            )
            c.execute(
                "UPDATE runs SET status = ?, updated_at = ?, state_json = ?"
                " WHERE run_id = ?",
                (RunStatus(state["status"]).value, now, payload, run_id),
            )

    def get_run(self, run_id: str) -> RunState | None:
        """Return the latest snapshot for a run, or None if unknown."""
        with self._conn() as c:
            row = c.execute(
                "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return state_from_json(row["state_json"]) if row else None

    def list_runs(self, limit: int = 50) -> list[dict[str, str]]:
        """Recent runs, newest first, as plain summary rows."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT run_id, user_goal, status, created_at, updated_at FROM runs"
                " ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def history(self, run_id: str) -> list[tuple[int, str, str]]:
        """The (seq, stage, at) audit trail for a run, in order."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT seq, stage, at FROM transitions WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        return [(r["seq"], r["stage"], r["at"]) for r in rows]

    def replay(self, run_id: str, upto_seq: int | None = None) -> list[RunState]:
        """Every recorded state for a run, in order — the replay of the run.

        `upto_seq` stops the replay early, which is how you inspect what the
        state looked like before a stage that later went wrong.
        """
        sql = "SELECT state_json FROM transitions WHERE run_id = ?"
        params: list[object] = [run_id]
        if upto_seq is not None:
            sql += " AND seq <= ?"
            params.append(upto_seq)
        sql += " ORDER BY seq"
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [state_from_json(r["state_json"]) for r in rows]
