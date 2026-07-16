"""Shared helpers for agent nodes.

Every node reads the dataframe the previous stage left behind and logs what it
did. Those two things live here rather than being copy-pasted eleven times.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..state.schema import AgentMessage, RunState


def log(agent: str, content: str, level: str = "info") -> AgentMessage:
    """Build one agent-log entry."""
    return AgentMessage(agent=agent, content=content, level=level)  # type: ignore[arg-type]


def active_df_path(state: RunState) -> str:
    """The most advanced dataframe available: engineered > clean > raw."""
    return state.get("clean_df_path") or state["raw_df_path"]


def load_active_df(state: RunState) -> pd.DataFrame:
    """Read the current working dataframe."""
    return pd.read_csv(active_df_path(state))


def describe_frame(df: pd.DataFrame, head_rows: int = 5) -> str:
    """A compact schema + sample block for prompts.

    Sends dtypes, null counts, cardinality and a few rows — enough for the model
    to reason about the data without pasting the whole CSV into the context.
    """
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", "", "Columns:"]
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        lines.append(
            f"  - {col}: dtype={df[col].dtype}, nulls={nulls}, "
            f"unique={df[col].nunique(dropna=True)}, "
            f"sample={df[col].dropna().head(3).tolist()}"
        )
    lines += ["", f"First {head_rows} rows:", df.head(head_rows).to_string()]
    return "\n".join(lines)


def relative_to_run(path: Path | str, run_dir: Path) -> str:
    """Store artifact paths relative to the run dir so they survive a move."""
    try:
        return str(Path(path).resolve().relative_to(Path(run_dir).resolve()))
    except ValueError:
        return str(path)
