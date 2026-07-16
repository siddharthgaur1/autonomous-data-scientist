"""Cleaning: missing values, type coercion, outliers, duplicates.

This node writes real pandas via the sandbox rather than applying a fixed recipe,
because the defects are dataset-specific — `total_charges` arriving as
`"3,750.84"` with blanks for new accounts isn't something a generic imputer
recognises. Every transformation is recorded with a reason, which is what the
follow-up Q&A answers "why did you drop X?" from.
"""

from __future__ import annotations

import pandas as pd

from ..state.schema import RunState, Transformation
from ..tools.codegen import generate_and_run, run_dir_for
from .common import describe_frame, log

NODE = "cleaning"

_TASK = """Clean this dataset and save the result.

Rules:
- Read the raw CSV from the path given below.
- Fix the obvious defects: text columns that are really numbers (strip currency
  symbols, thousands separators, K/M suffixes), inconsistent categorical
  spellings (whitespace, casing), mixed date formats, duplicate rows.
- Handle missing values with a strategy you can justify per column.
- Handle outliers only where they are clearly errors. Do NOT silently delete
  extreme-but-real values; cap or flag them instead and say so.
- NEVER drop or modify the target column's rows unless the target is missing.
- Save the cleaned dataframe to 'clean.csv' in the working directory
  (index=False).
- Set `result` to:
  {"transformations": [{"column": str, "action": str, "reason": str,
                        "rows_affected": int}, ...],
   "rows_before": int, "rows_after": int}

Be specific in "reason" — it is shown to a human who will ask why."""


def cleaning_agent(state: RunState) -> RunState:
    """Generate and run cleaning code, then record what it did."""
    raw_path = state["raw_df_path"]
    df = pd.read_csv(raw_path)
    run_dir = run_dir_for(state)

    context = (
        f"Raw CSV path: {raw_path!r}\n"
        f"Target column: {state.get('target_column')!r}\n"
        f"Task type: {state.get('task_type')}\n\n"
        f"{describe_frame(df)}"
    )

    result, code, messages = generate_and_run(state, NODE, _TASK, context)
    (run_dir / "cleaning_code.py").write_text(code, encoding="utf-8")

    clean_path = run_dir / "clean.csv"
    if not result.ok or not clean_path.exists():
        return RunState(
            current_stage=NODE,
            error=f"Cleaning failed: {result.error_feedback[:800]}",
            messages=[*messages, log(NODE, "Could not produce a clean dataset.", "error")],
        )

    payload = result.result if isinstance(result.result, dict) else {}
    transformations = [
        Transformation(
            column=str(t.get("column", "?")),
            action=str(t.get("action", "")),
            reason=str(t.get("reason", "")),
            rows_affected=int(t.get("rows_affected", 0) or 0),
        )
        for t in payload.get("transformations", [])
        if isinstance(t, dict)
    ]

    clean_df = pd.read_csv(clean_path)
    messages.append(
        log(
            NODE,
            f"Cleaned {payload.get('rows_before', len(df))} rows -> "
            f"{len(clean_df)} rows, {clean_df.shape[1]} columns; "
            f"{len(transformations)} transformations recorded.",
        )
    )

    target = state.get("target_column")
    if target and target not in clean_df.columns:
        return RunState(
            current_stage=NODE,
            error=f"Cleaning dropped the target column '{target}'.",
            messages=[*messages, log(NODE, f"Target '{target}' was lost.", "error")],
        )

    return RunState(
        clean_df_path=str(clean_path),
        transformations=transformations,
        current_stage=NODE,
        messages=messages,
    )
