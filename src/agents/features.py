"""Feature engineering: propose features with reasoning, build them, drop the bad ones.

Leakage is the failure mode that matters here. A model that gets `total_charges`
to predict churn when `total_charges` is computed from tenure is not a model, it's
a lookup — and it scores beautifully right up until production. So the prompt
names leakage explicitly, and a deterministic check afterwards drops any feature
that correlates with the target above 0.98 regardless of what the LLM claimed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..state.schema import EngineeredFeature, RunState
from ..tools.codegen import generate_and_run, run_dir_for
from .common import describe_frame, load_active_df, log

NODE = "features"

#: Above this |correlation| with the target, a feature is treated as leakage.
#: Genuine predictors essentially never reach it on real data; derived copies of
#: the target always do.
LEAKAGE_THRESHOLD = 0.98

_TASK = """Engineer features for this modelling task.

Rules:
- Read the cleaned CSV from the path given below.
- Propose and create features that plausibly help predict the target: ratios,
  interactions, date parts, aggregations, binned versions of skewed columns.
  For timeseries, lag and rolling features are appropriate — but they must only
  use PAST values (shift before rolling), never future ones.
- NEVER create a feature computed from the target column. That is leakage and it
  invalidates the whole run.
- Drop identifier-like columns (unique per row) and constant columns.
- Keep the target column in the output, unchanged.
- Save the result to 'featured.csv' in the working directory (index=False).
- Set `result` to:
  {"features": [{"name": str, "reasoning": str, "kept": bool,
                 "drop_reason": str}, ...]}
  Include both features you created and columns you dropped (kept=false, with a
  drop_reason).

"reasoning" must say why the feature should predict the target. It is shown to a
human who will ask."""


def _leaky_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Columns suspiciously correlated with the target — the leakage backstop."""
    numeric = df.select_dtypes(include=np.number)
    if target not in numeric.columns or numeric.shape[1] < 2:
        return []
    corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    return [str(c) for c in corr[corr.abs() > LEAKAGE_THRESHOLD].dropna().index]


def features_agent(state: RunState) -> RunState:
    """Build engineered features, then enforce the leakage rule independently."""
    df = load_active_df(state)
    run_dir = run_dir_for(state)
    target = state.get("target_column") or ""

    context = (
        f"Cleaned CSV path: {state['clean_df_path']!r}\n"
        f"Target column: {target!r}\n"
        f"Task type: {state.get('task_type')}\n"
        f"Time column: {state.get('time_column')!r}\n\n"
        f"{describe_frame(df)}"
    )

    result, code, messages = generate_and_run(state, NODE, _TASK, context)
    (run_dir / "features_code.py").write_text(code, encoding="utf-8")

    featured_path = run_dir / "featured.csv"
    if not result.ok or not featured_path.exists():
        # Feature engineering is an enhancement, not a prerequisite. The clean
        # data is still modellable, so the run continues on it rather than dying.
        messages.append(
            log(NODE, "Feature engineering failed; continuing with cleaned data.",
                "warning")
        )
        return RunState(current_stage=NODE, messages=messages)

    featured = pd.read_csv(featured_path)
    payload = result.result if isinstance(result.result, dict) else {}

    features = [
        EngineeredFeature(
            name=str(f.get("name", "?")),
            reasoning=str(f.get("reasoning", "")),
            kept=bool(f.get("kept", True)),
            drop_reason=str(f.get("drop_reason", "")),
        )
        for f in payload.get("features", [])
        if isinstance(f, dict)
    ]

    if target not in featured.columns:
        messages.append(
            log(NODE, f"Target '{target}' missing from engineered data; "
                      "keeping cleaned data instead.", "error")
        )
        return RunState(current_stage=NODE, messages=messages)

    leaky = _leaky_columns(featured, target)
    if leaky:
        featured = featured.drop(columns=leaky)
        featured.to_csv(featured_path, index=False)
        for col in leaky:
            features.append(
                EngineeredFeature(
                    name=col,
                    reasoning="Flagged by the automatic leakage check.",
                    kept=False,
                    drop_reason=(
                        f"|correlation| with '{target}' exceeded "
                        f"{LEAKAGE_THRESHOLD}, which means it encodes the answer."
                    ),
                )
            )
        messages.append(
            log(NODE, f"Dropped {len(leaky)} leaking column(s): {', '.join(leaky)}",
                "warning")
        )

    kept = [f.name for f in features if f.kept]
    messages.append(
        log(NODE, f"Engineered dataset has {featured.shape[1]} columns; "
                  f"{len(kept)} features kept, "
                  f"{len(features) - len(kept)} dropped.")
    )

    return RunState(
        clean_df_path=str(featured_path),
        engineered_features=features,
        current_stage=NODE,
        messages=messages,
    )
