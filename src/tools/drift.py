"""Input-drift baseline for an exported model, and the check against it.

At export, `build_profile` summarises the training features into a small JSON:
decile bin edges + bin shares for numeric columns, category shares for the rest.
No raw rows are stored — categories seen in fewer than `MIN_CATEGORY_COUNT` rows
are folded into `__other__`, so an ID-like column can't leak its values.

`check_drift` scores new data against that profile with the Population Stability
Index per feature. PSI < 0.1 is stable, 0.1-0.2 is worth watching, and above
`PSI_THRESHOLD` (0.2, the conventional "significant shift" cut-off) the feature
is flagged. The overall flag is set when any feature is flagged.

CLI:  python -m src.tools.drift runs/<run_id> new_data.csv   (exit 1 on drift)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PSI_THRESHOLD = 0.2
N_BINS = 10
MAX_CATEGORIES = 20
MIN_CATEGORY_COUNT = 5
MISSING = "__missing__"
OTHER = "__other__"
_EPS = 1e-4  # floor for empty bins so PSI stays finite

PROFILE_FILENAME = "reference_profile.json"


def _is_numeric(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)


def _as_float(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _numeric_shares(col: pd.Series, edges: list[float]) -> list[float]:
    values = _as_float(col)
    present = values[~np.isnan(values)]
    counts = np.bincount(
        np.searchsorted(edges, present, side="right"), minlength=len(edges) + 1
    ).tolist()
    counts.append(int(np.isnan(values).sum()))  # last bucket = missing
    return (np.asarray(counts, dtype=float) / max(len(values), 1)).tolist()


def _category_shares(col: pd.Series, categories: list[str]) -> list[float]:
    labels = col.astype("string").fillna(MISSING)
    labels = labels.where(labels.isin(categories), OTHER)
    return (labels.value_counts(normalize=True).reindex(categories + [OTHER], fill_value=0.0)
            .astype(float).tolist())


def build_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise training features into bins/frequencies. Contains no raw rows."""
    features: dict[str, Any] = {}
    for name in df.columns:
        col = df[name]
        if _is_numeric(col):
            values = _as_float(col)
            values = values[~np.isnan(values)]
            qs = np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1]) if len(values) else []
            edges = np.unique(qs).tolist()
            features[name] = {"kind": "numeric", "edges": edges,
                              "shares": _numeric_shares(col, edges)}
        else:
            counts = col.astype("string").fillna(MISSING).value_counts()
            kept = counts[counts >= MIN_CATEGORY_COUNT].head(MAX_CATEGORIES)
            categories = [str(c) for c in kept.index]
            features[name] = {"kind": "categorical", "categories": categories,
                              "shares": _category_shares(col, categories)}
    return {"n_rows": len(df), "psi_threshold": PSI_THRESHOLD, "features": features}


def psi(expected: list[float], actual: list[float]) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def check_drift(new_df: pd.DataFrame, reference: dict[str, Any]) -> dict[str, Any]:
    """Per-feature PSI against `reference` plus an overall `drift` flag.

    A feature missing from `new_df` is reported as missing and counts as drift.
    """
    threshold = reference.get("psi_threshold", PSI_THRESHOLD)
    scores: dict[str, float] = {}
    missing: list[str] = []
    for name, spec in reference["features"].items():
        if name not in new_df.columns:
            missing.append(name)
            continue
        if spec["kind"] == "numeric":
            actual = _numeric_shares(new_df[name], spec["edges"])
        else:
            actual = _category_shares(new_df[name], spec["categories"])
        scores[name] = round(psi(spec["shares"], actual), 4)
    drifted = sorted(n for n, s in scores.items() if s > threshold)
    return {
        "threshold": threshold,
        "psi": scores,
        "drifted_features": drifted,
        "missing_features": missing,
        "drift": bool(drifted or missing),
    }


def save_profile(df: pd.DataFrame, run_dir: Path) -> Path:
    path = Path(run_dir) / PROFILE_FILENAME
    path.write_text(json.dumps(build_profile(df), indent=2), encoding="utf-8")
    return path


def load_profile(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / PROFILE_FILENAME).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m src.tools.drift <run_dir> <new_data.csv>", file=sys.stderr)
        return 2
    result = check_drift(pd.read_csv(args[1]), load_profile(Path(args[0])))
    print(json.dumps(result, indent=2))
    return 1 if result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
