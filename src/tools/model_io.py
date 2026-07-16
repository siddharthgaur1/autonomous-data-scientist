"""Persist the fitted pipeline and its model card.

The .pkl is the whole Pipeline (preprocessing + estimator), not a bare estimator,
so loading it gives something you can call `.predict(raw_df)` on. The card sits
next to it because a pickle with no provenance is a liability: six months on, the
data hash and training date are the difference between a reusable model and an
unidentified binary.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def data_hash(path: Path | str) -> str:
    """SHA-256 of the training file, so a model can be tied back to its data."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(
    pipeline: Pipeline,
    run_dir: Path,
    *,
    task_type: str,
    target: str,
    features: list[str],
    metrics: dict[str, float],
    training_data_path: Path | str,
    best_params: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write `model.pkl` and `model_card.json`. Returns both paths."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.pkl"
    card_path = run_dir / "model_card.json"

    joblib.dump(pipeline, model_path)

    card = {
        "task_type": task_type,
        "target": target,
        "features": features,
        "metrics": metrics,
        "best_params": best_params or {},
        "estimator": type(pipeline.named_steps.get("model", pipeline)).__name__,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_hash": data_hash(training_data_path),
        "sklearn_pipeline": True,
    }
    card_path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    return model_path, card_path


def load_model(run_dir: Path) -> Pipeline:
    """Load a run's fitted pipeline."""
    return joblib.load(Path(run_dir) / "model.pkl")


def load_card(run_dir: Path) -> dict[str, Any]:
    """Load a run's model card."""
    return json.loads((Path(run_dir) / "model_card.json").read_text(encoding="utf-8"))
