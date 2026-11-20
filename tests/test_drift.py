"""Drift baseline: detects a shift, stays quiet on the same distribution, stores no rows."""

import json

import numpy as np
import pandas as pd

from src.tools.drift import build_profile, check_drift, main, save_profile


def _sample(n, seed, shift=0.0, plan_p=(0.6, 0.3, 0.1)):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "tenure": rng.normal(24 + shift, 6, n),
        "plan": rng.choice(["basic", "pro", "enterprise"], n, p=plan_p),
        "customer_id": [f"cust-{seed}-{i}" for i in range(n)],
    })


def test_no_drift_on_same_distribution():
    ref = build_profile(_sample(2000, seed=0))
    result = check_drift(_sample(2000, seed=1), ref)
    assert result["drift"] is False, result
    assert all(score < 0.1 for score in result["psi"].values())


def test_drift_detected_on_shifted_sample():
    ref = build_profile(_sample(2000, seed=0))
    result = check_drift(_sample(2000, seed=1, shift=12, plan_p=(0.1, 0.3, 0.6)), ref)
    assert result["drift"] is True
    assert {"tenure", "plan"} <= set(result["drifted_features"])
    assert "customer_id" not in result["drifted_features"]


def test_missing_feature_counts_as_drift():
    ref = build_profile(_sample(500, seed=0))
    result = check_drift(_sample(500, seed=1).drop(columns="plan"), ref)
    assert result["missing_features"] == ["plan"] and result["drift"] is True


def test_profile_contains_no_raw_rows(tmp_path):
    df = _sample(300, seed=0)
    text = save_profile(df, tmp_path).read_text()
    for value in df["customer_id"]:
        assert value not in text  # unique IDs are folded into __other__
    profile = json.loads(text)
    assert profile["features"]["customer_id"]["categories"] == []
    assert len(profile["features"]["tenure"]["edges"]) <= 9
    assert len(text) < 5000


def test_cli_exit_code(tmp_path):
    save_profile(_sample(1000, seed=0), tmp_path)
    same, shifted = tmp_path / "same.csv", tmp_path / "shifted.csv"
    _sample(1000, seed=1).to_csv(same, index=False)
    _sample(1000, seed=1, shift=12).to_csv(shifted, index=False)
    assert main([str(tmp_path), str(same)]) == 0
    assert main([str(tmp_path), str(shifted)]) == 1
