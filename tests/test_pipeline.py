"""Unit tests for the persisted production model artifact and metadata."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def test_artifact_bundle_exists_and_loads():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    bundle_path = models_dir / "pharmacy_forecaster.joblib"
    assert bundle_path.exists(), f"Unified artifact not found at {bundle_path}"

    bundle = joblib.load(bundle_path)
    assert "version" in bundle
    assert "demand" in bundle
    assert "expiry_risk" in bundle

    assert 0.50 in bundle["demand"]["primary_model"]
    assert hasattr(bundle["expiry_risk"]["primary_model"], "predict_proba")
    assert 0.0 < bundle["expiry_risk"]["optimal_threshold"] < 1.0


def test_metadata_json_schema():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    meta_path = models_dir / "metadata.json"
    assert meta_path.exists(), f"Metadata json not found at {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["package_version"] == "0.1.0"
    assert "environment_dependencies" in meta
    assert "lightgbm" in meta["environment_dependencies"]
    assert "forecasting_specification" in meta
    assert meta["forecasting_specification"]["horizon_days"] == 28
    assert "held_out_test_metrics" in meta


def test_standalone_artifact_predictions():
    """Verify inference without retraining."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    proc_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    bundle = joblib.load(models_dir / "pharmacy_forecaster.joblib")
    val_dem = pd.read_csv(proc_dir / "val_demand.csv").head(3)
    val_risk = pd.read_csv(proc_dir / "val_risk.csv").head(3)

    # Demand prediction
    d_cols = bundle["demand"]["feature_columns"]
    lgb_quantiles = bundle["demand"]["primary_model"]
    p50 = lgb_quantiles[0.50].predict(val_dem[d_cols])
    assert len(p50) == 3
    assert (p50 >= 0).all()

    # Risk prediction
    r_cols = bundle["expiry_risk"]["feature_columns"]
    lgb_clf = bundle["expiry_risk"]["primary_model"]
    thresh = bundle["expiry_risk"]["optimal_threshold"]
    probs = lgb_clf.predict_proba(val_risk[r_cols])[:, 1]
    assert len(probs) == 3
    assert (probs >= 0).all() and (probs <= 1.0).all()
