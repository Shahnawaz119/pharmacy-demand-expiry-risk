"""Unit tests for demand forecasting and expiry-risk models."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.models.demand_models import (
    MovingAverageBaseline,
    SeasonalNaiveBaseline,
    DEMAND_FEATURE_COLS,
)
from src.models.expiry_models import (
    RuleBasedExpiryModel,
    RISK_FEATURE_COLS,
)


def test_demand_baselines_output():
    df = pd.DataFrame({
        "lag_1": [10.0, 20.0],
        "rolling_mean_7": [12.0, 18.0],
        "rolling_mean_28": [11.0, 19.0],
    })
    ma7 = MovingAverageBaseline(window=7, horizon=28)
    preds7 = ma7.predict(df)
    assert len(preds7) == 2
    assert (preds7 >= 0).all()
    assert np.isclose(preds7[0], 12.0 * 28.0)

    snaive = SeasonalNaiveBaseline(horizon=28)
    preds_sn = snaive.predict(df)
    assert len(preds_sn) == 2
    assert (preds_sn >= 0).all()


def test_model_artifacts_load_and_predict():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    proc_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    # 1. Demand models
    lgb_quantiles = joblib.load(models_dir / "demand_lightgbm_quantiles.joblib")
    assert 0.10 in lgb_quantiles and 0.50 in lgb_quantiles and 0.90 in lgb_quantiles

    val_dem = pd.read_csv(proc_dir / "val_demand.csv").head(10)
    p10 = lgb_quantiles[0.10].predict(val_dem[DEMAND_FEATURE_COLS])
    p50 = lgb_quantiles[0.50].predict(val_dem[DEMAND_FEATURE_COLS])
    p90 = lgb_quantiles[0.90].predict(val_dem[DEMAND_FEATURE_COLS])

    assert len(p50) == 10
    assert (p50 >= 0).all()
    # Quantile monotonicity check: p10 <= p90
    assert (p10 <= p90 + 1e-3).all()

    # 2. Expiry risk model
    expiry_lgb = joblib.load(models_dir / "expiry_lightgbm.joblib")
    clf = expiry_lgb["model"]
    thresh = expiry_lgb["threshold"]
    assert 0.0 < thresh < 1.0

    val_risk = pd.read_csv(proc_dir / "val_risk.csv").head(10)
    probs = clf.predict_proba(val_risk[RISK_FEATURE_COLS])[:, 1]
    assert len(probs) == 10
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
