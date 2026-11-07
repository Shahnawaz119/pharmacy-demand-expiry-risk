"""Demand forecasting models and baselines.

Implements:
1. Moving Average baselines (7-day and 28-day historical velocity scaled to 28-day horizon)
2. Seasonal Naive baseline (cyclical 4-week repetition of recent day-of-week pattern)
3. Regularized Linear Model (Ridge with StandardScaler)
4. Gradient Boosted Trees (LightGBM quantile regressors for p10, p50, p90)
"""

import logging
from typing import Dict, Any, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEMAND_FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_28", "rolling_std_28", "rolling_max_28",
    "days_since_last_sale", "selling_price", "price_ratio",
    "is_promoted", "promo_rolling_28", "stock_on_hand",
    "stockout_flag", "stockout_rolling_28", "current_stock_cover",
    "category_code", "season_code", "velocity_code",
    "day_of_week", "month", "is_weekend", "is_holiday", "is_festival",
    "sin_day_of_year", "cos_day_of_year", "sin_dow", "cos_dow"
]


class MovingAverageBaseline:
    """Predicts forward 28-day demand as horizon_days * rolling_mean."""
    def __init__(self, window: int = 28, horizon: int = 28):
        self.window = window
        self.horizon = horizon
        self.col = f"rolling_mean_{window}"

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.col in X.columns:
            return np.maximum(0.0, X[self.col].values * float(self.horizon))
        # Fallback to lag_1
        return np.maximum(0.0, X["lag_1"].values * float(self.horizon))


class SeasonalNaiveBaseline:
    """Projects recent 4-week weekday pattern into forward 28-day demand."""
    def __init__(self, horizon: int = 28):
        self.horizon = horizon

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Blend 7-day rolling mean (captures immediate weekly cycle) with 28-day level
        # Project 4 repetitions of the 7-day pattern
        pred = (X["rolling_mean_7"].values * 0.6 + X["rolling_mean_28"].values * 0.4) * float(self.horizon)
        return np.maximum(0.0, pred)


def train_ridge_demand_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 10.0,
    random_state: int = 42,
) -> Pipeline:
    """Train regularized linear regression (Ridge) with standardized features."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=random_state)),
    ])
    pipe.fit(X_train[DEMAND_FEATURE_COLS], y_train)
    return pipe


def train_lightgbm_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles: list = [0.10, 0.50, 0.90],
    random_state: int = 42,
) -> Dict[float, lgb.LGBMRegressor]:
    """Train LightGBM quantile regression models for uncertainty intervals."""
    models = {}
    for q in quantiles:
        logger.info(f"Training LightGBM quantile model for alpha={q:.2f}...")
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=180,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbosity=-1,
        )
        model.fit(
            X_train[DEMAND_FEATURE_COLS],
            y_train,
            eval_set=[(X_val[DEMAND_FEATURE_COLS], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        models[q] = model
    return models


def evaluate_demand_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate demand error metrics: MAE, RMSE, WAPE, and Bias (Mean Error)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))

    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    total_actual = float(np.sum(y_true))
    wape = float(np.sum(np.abs(errors)) / max(1e-5, total_actual)) * 100.0
    bias = float(np.mean(errors))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "wape": round(wape, 2),
        "bias": round(bias, 2),
    }


def evaluate_interval_coverage(
    y_true: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
) -> Dict[str, float]:
    """Calculate empirical coverage rate and mean interval width."""
    y_true = np.asarray(y_true, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    p90 = np.asarray(p90, dtype=float)

    in_interval = (y_true >= p10) & (y_true <= p90)
    coverage = float(np.mean(in_interval)) * 100.0
    width = float(np.mean(p90 - p10))

    return {
        "coverage_pct": round(coverage, 2),
        "mean_interval_width": round(width, 2),
    }
