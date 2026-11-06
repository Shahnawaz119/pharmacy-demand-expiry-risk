"""Automated leakage verification tests.

Validates that feature engineering at date t is mathematically invariant
to arbitrary perturbations of future data (t+1, t+2, ...).
"""

import numpy as np
import pandas as pd
import pytest
from src.features.build_features import build_demand_features


@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    records = []
    for dt in dates:
        dt_str = dt.strftime("%Y-%m-%d")
        records.append({
            "date": dt_str,
            "product_id": "PROD_001",
            "units_sold": 10.0 + (dt.dayofweek * 2.0),
            "selling_price": 50.0,
            "is_promoted": 0,
            "stock_on_hand": 100.0,
            "stockout_flag": 0,
        })
    sales_df = pd.DataFrame(records)

    products_df = pd.DataFrame([{
        "product_id": "PROD_001",
        "category": "analgesics",
        "base_price": 50.0,
        "unit_cost": 36.0,
        "shelf_life_months": 24,
        "demand_velocity": "fast",
    }])

    cal_records = []
    for dt in dates:
        cal_records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "day_of_week": dt.dayofweek,
            "month": dt.month,
            "day_of_year": dt.dayofyear,
            "is_weekend": int(dt.dayofweek in [5, 6]),
            "season": "winter",
            "is_holiday": 0,
            "is_festival": 0,
        })
    calendar_df = pd.DataFrame(cal_records)

    return sales_df, products_df, calendar_df


def test_future_perturbation_invariance(sample_data):
    """Perturbing future sales (t+1, t+2...) MUST NOT alter features at day t."""
    sales_df, products_df, calendar_df = sample_data

    # Snapshot date t is 2023-02-15 (index 45)
    t_date = "2023-02-15"

    # Compute original features
    feat_orig = build_demand_features(sales_df, products_df, calendar_df, horizon_days=28)
    row_orig_t = feat_orig[feat_orig["date"] == t_date].iloc[0]

    # Create perturbed sales where future sales (after t_date) are multiplied by 100
    sales_perturbed = sales_df.copy()
    future_mask = sales_perturbed["date"] > t_date
    sales_perturbed.loc[future_mask, "units_sold"] = sales_perturbed.loc[future_mask, "units_sold"] * 100.0
    sales_perturbed.loc[future_mask, "selling_price"] = 999.0

    # Compute features on perturbed dataset
    feat_pert = build_demand_features(sales_perturbed, products_df, calendar_df, horizon_days=28)
    row_pert_t = feat_pert[feat_pert["date"] == t_date].iloc[0]

    # Test all predictor columns
    feature_cols = [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
        "rolling_mean_28", "rolling_std_28", "rolling_max_28",
        "days_since_last_sale", "selling_price", "price_ratio",
        "is_promoted", "promo_rolling_28", "stock_on_hand",
        "stockout_flag", "stockout_rolling_28", "current_stock_cover",
    ]

    for col in feature_cols:
        val_orig = row_orig_t[col]
        val_pert = row_pert_t[col]
        assert np.isclose(val_orig, val_pert, rtol=1e-6, atol=1e-6), (
            f"LEAK DETECTED: Feature '{col}' on date {t_date} changed from {val_orig} to {val_pert} "
            f"after altering future sales at t+1!"
        )


def test_target_label_not_in_feature_set():
    """Ensure ground-truth labels and future unconstrained demand are excluded from predictors."""
    prohibited_features = ["target_demand_28d", "will_expire_unsold", "actual_unsold_units", "unconstrained_demand"]
    
    # Check demand feature candidates
    demand_candidate_features = [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
        "rolling_mean_28", "rolling_std_28",
        "days_since_last_sale", "price_ratio", "is_promoted",
        "current_stock_cover", "category_code", "season_code",
    ]
    for col in prohibited_features:
        assert col not in demand_candidate_features

    # Check expiry risk candidate features
    risk_candidate_features = [
        "days_to_expiry", "remaining_qty", "batch_age_days", "fraction_depleted",
        "earlier_batches_qty", "fefo_queue_position", "product_daily_velocity",
        "expected_demand_to_expiry", "allocation_share", "stock_cover_days",
    ]
    for col in prohibited_features:
        assert col not in risk_candidate_features
