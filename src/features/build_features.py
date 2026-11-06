"""Feature engineering and temporal leakage safety module.

Constructs feature matrices for:
1. Demand Forecasting (product x day, predicting forward 28-day aggregate units sold)
2. Expiry-Risk Classification (batch x snapshot date, predicting whether batch expires unsold)

Strict temporal safety: all features at snapshot date t are computed using strictly
information available on or before day t. Future sales after day t are never used in features.
"""

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_demand_features(
    sales_df: pd.DataFrame,
    products_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    horizon_days: int = 28,
) -> pd.DataFrame:
    """Build leakage-safe demand forecasting features.
    
    Each row at date t represents product i. The target is the forward 28-day sum of sales:
    target_demand_28d = sum(units_sold[t+1 ... t+28]).
    All features use strictly data up to and including date t.
    """
    logger.info("Building demand features...")
    df = sales_df.sort_values(by=["product_id", "date"]).copy()
    df["date_dt"] = pd.to_datetime(df["date"])

    # 1. Target calculation: forward 28-day sum of units_sold (t+1 to t+28)
    # Using reverse rolling sum on shifted sales to avoid including day t in target
    # Note: Target is strictly used for training/eval, NEVER as a feature.
    grouped = df.groupby("product_id")["units_sold"]
    
    # We shift -horizon_days so that future sales are summed over [t+1, t+horizon]
    # In pandas: shift(-1).rolling(horizon_days, min_periods=horizon_days).sum() when reversed,
    # or computing rolling sum from the future:
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_days)
    df["target_demand_28d"] = (
        df.groupby("product_id")["units_sold"]
        .transform(lambda s: s.shift(-1).rolling(window=indexer, min_periods=horizon_days).sum())
    )

    # 2. Historical Lag Features (strictly on or before day t)
    # lag_1 is sales at day t; lag_7 is sales at day t-6, etc.
    df["lag_1"] = df.groupby("product_id")["units_sold"].shift(0)
    df["lag_7"] = df.groupby("product_id")["units_sold"].shift(6)
    df["lag_14"] = df.groupby("product_id")["units_sold"].shift(13)
    df["lag_28"] = df.groupby("product_id")["units_sold"].shift(27)

    # 3. Rolling Statistics (strictly on or before day t)
    # rolling(7) on s.shift(0) takes days [t-6 ... t]
    for w in [7, 14, 28]:
        df[f"rolling_mean_{w}"] = df.groupby("product_id")["units_sold"].transform(
            lambda s: s.rolling(window=w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = df.groupby("product_id")["units_sold"].transform(
            lambda s: s.rolling(window=w, min_periods=1).std().fillna(0.0)
        )
        df[f"rolling_max_{w}"] = df.groupby("product_id")["units_sold"].transform(
            lambda s: s.rolling(window=w, min_periods=1).max()
        )

    # 4. Days since last sale (intermittency feature, up to day t)
    def calc_days_since_sale(series: pd.Series) -> pd.Series:
        is_sale = series > 0
        sale_indices = np.where(is_sale)[0]
        result = np.zeros(len(series), dtype=float)
        
        last_idx = -999
        for i in range(len(series)):
            if is_sale.iloc[i]:
                last_idx = i
                result[i] = 0.0
            else:
                if last_idx >= 0:
                    result[i] = float(i - last_idx)
                else:
                    result[i] = 14.0  # default prior assumption
        return pd.Series(result, index=series.index)

    df["days_since_last_sale"] = df.groupby("product_id")["units_sold"].transform(calc_days_since_sale)

    # 5. Price & Promotion Features
    df["selling_price"] = df["selling_price"].astype(float)
    df["is_promoted"] = df["is_promoted"].astype(int)
    df["promo_rolling_28"] = df.groupby("product_id")["is_promoted"].transform(
        lambda s: s.rolling(window=28, min_periods=1).mean()
    )

    # 6. Inventory & Stockout features at day t
    df["stock_on_hand"] = df["stock_on_hand"].astype(float)
    df["stockout_flag"] = df["stockout_flag"].astype(int)
    df["stockout_rolling_28"] = df.groupby("product_id")["stockout_flag"].transform(
        lambda s: s.rolling(window=28, min_periods=1).mean()
    )
    df["current_stock_cover"] = df["stock_on_hand"] / (df["rolling_mean_28"] + 1e-4)

    # Merge product catalog attributes
    prod_cols = ["product_id", "category", "base_price", "unit_cost", "shelf_life_months", "demand_velocity"]
    df = df.merge(products_df[prod_cols], on="product_id", how="left")
    df["price_ratio"] = df["selling_price"] / (df["base_price"] + 1e-4)

    # Merge calendar features
    cal_cols = ["date", "day_of_week", "month", "day_of_year", "is_weekend", "season", "is_holiday", "is_festival"]
    df = df.merge(calendar_df[cal_cols], on="date", how="left")

    # Cyclical calendar features
    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # Categorical encoding preparation
    category_map = {cat: idx for idx, cat in enumerate(sorted(df["category"].unique()))}
    df["category_code"] = df["category"].map(category_map)
    season_map = {seas: idx for idx, seas in enumerate(["winter", "summer", "monsoon", "post_monsoon"])}
    df["season_code"] = df["season"].map(season_map)
    velocity_map = {"slow": 0, "medium": 1, "fast": 2}
    df["velocity_code"] = df["demand_velocity"].map(velocity_map)

    df = df.drop(columns=["date_dt"])
    logger.info(f"Demand feature matrix constructed with {len(df):,} rows and {len(df.columns)} columns.")
    return df


def build_expiry_risk_snapshots(
    sales_df: pd.DataFrame,
    batches_df: pd.DataFrame,
    products_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    config: dict,
    snapshot_frequency_days: int = 7,
) -> pd.DataFrame:
    """Build batch expiry risk classification dataset.
    
    For each snapshot date t (taken weekly), we identify all active batches:
    - received on or before date t
    - expiry_date > t
    - expiring within scoring_window_days (90 days)
    - remaining_quantity at date t > 0
    
    Label: will_expire_unsold = 1 if physical units remaining at batch expiry date > threshold.
    Ground-truth is derived from simulated sales between t and expiry date under FEFO.
    """
    logger.info("Building expiry risk dataset across weekly snapshots...")
    scoring_window = config["expiry_risk"]["scoring_window_days"]
    threshold_unsold = config["expiry_risk"]["threshold_units_unsold"]
    safety_factor = config["expiry_risk"]["safety_factor"]

    # Pre-parse dates
    sales = sales_df.copy()
    sales["date_dt"] = pd.to_datetime(sales["date"])
    batches = batches_df.copy()
    batches["recv_dt"] = pd.to_datetime(batches["received_date"])
    batches["exp_dt"] = pd.to_datetime(batches["expiry_date"])

    # Pre-calculate daily product sales lookup: (product_id, date_dt) -> units_sold
    sales_indexed = sales.set_index(["product_id", "date_dt"])["units_sold"].to_dict()

    # Pre-calculate product 28-day rolling mean velocity up to date t
    prod_velocity_28 = (
        sales.sort_values(["product_id", "date_dt"])
        .groupby("product_id")
        .apply(lambda g: g.set_index("date_dt")["units_sold"].rolling(28, min_periods=1).mean())
        .to_dict()
    )

    dates = sorted(sales["date_dt"].unique())
    # Take snapshots every 7 days (e.g. weekly Monday inventory review)
    snapshot_dates = dates[::snapshot_frequency_days]

    # Reconstruct dynamic batch depletion timeline under FEFO
    # To obtain exact ground-truth remaining qty at any day t and at expiry date:
    # We simulate day-by-day batch inventory depletion across the entire 2-year history
    product_batches = {p: [] for p in products_df["product_id"].unique()}
    for b_idx, b in batches.sort_values("recv_dt").iterrows():
        product_batches[b["product_id"]].append({
            "batch_id": b["batch_id"],
            "product_id": b["product_id"],
            "received_date": b["received_date"],
            "expiry_date": b["expiry_date"],
            "recv_dt": b["recv_dt"],
            "exp_dt": b["exp_dt"],
            "quantity_received": float(b["quantity_received"]),
            "unit_cost": float(b["unit_cost"]),
            "supplier": b["supplier"],
        })

    # Track remaining qty day by day for each batch
    batch_inventory_history: Dict[str, Dict[datetime, float]] = {}
    for b in batches["batch_id"]:
        batch_inventory_history[b] = {}

    # Run global FEFO depletion tracker
    active_batches_tracker = {p: [] for p in product_batches}
    all_batch_ptrs = {p: list(product_batches[p]) for p in product_batches}

    # Record remaining quantities at each day
    batch_final_unsold_at_expiry: Dict[str, float] = {}

    for dt in dates:
        # Check newly received batches today
        for p in active_batches_tracker:
            while all_batch_ptrs[p] and all_batch_ptrs[p][0]["recv_dt"] <= dt:
                nb = all_batch_ptrs[p].pop(0)
                nb["curr_qty"] = nb["quantity_received"]
                active_batches_tracker[p].append(nb)

            # Remove batches expired before today and record their unsold units
            surviving = []
            for b in active_batches_tracker[p]:
                if b["exp_dt"] <= dt:
                    batch_final_unsold_at_expiry[b["batch_id"]] = b["curr_qty"]
                else:
                    surviving.append(b)
            active_batches_tracker[p] = surviving

            # Sort surviving batches by expiry date (FEFO)
            active_batches_tracker[p].sort(key=lambda x: x["exp_dt"])

            # Deduct today's sales
            sold_today = sales_indexed.get((p, dt), 0.0)
            rem_to_deduct = sold_today
            for b in active_batches_tracker[p]:
                if rem_to_deduct <= 0:
                    break
                deduct = min(b["curr_qty"], rem_to_deduct)
                b["curr_qty"] -= deduct
                rem_to_deduct -= deduct

            # Snapshot current quantity for active batches
            for b in active_batches_tracker[p]:
                batch_inventory_history[b["batch_id"]][dt] = b["curr_qty"]

    # Any batch still active at the end of simulation
    for p in active_batches_tracker:
        for b in active_batches_tracker[p]:
            batch_final_unsold_at_expiry[b["batch_id"]] = b["curr_qty"]

    # Now construct snapshot records
    records = []
    prod_meta = products_df.set_index("product_id").to_dict(orient="index")

    for snap_dt in snapshot_dates:
        snap_str = snap_dt.strftime("%Y-%m-%d")

        # Find all batches eligible at this snapshot date
        # (received <= snap_dt, expiry_date > snap_dt, expiry_date <= snap_dt + 90 days, remaining_qty > 0)
        for _, b in batches.iterrows():
            b_id = b["batch_id"]
            p_id = b["product_id"]
            recv_dt = b["recv_dt"]
            exp_dt = b["exp_dt"]

            if recv_dt > snap_dt or exp_dt <= snap_dt:
                continue

            days_to_expiry = (exp_dt - snap_dt).days
            if days_to_expiry > scoring_window:
                continue

            rem_qty = batch_inventory_history.get(b_id, {}).get(snap_dt, 0.0)
            if rem_qty <= 0:
                continue

            qty_received = float(b["quantity_received"])
            batch_age = (snap_dt - recv_dt).days
            depleted_frac = 1.0 - (rem_qty / max(1.0, qty_received))

            # FEFO Queue positioning: sum of remaining quantities in earlier-expiring batches of same product
            earlier_qty = 0.0
            queue_position = 0
            for _, other_b in batches[batches["product_id"] == p_id].iterrows():
                if other_b["batch_id"] == b_id:
                    continue
                o_recv = other_b["recv_dt"]
                o_exp = other_b["exp_dt"]
                if o_recv <= snap_dt and snap_dt < o_exp < exp_dt:
                    o_rem = batch_inventory_history.get(other_b["batch_id"], {}).get(snap_dt, 0.0)
                    if o_rem > 0:
                        earlier_qty += o_rem
                        queue_position += 1

            # Product demand velocity up to snapshot date t
            daily_vel = prod_velocity_28.get((p_id, snap_dt), prod_meta[p_id]["base_daily_demand"])
            if np.isnan(daily_vel):
                daily_vel = prod_meta[p_id]["base_daily_demand"]

            expected_demand_to_expiry = daily_vel * float(days_to_expiry)

            # FEFO allocation share for this batch
            # If earlier batches already exceed expected demand, allocation_share approaches 0
            if expected_demand_to_expiry > 0:
                demand_remaining_for_batch = max(0.0, expected_demand_to_expiry - earlier_qty)
                allocation_share = min(1.0, demand_remaining_for_batch / expected_demand_to_expiry)
            else:
                allocation_share = 0.0

            # Explicit configurable Risk Rule:
            # batch is at risk if: remaining_qty > expected_demand * allocation_share * safety_factor
            threshold_capacity = expected_demand_to_expiry * allocation_share * safety_factor
            risk_rule_flag = int(rem_qty > threshold_capacity)
            risk_rule_gap = rem_qty - threshold_capacity

            # Ground truth label: units unsold when expiry was reached
            actual_unsold = batch_final_unsold_at_expiry.get(b_id, 0.0)
            will_expire_unsold = int(actual_unsold > threshold_unsold)

            p_info = prod_meta[p_id]
            records.append({
                "snapshot_date": snap_str,
                "batch_id": b_id,
                "product_id": p_id,
                "category": p_info["category"],
                "unit_cost": float(b["unit_cost"]),
                "quantity_received": qty_received,
                "remaining_qty": rem_qty,
                "batch_age_days": batch_age,
                "days_to_expiry": days_to_expiry,
                "fraction_depleted": round(depleted_frac, 4),
                "earlier_batches_qty": earlier_qty,
                "fefo_queue_position": queue_position,
                "product_daily_velocity": round(daily_vel, 2),
                "expected_demand_to_expiry": round(expected_demand_to_expiry, 2),
                "allocation_share": round(allocation_share, 4),
                "risk_rule_flag": risk_rule_flag,
                "risk_rule_gap": round(risk_rule_gap, 2),
                "stock_cover_days": round(rem_qty / max(0.1, daily_vel), 1),
                "will_expire_unsold": will_expire_unsold,  # Ground truth label
                "actual_unsold_units": round(actual_unsold, 1),
            })

    risk_df = pd.DataFrame(records)
    logger.info(f"Expiry risk dataset constructed with {len(risk_df):,} snapshot records across 90-day expiry windows.")
    return risk_df


def run_feature_engineering():
    repo_root = Path(__file__).resolve().parent.parent.parent
    cfg = load_config(repo_root / "configs" / "config.yaml")

    proc_dir = repo_root / "data" / "processed"
    sales_df = pd.read_csv(proc_dir / "daily_sales.csv")
    products_df = pd.read_csv(proc_dir / "products.csv")
    calendar_df = pd.read_csv(proc_dir / "calendar.csv")
    batches_df = pd.read_csv(proc_dir / "batches.csv")

    # 1. Build demand features
    demand_df = build_demand_features(
        sales_df=sales_df,
        products_df=products_df,
        calendar_df=calendar_df,
        horizon_days=cfg["forecasting"]["horizon_days"],
    )
    demand_out = proc_dir / "demand_features.csv"
    demand_df.to_csv(demand_out, index=False)
    logger.info(f"Saved demand features to {demand_out}")

    # 2. Build expiry risk features
    risk_df = build_expiry_risk_snapshots(
        sales_df=sales_df,
        batches_df=batches_df,
        products_df=products_df,
        calendar_df=calendar_df,
        config=cfg,
        snapshot_frequency_days=7,
    )
    risk_out = proc_dir / "expiry_risk_features.csv"
    risk_df.to_csv(risk_out, index=False)
    logger.info(f"Saved expiry risk features to {risk_out}")


if __name__ == "__main__":
    run_feature_engineering()
