"""Chronological dataset partitioning module.

Partitions demand and expiry-risk feature tables into train, validation, and test splits
with strict chronological separation and 28-day gap buffers matching the forecast horizon.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_chronological_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str,
    gap_train_val: int = 28,
    gap_val_test: int = 28,
) -> None:
    """Validate strict temporal non-overlap and required horizon gap buffers."""
    max_train = pd.to_datetime(train_df[date_col]).max()
    min_val = pd.to_datetime(val_df[date_col]).min()
    max_val = pd.to_datetime(val_df[date_col]).max()
    min_test = pd.to_datetime(test_df[date_col]).min()

    # Temporal ordering checks
    assert max_train < min_val, f"Train-val overlap: max train ({max_train}) >= min val ({min_val})"
    assert max_val < min_test, f"Val-test overlap: max val ({max_val}) >= min test ({min_test})"

    # Gap checks
    actual_gap_1 = (min_val - max_train).days - 1
    actual_gap_2 = (min_test - max_val).days - 1

    assert actual_gap_1 >= gap_train_val, (
        f"Train-val gap ({actual_gap_1} days) is less than required horizon ({gap_train_val} days)"
    )
    assert actual_gap_2 >= gap_val_test, (
        f"Val-test gap ({actual_gap_2} days) is less than required horizon ({gap_val_test} days)"
    )


def split_demand_data(
    df: pd.DataFrame,
    splits_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partition product demand features into chronological train/val/test splits."""
    t_start = splits_cfg["train_start"]
    t_end = splits_cfg["train_end"]
    v_start = splits_cfg["val_start"]
    v_end = splits_cfg["val_end"]
    test_start = splits_cfg["test_start"]
    test_end = splits_cfg["test_end"]

    train = df[(df["date"] >= t_start) & (df["date"] <= t_end)].copy()
    val = df[(df["date"] >= v_start) & (df["date"] <= v_end)].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    # Validate temporal integrity
    validate_chronological_splits(
        train, val, test, date_col="date",
        gap_train_val=splits_cfg["gap_train_val_days"],
        gap_val_test=splits_cfg["gap_val_test_days"],
    )

    # Verify no NaN in target
    assert not train["target_demand_28d"].isna().any(), "NaN found in train target_demand_28d"
    assert not val["target_demand_28d"].isna().any(), "NaN found in val target_demand_28d"
    assert not test["target_demand_28d"].isna().any(), "NaN found in test target_demand_28d"

    total_rows = len(train) + len(val) + len(test)
    stats = {
        "train": {
            "date_range": [t_start, t_end],
            "days": (pd.to_datetime(t_end) - pd.to_datetime(t_start)).days + 1,
            "rows": len(train),
            "share_pct": round(len(train) / total_rows * 100, 2),
            "target_mean": round(float(train["target_demand_28d"].mean()), 2),
            "target_std": round(float(train["target_demand_28d"].std()), 2),
        },
        "gap_1_days": splits_cfg["gap_train_val_days"],
        "validation": {
            "date_range": [v_start, v_end],
            "days": (pd.to_datetime(v_end) - pd.to_datetime(v_start)).days + 1,
            "rows": len(val),
            "share_pct": round(len(val) / total_rows * 100, 2),
            "target_mean": round(float(val["target_demand_28d"].mean()), 2),
            "target_std": round(float(val["target_demand_28d"].std()), 2),
        },
        "gap_2_days": splits_cfg["gap_val_test_days"],
        "test": {
            "date_range": [test_start, test_end],
            "days": (pd.to_datetime(test_end) - pd.to_datetime(test_start)).days + 1,
            "rows": len(test),
            "share_pct": round(len(test) / total_rows * 100, 2),
            "target_mean": round(float(test["target_demand_28d"].mean()), 2),
            "target_std": round(float(test["target_demand_28d"].std()), 2),
        },
    }
    return train, val, test, stats


def split_expiry_risk_data(
    df: pd.DataFrame,
    splits_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partition batch expiry risk snapshots into chronological train/val/test splits."""
    t_start = splits_cfg["train_start"]
    t_end = splits_cfg["train_end"]
    v_start = splits_cfg["val_start"]
    v_end = splits_cfg["val_end"]
    test_start = splits_cfg["test_start"]
    test_end = splits_cfg["test_end"]

    train = df[(df["snapshot_date"] >= t_start) & (df["snapshot_date"] <= t_end)].copy()
    val = df[(df["snapshot_date"] >= v_start) & (df["snapshot_date"] <= v_end)].copy()
    test = df[(df["snapshot_date"] >= test_start) & (df["snapshot_date"] <= test_end)].copy()

    # Validate temporal ordering
    validate_chronological_splits(
        train, val, test, date_col="snapshot_date",
        gap_train_val=splits_cfg["gap_train_val_days"],
        gap_val_test=splits_cfg["gap_val_test_days"],
    )

    total_rows = len(train) + len(val) + len(test)
    stats = {
        "train": {
            "date_range": [t_start, t_end],
            "rows": len(train),
            "share_pct": round(len(train) / total_rows * 100, 2),
            "positive_class_rate": round(float(train["will_expire_unsold"].mean()) * 100, 2),
        },
        "gap_1_days": splits_cfg["gap_train_val_days"],
        "validation": {
            "date_range": [v_start, v_end],
            "rows": len(val),
            "share_pct": round(len(val) / total_rows * 100, 2),
            "positive_class_rate": round(float(val["will_expire_unsold"].mean()) * 100, 2),
        },
        "gap_2_days": splits_cfg["gap_val_test_days"],
        "test": {
            "date_range": [test_start, test_end],
            "rows": len(test),
            "share_pct": round(len(test) / total_rows * 100, 2),
            "positive_class_rate": round(float(test["will_expire_unsold"].mean()) * 100, 2),
        },
    }
    return train, val, test, stats


def run_split_pipeline():
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "configs" / "config.yaml"
    cfg = load_config(config_path)

    proc_dir = repo_root / "data" / "processed"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading demand feature dataset...")
    demand_df = pd.read_csv(proc_dir / "demand_features.csv")
    train_dem, val_dem, test_dem, dem_stats = split_demand_data(demand_df, cfg["splits"])

    train_dem.to_csv(proc_dir / "train_demand.csv", index=False)
    val_dem.to_csv(proc_dir / "val_demand.csv", index=False)
    test_dem.to_csv(proc_dir / "test_demand.csv", index=False)
    logger.info("Demand splits saved: train, val, test.")

    logger.info("Loading expiry risk feature dataset...")
    risk_df = pd.read_csv(proc_dir / "expiry_risk_features.csv")
    train_risk, val_risk, test_risk, risk_stats = split_expiry_risk_data(risk_df, cfg["splits"])

    train_risk.to_csv(proc_dir / "train_risk.csv", index=False)
    val_risk.to_csv(proc_dir / "val_risk.csv", index=False)
    test_risk.to_csv(proc_dir / "test_risk.csv", index=False)
    logger.info("Expiry risk splits saved: train, val, test.")

    summary = {
        "demand_splits": dem_stats,
        "expiry_risk_splits": risk_stats,
    }

    summary_file = reports_dir / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Split pipeline completed successfully. Summary saved to {summary_file}")
    
    # Print formatted summary table to console
    print("\n" + "=" * 70)
    print("CHRONOLOGICAL SPLIT SUMMARY (HORIZON GAP = 28 DAYS)")
    print("=" * 70)
    print("1. DEMAND FORECASTING SPLITS:")
    print(f"   Train:      {dem_stats['train']['date_range'][0]} to {dem_stats['train']['date_range'][1]} | {dem_stats['train']['days']} days | {dem_stats['train']['rows']:,} rows ({dem_stats['train']['share_pct']}%)")
    print(f"   [Gap 1:     28 days buffer to prevent forward label leakage]")
    print(f"   Validation: {dem_stats['validation']['date_range'][0]} to {dem_stats['validation']['date_range'][1]} | {dem_stats['validation']['days']} days | {dem_stats['validation']['rows']:,} rows ({dem_stats['validation']['share_pct']}%)")
    print(f"   [Gap 2:     28 days buffer to prevent forward label leakage]")
    print(f"   Test:       {dem_stats['test']['date_range'][0]} to {dem_stats['test']['date_range'][1]} | {dem_stats['test']['days']} days | {dem_stats['test']['rows']:,} rows ({dem_stats['test']['share_pct']}%)")
    print("-" * 70)
    print("2. EXPIRY-RISK CLASSIFICATION SPLITS:")
    print(f"   Train:      {risk_stats['train']['rows']:,} snapshots ({risk_stats['train']['share_pct']}%) | Pos Rate: {risk_stats['train']['positive_class_rate']}%")
    print(f"   Validation: {risk_stats['validation']['rows']:,} snapshots ({risk_stats['validation']['share_pct']}%) | Pos Rate: {risk_stats['validation']['positive_class_rate']}%")
    print(f"   Test:       {risk_stats['test']['rows']:,} snapshots ({risk_stats['test']['share_pct']}%) | Pos Rate: {risk_stats['test']['positive_class_rate']}%")
    print("=" * 70 + "\n")
    return summary


if __name__ == "__main__":
    run_split_pipeline()
