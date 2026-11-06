"""Data quality validation and cleaning module.

Performs deduplication, missing value imputation, negative/impossible value
remediation, batch timeline consistency checks, and outlier capping.
Exports clean processed datasets to data/processed/ and saves an audit summary.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_products(products_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate product catalog entries."""
    stats = {
        "initial_rows": len(products_df),
        "duplicates_removed": 0,
        "final_rows": 0,
    }
    cleaned = products_df.drop_duplicates(subset=["product_id"])
    stats["duplicates_removed"] = stats["initial_rows"] - len(cleaned)
    stats["final_rows"] = len(cleaned)
    return cleaned, stats


def clean_batches(batches_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate batch records: remove duplicates and batches where expiry <= received_date."""
    stats = {
        "initial_rows": len(batches_df),
        "duplicates_removed": 0,
        "invalid_expiry_dropped": 0,
        "final_rows": 0,
    }
    df = batches_df.copy()
    
    # 1. Duplicates
    init_len = len(df)
    df = df.drop_duplicates(subset=["batch_id"])
    stats["duplicates_removed"] = init_len - len(df)

    # 2. Date consistency check
    df["received_dt"] = pd.to_datetime(df["received_date"])
    df["expiry_dt"] = pd.to_datetime(df["expiry_date"])
    
    invalid_mask = df["expiry_dt"] <= df["received_dt"]
    stats["invalid_expiry_dropped"] = int(invalid_mask.sum())
    
    df = df[~invalid_mask].copy()
    df = df.drop(columns=["received_dt", "expiry_dt"])
    stats["final_rows"] = len(df)

    return df, stats


def clean_daily_sales(
    sales_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean daily sales: deduplicate, handle missing values, rectify negative sales, cap outliers."""
    stats = {
        "initial_rows": len(sales_df),
        "duplicates_removed": 0,
        "missing_price_imputed": 0,
        "missing_sales_imputed": 0,
        "negative_sales_rectified": 0,
        "outliers_capped": 0,
        "final_rows": 0,
    }

    df = sales_df.copy()

    # 1. Deduplication on composite key (date, product_id)
    init_len = len(df)
    df = df.drop_duplicates(subset=["date", "product_id"], keep="first")
    stats["duplicates_removed"] = init_len - len(df)

    # 2. Impute missing selling_price using product base_price
    missing_price_mask = df["selling_price"].isna()
    stats["missing_price_imputed"] = int(missing_price_mask.sum())
    
    price_map = dict(zip(products_df["product_id"], products_df["base_price"]))
    df.loc[missing_price_mask, "selling_price"] = df.loc[missing_price_mask, "product_id"].map(price_map)

    # 3. Impute missing units_sold (fill with 0.0)
    missing_sales_mask = df["units_sold"].isna()
    stats["missing_sales_imputed"] = int(missing_sales_mask.sum())
    df["units_sold"] = df["units_sold"].fillna(0.0)

    # 4. Correct negative sales (sensor or keying errors)
    negative_mask = df["units_sold"] < 0
    stats["negative_sales_rectified"] = int(negative_mask.sum())
    df.loc[negative_mask, "units_sold"] = 0.0

    # 5. Outlier detection and treatment per product
    # Cap sales exceeding Q3 + 3.0 * IQR per product
    capped_count = 0
    df_cleaned_list = []
    
    for _, group in df.groupby("product_id", as_index=False):
        q25 = group["units_sold"].quantile(0.25)
        q75 = group["units_sold"].quantile(0.75)
        iqr = q75 - q25
        upper_bound = q75 + (3.0 * iqr)
        
        # Upper threshold must be at least reasonable minimum
        upper_bound = max(upper_bound, 20.0)
        
        outliers = group["units_sold"] > upper_bound
        capped_count += int(outliers.sum())
        
        group_cleaned = group.copy()
        group_cleaned.loc[outliers, "units_sold"] = upper_bound
        df_cleaned_list.append(group_cleaned)

    df = pd.concat(df_cleaned_list, ignore_index=True)
    df = df.sort_values(by=["date", "product_id"]).reset_index(drop=True)

    stats["outliers_capped"] = capped_count
    stats["final_rows"] = len(df)

    return df, stats


def clean_calendar(calendar_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate calendar records."""
    stats = {
        "initial_rows": len(calendar_df),
        "duplicates_removed": 0,
        "final_rows": 0,
    }
    df = calendar_df.drop_duplicates(subset=["date"])
    stats["duplicates_removed"] = len(calendar_df) - len(df)
    stats["final_rows"] = len(df)
    return df, stats


def run_pipeline() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "configs" / "config.yaml"
    cfg = load_config(config_path)

    raw_dir = repo_root / cfg["paths"]["raw_data_dir"]
    proc_dir = repo_root / cfg["paths"]["processed_data_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = repo_root / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading raw data tables...")
    products_raw = pd.read_csv(raw_dir / "products.csv")
    calendar_raw = pd.read_csv(raw_dir / "calendar.csv")
    batches_raw = pd.read_csv(raw_dir / "batches.csv")
    sales_raw = pd.read_csv(raw_dir / "daily_sales.csv")

    logger.info("Cleaning products...")
    products_clean, prod_stats = clean_products(products_raw)
    products_clean.to_csv(proc_dir / "products.csv", index=False)

    logger.info("Cleaning calendar...")
    calendar_clean, cal_stats = clean_calendar(calendar_raw)
    calendar_clean.to_csv(proc_dir / "calendar.csv", index=False)

    logger.info("Cleaning batches...")
    batches_clean, batch_stats = clean_batches(batches_raw)
    batches_clean.to_csv(proc_dir / "batches.csv", index=False)

    logger.info("Cleaning daily sales...")
    sales_clean, sales_stats = clean_daily_sales(sales_raw, products_clean)
    sales_clean.to_csv(proc_dir / "daily_sales.csv", index=False)

    summary = {
        "products": prod_stats,
        "calendar": cal_stats,
        "batches": batch_stats,
        "daily_sales": sales_stats,
    }

    summary_path = reports_dir / "data_cleaning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Data quality pipeline complete. Audit summary saved to {summary_path}")
    return summary


if __name__ == "__main__":
    run_pipeline()
