"""Tests for data cleaning and quality validation rules."""

import pandas as pd
from src.data.clean_data import clean_products, clean_batches, clean_daily_sales, clean_calendar


def test_clean_products():
    raw_df = pd.DataFrame([
        {"product_id": "P01", "name": "A"},
        {"product_id": "P01", "name": "A_duplicate"},
        {"product_id": "P02", "name": "B"},
    ])
    cleaned, stats = clean_products(raw_df)
    assert len(cleaned) == 2
    assert stats["duplicates_removed"] == 1


def test_clean_batches_invalid_expiry():
    raw_batches = pd.DataFrame([
        {"batch_id": "B01", "received_date": "2023-01-01", "expiry_date": "2024-01-01"},
        {"batch_id": "B02", "received_date": "2023-05-01", "expiry_date": "2023-04-01"},  # invalid
        {"batch_id": "B03", "received_date": "2023-06-01", "expiry_date": "2023-06-01"},  # invalid
    ])
    cleaned, stats = clean_batches(raw_batches)
    assert len(cleaned) == 1
    assert stats["invalid_expiry_dropped"] == 2
    assert cleaned.iloc[0]["batch_id"] == "B01"


def test_clean_daily_sales_rules():
    prods = pd.DataFrame([
        {"product_id": "P01", "base_price": 50.0},
    ])
    raw_sales = pd.DataFrame([
        {"date": "2023-01-01", "product_id": "P01", "units_sold": 10.0, "selling_price": 45.0},
        {"date": "2023-01-01", "product_id": "P01", "units_sold": 10.0, "selling_price": 45.0},  # duplicate
        {"date": "2023-01-02", "product_id": "P01", "units_sold": -5.0, "selling_price": 45.0},  # negative
        {"date": "2023-01-03", "product_id": "P01", "units_sold": None, "selling_price": None},  # missing
        {"date": "2023-01-04", "product_id": "P01", "units_sold": 500.0, "selling_price": 45.0},  # outlier
    ])
    cleaned, stats = clean_daily_sales(raw_sales, prods)
    
    assert stats["duplicates_removed"] == 1
    assert stats["negative_sales_rectified"] == 1
    assert stats["missing_price_imputed"] == 1
    assert stats["missing_sales_imputed"] == 1
    assert (cleaned["units_sold"] >= 0).all()
    assert not cleaned["selling_price"].isna().any()
