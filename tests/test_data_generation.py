"""Tests for synthetic data generation reproducibility and schemas."""

import pandas as pd
from src.data.generate_data import build_calendar, build_products, simulate_sales_and_batches


def test_calendar_generation():
    cal = build_calendar("2023-01-01", "2023-01-10")
    assert len(cal) == 10
    assert "season" in cal.columns
    assert "is_weekend" in cal.columns
    assert "is_holiday" in cal.columns
    assert cal.iloc[0]["date"] == "2023-01-01"


def test_products_catalog():
    prods = build_products()
    assert len(prods) == 60
    assert set(prods["category"].unique()) == {
        "analgesics", "antipyretics", "antacids", "vitamins_supplements",
        "cough_cold", "allergy", "first_aid", "diabetic_care", "baby_care", "skin_care"
    }
    assert (prods["base_price"] > prods["unit_cost"]).all()


def test_generator_determinism():
    """Ensure identical seed produces identical output."""
    cal = build_calendar("2023-01-01", "2023-01-31")
    prods = build_products().iloc[:5].copy()  # small subset for speed

    sales1, batches1 = simulate_sales_and_batches(prods, cal, seed=42)
    sales2, batches2 = simulate_sales_and_batches(prods, cal, seed=42)

    pd.testing.assert_frame_equal(sales1, sales2)
    pd.testing.assert_frame_equal(batches1, batches2)
