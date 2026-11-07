"""Unit tests for chronological split integrity and gap verification."""

import pandas as pd
import pytest
from src.features.split import validate_chronological_splits, split_demand_data, split_expiry_risk_data


def test_validate_chronological_splits_valid():
    train = pd.DataFrame({"date": ["2023-01-01", "2023-03-01"]})
    val = pd.DataFrame({"date": ["2023-03-30", "2023-05-01"]})  # 28-day gap (Mar 2 to Mar 29 = 28 days)
    test = pd.DataFrame({"date": ["2023-05-30", "2023-07-01"]})

    # Should pass without assertion error
    validate_chronological_splits(train, val, test, date_col="date", gap_train_val=28, gap_val_test=28)


def test_validate_chronological_splits_detects_overlap():
    train = pd.DataFrame({"date": ["2023-01-01", "2023-03-15"]})
    val = pd.DataFrame({"date": ["2023-03-10", "2023-05-01"]})  # Overlap!
    test = pd.DataFrame({"date": ["2023-05-30", "2023-07-01"]})

    with pytest.raises(AssertionError, match="overlap"):
        validate_chronological_splits(train, val, test, date_col="date", gap_train_val=28, gap_val_test=28)


def test_validate_chronological_splits_detects_insufficient_gap():
    train = pd.DataFrame({"date": ["2023-01-01", "2023-03-01"]})
    val = pd.DataFrame({"date": ["2023-03-10", "2023-05-01"]})  # Only 8 days gap
    test = pd.DataFrame({"date": ["2023-05-30", "2023-07-01"]})

    with pytest.raises(AssertionError, match="less than required horizon"):
        validate_chronological_splits(train, val, test, date_col="date", gap_train_val=28, gap_val_test=28)
