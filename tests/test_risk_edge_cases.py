"""Edge case tests for the FEFO batch expiry-risk rule and queue allocation."""

import pytest
import numpy as np


def compute_fefo_risk_rule(
    remaining_qty: float,
    days_to_expiry: int,
    daily_velocity: float,
    earlier_qty: float = 0.0,
    safety_factor: float = 1.15,
) -> dict:
    """Helper implementing the exact FEFO domain risk rule defined in config.yaml."""
    if days_to_expiry <= 0:
        return {
            "allocation_share": 0.0,
            "expected_demand": 0.0,
            "threshold_capacity": 0.0,
            "risk_rule_flag": 1 if remaining_qty > 0 else 0,
            "risk_rule_gap": remaining_qty,
            "is_expired": True,
        }

    expected_demand = daily_velocity * float(days_to_expiry)

    if expected_demand > 0:
        demand_remaining_for_batch = max(0.0, expected_demand - earlier_qty)
        allocation_share = min(1.0, demand_remaining_for_batch / expected_demand)
    else:
        allocation_share = 0.0

    threshold_capacity = expected_demand * allocation_share * safety_factor
    risk_rule_flag = 1 if remaining_qty > threshold_capacity else 0
    risk_rule_gap = remaining_qty - threshold_capacity

    return {
        "allocation_share": round(allocation_share, 4),
        "expected_demand": round(expected_demand, 2),
        "threshold_capacity": round(threshold_capacity, 2),
        "risk_rule_flag": risk_rule_flag,
        "risk_rule_gap": round(risk_rule_gap, 2),
        "is_expired": False,
    }


def test_risk_rule_zero_demand():
    """Edge Case 1: When product demand velocity is zero, any positive stock is at risk."""
    res = compute_fefo_risk_rule(
        remaining_qty=50.0,
        days_to_expiry=30,
        daily_velocity=0.0,
        earlier_qty=0.0,
        safety_factor=1.15,
    )
    assert res["allocation_share"] == 0.0
    assert res["expected_demand"] == 0.0
    assert res["threshold_capacity"] == 0.0
    assert res["risk_rule_flag"] == 1
    assert res["risk_rule_gap"] == 50.0


def test_risk_rule_expired_batch():
    """Edge Case 2: When days to expiry <= 0, batch is expired and flagged immediately."""
    res_zero = compute_fefo_risk_rule(
        remaining_qty=20.0,
        days_to_expiry=0,
        daily_velocity=2.5,
    )
    assert res_zero["is_expired"] is True
    assert res_zero["risk_rule_flag"] == 1

    res_negative = compute_fefo_risk_rule(
        remaining_qty=15.0,
        days_to_expiry=-10,
        daily_velocity=3.0,
    )
    assert res_negative["is_expired"] is True
    assert res_negative["risk_rule_flag"] == 1


def test_risk_rule_single_batch_gets_full_allocation():
    """Edge Case 3: A single batch with no earlier batches gets 100% allocation share."""
    res = compute_fefo_risk_rule(
        remaining_qty=60.0,
        days_to_expiry=30,
        daily_velocity=2.5,  # expected demand = 75
        earlier_qty=0.0,     # single batch
        safety_factor=1.15,  # capacity = 75 * 1.0 * 1.15 = 86.25
    )
    assert res["allocation_share"] == 1.0
    assert res["expected_demand"] == 75.0
    assert res["threshold_capacity"] == 86.25
    assert res["risk_rule_flag"] == 0  # 60 <= 86.25 -> Safe


def test_risk_rule_zero_remaining_stock():
    """Edge Case 4: When remaining quantity is 0, batch should not flag as risk."""
    res = compute_fefo_risk_rule(
        remaining_qty=0.0,
        days_to_expiry=45,
        daily_velocity=1.0,
        earlier_qty=0.0,
    )
    assert res["risk_rule_flag"] == 0
    assert res["risk_rule_gap"] <= 0.0


def test_risk_rule_overwhelming_demand_clears_batch():
    """Edge Case 5: When sales velocity far exceeds remaining stock, batch is completely safe."""
    res = compute_fefo_risk_rule(
        remaining_qty=20.0,
        days_to_expiry=60,
        daily_velocity=10.0,  # expected demand = 600
        earlier_qty=50.0,
        safety_factor=1.15,
    )
    assert res["risk_rule_flag"] == 0
    assert res["allocation_share"] > 0.9


def test_risk_rule_queue_blocked_by_earlier_batches():
    """Edge Case 6: When earlier queue batches exceed expected demand, allocation share is 0."""
    res = compute_fefo_risk_rule(
        remaining_qty=30.0,
        days_to_expiry=20,
        daily_velocity=1.0,   # expected demand = 20
        earlier_qty=50.0,     # 50 units already ahead in FEFO queue!
        safety_factor=1.15,
    )
    assert res["allocation_share"] == 0.0
    assert res["threshold_capacity"] == 0.0
    assert res["risk_rule_flag"] == 1
    assert res["risk_rule_gap"] == 30.0


def test_risk_rule_safety_factor_scaling():
    """Edge Case 7: Safety factor expands allowable capacity buffer linearly."""
    base = compute_fefo_risk_rule(50.0, 30, 2.0, earlier_qty=0.0, safety_factor=1.0)
    buffered = compute_fefo_risk_rule(50.0, 30, 2.0, earlier_qty=0.0, safety_factor=1.2)
    assert buffered["threshold_capacity"] == pytest.approx(base["threshold_capacity"] * 1.2, rel=1e-3)
