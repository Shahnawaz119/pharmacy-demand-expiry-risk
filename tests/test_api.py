"""Integration and unit tests for the FastAPI inference service."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, registry


@pytest.fixture(scope="module")
def client():
    """Create test client and ensure model registry is initialized."""
    registry.load()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health returns 200, valid schema, and model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["demand_features_count"] == 32
    assert data["expiry_features_count"] == 14
    assert "Advisory tool" in data["disclaimer"]


def test_predict_demand_valid_product(client):
    """Verify /predict/demand returns monotonic quantile predictions and context."""
    payload = {"product_id": "PROD_001"}
    response = client.post("/predict/demand", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["product_id"] == "PROD_001"
    assert "product_name" in data
    assert data["horizon_days"] == 28
    assert data["forecast_28d_p10"] >= 0.0
    assert data["forecast_28d_p10"] <= data["forecast_28d_p50"]
    assert data["forecast_28d_p50"] <= data["forecast_28d_p90"]
    assert "reorder_guidance" in data
    assert "Advisory tool" in data["disclaimer"]


def test_predict_demand_with_date(client):
    """Verify /predict/demand with specific historical as_of_date."""
    payload = {"product_id": "PROD_005", "as_of_date": "2024-05-01"}
    response = client.post("/predict/demand", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["product_id"] == "PROD_005"
    assert data["as_of_date"] == "2024-05-01"


def test_predict_demand_invalid_product_returns_404(client):
    """Verify /predict/demand rejects unknown product ID with 404."""
    payload = {"product_id": "PROD_9999"}
    response = client.post("/predict/demand", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_predict_demand_invalid_date_returns_422(client):
    """Verify /predict/demand rejects invalid date format with 422."""
    payload = {"product_id": "PROD_001", "as_of_date": "invalid-date-string"}
    response = client.post("/predict/demand", json=payload)
    assert response.status_code == 422


def test_predict_expiry_risk_by_batch_id(client):
    """Verify /predict/expiry-risk looks up existing batch."""
    payload = {"batch_id": "BAT_00024"}
    response = client.predict_risk = client.post("/predict/expiry-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_id"] == "BAT_00024"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert isinstance(data["risk_flag"], bool)
    assert data["suggested_action"] in {
        "reorder_review",
        "stock_rotation",
        "discount",
        "supplier_return",
        "monitor",
    }
    assert "Advisory tool" in data["disclaimer"]


def test_predict_expiry_risk_custom_attributes(client):
    """Verify /predict/expiry-risk with direct batch features."""
    payload = {
        "product_id": "PROD_002",
        "days_to_expiry": 25,
        "remaining_qty": 180.0,
        "unit_cost": 45.0,
        "quantity_received": 200.0,
        "product_daily_velocity": 1.2,
        "earlier_batches_qty": 50.0,
        "fefo_queue_position": 2,
    }
    response = client.post("/predict/expiry-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["product_id"] == "PROD_002"
    assert data["days_to_expiry"] == 25
    assert data["risk_flag"] is True
    assert data["suggested_action"] in {"discount", "stock_rotation", "supplier_return", "reorder_review"}


def test_predict_expiry_risk_already_expired(client):
    """Verify /predict/expiry-risk flags already-expired batch immediately."""
    payload = {
        "product_id": "PROD_002",
        "days_to_expiry": -5,
        "remaining_qty": 30.0,
        "unit_cost": 40.0,
    }
    response = client.post("/predict/expiry-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_flag"] is True
    assert data["risk_score"] == 1.0
    assert data["suggested_action"] == "supplier_return"
    assert "passed its expiry date" in data["reason"].lower()


def test_predict_expiry_risk_missing_required_fails(client):
    """Verify /predict/expiry-risk rejects request with missing fields."""
    payload = {"product_id": "PROD_002"}  # missing days_to_expiry and remaining_qty
    response = client.post("/predict/expiry-risk", json=payload)
    assert response.status_code == 422


def test_determinism_same_input_same_output(client):
    """Verify API produces strictly deterministic results across repeated calls."""
    payload = {"product_id": "PROD_003"}
    resp1 = client.post("/predict/demand", json=payload).json()
    resp2 = client.post("/predict/demand", json=payload).json()

    assert resp1["forecast_28d_p50"] == resp2["forecast_28d_p50"]
    assert resp1["forecast_28d_p10"] == resp2["forecast_28d_p10"]
    assert resp1["forecast_28d_p90"] == resp2["forecast_28d_p90"]
