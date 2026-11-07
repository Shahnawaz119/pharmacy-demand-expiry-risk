"""FastAPI Inference Service for Pharmacy Demand & Expiry-Risk Forecasting.

Advisory Decision-Support API providing:
- GET  /health: System and model artifact health check
- POST /predict/demand: Product-level 28-day demand forecast with p10/p50/p90 uncertainty intervals
- POST /predict/expiry-risk: Batch-level FEFO expiry-risk score, operational reason, and action
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("pharmacy_api")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "pharmacy_forecaster.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
PRODUCTS_PATH = BASE_DIR / "data" / "processed" / "products.csv"
DEMAND_FEATURES_PATH = BASE_DIR / "data" / "processed" / "demand_features.csv"
EXPIRY_FEATURES_PATH = BASE_DIR / "data" / "processed" / "expiry_risk_features.csv"
BATCHES_PATH = BASE_DIR / "data" / "processed" / "batches.csv"

DISCLAIMER_TEXT = (
    "Advisory tool. Predictions are estimates from synthetic-data training "
    "and must be reviewed by a pharmacist. This is not medical or treatment advice."
)


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_timestamp: str
    random_seed: int
    demand_features_count: int
    expiry_features_count: int
    disclaimer: str


class DemandPredictionRequest(BaseModel):
    product_id: str = Field(..., description="Unique product identifier (e.g. PROD_001)")
    as_of_date: Optional[str] = Field(None, description="Forecast cutoff date in YYYY-MM-DD format. Defaults to latest available.")
    features: Optional[Dict[str, float]] = Field(None, description="Optional custom feature overrides")


class DemandPredictionResponse(BaseModel):
    product_id: str
    product_name: str
    category: str
    as_of_date: str
    horizon_days: int = 28
    forecast_28d_p50: float = Field(..., description="Median point forecast for forward 28 days")
    forecast_28d_p10: float = Field(..., description="10th percentile forecast (lower uncertainty bound)")
    forecast_28d_p90: float = Field(..., description="90th percentile forecast (upper uncertainty bound)")
    baseline_ma28: Optional[float] = Field(None, description="28-day moving average benchmark")
    current_stock_on_hand: float
    stock_cover_days: float
    reorder_guidance: str
    disclaimer: str


class ExpiryRiskRequest(BaseModel):
    batch_id: Optional[str] = Field(None, description="Existing batch identifier (e.g. BAT_00024)")
    snapshot_date: Optional[str] = Field(None, description="Snapshot date in YYYY-MM-DD format")
    product_id: Optional[str] = Field(None, description="Product identifier if batch details are supplied directly")
    remaining_qty: Optional[float] = Field(None, ge=0.0, description="Current stock units remaining in batch")
    days_to_expiry: Optional[int] = Field(None, description="Days remaining until official expiration")
    unit_cost: Optional[float] = Field(None, ge=0.0, description="Cost per unit in INR")
    quantity_received: Optional[float] = Field(None, ge=0.0, description="Original quantity received")
    batch_age_days: Optional[int] = Field(None, ge=0, description="Days elapsed since batch was received")
    earlier_batches_qty: Optional[float] = Field(None, ge=0.0, description="Stock in queue ahead of this batch under FEFO")
    fefo_queue_position: Optional[int] = Field(None, ge=1, description="Queue position in FEFO sequence")
    product_daily_velocity: Optional[float] = Field(None, ge=0.0, description="Recent mean daily sales velocity")


class ExpiryRiskResponse(BaseModel):
    batch_id: Optional[str]
    product_id: str
    product_name: str
    days_to_expiry: int
    remaining_qty: float
    unit_cost: float
    potential_loss_value: float
    risk_flag: bool = Field(..., description="True if predicted risk exceeds cost-tuned decision threshold")
    risk_score: float = Field(..., description="Predicted probability of expiring unsold (0.0 to 1.0)")
    decision_threshold: float = Field(..., description="Optimal operational threshold (e.g. 0.93)")
    heuristic_rule_flag: bool = Field(..., description="Risk flag according to configurable domain rule")
    reason: str = Field(..., description="Transparent operational explanation of the risk rating")
    suggested_action: str = Field(..., description="Operational action: reorder_review, stock_rotation, discount, supplier_return, or monitor")
    disclaimer: str


# ==============================================================================
# Model & Data Registry
# ==============================================================================

class ModelRegistry:
    def __init__(self) -> None:
        self.bundle: Optional[Dict[str, Any]] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.demand_df: Optional[pd.DataFrame] = None
        self.expiry_df: Optional[pd.DataFrame] = None
        self.batches_df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. Run 'python -m src.train' first."
            )
        logger.info("Loading production model artifact from %s...", MODEL_PATH)
        self.bundle = joblib.load(MODEL_PATH)

        if PRODUCTS_PATH.exists():
            self.products_df = pd.read_csv(PRODUCTS_PATH).set_index("product_id")
        if DEMAND_FEATURES_PATH.exists():
            self.demand_df = pd.read_csv(DEMAND_FEATURES_PATH)
            self.demand_df["date"] = pd.to_datetime(self.demand_df["date"])
        if EXPIRY_FEATURES_PATH.exists():
            self.expiry_df = pd.read_csv(EXPIRY_FEATURES_PATH)
            self.expiry_df["snapshot_date"] = pd.to_datetime(self.expiry_df["snapshot_date"])
        if BATCHES_PATH.exists():
            self.batches_df = pd.read_csv(BATCHES_PATH).set_index("batch_id")

        logger.info("Model artifact and supporting feature tables loaded successfully.")


registry = ModelRegistry()


# ==============================================================================
# FastAPI Application Definition
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        registry.load()
    except Exception as exc:
        logger.error("Failed to load model registry on startup: %s", exc)
    yield


app = FastAPI(
    title="Pharmacy Medicine Demand & Expiry-Risk Forecasting API",
    description=(
        "Advisory decision-support REST API for retail pharmacy inventory management. "
        "Forecasts 28-day product demand with uncertainty intervals and flags batches "
        "at risk of expiration under FEFO rules. Always requires pharmacist review."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return system status, model artifact readiness, and configuration metadata."""
    if registry.bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Ensure 'python -m src.train' has been run.",
        )

    return HealthResponse(
        status="healthy",
        version=registry.bundle.get("version", "0.1.0"),
        model_loaded=True,
        model_timestamp=registry.bundle.get("created_at", "unknown"),
        random_seed=registry.bundle.get("random_seed", 42),
        demand_features_count=len(registry.bundle["demand"]["feature_columns"]),
        expiry_features_count=len(registry.bundle["expiry_risk"]["feature_columns"]),
        disclaimer=DISCLAIMER_TEXT,
    )


@app.post("/predict/demand", response_model=DemandPredictionResponse, tags=["Forecasting"])
def predict_demand(request: DemandPredictionRequest) -> DemandPredictionResponse:
    """Generate forward 28-day aggregate demand forecast with p10, p50, and p90 intervals."""
    if registry.bundle is None or registry.demand_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact or feature tables unavailable.",
        )

    product_id = request.product_id.strip().upper()
    if registry.products_df is not None and product_id not in registry.products_df.index:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found in pharmacy catalog. Valid IDs: PROD_001 to PROD_060.",
        )

    # Determine as_of_date
    prod_data = registry.demand_df[registry.demand_df["product_id"] == product_id]
    if prod_data.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sales history found for product '{product_id}'.",
        )

    if request.as_of_date:
        try:
            target_date = pd.to_datetime(request.as_of_date)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid date format for as_of_date: '{request.as_of_date}'. Expected YYYY-MM-DD.",
            )
        matched_rows = prod_data[prod_data["date"] == target_date]
        if matched_rows.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No record found for product '{product_id}' on {request.as_of_date}.",
            )
        feature_row = matched_rows.iloc[-1].to_dict()
        as_of_str = target_date.strftime("%Y-%m-%d")
    else:
        # Default to latest available date
        feature_row = prod_data.iloc[-1].to_dict()
        as_of_str = pd.to_datetime(feature_row["date"]).strftime("%Y-%m-%d")

    # Apply manual overrides if provided
    if request.features:
        feature_row.update(request.features)

    # Prepare feature vector
    feature_cols = registry.bundle["demand"]["feature_columns"]
    X = pd.DataFrame([{col: feature_row.get(col, 0.0) for col in feature_cols}])

    # Predict using saved LightGBM quantile models
    models = registry.bundle["demand"]["primary_model"]
    p10_pred = float(models[0.1].predict(X)[0])
    p50_pred = float(models[0.5].predict(X)[0])
    p90_pred = float(models[0.9].predict(X)[0])

    # Enforce non-negativity and monotone quantile constraints
    p10_val = max(0.0, p10_pred)
    p50_val = max(p10_val, p50_pred)
    p90_val = max(p50_val, p90_pred)

    # Compute operational context
    stock_on_hand = float(feature_row.get("stock_on_hand", 0.0))
    daily_velocity = max(0.1, float(feature_row.get("rolling_mean_7", 1.0)))
    stock_cover_days = stock_on_hand / daily_velocity
    baseline_ma28 = float(feature_row.get("rolling_mean_28", daily_velocity)) * 28.0

    # Operational reorder guidance logic
    if stock_cover_days < 7.0:
        reorder_guidance = (
            f"URGENT REORDER REVIEW: Critical stock cover ({stock_cover_days:.1f} days) "
            f"is below weekly replenishment cycle. Expedite purchase order."
        )
    elif stock_cover_days < 21.0:
        suggested_qty = max(0.0, round(p50_val * 1.10 - stock_on_hand, 0))
        reorder_guidance = (
            f"STANDARD REORDER REVIEW: Stock cover ({stock_cover_days:.1f} days) is moderate. "
            f"Suggested replenishment: ~{suggested_qty:.0f} units to cover 28-day horizon with safety buffer."
        )
    elif stock_cover_days > 60.0:
        reorder_guidance = (
            f"OVERSTOCK ALERT: Stock cover ({stock_cover_days:.1f} days) exceeds 2 months. "
            f"Suspend pending orders and monitor for slow-moving expiry risk."
        )
    else:
        reorder_guidance = (
            f"ADEQUATE STOCK: Current inventory ({stock_cover_days:.1f} days cover) "
            f"satisfies projected 28-day demand ({p50_val:.1f} units)."
        )

    product_name = (
        registry.products_df.loc[product_id, "product_name"]
        if registry.products_df is not None and product_id in registry.products_df.index
        else product_id
    )
    category = (
        registry.products_df.loc[product_id, "category"]
        if registry.products_df is not None and product_id in registry.products_df.index
        else str(feature_row.get("category", "general"))
    )

    return DemandPredictionResponse(
        product_id=product_id,
        product_name=str(product_name),
        category=str(category),
        as_of_date=as_of_str,
        horizon_days=28,
        forecast_28d_p50=round(p50_val, 2),
        forecast_28d_p10=round(p10_val, 2),
        forecast_28d_p90=round(p90_val, 2),
        baseline_ma28=round(baseline_ma28, 2),
        current_stock_on_hand=round(stock_on_hand, 2),
        stock_cover_days=round(stock_cover_days, 1),
        reorder_guidance=reorder_guidance,
        disclaimer=DISCLAIMER_TEXT,
    )


@app.post("/predict/expiry-risk", response_model=ExpiryRiskResponse, tags=["Expiry Risk"])
def predict_expiry_risk(request: ExpiryRiskRequest) -> ExpiryRiskResponse:
    """Predict expiry risk for a batch under FEFO rules using cost-tuned LightGBM classifier."""
    if registry.bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded.",
        )

    bundle_exp = registry.bundle["expiry_risk"]
    clf = bundle_exp["primary_model"]
    optimal_threshold = float(bundle_exp["optimal_threshold"])
    safety_factor = float(bundle_exp.get("safety_factor", 1.15))
    feature_cols = bundle_exp["feature_columns"]

    # Resolution path 1: batch_id lookup
    if request.batch_id:
        batch_id = request.batch_id.strip().upper()
        matched = pd.DataFrame()
        if registry.expiry_df is not None:
            if request.snapshot_date:
                matched = registry.expiry_df[
                    (registry.expiry_df["batch_id"] == batch_id)
                    & (registry.expiry_df["snapshot_date"] == pd.to_datetime(request.snapshot_date))
                ]
            else:
                matched = registry.expiry_df[registry.expiry_df["batch_id"] == batch_id]

        if not matched.empty:
            feature_row = matched.iloc[-1].to_dict()
            product_id = str(feature_row["product_id"])
            days_to_expiry = int(feature_row["days_to_expiry"])
            remaining_qty = float(feature_row["remaining_qty"])
            unit_cost = float(feature_row["unit_cost"])
        elif registry.batches_df is not None and batch_id in registry.batches_df.index:
            batch_meta = registry.batches_df.loc[batch_id]
            product_id = str(batch_meta["product_id"])
            unit_cost = float(batch_meta["unit_cost"])
            exp_date = pd.to_datetime(batch_meta["expiry_date"])
            ref_date = pd.to_datetime(request.snapshot_date) if request.snapshot_date else pd.to_datetime("2024-12-03")
            days_to_expiry = int((exp_date - ref_date).days)
            remaining_qty = float(request.remaining_qty or batch_meta["quantity_received"])
            feature_row = {
                "days_to_expiry": days_to_expiry,
                "remaining_qty": remaining_qty,
                "unit_cost": unit_cost,
                "quantity_received": float(batch_meta["quantity_received"]),
                "batch_age_days": max(0, int((ref_date - pd.to_datetime(batch_meta["received_date"])).days)),
                "fraction_depleted": 0.0,
                "earlier_batches_qty": float(request.earlier_batches_qty or 0.0),
                "fefo_queue_position": int(request.fefo_queue_position or 1),
                "product_daily_velocity": float(request.product_daily_velocity or 2.0),
                "expected_demand_to_expiry": float(request.product_daily_velocity or 2.0) * max(0, days_to_expiry),
                "allocation_share": 1.0,
                "stock_cover_days": max(0, days_to_expiry),
                "risk_rule_flag": 0,
                "risk_rule_gap": 0.0,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch '{batch_id}' not found in historical records.",
            )
    else:
        # Resolution path 2: direct batch details
        if request.product_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Must provide either 'batch_id' or 'product_id' with batch attributes.",
            )
        if request.days_to_expiry is None or request.remaining_qty is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="When specifying custom batch details, 'days_to_expiry' and 'remaining_qty' are required.",
            )
        batch_id = None
        product_id = request.product_id.strip().upper()
        days_to_expiry = int(request.days_to_expiry)
        remaining_qty = float(request.remaining_qty)
        unit_cost = float(request.unit_cost or 50.0)
        qty_received = float(request.quantity_received or remaining_qty)
        velocity = float(request.product_daily_velocity or 2.5)
        earlier_qty = float(request.earlier_batches_qty or 0.0)
        expected_demand = velocity * max(0, days_to_expiry)
        alloc_share = max(0.1, min(1.0, days_to_expiry / (days_to_expiry + 10)))

        # Rule evaluation
        rule_needed = expected_demand * alloc_share * safety_factor
        risk_rule_flag = 1 if remaining_qty > rule_needed else 0
        risk_rule_gap = remaining_qty - rule_needed

        feature_row = {
            "days_to_expiry": days_to_expiry,
            "remaining_qty": remaining_qty,
            "unit_cost": unit_cost,
            "quantity_received": qty_received,
            "batch_age_days": int(request.batch_age_days or 30),
            "fraction_depleted": max(0.0, min(1.0, 1.0 - (remaining_qty / max(1.0, qty_received)))),
            "earlier_batches_qty": earlier_qty,
            "fefo_queue_position": int(request.fefo_queue_position or 1),
            "product_daily_velocity": velocity,
            "expected_demand_to_expiry": expected_demand,
            "allocation_share": alloc_share,
            "risk_rule_flag": risk_rule_flag,
            "risk_rule_gap": risk_rule_gap,
            "stock_cover_days": remaining_qty / max(0.1, velocity),
        }

    # Immediate guard: already expired
    if days_to_expiry <= 0:
        return ExpiryRiskResponse(
            batch_id=batch_id,
            product_id=product_id,
            product_name=str(registry.products_df.loc[product_id, "product_name"]) if (registry.products_df is not None and product_id in registry.products_df.index) else product_id,
            days_to_expiry=days_to_expiry,
            remaining_qty=remaining_qty,
            unit_cost=unit_cost,
            potential_loss_value=round(remaining_qty * unit_cost, 2),
            risk_flag=True,
            risk_score=1.0,
            decision_threshold=optimal_threshold,
            heuristic_rule_flag=True,
            reason="CRITICAL: Batch has already passed its expiry date. Immediate quarantine required.",
            suggested_action="supplier_return",
            disclaimer=DISCLAIMER_TEXT,
        )

    # Feature vector
    X = pd.DataFrame([{col: feature_row.get(col, 0.0) for col in feature_cols}])

    # Predict probability
    prob = float(clf.predict_proba(X)[0, 1])
    is_at_risk = bool(prob >= optimal_threshold)
    rule_flag = bool(feature_row.get("risk_rule_flag", 0) == 1)

    # Financial exposure
    loss_val = round(remaining_qty * unit_cost, 2)
    earlier_q = float(feature_row.get("earlier_batches_qty", 0.0))
    exp_demand = float(feature_row.get("expected_demand_to_expiry", 10.0))

    # Formulate transparent operational explanation
    if is_at_risk:
        reason = (
            f"HIGH RISK of unsold expiration ({prob*100:.1f}% probability vs {optimal_threshold*100:.1f}% threshold). "
            f"Remaining {remaining_qty:.0f} units with {days_to_expiry} days to expiry. "
            f"With {earlier_q:.0f} units ahead in FEFO queue and expected demand of {exp_demand:.0f} units, "
            f"estimated ~{max(0.0, remaining_qty - exp_demand):.0f} units risk being unsold."
        )
    else:
        reason = (
            f"LOW RISK of unsold expiration ({prob*100:.1f}% probability below {optimal_threshold*100:.1f}% threshold). "
            f"Expected sales of {exp_demand:.0f} units comfortably absorb the remaining {remaining_qty:.0f} units "
            f"within the next {days_to_expiry} days under standard FEFO dispensing."
        )

    # Formulate strictly operational suggested action
    if days_to_expiry <= 20 and is_at_risk:
        suggested_action = "supplier_return"
    elif days_to_expiry <= 45 and is_at_risk:
        suggested_action = "discount"
    elif feature_row.get("fefo_queue_position", 1) > 1 and is_at_risk:
        suggested_action = "stock_rotation"
    elif is_at_risk:
        suggested_action = "reorder_review"
    else:
        suggested_action = "monitor"

    product_name = (
        registry.products_df.loc[product_id, "product_name"]
        if registry.products_df is not None and product_id in registry.products_df.index
        else product_id
    )

    return ExpiryRiskResponse(
        batch_id=batch_id,
        product_id=product_id,
        product_name=str(product_name),
        days_to_expiry=days_to_expiry,
        remaining_qty=round(remaining_qty, 2),
        unit_cost=round(unit_cost, 2),
        potential_loss_value=loss_val,
        risk_flag=is_at_risk,
        risk_score=round(prob, 4),
        decision_threshold=optimal_threshold,
        heuristic_rule_flag=rule_flag,
        reason=reason,
        suggested_action=suggested_action,
        disclaimer=DISCLAIMER_TEXT,
    )
