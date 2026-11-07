"""Expiry-risk classification models, heuristic rules, and cost-aware threshold tuning.

Implements:
1. Rule-based heuristic decision flag
2. Regularized Logistic Regression with class-weight balancing
3. Gradient Boosted Trees (LightGBM Classifier)
4. Cost-sensitive threshold optimization on validation data
"""

import logging
from typing import Dict, Any, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RISK_FEATURE_COLS = [
    "days_to_expiry",
    "remaining_qty",
    "unit_cost",
    "quantity_received",
    "batch_age_days",
    "fraction_depleted",
    "earlier_batches_qty",
    "fefo_queue_position",
    "product_daily_velocity",
    "expected_demand_to_expiry",
    "allocation_share",
    "stock_cover_days",
    "risk_rule_flag",
    "risk_rule_gap",
]


class RuleBasedExpiryModel:
    """Uses the explicit FEFO runout rule directly as an advisory flag."""
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["risk_rule_flag"].values.astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        flags = X["risk_rule_flag"].values.astype(float)
        # Approximate pseudo-probability: 0.85 if flag=1, 0.15 if flag=0
        probs = np.where(flags == 1, 0.85, 0.15)
        return np.column_stack([1.0 - probs, probs])


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Train balanced L2-regularized logistic regression."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            penalty="l2",
            class_weight="balanced",
            max_iter=500,
            random_state=random_state,
        )),
    ])
    pipe.fit(X_train[RISK_FEATURE_COLS], y_train)
    return pipe


def train_lightgbm_risk_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """Train gradient-boosted tree classifier for batch expiry risk."""
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=15,
        class_weight="balanced",
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(
        X_train[RISK_FEATURE_COLS],
        y_train,
        eval_set=[(X_val[RISK_FEATURE_COLS], y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )
    return model


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> Tuple[float, float]:
    """Search for decision threshold that minimizes total asymmetric business cost.
    
    Total Cost = cost_fn * FN + cost_fp * FP.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.50
    best_cost = float("inf")

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        cost = (cost_fn * fn) + (cost_fp * fp)
        if cost < best_cost:
            best_cost = cost
            best_thresh = float(t)

    return round(best_thresh, 3), round(best_cost, 2)


def evaluate_risk_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> Dict[str, Any]:
    """Calculate classification metrics and business loss."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    pr_auc = 0.0
    if y_prob is not None:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_curve, p_curve))

    business_cost = float((cost_fn * fn) + (cost_fp * fp))

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "business_cost": round(business_cost, 2),
    }
