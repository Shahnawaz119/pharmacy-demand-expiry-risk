"""Training orchestrator and experiment logging module.

Fits all demand and expiry-risk models on the training split, tunes decision thresholds
on the validation split, logs results to reports/experiments.csv, and persists model artifacts.
The test split is held completely untouched.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
import yaml

from src.models.demand_models import (
    MovingAverageBaseline,
    SeasonalNaiveBaseline,
    train_ridge_demand_model,
    train_lightgbm_quantile_models,
    evaluate_demand_predictions,
    evaluate_interval_coverage,
    DEMAND_FEATURE_COLS,
)
from src.models.expiry_models import (
    RuleBasedExpiryModel,
    train_logistic_regression,
    train_lightgbm_risk_classifier,
    find_cost_optimal_threshold,
    evaluate_risk_predictions,
    RISK_FEATURE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_experiment(csv_path: Path, entry: dict):
    """Append experiment run metadata and metrics to reports/experiments.csv."""
    df_new = pd.DataFrame([entry])
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)


def run_training_and_validation() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent.parent
    cfg = load_config(repo_root / "configs" / "config.yaml")

    proc_dir = repo_root / "data" / "processed"
    models_dir = repo_root / cfg["paths"]["models_dir"]
    reports_dir = repo_root / cfg["paths"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    experiments_csv = repo_root / cfg["paths"]["experiments_file"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {"timestamp": timestamp, "demand": {}, "expiry_risk": {}}

    # =========================================================================
    # 1. DEMAND FORECASTING MODELS
    # =========================================================================
    logger.info("Loading demand train and validation sets...")
    train_demand = pd.read_csv(proc_dir / "train_demand.csv")
    val_demand = pd.read_csv(proc_dir / "val_demand.csv")

    X_train_dem = train_demand[DEMAND_FEATURE_COLS]
    y_train_dem = train_demand["target_demand_28d"]
    X_val_dem = val_demand[DEMAND_FEATURE_COLS]
    y_val_dem = val_demand["target_demand_28d"]

    # Baseline 1A: Moving Average (7-day velocity projected)
    logger.info("Evaluating Baseline 1A: Moving Average (7-day)...")
    ma7 = MovingAverageBaseline(window=7, horizon=28)
    preds_ma7 = ma7.predict(val_demand)
    metrics_ma7 = evaluate_demand_predictions(y_val_dem, preds_ma7)
    results["demand"]["baseline_ma7"] = metrics_ma7
    log_experiment(experiments_csv, {
        "timestamp": timestamp, "task": "demand", "model": "Baseline_MA7",
        "params": "window=7,horizon=28", "split": "validation", **metrics_ma7
    })

    # Baseline 1B: Moving Average (28-day velocity projected)
    logger.info("Evaluating Baseline 1B: Moving Average (28-day)...")
    ma28 = MovingAverageBaseline(window=28, horizon=28)
    preds_ma28 = ma28.predict(val_demand)
    metrics_ma28 = evaluate_demand_predictions(y_val_dem, preds_ma28)
    results["demand"]["baseline_ma28"] = metrics_ma28
    log_experiment(experiments_csv, {
        "timestamp": timestamp, "task": "demand", "model": "Baseline_MA28",
        "params": "window=28,horizon=28", "split": "validation", **metrics_ma28
    })

    # Baseline 2: Seasonal Naive
    logger.info("Evaluating Baseline 2: Seasonal Naive...")
    snaive = SeasonalNaiveBaseline(horizon=28)
    preds_snaive = snaive.predict(val_demand)
    metrics_snaive = evaluate_demand_predictions(y_val_dem, preds_snaive)
    results["demand"]["baseline_seasonal_naive"] = metrics_snaive
    log_experiment(experiments_csv, {
        "timestamp": timestamp, "task": "demand", "model": "Baseline_SeasonalNaive",
        "params": "horizon=28", "split": "validation", **metrics_snaive
    })

    # Model A: Regularized Ridge Regression
    logger.info("Training Model A: Ridge Regression...")
    ridge_model = train_ridge_demand_model(train_demand, y_train_dem, alpha=10.0, random_state=cfg["project"]["random_seed"])
    preds_ridge = ridge_model.predict(X_val_dem)
    metrics_ridge = evaluate_demand_predictions(y_val_dem, preds_ridge)
    results["demand"]["model_a_ridge"] = metrics_ridge
    log_experiment(experiments_csv, {
        "timestamp": timestamp, "task": "demand", "model": "ModelA_Ridge",
        "params": "alpha=10.0,scaler=StandardScaler", "split": "validation", **metrics_ridge
    })
    joblib.dump(ridge_model, models_dir / "demand_ridge.joblib")

    # Model B: LightGBM Quantile Regressors (p10, p50, p90)
    logger.info("Training Model B: LightGBM Quantile Models (p10, p50, p90)...")
    quantiles = cfg["forecasting"]["quantiles"]
    lgb_models = train_lightgbm_quantile_models(
        train_demand, y_train_dem, val_demand, y_val_dem,
        quantiles=quantiles, random_state=cfg["project"]["random_seed"]
    )
    p10_preds = lgb_models[0.10].predict(X_val_dem)
    p50_preds = lgb_models[0.50].predict(X_val_dem)
    p90_preds = lgb_models[0.90].predict(X_val_dem)

    metrics_lgb_p50 = evaluate_demand_predictions(y_val_dem, p50_preds)
    interval_coverage = evaluate_interval_coverage(y_val_dem, p10_preds, p90_preds)
    metrics_lgb_p50.update(interval_coverage)
    results["demand"]["model_b_lightgbm_quantile"] = metrics_lgb_p50
    log_experiment(experiments_csv, {
        "timestamp": timestamp, "task": "demand", "model": "ModelB_LightGBM_p50",
        "params": "quantiles=[0.1,0.5,0.9],lr=0.04,num_leaves=31", "split": "validation", **metrics_lgb_p50
    })
    joblib.dump(lgb_models, models_dir / "demand_lightgbm_quantiles.joblib")

    # =========================================================================
    # 2. EXPIRY-RISK CLASSIFICATION MODELS
    # =========================================================================
    logger.info("Loading expiry risk train and validation sets...")
    train_risk = pd.read_csv(proc_dir / "train_risk.csv")
    val_risk = pd.read_csv(proc_dir / "val_risk.csv")

    X_train_risk = train_risk[RISK_FEATURE_COLS]
    y_train_risk = train_risk["will_expire_unsold"]
    X_val_risk = val_risk[RISK_FEATURE_COLS]
    y_val_risk = val_risk["will_expire_unsold"]

    cost_fn = cfg["expiry_risk"]["cost_fn"]
    cost_fp = cfg["expiry_risk"]["cost_fp"]

    # Baseline: Rule-based Heuristic Flag
    logger.info("Evaluating Rule-Based Expiry Baseline...")
    rule_model = RuleBasedExpiryModel()
    rule_preds = rule_model.predict(val_risk)
    rule_probs = rule_model.predict_proba(val_risk)[:, 1]
    metrics_rule = evaluate_risk_predictions(y_val_risk, rule_preds, rule_probs, cost_fn, cost_fp)
    results["expiry_risk"]["baseline_rule"] = metrics_rule
    log_experiment(experiments_csv, {
        "timestamp": timestamp, "task": "expiry_risk", "model": "Baseline_Rule",
        "params": "safety_factor=1.15,FEFO", "split": "validation", **metrics_rule
    })

    # Classifier A: Logistic Regression (balanced class weight)
    logger.info("Training Classifier A: Logistic Regression...")
    logreg = train_logistic_regression(train_risk, y_train_risk, C=1.0, random_state=cfg["project"]["random_seed"])
    lr_probs = logreg.predict_proba(X_val_risk[RISK_FEATURE_COLS])[:, 1]
    
    # Standard threshold 0.50
    metrics_lr_std = evaluate_risk_predictions(y_val_risk, (lr_probs >= 0.50).astype(int), lr_probs, cost_fn, cost_fp)
    # Cost-tuned threshold
    lr_opt_thresh, lr_opt_cost = find_cost_optimal_threshold(y_val_risk.values, lr_probs, cost_fn, cost_fp)
    metrics_lr_opt = evaluate_risk_predictions(y_val_risk, (lr_probs >= lr_opt_thresh).astype(int), lr_probs, cost_fn, cost_fp)
    metrics_lr_opt["optimal_threshold"] = lr_opt_thresh
    results["expiry_risk"]["classifier_logistic_regression"] = metrics_lr_opt
    log_experiment(experiments_csv, {
        "timestamp": timestamp, "task": "expiry_risk", "model": "Classifier_LogisticRegression",
        "params": f"C=1.0,thresh={lr_opt_thresh}", "split": "validation", **metrics_lr_opt
    })
    joblib.dump({"model": logreg, "threshold": lr_opt_thresh}, models_dir / "expiry_logreg.joblib")

    # Classifier B: LightGBM Risk Classifier
    logger.info("Training Classifier B: LightGBM Classifier...")
    lgb_clf = train_lightgbm_risk_classifier(
        train_risk, y_train_risk, val_risk, y_val_risk, random_state=cfg["project"]["random_seed"]
    )
    lgb_probs = lgb_clf.predict_proba(X_val_risk[RISK_FEATURE_COLS])[:, 1]
    
    # Cost-tuned threshold
    lgb_opt_thresh, lgb_opt_cost = find_cost_optimal_threshold(y_val_risk.values, lgb_probs, cost_fn, cost_fp)
    metrics_lgb_opt = evaluate_risk_predictions(y_val_risk, (lgb_probs >= lgb_opt_thresh).astype(int), lgb_probs, cost_fn, cost_fp)
    metrics_lgb_opt["optimal_threshold"] = lgb_opt_thresh
    results["expiry_risk"]["classifier_lightgbm"] = metrics_lgb_opt
    log_experiment(experiments_csv, {
        "timestamp": timestamp, "task": "expiry_risk", "model": "Classifier_LightGBM",
        "params": f"n_est=150,lr=0.03,thresh={lgb_opt_thresh}", "split": "validation", **metrics_lgb_opt
    })
    joblib.dump({"model": lgb_clf, "threshold": lgb_opt_thresh}, models_dir / "expiry_lightgbm.joblib")

    # Save validation results summary
    val_summary_path = reports_dir / "val_results.json"
    with open(val_summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Model training and validation tuning completed.")

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SET RESULTS (UNTRAINED TUNING SET)")
    print("=" * 80)
    print("DEMAND FORECASTING (28-DAY AGGREGATE):")
    print(f"  Baseline MA7:         MAE: {metrics_ma7['mae']:.2f} | RMSE: {metrics_ma7['rmse']:.2f} | WAPE: {metrics_ma7['wape']:.2f}% | Bias: {metrics_ma7['bias']:.2f}")
    print(f"  Baseline MA28:        MAE: {metrics_ma28['mae']:.2f} | RMSE: {metrics_ma28['rmse']:.2f} | WAPE: {metrics_ma28['wape']:.2f}% | Bias: {metrics_ma28['bias']:.2f}")
    print(f"  Baseline Seas. Naive: MAE: {metrics_snaive['mae']:.2f} | RMSE: {metrics_snaive['rmse']:.2f} | WAPE: {metrics_snaive['wape']:.2f}% | Bias: {metrics_snaive['bias']:.2f}")
    print(f"  Model A (Ridge):      MAE: {metrics_ridge['mae']:.2f} | RMSE: {metrics_ridge['rmse']:.2f} | WAPE: {metrics_ridge['wape']:.2f}% | Bias: {metrics_ridge['bias']:.2f}")
    print(f"  Model B (LightGBM):   MAE: {metrics_lgb_p50['mae']:.2f} | RMSE: {metrics_lgb_p50['rmse']:.2f} | WAPE: {metrics_lgb_p50['wape']:.2f}% | Bias: {metrics_lgb_p50['bias']:.2f}")
    print(f"                        Interval Coverage (p10-p90): {metrics_lgb_p50['coverage_pct']:.2f}% (Mean Width: {metrics_lgb_p50['mean_interval_width']:.2f} units)")
    print("-" * 80)
    print("EXPIRY-RISK CLASSIFICATION (90-DAY WINDOW):")
    print(f"  Baseline Rule:        Precision: {metrics_rule['precision']:.4f} | Recall: {metrics_rule['recall']:.4f} | F1: {metrics_rule['f1']:.4f} | PR-AUC: {metrics_rule['pr_auc']:.4f} | Cost: {metrics_rule['business_cost']}")
    print(f"  LogReg (Thresh={lr_opt_thresh}):   Precision: {metrics_lr_opt['precision']:.4f} | Recall: {metrics_lr_opt['recall']:.4f} | F1: {metrics_lr_opt['f1']:.4f} | PR-AUC: {metrics_lr_opt['pr_auc']:.4f} | Cost: {metrics_lr_opt['business_cost']}")
    print(f"  LightGBM (Thresh={lgb_opt_thresh}): Precision: {metrics_lgb_opt['precision']:.4f} | Recall: {metrics_lgb_opt['recall']:.4f} | F1: {metrics_lgb_opt['f1']:.4f} | PR-AUC: {metrics_lgb_opt['pr_auc']:.4f} | Cost: {metrics_lgb_opt['business_cost']}")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    run_training_and_validation()
