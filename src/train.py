"""Reproducible end-to-end training and artifact generation pipeline.

Executes:
1. Synthetic data verification / generation
2. Quality cleaning and anomaly remediation
3. Temporal feature engineering for demand and batch expiry
4. Chronological dataset partitioning with 28-day gap buffers
5. Baseline and candidate model fitting & validation tuning
6. Final evaluation on the untouched test set
7. Model artifact bundle serialization and metadata generation
"""

from datetime import datetime
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Any
import joblib
import pandas as pd
import yaml

from src.data.generate_data import main as generate_raw_data
from src.data.clean_data import run_pipeline as run_cleaning_pipeline
from src.features.build_features import run_feature_engineering
from src.features.split import run_split_pipeline
from src.models.demand_models import DEMAND_FEATURE_COLS
from src.models.expiry_models import RISK_FEATURE_COLS
from src.models.train_models import run_training_and_validation
from src.evaluation.evaluate import run_full_evaluation
import src

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_installed_versions() -> Dict[str, str]:
    """Inspect environment package versions for reproducibility audit."""
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn",
        "lightgbm", "statsmodels", "joblib", "fastapi",
        "uvicorn", "streamlit", "pydantic", "pyyaml", "pytest"
    ]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def build_and_save_unified_artifact(
    repo_root: Path,
    config: dict,
    val_results: dict,
    test_results: dict,
) -> Path:
    """Bundle trained models and preprocessing metadata into a single deployment artifact."""
    models_dir = repo_root / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading component models for unified packaging...")
    demand_quantiles = joblib.load(models_dir / "demand_lightgbm_quantiles.joblib")
    demand_ridge = joblib.load(models_dir / "demand_ridge.joblib")
    expiry_lgb_data = joblib.load(models_dir / "expiry_lightgbm.joblib")
    expiry_lr_data = joblib.load(models_dir / "expiry_logreg.joblib")

    artifact_bundle = {
        "version": src.__version__,
        "created_at": datetime.now().isoformat(),
        "random_seed": config["project"]["random_seed"],
        "config": config,
        "demand": {
            "primary_model": demand_quantiles,  # dict of {0.10: m, 0.50: m, 0.90: m}
            "linear_model": demand_ridge,
            "feature_columns": DEMAND_FEATURE_COLS,
            "horizon_days": config["forecasting"]["horizon_days"],
        },
        "expiry_risk": {
            "primary_model": expiry_lgb_data["model"],
            "optimal_threshold": expiry_lgb_data["threshold"],
            "linear_model": expiry_lr_data["model"],
            "linear_threshold": expiry_lr_data["threshold"],
            "feature_columns": RISK_FEATURE_COLS,
            "scoring_window_days": config["expiry_risk"]["scoring_window_days"],
            "safety_factor": config["expiry_risk"]["safety_factor"],
            "threshold_units_unsold": config["expiry_risk"]["threshold_units_unsold"],
        },
    }

    bundle_path = models_dir / "pharmacy_forecaster.joblib"
    joblib.dump(artifact_bundle, bundle_path)
    logger.info(f"Unified deployment artifact saved to {bundle_path}")

    # Generate metadata.json
    metadata = {
        "project_name": config["project"]["name"],
        "package_version": src.__version__,
        "pipeline_execution_time": datetime.now().isoformat(),
        "random_seed": config["project"]["random_seed"],
        "python_version": sys.version,
        "environment_dependencies": get_installed_versions(),
        "forecasting_specification": {
            "horizon_days": config["forecasting"]["horizon_days"],
            "prediction_unit": "product x day aggregated to 28-day horizon",
            "demand_feature_count": len(DEMAND_FEATURE_COLS),
            "demand_features": DEMAND_FEATURE_COLS,
        },
        "expiry_risk_specification": {
            "scoring_window_days": config["expiry_risk"]["scoring_window_days"],
            "optimal_decision_threshold": expiry_lgb_data["threshold"],
            "cost_ratio_fn_to_fp": config["expiry_risk"]["cost_ratio"],
            "risk_feature_count": len(RISK_FEATURE_COLS),
            "risk_features": RISK_FEATURE_COLS,
        },
        "validation_metrics": val_results,
        "held_out_test_metrics": test_results,
        "saved_artifacts": [
            "models/pharmacy_forecaster.joblib",
            "models/demand_lightgbm_quantiles.joblib",
            "models/demand_ridge.joblib",
            "models/expiry_lightgbm.joblib",
            "models/expiry_logreg.joblib",
            "models/metadata.json",
        ],
    }

    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model card metadata saved to {metadata_path}")

    return bundle_path


def verify_saved_artifact(bundle_path: Path, proc_dir: Path):
    """Verify that the persisted model artifact loads and executes predictions without retraining."""
    logger.info(f"Verifying standalone loading of {bundle_path.name}...")
    bundle = joblib.load(bundle_path)

    assert "demand" in bundle and "expiry_risk" in bundle
    assert bundle["version"] == "0.1.0"

    # Verify demand inference
    val_dem = pd.read_csv(proc_dir / "val_demand.csv").head(5)
    d_feat_cols = bundle["demand"]["feature_columns"]
    lgb_quantiles = bundle["demand"]["primary_model"]
    
    p10 = lgb_quantiles[0.10].predict(val_dem[d_feat_cols])
    p50 = lgb_quantiles[0.50].predict(val_dem[d_feat_cols])
    p90 = lgb_quantiles[0.90].predict(val_dem[d_feat_cols])
    
    assert len(p50) == 5
    assert (p50 >= 0).all()
    assert (p10 <= p90 + 1e-3).all()

    # Verify expiry risk inference
    val_risk = pd.read_csv(proc_dir / "val_risk.csv").head(5)
    r_feat_cols = bundle["expiry_risk"]["feature_columns"]
    lgb_clf = bundle["expiry_risk"]["primary_model"]
    thresh = bundle["expiry_risk"]["optimal_threshold"]

    probs = lgb_clf.predict_proba(val_risk[r_feat_cols])[:, 1]
    preds = (probs >= thresh).astype(int)

    assert len(probs) == 5
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert set(preds).issubset({0, 1})

    logger.info("Verification passed: Artifact loaded independently and returned valid predictions.")


def main():
    logger.info("=" * 80)
    logger.info("STARTING REPRODUCIBLE END-TO-END PIPELINE (python -m src.train)")
    logger.info("=" * 80)

    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "configs" / "config.yaml"
    cfg = load_config(config_path)

    raw_dir = repo_root / cfg["paths"]["raw_data_dir"]
    proc_dir = repo_root / cfg["paths"]["processed_data_dir"]

    # Step 1: Check raw data files
    raw_files = [raw_dir / "calendar.csv", raw_dir / "products.csv", raw_dir / "daily_sales.csv", raw_dir / "batches.csv"]
    if not all(p.exists() for p in raw_files):
        logger.info("Raw data tables missing. Executing synthetic data generator...")
        generate_raw_data()
    else:
        logger.info("Raw data tables verified in data/raw/.")

    # Step 2: Quality cleaning
    logger.info("Executing data quality cleaning pipeline...")
    run_cleaning_pipeline()

    # Step 3: Feature engineering
    logger.info("Executing temporal feature engineering...")
    run_feature_engineering()

    # Step 4: Chronological partitioning
    logger.info("Executing chronological dataset splitting...")
    run_split_pipeline()

    # Step 5: Training and validation
    logger.info("Training models and tuning thresholds on validation set...")
    val_results = run_training_and_validation()

    # Step 6: Test evaluation
    logger.info("Evaluating models on held-out test set...")
    test_results = run_full_evaluation()

    # Step 7: Packaging and metadata
    logger.info("Packaging unified artifact and metadata...")
    bundle_path = build_and_save_unified_artifact(repo_root, cfg, val_results, test_results)

    # Step 8: Standalone verification
    verify_saved_artifact(bundle_path, proc_dir)

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
