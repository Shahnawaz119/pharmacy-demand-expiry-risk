# Pharmacy Medicine Demand & Expiry-Risk Forecasting

An advisory decision-support machine learning solution designed for retail pharmacy managers.

> **Operational Advisory Notice**: This tool is an operational forecasting and inventory decision-support system. It produces estimates of product demand and identifies stock batches at risk of expiring unsold. It never provides clinical, dosage, or medical advice. All inventory recommendations must be reviewed and authorized by pharmacy personnel.

## Project Overview

- **Decision**: Weekly inventory replenishment and stock-rotation review.
- **Prediction Unit**: 
  - (a) Demand: product $\times$ day, aggregated to a 28-day horizon.
  - (b) Expiry Risk: batch $\times$ snapshot date.
- **Horizon**: 28 days for product demand; up to 90 days for expiring batches.
- **Methodology**: Moving average and seasonal naive baselines compared against regularized linear models, LightGBM quantile regression for demand uncertainty, and cost-sensitive gradient boosted classifiers for batch expiry risk.

## Quickstart

```bash
# 1. Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .

# 2. Generate synthetic data and run quality cleaning
make data

# 3. Train models and generate serialized artifacts
make train

# 4. Evaluate models on the held-out test set
make evaluate

# 5. Run test suite
make test

# 6. Launch FastAPI service
make api

# 7. Launch Streamlit advisory dashboard
make app
```
