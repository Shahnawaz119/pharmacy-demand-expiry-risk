# Pharmacy Medicine Demand & Expiry-Risk Forecasting

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Tests: 38 Passed](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Advisory Only](https://img.shields.io/badge/system-decision--support%20advisory-orange.svg)](#mandatory-advisory-framing)

An end-to-end machine learning decision-support system designed for retail pharmacy managers. The system forecasts **product-level demand across a forward 28-day replenishment horizon** with uncertainty intervals and predicts **individual batch-level expiration risk under First-Expiry-First-Out (FEFO) dispensing rules**.

---

## Mandatory Advisory Framing

> **CRITICAL OPERATIONAL NOTICE**:
> This system is strictly an **advisory decision-support tool** for inventory replenishment and FEFO stock rotation.
> - **No Medical Advice**: It never generates clinical, dosage, diagnosis, or patient-specific treatment guidance.
> - **Mandatory Human-in-the-Loop**: All automated reorders, shelf rotations, clearance discounts, or vendor returns **must be reviewed and authorized by a licensed pharmacist**.
> - **Synthetic Benchmark**: All data was procedurally generated using realistic retail seasonality, shelf-life distributions, and stock-out dynamics. It contains no real patient records or proprietary retail trade secrets.

---

## Business Problem & Decision Context

Retail community pharmacies face two conflicting inventory pressures:
1. **Stockouts & Lost Care**: Under-stocking essential medications (such as analgesics, antipyretics, and insulin needles) disrupts acute patient care and incurs lost retail revenue.
2. **Expired Inventory Scrap**: Over-stocking causes medicines to exceed their shelf-life before dispensing, requiring total financial write-offs and hazardous pharmaceutical disposal.

### Operational Decision Units
| Dimension | Demand Forecasting | Expiry Risk Classification |
| :--- | :--- | :--- |
| **Primary Decision** | Weekly replenishment purchase orders ($Q$) | Weekly stock rotation, markdowns, and returns |
| **Prediction Unit** | $\text{Product} \times \text{Day}$, evaluated on 28-day sum | $\text{Batch} \times \text{Snapshot Date}$ |
| **Horizon** | Fixed forward 28 days | Days remaining to expiry (scoring batches $\le 90$ days) |
| **Target Label** | Cumulative units sold in forward 28 days | Binary $\mathbb{I}(\text{actual unsold units at expiry} > 5.0)$ |
| **Dispensing Policy** | Observed retail sell-through | First-Expiry-First-Out (FEFO) simulation |
| **Cost Matrix** | Symmetric WAPE / MAE penalty | Asymmetric: $5\times$ penalty for missed expiry (FN) vs false alarm (FP) |

---

## System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[Raw Generator: generate_data.py<br/>Seed 42, 731 Days, 60 SKUs] --> B[Data Quality & Cleaning: clean_data.py<br/>Deduplication, Imputation, IQR Clipping]
        B --> C[(Cleaned Tables<br/>Sales, Products, Batches, Calendar)]
    end

    subgraph Feature Engineering & Leakage Shield
        C --> D[build_features.py<br/>Lagged Shifts t-1, Backward Rolling Windows]
        D --> E[Temporal Leakage Guard<br/>Strict t Shift: Zero Future Leakage]
    end

    subgraph Chronological Partitioning
        E --> F[Train Split: 2023-01-01 to 2024-03-15<br/>26,400 Rows | 68.0%]
        F --> G[Gap Buffer: 28 Days<br/>Prevents Label Overlap]
        G --> H[Val Split: 2024-04-13 to 2024-08-08<br/>7,080 Rows | 18.2%]
        H --> I[Gap Buffer: 28 Days<br/>Prevents Label Overlap]
        I --> J[Held-Out Test: 2024-09-06 to 2024-12-03<br/>5,340 Rows | 13.8%]
    end

    subgraph Model Training & Tuning
        F & H --> K[Demand Models<br/>Baselines: MA7, MA28, SNaive<br/>Model A: Ridge Regressor<br/>Model B: LightGBM Quantiles p10, p50, p90]
        F & H --> L[Expiry Classifiers<br/>Heuristic FEFO Rule Flag<br/>Cost-Sensitive Logistic Regression<br/>Cost-Tuned LightGBM Classifier: Threshold 0.93]
    end

    subgraph Artifact Serialization
        K & L --> M[models/pharmacy_forecaster.joblib<br/>Unified Production Bundle]
        K & L --> N[models/metadata.json<br/>Manifest & Dependency Checksums]
    end

    subgraph Advisory Serving Layer
        M & N --> O[FastAPI Microservice: src.api.main<br/>POST /predict/demand<br/>POST /predict/expiry-risk<br/>GET /health]
        M & N --> P[Streamlit Advisory App: src.app.main<br/>Uncertainty Ribbon, FEFO Queue, Disclaimers]
    end
```

---

## Empirical Results Summary

All models were evaluated on the **untouched held-out test dataset** (5,340 samples from September 6, 2024 to December 3, 2024) across 60 products and 10 categories.

### 1. 28-Day Demand Forecasting Benchmark

$$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i} \times 100\%$$

| Model Architecture | MAE (Units) | RMSE (Units) | WAPE (%) | Signed Bias | 80% Coverage | Mean Interval Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Moving Average (MA7)** | 92.68 | 132.41 | 21.73% | +1.54 | N/A | N/A |
| **Baseline 2: Moving Average (MA28)** | 86.67 | 126.42 | 20.32% | +2.95 | N/A | N/A |
| **Baseline 3: Seasonal Naive** | 83.13 | 121.16 | 19.49% | +2.10 | N/A | N/A |
| **Model A: Category Ridge Regression** | 71.70 | 105.71 | 16.81% | -8.68 | N/A | N/A |
| **Model B: LightGBM Quantile ($p_{50}$) [PROD]** | **66.82** | **102.39** | **15.67%** | -19.35 | **72.04%** | 172.35 units |

> **Honest Scientific Takeaway**: LightGBM achieved the lowest overall test WAPE (**15.67%** vs 20.32% for MA28), providing its largest accuracy gains on seasonal categories (Cough/Cold, Analgesics, Antipyretics). However, on highly stable categories like **Diabetic Care** and **Baby Care**, the simple 28-day moving average proved remarkably competitive (12.91% and 15.18% WAPE), demonstrating that non-seasonal staple lines require little complex modeling.

### 2. FEFO Expiry-Risk Classification Benchmark

Evaluated on 77 test-period active batch snapshots expiring within 90 days. Decision threshold $\theta^* = 0.93$ was tuned on the validation set using a cost ratio of $c_{\text{fn}}=5.0$ vs $c_{\text{fp}}=1.0$.

| Classifier / Strategy | Precision | Recall | F1 Score | PR-AUC | Cost Penalty ($5\cdot\text{FN} + 1\cdot\text{FP}$) | Operational Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Heuristic FEFO Risk Rule** | 1.0000 | 0.5000 | 0.6667 | 0.7565 | 5.0 | Deterministic Flag |
| **Logistic Regression ($\theta^*=0.34$)** | 1.0000 | 0.5000 | 0.6667 | 0.6625 | 5.0 | Linear Scoring |
| **LightGBM Classifier ($\theta^*=0.93$) [PROD]** | **1.0000** | **0.5000** | **0.6667** | **0.5593** | **5.0** | Non-Linear Probabilities |

---

## Repository Directory Structure

```text
pharmacy-demand-expiry-risk/
├── configs/
│   └── config.yaml                 # Central hyperparameter, horizon, and cost configuration
├── data/
│   ├── raw/                        # Synthetic generated CSVs (daily_sales, products, batches, calendar)
│   └── processed/                  # Cleaned tables and engineered feature matrices
├── docs/
│   ├── DATA_CARD.md                # Data provenance, generation logic, and limitations
│   ├── LEAKAGE_ANALYSIS.md         # Temporal safety proof and perturbation test documentation
│   ├── TECHNICAL_REPORT.md         # Exhaustive ML technical report and error analysis
│   ├── MODEL_CARD.md               # Standardized model card per Mitchell et al.
│   └── DEMO_SCRIPT.md              # 4-minute screen-recording walkthrough script
├── models/
│   ├── pharmacy_forecaster.joblib  # Unified serialized model bundle
│   └── metadata.json               # Training execution manifest, library versions, and metrics
├── notebooks/
│   ├── 01_eda.ipynb                # Publication-grade exploratory analysis notebook
│   └── build_eda_notebook.py       # Programmatic notebook compiler
├── reports/
│   ├── experiments.csv             # Experiment tracking log with historical runs
│   ├── metrics.json                # Complete held-out test evaluation scorecard
│   ├── split_summary.json          # Temporal split date boundaries and row counts
│   └── figures/                    # 14 publication-grade PNG visualization charts
├── src/
│   ├── __init__.py
│   ├── train.py                    # End-to-end reproducible training orchestrator
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                 # FastAPI REST inference microservice
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py                 # Streamlit interactive advisory dashboard
│   ├── data/
│   │   ├── generate_data.py        # Realistic synthetic pharmacy generator (Seed 42)
│   │   └── clean_data.py           # Automated data quality audit and cleaning
│   ├── features/
│   │   ├── build_features.py       # Demand & expiry feature engineering with strict shift
│   │   └── split.py                # Chronological partitioning with 28-day gap buffers
│   ├── models/
│   │   ├── demand_models.py        # Quantile LightGBM, Ridge, and baseline regressors
│   │   ├── expiry_models.py        # Cost-sensitive LightGBM, LogReg, and rule classifiers
│   │   └── train_models.py         # Model training, validation tuning, and logging
│   └── evaluation/
│       ├── evaluate.py             # Test evaluation, error diagnostics, and feature importance
│       └── generate_eda_plots.py   # Publication figures generator
├── tests/                          # 38 automated unit & integration tests
│   ├── test_setup.py
│   ├── test_data_generation.py
│   ├── test_data_quality.py
│   ├── test_leakage.py
│   ├── test_split.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   ├── test_pipeline.py
│   ├── test_api.py
│   └── test_risk_edge_cases.py
├── Makefile                        # Unified command runner for all developer targets
├── pyproject.toml                  # Modern package build configuration
├── requirements.txt                # Pinned production dependencies
├── .env.example                    # Environment variable template
└── README.md                       # Main repository guide
```

---

## Quickstart & Reproduction Guide

### Prerequisites
- Python 3.11.x installed (verified on Python 3.11.9)
- Git

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Shahnawaz119/pharmacy-demand-expiry-risk.git
cd pharmacy-demand-expiry-risk

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Execution via Makefile

The repository includes a complete `Makefile` automating every workflow stage:

```bash
# Generate synthetic dataset and execute data quality cleaning
make data

# Execute reproducible end-to-end training and serialize artifacts
make train

# Evaluate the serialized model on the untouched held-out test set
make evaluate

# Execute full pytest suite (38 tests)
make test

# Launch the FastAPI REST inference microservice (http://127.0.0.1:8000/docs)
make api

# Launch the Streamlit interactive advisory dashboard (http://localhost:8501)
make app
```

---

## FastAPI Microservice Endpoints

Run `make api` or:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation is automatically served at `http://127.0.0.1:8000/docs`.

### Key Endpoints:
- `GET /health`: Service health check, model version, timestamp, and feature list.
- `POST /predict/demand`:
  ```json
  // Request
  {
    "product_id": "PROD_001",
    "as_of_date": "2024-11-20"
  }
  
  // Response
  {
    "product_id": "PROD_001",
    "product_name": "paracetamol_500mg",
    "category": "analgesics",
    "as_of_date": "2024-11-20",
    "horizon_days": 28,
    "forecast_28d_p50": 742.15,
    "forecast_28d_p10": 589.40,
    "forecast_28d_p90": 912.80,
    "current_stock_on_hand": 310.0,
    "stock_cover_days": 11.7,
    "reorder_guidance": "STANDARD REORDER REVIEW: Stock cover (11.7 days) is moderate. Suggested replenishment: ~506 units to cover 28-day horizon with safety buffer.",
    "disclaimer": "Advisory tool. Predictions are estimates from synthetic-data training and must be reviewed by a pharmacist. This is not medical or treatment advice."
  }
  ```
- `POST /predict/expiry-risk`:
  ```json
  // Request (by batch ID or custom features)
  {
    "batch_id": "BAT_00024"
  }
  
  // Response
  {
    "batch_id": "BAT_00024",
    "product_id": "PROD_001",
    "product_name": "paracetamol_500mg",
    "days_to_expiry": 32,
    "remaining_qty": 199.0,
    "unit_cost": 15.0,
    "potential_loss_value": 2985.0,
    "risk_flag": true,
    "risk_score": 0.9842,
    "decision_threshold": 0.93,
    "heuristic_rule_flag": true,
    "reason": "HIGH RISK of unsold expiration (98.4% probability vs 93.0% threshold). Remaining 199 units with 32 days to expiry. With 0 units ahead in FEFO queue and expected demand of 72 units, estimated ~127 units risk being unsold.",
    "suggested_action": "discount",
    "disclaimer": "Advisory tool. Predictions are estimates from synthetic-data training and must be reviewed by a pharmacist. This is not medical or treatment advice."
  }
  ```

---

## Streamlit Advisory Dashboard

Run `make app` or:
```bash
streamlit run src/app/main.py
```
Key features:
1. **Interactive Uncertainty Ribbon**: Visualizes past 90 days of actual daily sales alongside forward 28 days of daily forecasts with shaded $[p_{10}, p_{90}]$ prediction intervals.
2. **FEFO Priority Queue Table**: Highlights all active batches expiring within 90 days, color-coded by operational action (`🔴 Supplier Return`, `🟠 Discount Clearance`, `🟡 Stock Rotation`, `🟢 Monitor`).
3. **Drill-Down Inspector**: Explains why a specific batch is flagged (earlier queue stock, runout demand shortfall).
4. **Model Transparency**: Features importance charts and baseline comparison scorecard.

---

## Limitations & Operational Boundaries

1. **Synthetic Data Constraint**: The models are trained on synthetic data simulating Indian seasonal retail patterns. While realistic, the data does not capture macroeconomic supply chain shocks, regulatory recalls, or sudden clinic closures.
2. **Lag Burn-In**: Features rely on 28-day rolling lags; new SKU introductions require a minimum 28-day warm-up period or fallback to category-median baselines.
3. **Stockout Censoring**: During severe stockouts, true latent customer demand exceeds recorded sales. The model uses backward stockout flags to mitigate distortion, but prolonged stockouts may suppress forecasts.
4. **Pharmacist Sign-Off**: Automated output is strictly advisory. Final decisions regarding purchase orders, markdowns, and inventory destocking must be validated by licensed pharmacy staff.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
