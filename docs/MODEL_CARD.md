# Model Card: Pharmacy Medicine Demand & Expiry-Risk Forecaster

## Model Details
- **Model Name**: Retail Pharmacy Demand & FEFO Expiry-Risk Decision Support Engine
- **Model Version**: `0.1.0` (Production Bundle: `models/pharmacy_forecaster.joblib`)
- **Developers**: Senior Machine Learning Engineer (Assessment Project)
- **Model Release Date**: November 2026
- **Model Architecture**:
  1. **Demand Forecasting**: Ensemble of 3 LightGBM Quantile Regressors ($\alpha \in \{0.10, 0.50, 0.90\}$) trained with histogram gradient boosting and 32 engineered lag/rolling/calendar features.
  2. **Baseline Demand**: Category-Aware Ridge Regressor ($\alpha=10.0$) and 28-day Moving Average.
  3. **Expiry-Risk Classifier**: Cost-Tuned LightGBM Binary Classifier with optimal operational threshold ($\theta^* = 0.93$) trained with `class_weight='balanced'` on 14 FEFO queue features.
- **License**: MIT
- **Dependencies**: Python 3.11.9, `lightgbm==4.7.0`, `scikit-learn==1.8.0`, `pandas==2.3.3`, `numpy==2.3.5`, `joblib==1.5.2`.

---

## Intended Use
- **Primary Intended Uses**:
  - Operational replenishment advisory: Generating 28-day forward point forecasts ($p_{50}$) and uncertainty intervals ($[p_{10}, p_{90}]$) to assist pharmacy inventory managers in formulating weekly purchase orders ($Q$).
  - Stock rotation and markdown advisory: Identifying active inventory batches that risk expiring unsold within a 90-day window under First-Expiry-First-Out (FEFO) dispensing rules, suggesting physical shelf rotation, clearance discounts ($15-30\%$), or supplier returns.
- **Primary Users**: Licensed retail pharmacists, pharmacy store managers, inventory procurement coordinators.
- **Mandatory Human-in-the-Loop Protocol**:
  - The system is strictly **decision-support**.
  - All reorders, transfers, markdowns, and returns must be verified and authorized by a licensed pharmacist.

---

## Out-of-Scope & Prohibited Uses
- **NO Medical Advice**: This system MUST NOT be used to provide patient care, prescription verification, drug-drug interaction warnings, dosage calculations, or clinical diagnosis.
- **NO Autonomous Stock Disposal**: Automated disposal or vendor return without human inspection is strictly prohibited.
- **NO Uncalibrated Geographic Transfer**: The model was trained on Indian retail seasonality (monsoons, winter cough peaks, Diwali festival surges). Deploying in different climatic or macroeconomic contexts without retraining will cause systematic forecasting errors.

---

## Factors & Performance Subgroups

Performance was evaluated on the **untouched held-out test dataset** (5,340 samples from September 6, 2024 to December 3, 2024):

### 1. Demand Forecasting Overall & Quantile Calibration
- **Point Forecast ($p_{50}$)**:
  - **WAPE**: `15.67%` (vs Baseline MA28 of `20.32%` and Seasonal Naive of `19.49%`)
  - **MAE**: `66.82` units / 28-day period
  - **RMSE**: `102.39` units / 28-day period
  - **Signed Bias**: `-19.35` units (slight conservative under-forecast during acute winter surge)
- **Uncertainty Interval Calibration**:
  - **Nominal Coverage**: $80.0\%$
  - **Empirical Test Coverage**: `72.04%`
  - **Mean Prediction Interval Width**: `172.35` units

### 2. Performance by Product Category
| Category | LightGBM $p_{50}$ WAPE (%) | Baseline MA28 WAPE (%) | LightGBM MAE (Units) | Signed Bias |
| :--- | :---: | :---: | :---: | :---: |
| **Allergy** | 19.23% | 24.14% | 97.90 | +18.84 |
| **Analgesics** | 14.23% | 21.79% | 65.09 | -10.89 |
| **Antacids** | 15.96% | 19.73% | 97.88 | -59.07 |
| **Antipyretics** | 13.74% | 20.32% | 66.05 | +22.30 |
| **Baby Care** | 12.84% | 15.18% | 30.25 | +1.05 |
| **Cough & Cold** | 25.14% | 31.50% | 141.36 | -114.90 |
| **Diabetic Care** | 11.15% | 12.91% | 27.16 | -9.32 |
| **First Aid** | 10.97% | 12.49% | 35.76 | -3.87 |
| **Skin Care** | 17.03% | 22.26% | 52.56 | -14.84 |
| **Vitamins & Supplements** | 10.26% | 13.59% | 54.16 | -22.80 |

### 3. Expiry Risk Classification ($N=77$ Batches)
- **Decision Threshold**: $\theta^* = 0.93$ (tuned on validation set for $5\times$ cost of FN vs FP)
- **Precision**: `1.0000` (100% of flagged batches were genuine expiration risks)
- **Recall**: `0.5000`
- **F1 Score**: `0.6667`
- **Business Cost Penalty**: `5.0`

---

## Training & Evaluation Data
- **Source**: Procedural synthetic pharmacy retail data generator (`src/data/generate_data.py`, Seed `42`).
- **Total Data Volume**: 43,860 daily sales rows, 60 SKUs, 977 batches over 731 days (2023-01-01 to 2024-12-31).
- **Data Quality Pipeline**: Automated deduplication, base-price imputation for missing values, negative sales clipping, corrupted batch dropping, and robust outlier winsorization.
- **Partitioning**: Chronological split with 28-day gap buffers matching the prediction horizon:
  - Train: `2023-01-01` to `2024-03-15` (26,400 rows, 68.0%)
  - Validation: `2024-04-13` to `2024-08-08` (7,080 rows, 18.2%)
  - Held-Out Test: `2024-09-06` to `2024-12-03` (5,340 rows, 13.8%)

---

## Ethical Considerations & Privacy
- **Zero Patient Data**: Operates strictly on SKU-level aggregates and warehouse lot numbers. No patient PHI/PII is ingested or stored.
- **Fairness Across SKUs**: Slow-moving medicines (vital for niche therapeutic needs, e.g. diabetic consumables) are modeled without bias towards high-velocity OTC analgesics through log/velocity normalization.

---

## Caveats, Limitations & Failure Modes
1. **Acute Weather & Pollution Inversions**: Sudden regional events (e.g. November Delhi smog spikes) cause demand surges that historical 28-day lags cannot anticipate, resulting in temporary under-forecasting (see technical case study on `PROD_030`).
2. **Cold-Start SKUs**: Formulations on shelf for <28 days lack lag history and must use category-median fallback rules.
3. **Prolonged Stockout Censoring**: If an item is unstocked for >28 days, observed sales fall to zero; human replenishment review must reset baseline orders.
