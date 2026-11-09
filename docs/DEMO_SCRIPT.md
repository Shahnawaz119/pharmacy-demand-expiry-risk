# Demo Video Walkthrough Script: Pharmacy Demand & Expiry-Risk Forecasting

**Format**: 4-Minute Screen-Recording Walkthrough  
**Presenter**: Senior Machine Learning Engineer  
**Target Audience**: Technical Assessors, Pharmacy Store Managers, Procurement Directors  
**Repository**: `https://github.com/Shahnawaz119/pharmacy-demand-expiry-risk`

---

## Overview Timing & Scene Breakdown

| Timestamp | Segment Title | On-Screen Visual | Key Talking Points |
| :---: | :--- | :--- | :--- |
| **0:00 – 0:45** | **1. Problem Statement & Advisory Notice** | Repository README & Mermaid architecture diagram | Dual inventory challenge; strictly decision-support framing; no medical advice. |
| **0:45 – 1:30** | **2. Data Quality & EDA Insights** | EDA notebook & figures (`fig01`, `fig03`, `fig07`) | 43.8k rows; Indian seasonality; FEFO queue bottleneck analysis. |
| **1:30 – 2:30** | **3. Modeling, Baselines & Results** | Terminal test run & Held-out results scorecard | Chronological gap split; LightGBM beats MA28 (15.67% vs 20.32% WAPE); cost tuning. |
| **2:30 – 3:30** | **4. Live Streamlit Advisory App Walkthrough** | Streamlit UI at `http://localhost:8501` | Product picker, uncertainty ribbon $[p_{10}, p_{90}]$, FEFO at-risk batch table, action badges. |
| **3:30 – 4:00** | **5. FastAPI Swagger & Responsible Conclusion** | FastAPI Swagger UI at `http://127.0.0.1:8000/docs` | Live REST prediction; 38 passing unit tests; mandatory pharmacist oversight. |

---

## Detailed Script & Presenter Cues

### Segment 1: Problem Statement & Operational Framing (0:00 – 0:45)
**[VISUAL]**: Show GitHub repository front page, scrolling to the Mermaid architecture flowchart and the amber Advisory Notice box.

**[NARRATION]**:
> "Hello! Today I am presenting our **Pharmacy Medicine Demand & Expiry-Risk Forecasting System**, an end-to-end decision-support solution built for retail community pharmacies.
>
> In community pharmacy retail, store managers navigate two severe inventory risks: under-stocking critical medicines like antipyretics or analgesics, which disrupts patient care, and over-stocking, which leads to expired medicines that must be discarded as hazardous medical waste.
>
> To solve this, we formulated two distinct prediction tasks:
> First, forecasting product-level demand for the next 28 days to support replenishment purchase orders.
> Second, identifying individual stock batches that risk expiring unsold under First-Expiry-First-Out (FEFO) dispensing rules.
>
> Crucially, our system is strictly **advisory decision-support**. It never provides clinical or dosage advice, and every operational recommendation—reordering, rotating stock, or marking down—must be reviewed and authorized by a licensed pharmacist."

---

### Segment 2: Data Architecture & Exploratory Insights (0:45 – 1:30)
**[VISUAL]**: Open `notebooks/01_eda.ipynb` and display Figure 3 (Seasonality Patterns) and Figure 7 (Batch Shelf-Life vs Sell-Through).

**[NARRATION]**:
> "Let's examine the data foundation. We generated 731 days of synthetic retail data across 60 products in 10 categories, yielding over 43,800 daily records and 977 batches with seed 42.
>
> In our exploratory analysis, three critical operational patterns emerged:
> First, strong Indian seasonality. As seen in Figure 3, Cough and Cold remedies surge by over 180% during North Indian winter months, while gastrointestinal antacids peak during the monsoon.
> Second, retail weekend footfall creates an 18.4% sales lift on Saturdays and Sundays.
> Third, in Figure 7, we analyzed batch sell-through under FEFO dispensing. Batches with over 18 months of shelf-life achieved 99.8% sell-through. However, short-dated supplier deliveries and batches positioned deep in the dispensing queue accounted for 94% of all expired units.
>
> Our data quality pipeline cleaned this data with deduplication, price imputation, negative sales clipping, and robust outlier winsorization."

---

### Segment 3: Modeling Protocol, Baselines & Test Results (1:30 – 2:30)
**[VISUAL]**: Switch to VS Code terminal. Show `reports/metrics.json` and the test scorecard table in `README.md`.

**[NARRATION]**:
> "To prevent temporal data leakage, we implemented a strict chronological partition with **28-day gap buffers** matching our forecasting horizon. We trained on the first 68% of data, tuned hyperparameters strictly on the 18% validation split, and evaluated on the final 14% held-out test set from September to December 2024.
>
> Here are our test results:
> Against a standard 28-day moving average baseline of **20.32% WAPE**, and a Seasonal Naive baseline of **19.49% WAPE**, our production **LightGBM Quantile Regressor ($p_{50}$)** achieved **15.67% WAPE**—an absolute improvement of 4.65 percentage points. Furthermore, our 10th and 90th percentile models achieved **72.04% empirical coverage**, giving managers calibrated risk bands.
>
> For expiry risk, we optimized our LightGBM classifier with an asymmetric business cost matrix where letting a batch expire unsold costs 5 times more than a false alarm. On our test set, our model achieved **100% precision** with zero false alarms."

---

### Segment 4: Live Streamlit Advisory App Walkthrough (2:30 – 3:30)
**[VISUAL]**: Switch to browser tab running Streamlit at `http://localhost:8501`.

**[NARRATION]**:
> "Now, let's look at the decision-support interface in Streamlit.
>
> At the top, you notice our prominent warning banner reminding staff that pharmacist verification is required before taking action.
>
> In Tab 1, we select a product—for example, `PROD_001: paracetamol_500mg`. We instantly see our key operational metrics:
> A 28-day forecast of **742 units**, with an 80% uncertainty band from **589 to 912 units**, current stock of **310 units**, and **11.7 days of stock cover**.
> Notice our clear operational banner: *'STANDARD REORDER REVIEW: Suggested replenishment ~506 units'*.
> In the chart below, managers can see past actual sales and the forward 28-day trajectory with the shaded uncertainty ribbon.
>
> In Tab 2, we switch to our **FEFO Expiry-Risk Queue**. Here we see batches expiring within 90 days. The app highlights high-risk batches and suggests strictly operational actions:
> For batch `BAT_00024`, expiring in 32 days with 199 units left, our model flags a 98.4% risk score, recommending an orange **Discount Clearance** badge.
> In our single-batch deep dive, we transparently explain the math: daily velocity cannot absorb the remaining stock before the expiry date."

---

### Segment 5: FastAPI REST Endpoints & Conclusion (3:30 – 4:00)
**[VISUAL]**: Switch to browser tab showing FastAPI Swagger docs at `http://127.0.0.1:8000/docs`, execute a sample `POST /predict/demand` call, then toggle to terminal showing `38 passed in pytest`.

**[NARRATION]**:
> "For automated ERP integration, all predictions are served via a production **FastAPI microservice**.
> Here at `/docs`, we can test `POST /predict/demand` and `POST /predict/expiry-risk`. Every response returns validated Pydantic schemas, quantile intervals, operational recommendations, and decision-support disclaimers.
>
> In our terminal, running `pytest` executes our complete test suite: **all 38 automated unit and integration tests pass cleanly**, verifying data determinism, zero feature leakage, split gaps, and risk rule edge cases.
>
> In conclusion, this project delivers an accurate, explainable, and ethically sound decision-support tool that protects community pharmacy margins while maintaining essential medication availability.
>
> Thank you!"
