# Data Card: Synthetic Pharmacy Sales and Inventory Batches

## 1. Overview and Synthetic Declaration
- **Dataset Name**: Synthetic Retail Pharmacy Daily Sales and Batch Inventory
- **Version**: 1.0.0
- **Generator**: `src/data/generate_data.py` (Random Seed: `42`)
- **Synthetic Statement**: This dataset is **entirely synthetically generated** for machine learning research and internship assessment. It contains **no real patient data, no protected health information (PHI), no proprietary prescription records, and no genuine pharmacy transaction logs**. All product titles are generic pharmacological names.

## 2. Dataset Motivation and Scope
The dataset was designed to support an operational decision-support tool for retail pharmacy inventory managers. It addresses two interdependent forecasting objectives:
1. **Demand Forecasting**: Estimating daily unit sales per product aggregated over a 28-day forward horizon.
2. **Expiry-Risk Forecasting**: Identifying specific physical inventory batches that risk passing their expiry date before full sell-through under First-Expiry-First-Out (FEFO) dispensing rules.

## 3. Schema and Table Specifications

### 3.1 `products.csv` (60 Products)
| Field | Type | Description |
|---|---|---|
| `product_id` | String | Unique product identifier (`PROD_001` to `PROD_060`) |
| `product_name` | String | Generic medicine or consumable description (e.g. `paracetamol_650mg`) |
| `category` | String | 10 therapeutic categories (analgesics, cough_cold, allergy, etc.) |
| `base_price` | Float | Nominal retail selling price in INR (₹12.0 to ₹980.0) |
| `unit_cost` | Float | Wholesale acquisition cost (~72% of retail price) |
| `shelf_life_months` | Integer | Standard manufacturing shelf life (12 to 36 months) |
| `base_daily_demand` | Float | Latent average daily consumption rate |
| `demand_velocity` | String | Categorization (`fast`, `medium`, `slow`) |
| `intermittency_prob`| Float | Probability of zero-demand days for slow-moving products |

### 3.2 `calendar.csv` (731 Days: 2023-01-01 to 2024-12-31)
| Field | Type | Description |
|---|---|---|
| `date` | Date (YYYY-MM-DD) | Calendar date |
| `day_of_week` | Integer | 0 (Monday) to 6 (Sunday) |
| `day_name` | String | Weekday name |
| `month`, `year` | Integer | Month (1-12) and Gregorian year |
| `day_of_year` | Integer | Day index within calendar year (1-366) |
| `is_weekend` | Integer | Binary indicator (1 for Saturday and Sunday) |
| `season` | String | Indian meteorological seasons: `winter`, `summer`, `monsoon`, `post_monsoon` |
| `is_holiday` | Integer | Binary indicator for major national holidays |
| `holiday_name` | String | Name of holiday (e.g. `Diwali`, `Independence Day`) |
| `is_festival` | Integer | Binary indicator for festive shopping periods |

### 3.3 `daily_sales.csv` (~43,900 Rows)
| Field | Type | Description |
|---|---|---|
| `date` | Date | Transaction date |
| `product_id` | String | Foreign key to `products.csv` |
| `units_sold` | Float | Observed physical sales dispensed to customers |
| `unconstrained_demand` | Float | True latent customer demand before stockout censoring |
| `selling_price` | Float | Effective daily retail price after any promotional discount |
| `is_promoted` | Integer | Binary flag indicating a promotional campaign |
| `stock_on_hand` | Float | Beginning-of-day available inventory across active batches |
| `stockout_flag` | Integer | 1 if beginning stock was 0, resulting in zero sales |

### 3.4 `batches.csv` (~2,000 Batches)
| Field | Type | Description |
|---|---|---|
| `batch_id` | String | Unique batch code (`BAT_00001` ...) |
| `product_id` | String | Foreign key to `products.csv` |
| `received_date` | Date | Warehouse delivery receipt date |
| `expiry_date` | Date | Expiry date stamped on packaging |
| `quantity_received` | Integer | Total units delivered in batch |
| `unit_cost` | Float | Wholesale acquisition cost per unit |
| `supplier` | String | Name of logistics distributor |

## 4. Simulation Mechanisms and Assumptions

1. **Indian Meteorological Seasonality**:
   - *Cough and Cold*: Peaks sharply in winter (Dec-Jan: +85%) with a secondary lift during monsoon chill (+20%).
   - *Antipyretics / Analgesics*: Surge during monsoon months (Jul-Sep: +65%) due to vector-borne fevers (dengue, malaria, viral illness).
   - *Allergy*: Peaks during spring pollen release (Feb-Mar: +55%) and post-monsoon crop harvest/dust (Sep-Oct: +45%).
   - *Antacids*: Exhibit prominent demand spikes around festive culinary periods (Diwali, Holi, New Year: +25% to +40%).
   - *Vitamins & Diabetic Care*: Exhibit stable year-round baseline with steady long-term drift (+6% annually).

2. **Censored Demand and Stock-outs**:
   - Retail sales can never exceed `stock_on_hand`. When inventory drops to zero, `units_sold` is forced to 0 regardless of latent demand.
   - The simulation logs both `unconstrained_demand` and `units_sold` so researchers can analyze stock-out distortion.

3. **FEFO Inventory Depletion**:
   - Active stock is consumed strictly according to First-Expiry-First-Out (FEFO): batches with earlier expiry dates are dispensed first.
   - Any batch whose expiry date is reached before full depletion expires unsold and is discarded from usable stock.

4. **Injected Quality Flaws (for data pipeline verification)**:
   - **Missing values**: ~0.5% missing prices and sales values.
   - **Duplicate rows**: ~0.2% exact duplicate transactions injected.
   - **Outliers**: ~0.3% extreme transcription spikes (8x to 15x normal volume).
   - **Negative sales**: 0.1% negative values to simulate faulty keyboard inputs.
   - **Corrupted batches**: 4 batches with impossible date ordering (`expiry_date <= received_date`).

## 5. What This Dataset Does NOT Represent
- It does **not** model prescription insurance adjudication, doctor prescription networks, or clinical dosage guidelines.
- It does **not** reflect individualized patient purchasing patterns, repeat customer identities, or chronic patient retention.
- It does **not** simulate multi-echelon supply chain logistics (lead time is simplified to batch arrivals).
- It must **not** be used for clinical or therapeutic research.
