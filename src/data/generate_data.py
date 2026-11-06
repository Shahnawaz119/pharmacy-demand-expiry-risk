"""Synthetic pharmacy data generator.

Simulates 2 years of daily retail pharmacy sales and inventory batches across
60 products in 10 categories, capturing realistic Indian seasonal patterns,
weekly cycles, stock-outs, FEFO batch consumption, and controlled anomalies.
"""

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES_SPEC = {
    "analgesics": {
        "products": [
            ("paracetamol_500mg", 15.0, 24, 25.0, 0.15),
            ("ibuprofen_400mg", 22.0, 24, 18.0, 0.18),
            ("tramadol_50mg", 45.0, 36, 6.0, 0.35),
            ("diclofenac_50mg", 18.0, 24, 16.0, 0.20),
            ("naproxen_250mg", 32.0, 24, 9.0, 0.25),
            ("mefenamic_acid_250mg", 28.0, 24, 12.0, 0.22),
        ],
        "shelf_life_months": (18, 36),
    },
    "antipyretics": {
        "products": [
            ("paracetamol_650mg", 30.0, 24, 32.0, 0.12),
            ("paracetamol_suspension_120mg", 42.0, 18, 14.0, 0.20),
            ("mefenamic_suspension", 55.0, 18, 8.0, 0.28),
            ("paracetamol_drops_15ml", 35.0, 12, 10.0, 0.25),
            ("ibuprofen_paracetamol_tab", 32.0, 24, 20.0, 0.15),
            ("mefenamic_paracetamol_syrup", 60.0, 18, 9.0, 0.25),
        ],
        "shelf_life_months": (12, 24),
    },
    "antacids": {
        "products": [
            ("omeprazole_20mg", 45.0, 24, 22.0, 0.15),
            ("pantoprazole_40mg", 65.0, 24, 28.0, 0.14),
            ("rabeprazole_20mg", 70.0, 24, 16.0, 0.18),
            ("antacid_liquid_gel_200ml", 95.0, 18, 19.0, 0.18),
            ("ranitidine_150mg", 25.0, 24, 14.0, 0.22),
            ("esomeprazole_40mg", 80.0, 24, 15.0, 0.20),
        ],
        "shelf_life_months": (18, 30),
    },
    "vitamins_supplements": {
        "products": [
            ("vitamin_c_500mg_chewable", 35.0, 24, 24.0, 0.15),
            ("vitamin_d3_60k_capsules", 110.0, 24, 21.0, 0.16),
            ("multivitamin_daily_tablets", 140.0, 24, 26.0, 0.14),
            ("calcium_vitamin_d3_tablets", 90.0, 24, 20.0, 0.16),
            ("iron_folic_acid_tablets", 50.0, 24, 15.0, 0.20),
            ("zinc_gluconate_50mg", 40.0, 24, 17.0, 0.18),
        ],
        "shelf_life_months": (18, 36),
    },
    "cough_cold": {
        "products": [
            ("cetirizine_cold_relief_tab", 40.0, 24, 22.0, 0.18),
            ("dextromethorphan_syrup_100ml", 85.0, 18, 16.0, 0.22),
            ("ambroxol_syrup_100ml", 78.0, 18, 15.0, 0.24),
            ("phenylephrine_nasal_drops", 52.0, 12, 11.0, 0.25),
            ("xylometazoline_nasal_spray", 90.0, 18, 14.0, 0.22),
            ("menthol_cough_lozenges_strip", 25.0, 24, 30.0, 0.15),
        ],
        "shelf_life_months": (12, 24),
    },
    "allergy": {
        "products": [
            ("cetirizine_10mg", 20.0, 24, 25.0, 0.15),
            ("fexofenadine_120mg", 95.0, 24, 18.0, 0.18),
            ("levocetirizine_5mg", 45.0, 24, 22.0, 0.16),
            ("loratadine_10mg", 38.0, 24, 12.0, 0.22),
            ("montelukast_10mg", 115.0, 24, 14.0, 0.20),
            ("chlorpheniramine_maleate_4mg", 12.0, 24, 16.0, 0.20),
        ],
        "shelf_life_months": (18, 36),
    },
    "first_aid": {
        "products": [
            ("povidone_iodine_ointment_20g", 60.0, 24, 12.0, 0.20),
            ("antiseptic_liquid_100ml", 75.0, 36, 16.0, 0.18),
            ("adhesive_bandages_pack20", 45.0, 36, 22.0, 0.16),
            ("cotton_absorbent_roll_100g", 55.0, 36, 14.0, 0.20),
            ("sterile_gauze_swabs_pack10", 35.0, 36, 11.0, 0.22),
            ("burn_relief_cream_30g", 85.0, 24, 8.0, 0.25),
        ],
        "shelf_life_months": (24, 36),
    },
    "diabetic_care": {
        "products": [
            ("glucometer_test_strips_50s", 650.0, 18, 12.0, 0.16),
            ("sterile_lancets_pack100", 180.0, 36, 9.0, 0.22),
            ("alcohol_swabs_pack100", 120.0, 36, 14.0, 0.18),
            ("diabetic_foot_cream_100g", 210.0, 24, 6.0, 0.28),
            ("blood_glucose_monitor_kit", 980.0, 36, 3.0, 0.40),
            ("insulin_syringes_31g_pack10", 110.0, 36, 15.0, 0.18),
        ],
        "shelf_life_months": (18, 36),
    },
    "baby_care": {
        "products": [
            ("baby_diaper_rash_cream_50g", 145.0, 24, 9.0, 0.22),
            ("gripe_water_liquid_150ml", 65.0, 12, 11.0, 0.24),
            ("baby_nasal_saline_drops_20ml", 48.0, 12, 8.0, 0.26),
            ("infant_vitamin_d3_drops_15ml", 85.0, 12, 10.0, 0.22),
            ("pediatric_electrolyte_powder_sachet", 22.0, 24, 18.0, 0.20),
            ("digital_flexible_thermometer", 175.0, 36, 4.0, 0.35),
        ],
        "shelf_life_months": (12, 24),
    },
    "skin_care": {
        "products": [
            ("calamine_lotion_100ml", 85.0, 24, 14.0, 0.20),
            ("clotrimazole_antifungal_cream_20g", 65.0, 24, 16.0, 0.18),
            ("hydrocortisone_cream_15g", 75.0, 24, 9.0, 0.24),
            ("barrier_repair_moisturizer_100g", 220.0, 24, 12.0, 0.22),
            ("antibacterial_dusting_powder_100g", 110.0, 24, 15.0, 0.20),
            ("sunscreen_spf30_gel_50g", 260.0, 24, 10.0, 0.24),
        ],
        "shelf_life_months": (18, 30),
    },
}

INDIAN_HOLIDAYS = {
    "2023-01-26": "Republic Day",
    "2023-03-08": "Holi",
    "2023-04-07": "Good Friday",
    "2023-04-22": "Eid-ul-Fitr",
    "2023-08-15": "Independence Day",
    "2023-09-19": "Ganesh Chaturthi",
    "2023-10-02": "Gandhi Jayanti",
    "2023-10-24": "Dussehra",
    "2023-11-12": "Diwali",
    "2023-12-25": "Christmas",
    "2024-01-26": "Republic Day",
    "2024-03-25": "Holi",
    "2024-04-11": "Eid-ul-Fitr",
    "2024-08-15": "Independence Day",
    "2024-09-07": "Ganesh Chaturthi",
    "2024-10-02": "Gandhi Jayanti",
    "2024-10-12": "Dussehra",
    "2024-11-01": "Diwali",
    "2024-12-25": "Christmas",
}


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate daily calendar records with Indian seasons and festival flags."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    for dt in dates:
        dt_str = dt.strftime("%Y-%m-%d")
        month = dt.month
        day_of_week = dt.dayofweek  # 0=Monday, 6=Sunday
        is_weekend = int(day_of_week in [5, 6])

        # Indian seasons: Winter (Nov-Feb), Summer (Mar-Jun), Monsoon (Jul-Sep), Post-Monsoon (Oct)
        if month in [11, 12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5, 6]:
            season = "summer"
        elif month in [7, 8, 9]:
            season = "monsoon"
        else:
            season = "post_monsoon"

        holiday_name = INDIAN_HOLIDAYS.get(dt_str, "")
        is_holiday = int(bool(holiday_name))

        # Festivals with elevated retail footfall
        is_festival = int(holiday_name in ["Diwali", "Holi", "Eid-ul-Fitr", "Dussehra", "Christmas", "Ganesh Chaturthi"])

        records.append({
            "date": dt_str,
            "day_of_week": day_of_week,
            "day_name": dt.strftime("%A"),
            "month": month,
            "year": dt.year,
            "day_of_year": dt.dayofyear,
            "is_weekend": is_weekend,
            "season": season,
            "is_holiday": is_holiday,
            "holiday_name": holiday_name,
            "is_festival": is_festival,
        })

    return pd.DataFrame(records)


def build_products() -> pd.DataFrame:
    """Compile product catalog metadata."""
    records = []
    pid_counter = 1

    for category, spec in CATEGORIES_SPEC.items():
        min_shelf, max_shelf = spec["shelf_life_months"]
        for prod_name, base_price, default_shelf, base_demand, intermittency in spec["products"]:
            product_id = f"PROD_{pid_counter:03d}"
            cost_price = round(base_price * 0.72, 2)
            velocity = "slow" if base_demand < 8 else ("fast" if base_demand > 20 else "medium")

            records.append({
                "product_id": product_id,
                "product_name": prod_name,
                "category": category,
                "base_price": base_price,
                "unit_cost": cost_price,
                "shelf_life_months": default_shelf,
                "base_daily_demand": base_demand,
                "demand_velocity": velocity,
                "intermittency_prob": intermittency,
            })
            pid_counter += 1

    return pd.DataFrame(records)


def get_seasonal_multiplier(category: str, month: int, day_of_year: int) -> float:
    """Simulate realistic Indian seasonality curves per pharmaceutical category."""
    if category == "cough_cold":
        # Peaks in winter (Dec-Jan: ~1.85) and autumn turn
        if month in [12, 1]:
            return 1.85
        elif month in [11, 2]:
            return 1.50
        elif month in [7, 8]:
            return 1.20  # Monsoon chill/flu
        return 0.75

    elif category in ["antipyretics", "analgesics"]:
        # Monsoon spike (fevers, dengue, flu: Jul-Sep ~1.65)
        if month in [7, 8, 9]:
            return 1.65
        elif month in [10, 11]:
            return 1.25
        return 0.90

    elif category == "allergy":
        # Spring pollen (Feb-Mar ~1.55) and post-monsoon harvest/dust (Sep-Oct ~1.45)
        if month in [2, 3]:
            return 1.55
        elif month in [9, 10]:
            return 1.45
        return 0.85

    elif category == "antacids":
        # Elevated during festive dining (Oct-Nov Diwali: ~1.40, Mar Holi: ~1.25)
        if month in [10, 11]:
            return 1.40
        elif month in [3, 4]:
            return 1.25
        return 0.95

    elif category == "skin_care":
        # Antifungals & powders peak in humid monsoon (Jul-Aug ~1.60); lotions peak in dry winter (Dec-Jan ~1.40)
        if month in [6, 7, 8]:
            return 1.50
        elif month in [12, 1]:
            return 1.30
        return 0.90

    elif category in ["vitamins_supplements", "diabetic_care"]:
        # Stable year-round baseline with minor winter health-check lift
        if month in [1, 2]:
            return 1.10
        return 0.98

    elif category == "baby_care":
        # Relatively steady demand, slight summer dehydration / prickly heat lift
        if month in [4, 5, 6]:
            return 1.15
        return 1.00

    elif category == "first_aid":
        # Summer outdoor activities lift
        if month in [4, 5, 6]:
            return 1.20
        return 0.95

    return 1.0


def simulate_sales_and_batches(
    products_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    seed: int,
    missing_rate: float = 0.005,
    duplicate_rate: float = 0.002,
    outlier_rate: float = 0.003,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate daily sales and batch inventory under FEFO allocation."""
    rng = np.random.default_rng(seed)
    sales_rows: List[dict] = []
    batch_rows: List[dict] = []

    suppliers = ["Apex Pharma Ltd", "BioCare Distribution", "Zenith Health Logistics", "MedSource Direct", "Sanitas Wholesale"]

    # Active inventory tracker per product: list of dicts [ {batch_id, expiry_date, remaining_qty} ]
    inventory: Dict[str, List[dict]] = {row["product_id"]: [] for _, row in products_df.iterrows()}
    batch_counter = 1

    dates = calendar_df["date"].tolist()
    total_days = len(dates)

    # Initial batch seeding before start_date so inventory is ready on Day 1
    start_dt = datetime.strptime(dates[0], "%Y-%m-%d")
    for _, prod in products_df.iterrows():
        p_id = prod["product_id"]
        shelf_months = prod["shelf_life_months"]

        # Initial 2 batches with staggered expiry dates
        for b_idx in range(2):
            recv_date = start_dt - timedelta(days=int(rng.integers(15, 60)))
            shelf_days = int(shelf_months * 30.4)
            expiry_date = recv_date + timedelta(days=shelf_days)
            qty = int(prod["base_daily_demand"] * rng.integers(25, 45))
            b_id = f"BAT_{batch_counter:05d}"
            batch_counter += 1

            batch_record = {
                "batch_id": b_id,
                "product_id": p_id,
                "received_date": recv_date.strftime("%Y-%m-%d"),
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "quantity_received": qty,
                "unit_cost": prod["unit_cost"],
                "supplier": rng.choice(suppliers),
            }
            batch_rows.append(batch_record)
            inventory[p_id].append({
                "batch_id": b_id,
                "expiry_date": expiry_date,
                "remaining_qty": qty,
            })

    # Day-by-day simulation
    for day_idx, date_str in enumerate(dates):
        current_dt = datetime.strptime(date_str, "%Y-%m-%d")
        cal_row = calendar_df.iloc[day_idx]
        month = cal_row["month"]
        day_of_year = cal_row["day_of_year"]
        is_weekend = cal_row["is_weekend"]
        is_holiday = cal_row["is_holiday"]
        is_festival = cal_row["is_festival"]

        for _, prod in products_df.iterrows():
            p_id = prod["product_id"]
            cat = prod["category"]
            base_demand = prod["base_daily_demand"]
            intermittency = prod["intermittency_prob"]
            base_price = prod["base_price"]

            # Multipliers
            seas_mult = get_seasonal_multiplier(cat, month, day_of_year)
            wknd_mult = 1.25 if is_weekend else 0.95
            fest_mult = 1.30 if is_festival else (1.10 if is_holiday else 1.0)
            
            # Trend: gentle 3-8% annual expansion in category demand
            trend_mult = 1.0 + (day_idx / total_days) * 0.06

            # Promotion: 4% chance of promotion giving 10-20% discount and 35% demand boost
            has_promo = int(rng.random() < 0.04)
            if has_promo:
                promo_discount = rng.uniform(0.10, 0.20)
                selling_price = round(base_price * (1.0 - promo_discount), 2)
                promo_mult = 1.35
            else:
                selling_price = base_price
                promo_mult = 1.0

            # Intermittent zero-demand probability for slow products
            is_zero_day = rng.random() < intermittency

            if is_zero_day:
                latent_demand = 0.0
            else:
                expected_rate = base_demand * seas_mult * wknd_mult * fest_mult * trend_mult * promo_mult
                # Poisson / negative-binomial variation
                latent_demand = float(rng.poisson(lam=max(0.5, expected_rate)))

            # Stock check and FEFO consumption
            # Remove any batches that reached expiration on or before today
            inventory[p_id] = [b for b in inventory[p_id] if b["expiry_date"] > current_dt]
            # Sort batches by expiry_date (FEFO: First Expiry First Out)
            inventory[p_id].sort(key=lambda x: x["expiry_date"])

            total_stock_on_hand = sum(b["remaining_qty"] for b in inventory[p_id])

            if total_stock_on_hand <= 0:
                stockout_flag = 1
                observed_sales = 0.0
            else:
                stockout_flag = 0
                observed_sales = min(latent_demand, float(total_stock_on_hand))

                # Deduct sales using FEFO across active batches
                remaining_to_deduct = observed_sales
                for b in inventory[p_id]:
                    if remaining_to_deduct <= 0:
                        break
                    deduct = min(b["remaining_qty"], remaining_to_deduct)
                    b["remaining_qty"] -= deduct
                    remaining_to_deduct -= deduct

            # Reordering logic: if stock drops below 14 days of expected base demand, place order
            reorder_threshold = base_demand * 14.0
            remaining_stock_after_sales = sum(b["remaining_qty"] for b in inventory[p_id])

            if remaining_stock_after_sales < reorder_threshold:
                # Place new batch replenishment
                order_qty = int(base_demand * rng.integers(30, 60))
                shelf_days = int(prod["shelf_life_months"] * 30.4)
                # Occasional shorter shelf life batch delivered by supplier
                if rng.random() < 0.12:
                    shelf_days = int(shelf_days * rng.uniform(0.4, 0.7))

                expiry_dt = current_dt + timedelta(days=shelf_days)
                b_id = f"BAT_{batch_counter:05d}"
                batch_counter += 1

                batch_record = {
                    "batch_id": b_id,
                    "product_id": p_id,
                    "received_date": current_dt.strftime("%Y-%m-%d"),
                    "expiry_date": expiry_dt.strftime("%Y-%m-%d"),
                    "quantity_received": order_qty,
                    "unit_cost": prod["unit_cost"],
                    "supplier": rng.choice(suppliers),
                }
                batch_rows.append(batch_record)
                inventory[p_id].append({
                    "batch_id": b_id,
                    "expiry_date": expiry_dt,
                    "remaining_qty": order_qty,
                })

            sales_rows.append({
                "date": date_str,
                "product_id": p_id,
                "units_sold": observed_sales,
                "unconstrained_demand": latent_demand,
                "selling_price": selling_price,
                "is_promoted": has_promo,
                "stock_on_hand": total_stock_on_hand,
                "stockout_flag": stockout_flag,
            })

    sales_df = pd.DataFrame(sales_rows)
    batches_df = pd.DataFrame(batch_rows)

    # Inject controlled anomalies into raw sales data to require realistic cleaning
    # 1. Missing values
    missing_mask = rng.random(len(sales_df)) < missing_rate
    sales_df.loc[missing_mask, "selling_price"] = np.nan

    missing_sales_mask = rng.random(len(sales_df)) < (missing_rate * 0.5)
    sales_df.loc[missing_sales_mask, "units_sold"] = np.nan

    # 2. Outliers: transcription spikes
    outlier_mask = rng.random(len(sales_df)) < outlier_rate
    sales_df.loc[outlier_mask, "units_sold"] = sales_df.loc[outlier_mask, "units_sold"].fillna(10.0) * rng.uniform(8.0, 15.0)

    # 3. Impossible values: negative sales (e.g. keying error)
    impossible_idx = rng.choice(len(sales_df), size=int(len(sales_df) * 0.001), replace=False)
    sales_df.loc[impossible_idx, "units_sold"] = -1.0 * rng.integers(1, 10, size=len(impossible_idx))

    # 4. Injected duplicates
    num_duplicates = int(len(sales_df) * duplicate_rate)
    duplicate_rows = sales_df.sample(n=num_duplicates, random_state=seed)
    sales_df = pd.concat([sales_df, duplicate_rows], ignore_index=True)

    # 5. Injected invalid batches: 4 batches with expiry before received date
    corrupt_batch_idx = rng.choice(len(batches_df), size=4, replace=False)
    for idx in corrupt_batch_idx:
        rec_dt = datetime.strptime(batches_df.loc[idx, "received_date"], "%Y-%m-%d")
        batches_df.loc[idx, "expiry_date"] = (rec_dt - timedelta(days=int(rng.integers(5, 30)))).strftime("%Y-%m-%d")

    return sales_df, batches_df


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "configs" / "config.yaml"
    cfg = load_config(config_path)

    seed = cfg["project"]["random_seed"]
    start_date = cfg["data_generation"]["start_date"]
    end_date = cfg["data_generation"]["end_date"]
    missing_rate = cfg["data_generation"]["missing_rate"]
    duplicate_rate = cfg["data_generation"]["duplicate_rate"]
    outlier_rate = cfg["data_generation"]["outlier_rate"]

    raw_dir = repo_root / cfg["paths"]["raw_data_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating calendar data...")
    calendar_df = build_calendar(start_date, end_date)
    calendar_path = raw_dir / "calendar.csv"
    calendar_df.to_csv(calendar_path, index=False)
    logger.info(f"Saved calendar with {len(calendar_df)} days to {calendar_path}")

    logger.info("Generating products catalog...")
    products_df = build_products()
    products_path = raw_dir / "products.csv"
    products_df.to_csv(products_path, index=False)
    logger.info(f"Saved {len(products_df)} products to {products_path}")

    logger.info(f"Simulating daily sales and inventory batches (seed={seed})...")
    sales_df, batches_df = simulate_sales_and_batches(
        products_df=products_df,
        calendar_df=calendar_df,
        seed=seed,
        missing_rate=missing_rate,
        duplicate_rate=duplicate_rate,
        outlier_rate=outlier_rate,
    )

    sales_path = raw_dir / "daily_sales.csv"
    sales_df.to_csv(sales_path, index=False)
    logger.info(f"Saved {len(sales_df)} daily sales records to {sales_path}")

    batches_path = raw_dir / "batches.csv"
    batches_df.to_csv(batches_path, index=False)
    logger.info(f"Saved {len(batches_df)} batch records to {batches_path}")


if __name__ == "__main__":
    main()
