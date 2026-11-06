"""Build the 01_eda.ipynb notebook programmatically."""

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01 - Exploratory Data Analysis (EDA)\n",
            "## Pharmacy Medicine Demand & Expiry-Risk Forecasting\n",
            "\n",
            "This notebook presents the exploratory data analysis of 2 years of daily retail pharmacy sales and batch inventory across 60 products in 10 categories.\n",
            "All data is synthetically simulated based on Indian seasonality patterns, weekly retail cycles, stockout censoring, and FEFO inventory depletion."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from IPython.display import Image, display\n",
            "\n",
            "DATA_DIR = Path('../data/processed')\n",
            "FIG_DIR = Path('../reports/figures')\n",
            "\n",
            "sales_df = pd.read_csv(DATA_DIR / 'daily_sales.csv')\n",
            "products_df = pd.read_csv(DATA_DIR / 'products.csv')\n",
            "calendar_df = pd.read_csv(DATA_DIR / 'calendar.csv')\n",
            "batches_df = pd.read_csv(DATA_DIR / 'batches.csv')\n",
            "\n",
            "merged_df = sales_df.merge(products_df, on='product_id', how='left').merge(calendar_df, on='date', how='left')\n",
            "print(f'Sales records: {len(sales_df):,}')\n",
            "print(f'Products count: {len(products_df)}')\n",
            "print(f'Batches count: {len(batches_df)}')\n",
            "print(f'Calendar days: {len(calendar_df)}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Overall Sales Distribution & Zero Inflation\n",
            "Examining the empirical distribution of daily product sales to identify skewness and zero proportions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "zero_pct = (sales_df['units_sold'] == 0).mean() * 100\n",
            "print(f\"Median daily sales: {sales_df['units_sold'].median():.2f}\")\n",
            "print(f\"Mean daily sales: {sales_df['units_sold'].mean():.2f}\")\n",
            "print(f\"95th percentile: {sales_df['units_sold'].quantile(0.95):.2f}\")\n",
            "print(f\"Zero-sales percentage: {zero_pct:.2f}%\")\n",
            "display(Image(filename=str(FIG_DIR / 'fig01_sales_distribution.png')))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Category Comparisons\n",
            "Evaluating baseline velocity differences across the 10 therapeutic categories."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / 'fig02_category_comparison.png')))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Meteorological Seasonality in India\n",
            "Visualizing seasonal peaks: winter cough/cold, monsoon fever and stomach infections, spring allergy."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / 'fig03_seasonality_patterns.png')))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. Day-of-Week Cycles (Weekend Effect)\n",
            "Retail pharmacy sales exhibit notable weekend volume expansion compared to mid-week dispensing."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / 'fig04_day_of_week_pattern.png')))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. Zero-Inflation by Product Velocity\n",
            "Fast-moving lines rarely record zero-demand days, while slow-moving diabetic kits and baby monitors experience intermittent demand."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / 'fig05_zero_inflation_intermittency.png')))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 6. Stockout Impact & Censored Demand\n",
            "When stock is exhausted, observed sales fall to zero regardless of customer intent, creating a downward bias in raw historical demand."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / 'fig06_stockout_impact.png')))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 7. Batch Shelf Life vs. Expiry Risk\n",
            "Examining received quantities versus manufacturing shelf life to understand batch expiration exposure."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / 'fig07_batch_shelflife_sellthrough.png')))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 8. Product Pareto: Top 10 vs. Bottom 10 Products\n",
            "High volume concentrates in OTC analgesics and antipyretics, whereas specialized items represent long-tail sales."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / 'fig08_top_bottom_products.png')))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9. Feature Correlation Heatmap\n",
            "Evaluating correlations between price, promotions, calendar flags, and daily unit sales."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 10,
        "metadata": {},
        "outputs": [],
        "source": [
            "display(Image(filename=str(FIG_DIR / 'fig09_correlation_heatmap.png')))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 10. Summary Takeaways for Modeling\n",
            "1. **Lag & Rolling Features**: Strong day-of-week seasonality necessitates 7-day and 14-day lags.\n",
            "2. **Calendar Flags**: Indian holiday and festival indicators provide key demand lift signals.\n",
            "3. **Stockout Awareness**: Stockout indicator features prevent models from misinterpreting inventory shortages as structural demand drops.\n",
            "4. **Uncertainty Bands**: High skewness and intermittency justify quantile regression (p10, p50, p90) over mean-only forecasting."
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.11.9"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path(__file__).resolve().parent / "01_eda.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated {out_path}")
