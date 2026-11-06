# Temporal Leakage Analysis and Prevention

## 1. Principles of Temporal Integrity
In time-series and inventory forecasting, **data leakage** occurs whenever information from the future (relative to the forecast origin $t$) is inadvertently incorporated into the feature matrix. 

In this system:
1. **Decision Time**: The end of day $t$. All operational replenishment and stock-rotation reviews are performed using information recorded up to and including date $t$.
2. **Forecast Origin**: Date $t$.
3. **Forecast Horizon**: $[t+1, t+28]$ for demand, and $[t+1, expiry\_date]$ for expiring inventory batches.
4. **Strict Temporal Boundary**: No feature vector $X_t$ may consume or depend upon any measurement $y_{t+k}$ for $k \ge 1$.

---

## 2. Demand Forecasting Feature Matrix Audit

| Feature Name | Computation Formula / Source | Temporal Window | Mathematical Safety Justification |
|---|---|---|---|
| `lag_1` | $s_{i, t}$ | Day $t$ | Units dispensed on day $t$. Available at end of day $t$. |
| `lag_7` | $s_{i, t-6}$ | Day $t-6$ | Exactly 7 days prior. Fully historical. |
| `lag_14` | $s_{i, t-13}$ | Day $t-13$ | Exactly 14 days prior. Fully historical. |
| `lag_28` | $s_{i, t-27}$ | Day $t-27$ | Exactly 28 days prior. Fully historical. |
| `rolling_mean_7` | $\frac{1}{7}\sum_{k=0}^{6} s_{i, t-k}$ | $[t-6, t]$ | Rolling backward window. Computed with `min_periods=1`. Uses strictly $\le t$. |
| `rolling_std_7` | $\text{std}(s_{i, t-6 \dots t})$ | $[t-6, t]$ | Backward sample standard deviation up to day $t$. |
| `rolling_mean_14` | $\frac{1}{14}\sum_{k=0}^{13} s_{i, t-k}$ | $[t-13, t]$ | Backward rolling mean over past 2 weeks up to day $t$. |
| `rolling_std_14` | $\text{std}(s_{i, t-13 \dots t})$ | $[t-13, t]$ | Backward rolling std over past 2 weeks up to day $t$. |
| `rolling_mean_28` | $\frac{1}{28}\sum_{k=0}^{27} s_{i, t-k}$ | $[t-27, t]$ | Backward rolling mean over past 4 weeks up to day $t$. |
| `rolling_std_28` | $\text{std}(s_{i, t-27 \dots t})$ | $[t-27, t]$ | Backward rolling std over past 4 weeks up to day $t$. |
| `rolling_max_28` | $\max(s_{i, t-27 \dots t})$ | $[t-27, t]$ | Peak daily sales in preceding 28 days. |
| `days_since_last_sale` | $t - \max\{\tau \le t : s_{i, \tau} > 0\}$ | $(-\infty, t]$ | Evaluated sequentially forward. Only checks historical sales on or before $t$. |
| `selling_price` | Price on day $t$ | Day $t$ | Active retail price at day $t$. |
| `price_ratio` | $\text{price}_t / \text{base\_price}$ | Day $t$ | Static catalog price denominator; numerator is price at day $t$. |
| `is_promoted` | Promotion flag at day $t$ | Day $t$ | Known promotional schedule at day $t$. |
| `promo_rolling_28` | $\frac{1}{28}\sum_{k=0}^{27} \text{promo}_{i, t-k}$ | $[t-27, t]$ | Historical promotion frequency over past 4 weeks. |
| `stock_on_hand` | Available inventory at day $t$ | Day $t$ | Recorded inventory level at day $t$. |
| `stockout_flag` | $\mathbb{I}(\text{stock}_t \le 0)$ | Day $t$ | Binary flag recorded at day $t$. |
| `stockout_rolling_28`| $\frac{1}{28}\sum_{k=0}^{27} \text{stockout}_{i, t-k}$| $[t-27, t]$ | Frequency of stockout occurrences in past 28 days. |
| `current_stock_cover`| $\text{stock}_t / (\text{rolling\_mean\_28} + 10^{-4})$ | $[t-27, t]$ | Ratio of current stock to past 28-day historical velocity. |
| `day_of_week`, `month`| Calendar timestamp of day $t$ | Day $t$ | Deterministic calendar attributes of date $t$. |
| `sin_day_of_year`, etc.| Cyclical transform of day $t$ | Day $t$ | Trigonometric transformation of calendar date $t$. |
| `is_weekend`, `is_holiday`| Calendar flags of day $t$ | Day $t$ | Official national holidays and weekend indicators. |

---

## 3. Expiry-Risk Classification Feature Matrix Audit

| Feature Name | Computation Formula / Source | Temporal Window | Mathematical Safety Justification |
|---|---|---|---|
| `days_to_expiry` | $(\text{expiry\_date} - t).\text{days}$ | Snapshot $t$ | Fixed expiration date on batch label minus snapshot date $t$. |
| `remaining_qty` | Inventory remaining at end of day $t$ | $\le t$ | Evaluated from receipt date up to day $t$ under FEFO. |
| `batch_age_days` | $(t - \text{received\_date}).\text{days}$ | $\le t$ | Days elapsed since receipt. |
| `fraction_depleted` | $1 - (\text{rem\_qty}_t / \text{qty\_received})$ | $\le t$ | Fraction of batch sold up to snapshot date $t$. |
| `earlier_batches_qty`| Sum of remaining stock in batches with earlier expiry | $\le t$ | Strictly inventory remaining at day $t$ in preceding FEFO queue. |
| `fefo_queue_position`| Rank of batch among active batches of same product | $\le t$ | Rank ordered by expiry date for batches active at day $t$. |
| `product_daily_velocity`| Past 28-day mean daily sales | $[t-27, t]$ | Rolling historical consumption rate prior to snapshot date $t$. |
| `expected_demand_to_expiry`| $\text{velocity} \times \text{days\_to\_expiry}$ | Historical velocity extrapolated forward | Point estimate based strictly on historical rate; does not peek into future sales. |
| `allocation_share` | Share of expected demand assigned to this batch | $\le t$ | Derived from `earlier_batches_qty` and `expected_demand_to_expiry`. |
| `risk_rule_flag` | Heuristic threshold rule | $\le t$ | Evaluated strictly using current remaining stock vs historical expected runout. |
| `stock_cover_days` | $\text{rem\_qty}_t / \text{daily\_velocity}$ | $\le t$ | Operational ratio of stock to past velocity. |
| `will_expire_unsold` | Ground-truth binary label ($\mathbb{I}(\text{actual\_unsold} > 5.0)$) | $[t+1, \text{expiry}]$ | **Target label only**. Used strictly for loss computation and evaluation. **Never included in the feature set $X$**. |

---

## 4. Automated Leakage Verification Tests
To mathematically guarantee no feature leaks future information, we implement automated tests in `tests/test_leakage.py`:
1. **Future Perturbation Invariance Test**:
   - For an arbitrary test date $t$, we compute feature row $X_t$.
   - We then arbitrarily alter future sales for all dates $t+k$ ($k \in [1, 50]$) by multiplying sales by a random factor of $100\times$.
   - We recompute the feature extraction pipeline on the perturbed dataset.
   - We assert that $X_t^{\text{perturbed}} \equiv X_t^{\text{original}}$ to float precision ($10^{-9}$).
2. **Label Exclusion Test**:
   - We assert that neither `target_demand_28d` nor `will_expire_unsold` nor `unconstrained_demand` appears in any candidate model feature list.
3. **Split Buffer Verification**:
   - We assert that the chronological gap between train, validation, and test splits is at least 28 days (equal to the forecasting horizon), ensuring no forward target overlapping across partitions.
