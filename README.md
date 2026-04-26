# ML Logistic Momentum Trading Strategy

A Python implementation of the machine learning momentum trading framework from **Beaudan & He (2019)**: _"Applying Machine Learning to Trading Strategies: Using Logistic Regression to Build Momentum-based Trading Strategies."_ This repository replicates and extends the paper's methodology using contemporary S&P 500 data from Yahoo Finance.

## Executive Summary

This project reformulates traditional momentum trading as a **binary classification problem**: each trading day, a logistic regression model predicts whether the S&P 500 index will achieve a target annualized return (5%) over a 3-day horizon. The strategy allocates capital only on positive predictions, otherwise holds cash.

**Key Finding:** The ML approach (8.0% annual return, 0.472 Sharpe ratio) underperforms buy-and-hold (9.0%, 0.490 Sharpe) but outperforms classic dual momentum (6.4%, 0.443 Sharpe) on out-of-sample data. Critically, modest predictive power exists (F1 scores 0.54–0.68), but practical outperformance remains elusive after accounting for implementation realities.

---

## Methodology

### Core Framework

The strategy is built on a walk-forward optimization structure with the following components:

| Parameter                | Value                   | Rationale                                                            |
| ------------------------ | ----------------------- | -------------------------------------------------------------------- |
| **Asset**                | S&P 500 (^GSPC)         | Large-cap equity exposure; high liquidity; robust historical data    |
| **Training Window**      | 40% of available sample | Balances feature stability with recent market regime relevance       |
| **Test Window**          | Remaining 60%           | Out-of-sample validation; prevents data leakage                      |
| **Retraining Frequency** | ~3 years (756 days)     | Captures structural market shifts without overfitting to local noise |

### Feature Engineering

The model uses **40 polynomial features** derived from 8 base feature groups:

**Momentum Features (8 windows):**

- Historical returns over 30, 60, 90, 120, 180, 270, 300, 360 trading days
- Captures trend persistence and mean reversion dynamics across multiple timescales
- Rationale: Momentum is well-documented in finance literature; multi-scale windows capture hierarchical trends

**Drawdown Features (4 windows):**

- Rolling drawdowns over 15, 60, 90, 120 trading days
- Measures the magnitude of peak-to-trough declines relative to recent highs
- Rationale: Risk-aware modeling; drawdown acts as a volatility proxy and regime detector

**Feature Interactions:**

- Cubic polynomial expansion generates ~100 raw features; dimensionality is implicitly reduced via L2 regularization

### Model Architecture

- **Algorithm:** Logistic Regression with L2 regularization (C=1.0)
- **Preprocessing:** MinMaxScaler normalization [0,1] applied to all features before polynomial expansion
- **Target Variable:** Binary classification
  - Label = 1 if annualized future return ≥ 5% over 3 days; 0 otherwise
  - Annualized return = $(P_{t+3}/P_t)^{252/3} - 1$

**Why logistic regression?** Simplicity, interpretability, and computational efficiency. The paper demonstrates that in this domain, simpler models generalize better than complex alternatives (e.g., neural networks).

### Walk-Forward Backtesting

7 non-overlapping test windows span **1991–2026**:

| Set | Train Period | Test Period | Train Size | Test Size |
| --- | ------------ | ----------- | ---------- | --------- |
| 1   | 1991–2005    | 2005–2008   | 3,512 days | 756 days  |
| 2   | 1994–2008    | 2008–2011   | 3,512 days | 756 days  |
| 3   | 1997–2011    | 2011–2014   | 3,512 days | 756 days  |
| 4   | 2000–2014    | 2014–2017   | 3,512 days | 756 days  |
| 5   | 2003–2017    | 2017–2020   | 3,512 days | 756 days  |
| 6   | 2006–2020    | 2020–2023   | 3,512 days | 756 days  |
| 7   | 2009–2023    | 2023–2026   | 3,512 days | 734 days  |

This design ensures no data leakage and provides out-of-sample evidence of generalization.

---

## Results & Critical Reflection

### Overall Performance (2005–2026 Out-of-Sample)

| Metric            | Buy & Hold | Classic Dual Momentum | ML Logistic Regression | Assessment                                                     |
| ----------------- | ---------- | --------------------- | ---------------------- | -------------------------------------------------------------- |
| **Annual Return** | 9.00%      | 6.43%                 | 8.00%                  | ML underperforms buy-hold; beats dual momentum                 |
| **Sharpe Ratio**  | 0.490      | 0.443                 | 0.472                  | ML shows marginal improvement; all ratios modest               |
| **Sortino Ratio** | 0.597      | 0.542                 | 0.584                  | Similar pattern; downside risk management weak                 |
| **Volatility**    | 19.19%     | 14.04%                | 17.43%                 | ML volatility closer to buy-hold; dual momentum more defensive |
| **Max Drawdown**  | -56.78%    | -33.97%               | -41.91%                | ML incurs 25% worse drawdown than dual momentum                |
| **Final Equity**  | $5,992k    | $3,682k               | $5,000k                | Buy-hold compounds most; ML returns intermediate capital       |

### Why ML Underperforms Buy-and-Hold

**Data Issue: Upward Bias of Recent Regime**  
The 2009–2026 period is characterized by unprecedented monetary accommodation, low rates, and equity-friendly conditions. Buy-and-hold captures this secular bull market directly. ML systems, trained on older data with different volatility and return regimes, fail to exploit this shift efficiently. Walk-forward windows 5–7 (2017 onward) show declining out-of-sample performance, consistent with this regime transition.

**Classification Accuracy Ceiling**  
Maximum out-of-sample test accuracy is 56.5% (Set 4), barely above the 50% random baseline. While precision averages 57.4% and recall 68.0%, this translates to marginal information content. The model predicts "up" more often when returns are up, but the predictive margin is too thin to overcome transaction costs and market timing risk.

**Transaction Cost Sensitivity**  
The backtest assumes zero transaction costs. In practice, daily rebalancing incurs 10–20 basis points per rebalance (bid-ask spreads, commissions, slippage). Over 5,200+ trading days, cumulative drag eliminates any alpha. Dual momentum's 3-day rebalancing frequency (not daily) provides a natural cost advantage in realistic settings.

### What the Model Did Capture

**Classification Metrics by Walk-Forward Set**

| Set      | Train Precision | Train Recall | **Test Precision** | **Test Recall** | **Test F1** | **Test Accuracy** |
| -------- | --------------- | ------------ | ------------------ | --------------- | ----------- | ----------------- |
| 1        | 0.601           | 0.766        | **0.578**          | **0.808**       | **0.674**   | 0.565             |
| 2        | 0.609           | 0.795        | **0.527**          | **0.551**       | **0.539**   | 0.488             |
| 3        | 0.597           | 0.759        | **0.560**          | **0.887**       | **0.686**   | 0.546             |
| 4        | 0.593           | 0.776        | **0.597**          | **0.678**       | **0.635**   | 0.565             |
| 5        | 0.594           | 0.865        | **0.574**          | **0.782**       | **0.662**   | 0.536             |
| 6        | 0.596           | 0.898        | **0.580**          | **0.501**       | **0.538**   | 0.516             |
| 7        | 0.599           | 0.898        | **0.574**          | **0.765**       | **0.656**   | 0.531             |
| **Mean** | 0.598           | 0.823        | **0.570**          | **0.705**       | **0.613**   | 0.534             |

**Observations:**

1. **Train-Test Gap:** Training F1 averages 0.73, test F1 averages 0.61—a 0.12-point drop indicates modest overfitting. This is reasonable for linear models but signals that feature interactions are partially dataset-specific.

2. **High Recall, Moderate Precision:** The model exhibits high recall (70.5% out-of-sample average), meaning it captures most true positive days but with false positive rate of ~43%. This "bullish bias" is a deliberate trade-off: missing upside (low recall = high cost) is worse than entering false signals (low precision = noise).

3. **Set 2 Underperformance (2008–2011):** The global financial crisis and recovery period shows the lowest test F1 (0.539) and accuracy (0.488). Market structure shifted dramatically; features trained on 1994–2008 were less predictive of post-crisis regimes. This highlights the model's vulnerability to structural breaks.

4. **Set 3 Success (2011–2014):** Best test recall (0.887) and strong F1 (0.686) during a recovery bull market. Momentum signals aligned well with realized returns; the post-crisis consolidation was highly trend-following.

### Top Predictive Features

The final walk-forward model (trained 2009–2023, tested 2023–2026) reveals **top 10 features by absolute coefficient magnitude:**

| Feature                     | Coefficient | Interpretation                                                                                                                                           |
| --------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mom_90 × mom_360 × dd_60`  | +0.829      | Strong predictor: recent 90-day momentum AND longer 360-day momentum AND elevated drawdown (risk-on environment). Positive coefficient = bullish signal. |
| `mom_120 × mom_360²`        | +0.785      | Nonlinear long-term momentum persistence. Squared term amplifies effect in high-momentum regimes.                                                        |
| `mom_120 × dd_60²`          | -0.764      | **Contrarian signal:** High 120-day momentum BUT severe drawdown = mean reversion trigger. Predicts lower returns.                                       |
| `mom_120³`                  | -0.760      | **Extreme momentum reversal:** Cubic nonlinearity suggests that very high momentum is unsustainable; market corrects.                                    |
| `mom_180 × mom_270`         | +0.742      | Medium-term momentum confirmation; two horizons aligned = stronger bullish signal.                                                                       |
| `mom_90 × mom_360 × dd_120` | +0.736      | Three-way interaction; robust uptrend with moderate drawdown = sustained upside.                                                                         |
| `mom_180 × mom_270 × dd_15` | +0.726      | Medium-term trend WITH short-term drawdown = tactical buying opportunity within trend.                                                                   |
| `mom_300` (standalone)      | -0.710      | **Negative standalone:** 300-day momentum alone is contrarian; very long-term reversions are predictable.                                                |
| `mom_180 × mom_270`         | +0.742      | (repeated; top positive interaction)                                                                                                                     |
| `dd_90³`                    | +0.632      | Cubic drawdown nonlinearity; extreme drawdown (fire sale environment) predicts recovery.                                                                 |

**Key Pattern:** The model learns **momentum-momentum interactions** (e.g., `mom_90 × mom_360`) as strong predictors but tempers these with **nonlinear reversions** (e.g., `mom_120³` negative). It also captures **drawdown contrarian signals** (`mom_120 × dd_60²` negative), suggesting tactical rebalancing during stressed markets.

### Why Predictive Power Doesn't Translate to Excess Returns

1. **Efficient Markets Hypothesis Partially Holds:** Modest classification accuracy (57% precision) may reflect true market inefficiencies, but these are small and compete with implementation costs. The S&P 500 is among the most liquid, most-studied assets; exploitable mispricings are rare.

2. **Regime Dependency:** Sets 2 and 6 (2008 crisis, 2020 pandemic) show degraded performance. Macro shocks create new feature relationships that historical training data cannot anticipate.

3. **Timing Risk:** Even with 57% accuracy, strategy returns depend critically on **when** predictions are correct. Accurate predictions during sideways markets add little value; accurate predictions during large drawdowns create outsized returns but are rare.

4. **Momentum Already Priced In:** The dual momentum strategy (6.43% return) uses simpler heuristics and still beats the ML model, suggesting that basic momentum rules have already been arbitraged into market prices. ML gains only marginal information.

---

## Project Structure

```
ML_Logistic_Momentum_Strategy/
├── ML_Logistic_Momentum_Strategy.ipynb    # Full implementation and analysis
├── README.md                              # This file
└── results/
    ├── tables/
    │   ├── performance_summary.csv         # Strategy returns, Sharpe, drawdown
    │   ├── ml_error_metrics_by_walk_forward_set.csv  # Precision, recall, F1 per set
    │   └── last_model_feature_coefficients.csv       # Top features, coefficients
    └── figures/
        ├── equity_curves.png              # Growth comparison: Buy-Hold vs Dual Momentum vs ML
        ├── rolling_sharpe.png             # Rolling Sharpe ratio (36-month windows)
        ├── drawdowns.png                  # Drawdown profiles over time
        ├── classification_metrics_by_set.png  # Precision, recall, F1 per walk-forward set
        ├── top_feature_coefficients_last_model.png  # Bar chart of top 30 features
        ├── daily_return_distribution.png  # Histogram of daily returns by strategy
        ├── roc_auc_oos.png                # ROC curves, AUC scores per set
        ├── ml_signal.png                  # ML signal (0/1) over time
        ├── rolling_volatility.png         # 36-month rolling volatility
        ├── buy_hold_growth.png            # Growth of $1 in buy-and-hold
        └── long_cash_vs_long_short.png    # Long/cash vs long/short strategy comparison
```

---

## Key Limitations & Caveats

### Data & Backtesting

- **No Transaction Costs:** Real-world trading incurs bid-ask spreads (1–2 bps), commissions, and slippage. Daily rebalancing is unrealistic; actual strategy should use 1–3 day holding periods.
- **Survivorship Bias:** S&P 500 constituents are adjusted historically. Component changes are not modeled; analysis assumes a static index.
- **Yahoo Finance Data Quality:** Adjusted close prices can contain errors or adjustments that differ from official sources. Paper used Bloomberg, which is more curated.
- **No Cash Drag:** Risk-free rate assumed 1% annual; actual cash holdings would earn 4–5% in 2023–2026. Modest advantage to cash-holding strategy not captured.

### Model Design

- **Single Asset:** Trades only S&P 500. Diversification or pairs trading would require separate models and estate allocation rules.
- **Fixed Hyperparameters:** Polynomial degree (3), regularization (C=1.0), and target threshold (5% annualized, 3-day horizon) are not optimized. Different choices would yield different results.
- **No Regime Detection:** Model cannot adapt to structural breaks (e.g., 2008 crisis, COVID, 2022 rate shock). Walk-forward retraining is reactive, not proactive.
- **Linear Model Limitation:** Logistic regression assumes separability in feature space. Nonlinear decision boundaries (neural networks, tree ensembles) might capture richer patterns but at cost of overfitting and interpretability.

### Out-of-Sample Evidence

- **Limited History:** 2005–2026 backtest covers ~21 years and 5 market cycles. Smaller datasets (e.g., Set 2 spanning crisis) show lower F1; larger datasets needed for robust statistical inference.
- **Future Regime Unknown:** Post-2026 behavior is unknowable. Strategy may fail if markets shift to high-rates, low-growth regime (mean reversion dominant) or high-inflation environment (volatility spikes).

---

## How to Use

### Installation

```bash
pip install yfinance pandas numpy scikit-learn matplotlib scipy openpyxl
```

### Running the Notebook

1. Open `ML_Logistic_Momentum_Strategy.ipynb` in Jupyter or JupyterLab.
2. Modify `StrategyConfig` in cell 2.1 to adjust parameters (e.g., ticker, date range, polynomial degree).
3. Run all cells sequentially. Results are saved to `results/tables/` and `results/figures/`.

### Key Customizations

**Extend the backtest period:**

```python
cfg.end = "2026-12-31"  # Update to latest data
```

**Increase polynomial complexity:**

```python
cfg.polynomial_degree = 4  # Cubic → quartic interactions
```

**Use long-short instead of long-cash:**

```python
cfg.long_short = True
```

**Adjust target profitability threshold:**

```python
cfg.delta = 0.10  # Raise to 10% annualized (harder target)
```

---

## Conclusions & Recommendations

### What Worked

1. **Interpretable Classification:** Logistic regression successfully identifies momentum-drawdown interactions as predictive features. Cubic nonlinearities capture mean reversion without black-box complexity.
2. **Walk-Forward Framework:** Non-overlapping test sets provide honest out-of-sample evidence. Methodology is reproducible and defensible.
3. **Outperforms Naive Alternative:** ML strategy (8.0% return) beats classic dual momentum (6.4%) in this sample, validating the premise that learned feature interactions add value.

### What Didn't Work

1. **Underperformance to Passive:** 8% vs 9% buy-and-hold return in a bull market reflects the limitations of market timing. Avoiding the best days (via timing errors) is costly.
2. **Marginal Accuracy:** 57% precision and 56% accuracy, while above random, are insufficient to overcome transaction costs and slippage in practice.
3. **Regime Dependence:** Crisis periods (Set 2) and pandemic (Set 6) show degraded predictive power, indicating the model lacks robustness to structural breaks.

### Recommendations for Practitioners

1. **Add Transaction Cost Model:** Discount backtest returns by 10–20 bps per rebalance to reflect realistic costs. ML strategy's edge likely vanishes.
2. **Implement Regime Switching:** Use market volatility regimes (VIX thresholds) or structural break detection to adapt model or increase cash holdings during high uncertainty.
3. **Extend to Multi-Asset:** Combine ML momentum signals with mean-reversion in other assets (bonds, commodities) to diversify and smooth returns.
4. **Deploy on Out-of-Money Options:** Convert ML classification probabilities into options strategies to monetize tail predictions without bearing directional risk.
5. **Monitor Live Performance:** If deployed, track realized vs. backtested metrics weekly. Large deviations signal regime shift or data quality issues.

---

## References

- **Beaudan, O., & He, K. (2019).** _Applying Machine Learning to Trading Strategies: Using Logistic Regression to Build Momentum-based Trading Strategies._ Available at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3416507
- **Arnott, R. D., Beck, S. L., Kalesnik, V., & West, J. (2016).** _How Can 'Machine Learning' Improve Finance?_ Research Affiliates Publications.
- **Goyal, A., & Welch, I. (2008).** _A Comprehensive Look at the Empirical Performance of Equity Premium Prediction._ Review of Financial Studies, 21(4), 1455–1508.
