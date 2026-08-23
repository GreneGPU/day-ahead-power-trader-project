# Project Summary

## What This Is

`Day-Ahead Power Trader Project` is a portfolio-grade extension of the master-thesis notebook. It converts the one-cell notebook into a reusable Python project for 15-minute day-ahead DK1 electricity-price forecasting, transfer-learning evaluation, and forecast-driven flexibility simulation.

## Validation Setup

The project now supports two chronological test split methods:

```text
ratio         Use a fixed percentage split, such as the first 85% for training.
last_n_days   Use the final N calendar days of the dataset as the test period.
```

`configs/jakob-local.json` is set to:

```json
"test_split_method": "last_n_days",
"test_period_days": 30
```

With the current 15-minute thesis dataset, this produces:

```text
Train: 2025-10-09 through 2026-02-02 22:30:00
Test:  2026-02-02 22:45:00 through 2026-03-04 22:45:00
Rows:  2,881 test intervals, 100% 15-minute coverage
```

## Final-30-Day Holdout Result

The refactored pipeline was run successfully against the local thesis datasets using the final month as the out-of-sample period.

Best model by MAE:

```text
Hourly baseline only
MAE   17.1621
RMSE  23.6360
sMAPE 18.3946
R2     0.4802
```

Key comparison:

```text
Hourly baseline only          MAE 17.1621  RMSE 23.6360  R2 0.4802
TL residual pure stacking     MAE 17.9628  RMSE 24.6335  R2 0.4355
TL residual simple average    MAE 19.4379  RMSE 27.1662  R2 0.3134
Direct 15-min pure stacking   MAE 19.8300  RMSE 27.3065  R2 0.3063
```

Interpretation: the final-month holdout is a stricter and more recent robustness check. On this particular month, the hourly baseline is strongest by MAE, while the transfer-learning models remain useful comparison candidates and strategy inputs.

## Broader Full-Run Result

The earlier ratio-split run remains available in `outputs/model_run`.

Best model by MAE:

```text
TL residual simple average
MAE   17.0951
RMSE  24.6850
sMAPE 19.0252
R2     0.4911
```

Key comparison:

```text
TL residual simple average  MAE 17.0951
Hourly baseline only        MAE 18.5318
Direct 15-min XGBoost       MAE 19.0388
Direct 15-min pure stacking MAE 21.2197
```

Baseline-feature ablation:

```text
With hourly baseline as residual feature     MAE 17.0951  RMSE 24.6850  R2 0.4911
Without hourly baseline as residual feature  MAE 17.0965  RMSE 23.9013  R2 0.5229
```

Interpretation: the explicit hourly-baseline feature gives the smallest MAE by a very narrow margin, while the no-baseline-feature residual model has better RMSE/R2. Both ablation variants select the TL residual simple average as the best model.

## Generated Artifacts

- Last-30-day forecasts: `outputs/model_run_last30/transfer_15min_forecasts_detailed.csv`
- Last-30-day metrics: `outputs/model_run_last30/transfer_15min_metrics_detailed.csv`
- Last-30-day run summary: `outputs/model_run_last30/run_summary.json`
- Last-30-day dashboard: `reports/day_ahead_power_trader_project_dashboard_last30.html`
- Last-30-day report: `reports/day_ahead_power_trader_project_report_last30.md`
- Original full-run forecasts: `outputs/model_run/transfer_15min_forecasts_detailed.csv`
- Original full-run metrics: `outputs/model_run/transfer_15min_metrics_detailed.csv`
- Original full-run dashboard: `reports/intraday_power_quant_dashboard_model_run.html`
- Original full-run report: `reports/intraday_power_quant_report_model_run.md`
- Ablation comparison: `outputs/ablation/baseline_feature_ablation.csv`

## Strategy Suite

The dashboard supports ten strategy families:

```text
Forecast quantile              Charge in low forecast quantiles, discharge in high forecast quantiles
Weekly average band            Last-week moving average; charge below MA - X, discharge above MA + X
Forecast edge                  Trade the 15-minute forecast edge versus the hourly baseline
Volatility filtered average    Only trade below/above rolling average when rolling std is high
Daily spread rank              Rank each day's curve and optionally require a minimum daily spread
Ensemble agreement             Trade only when enough ensemble members agree on cheap/expensive intervals
Mean reversion                 Use rolling forecast z-score extremes as charge/discharge signals
Momentum                       Charge on downward forecast momentum, discharge on upward forecast momentum
Momentum spread                Combine daily spread rank with momentum confirmation
Channel breakout               Trade breaks outside recent rolling high/low forecast channels
```

All strategies use forecast prices for dispatch decisions and realized prices for cashflow. The available price data is day-ahead data, so the project is framed as day-ahead forecast-to-dispatch research rather than true intraday order-book execution.

The dashboard/report also includes an optimization sweep. For each strategy family, it tests a compact grid of important parameters and displays the best cashflow setting, max drawdown, active intervals, and number of parameter runs. This is designed as a transparent research backtest rather than a live-trading promise.

Additional CV-grade research checks now included:

```text
Risk-adjusted ranking       Ranks settings by cashflow minus max drawdown, Calmar, Sharpe-style, Sortino-style, profit factor, and win rate
Sharpe-style ranking        Annualized daily cashflow Sharpe proxy for strategy comparison
Walk-forward validation     Optimizes on one rolling period and tests on the next unseen period
Execution-cost stress       Uses fee per MWh as a slippage/transaction-cost proxy
Robustness grid             Shows whether nearby daily-spread parameters also perform well
Regime analysis             Splits performance by spread, price, weekday/weekend, and hour bucket
Uncertainty diagnostics     Residual bias/std and post-hoc 80% interval coverage
Latest decision output      Shows the latest action, dispatch, SOC, forecast, and signal reason
```

`Daily spread rank` can now be expressed directly as:

```text
Charge when daily rank <= X
Discharge when daily rank >= Y
```

Daily rank `1` is the cheapest forecast interval of the day.

The daily spread rule can also require a minimum daily forecast spread before any daily rank trades are allowed.

`Ensemble agreement` uses the available transfer-learning ensemble members first. The key controls are the cheap/expensive quantiles, the minimum model-agreement share, and an optional maximum model-spread filter.

`Forecast edge` is a thesis-specific diagnostic strategy: it tests whether the 15-minute forecast has tradable edge versus the hourly baseline. On the current final-month holdout, the default threshold is not profitable, which is useful evidence rather than a problem to hide.

`Volatility filtered average` combines regime filtering and price level: it only trades when rolling forecast std is above the configured threshold, then buys below the rolling average and sells above it.

`Momentum spread` requires spread and direction to agree: cheap daily ranks must also have falling forecast momentum, and expensive daily ranks must also have rising forecast momentum.

Current last-30-day strategy results:

```text
Daily spread rank              cashflow 106,561.62  max drawdown 10,542.34  active intervals 1030
Momentum spread                cashflow  98,338.23  max drawdown 10,542.34  active intervals 947
Momentum                       cashflow  94,447.87  max drawdown 12,654.52  active intervals 1578
Forecast quantile              cashflow  59,551.76  max drawdown  9,637.40  active intervals 388
Ensemble agreement             cashflow  58,053.04  max drawdown  9,624.39  active intervals 373
Weekly average band            cashflow  39,136.40  max drawdown 10,480.32  active intervals 369
Mean reversion                 cashflow  32,843.85  max drawdown 10,437.73  active intervals 317
Volatility filtered average    cashflow  17,415.29  max drawdown 12,308.43  active intervals 455
Channel breakout               cashflow   8,294.37  max drawdown  7,529.19  active intervals 77
Forecast edge                  cashflow -43,431.96  max drawdown 47,928.20  active intervals 489
```

## Verification

- Source data resolved from `configs/jakob-local.json`
- Hourly data: 47,066 rows, 79 columns, no duplicate timestamps, no missing target
- 15-minute data: 14,108 rows, 77 columns, no duplicate timestamps, no missing target
- ML dependencies installed in `.venv`
- Final-30-day transfer-learning run completed
- Last-30-day static dashboard and markdown report generated
- Baseline-feature ablation completed
- Smoke tests passed

## Main Commands

```powershell
.venv\Scripts\python.exe -m intraday_power_quant.cli --config configs/jakob-local.json check-data
.venv\Scripts\python.exe -m intraday_power_quant.cli --config configs/jakob-local.json run --output-dir outputs\model_run_last30
.venv\Scripts\python.exe -m intraday_power_quant.cli --config configs/jakob-local.json report --results-dir outputs\model_run_last30 --output-html reports\day_ahead_power_trader_project_dashboard_last30.html --output-report reports\day_ahead_power_trader_project_report_last30.md
.venv\Scripts\streamlit.exe run src\intraday_power_quant\streamlit_app.py
```
