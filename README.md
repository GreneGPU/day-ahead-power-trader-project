# Day-Ahead Power Trader Project

Transfer-learning 15-minute day-ahead electricity-price forecasting, physical battery optimization, and synthetic proprietary-trading research for DK1/DK2-style power markets.

Live dashboard: https://greneportfolio.vercel.app/

The live dashboard offers two accounting setups. `Physical battery` defaults to 90% round-trip efficiency and a 2026 DK1 distribution-connected fee assumption: 115.41 DKK/MWh while charging and 10.71 DKK/MWh while discharging. `Prop proxy` maps buy/charge signals to long positions and sell/discharge signals to short positions for the next observed DK1 price move, with editable capital, position size, switching cost, and daily loss limit.

The prop setup is deliberately labeled as a synthetic research proxy. The thesis dataset does not contain historical financial-contract entry quotes, bid/ask spreads, margin, collateral, liquidity, or imbalance settlement, so its PnL is not presented as executable Nord Pool spot arbitrage.

Its default evaluation mode reserves the final six complete DK1 calendar days as a chronological holdout: every strategy searches its parameter grid for the highest net cashflow on all earlier available observations, then the dashboard ranks strategies, calculates daily Sharpe, and shows trade logs using only those six unseen days. A fixed-default mode remains available for comparison.

This repo turns the thesis notebook into a maintainable project:

- hourly source ensemble trained before the 15-minute transition
- minute-0 feature mapping from 15-minute rows into hourly feature space
- hourly baseline forecast merged back to all 15-minute intervals
- residual transfer-learning model
- direct 15-minute benchmark model
- configurable chronological validation split: fixed ratio or final N days
- model-ranking metrics, coverage checks, leakage warnings, and ablation entry point
- forecast-driven battery arbitrage simulator
- asset-free long/short research proxy with capital return, transaction costs, loss limits, and forced flat close
- separate perfect-foresight benchmarks for battery dispatch and synthetic directional positions
- selectable strategy suite with user-adjustable parameters
- live strategy selection with per-interval charge/discharge logs and CSV export
- DK1 actual-versus-prediction chart with charge and discharge execution markers
- empty battery at the start of every default simulation, avoiding free initial inventory
- predicted-best-hours strategy that pairs cheap forecast hours with later expensive hours only when the efficiency- and fee-adjusted paper spread is positive
- parameter sweep that finds the best cashflow setting for each strategy
- risk-adjusted strategy ranking, walk-forward validation, execution-cost stress testing, robustness grids, regime analysis, uncertainty diagnostics, and latest-decision output
- standalone HTML dashboard and markdown report

The code is research tooling, not financial advice or a production trading system. The current data is day-ahead price data, so execution-cost/slippage checks are proxy stress tests rather than true intraday order-book simulations.

## Project Structure

```text
intraday-power-quant/
|-- configs/
|   |-- default.json
|   `-- jakob-local.json
|-- data/
|   |-- raw/
|   `-- processed/
|-- reports/
|-- outputs/
|-- scripts/
|-- src/intraday_power_quant/
|   |-- cli.py
|   |-- config.py
|   |-- dashboard.py
|   |-- data.py
|   |-- evaluation.py
|   |-- experiments.py
|   |-- models.py
|   |-- optimization.py
|   |-- plots.py
|   |-- prop_trading.py
|   |-- research.py
|   |-- risk.py
|   |-- trading.py
|   |-- transfer.py
|   `-- validation.py
`-- tests/
```

## Data Inputs

The config resolves the source files from these candidate names:

```python
hourly_candidates = [
    "final_day_ahead_safe_modeling_dataset_hourly.csv",
    "final_day_ahead_safe_modeling_dataset.csv",
    "final_day_ahead_safe_modeling_dataset.xlsx",
]

min15_candidates = [
    "final_15min_day_ahead_safe_modeling_dataset.csv",
    "final_15min_day_ahead_safe_modeling_dataset.xlsx",
]
```

For this machine, `configs/jakob-local.json` points to the thesis folder on the Desktop. For a portfolio upload, place the data files under `data/raw/` or edit `configs/default.json`.

## Setup

From this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

The Vercel API installs the lightweight base dependencies only. For a local
editable install that also includes the model-training stack, use:

```powershell
.venv\Scripts\python -m pip install -e ".[training,dashboard,dev]"
```

If `python` is not on PATH, use the Python executable you normally use for the thesis environment.

## Testing Window

The local config is set to test on the final 30 days of the 15-minute dataset:

```json
"test_split_method": "last_n_days",
"test_period_days": 30
```

With the current local data, that means:

```text
Train: 2025-10-09 through 2026-02-02 22:30:00
Test:  2026-02-02 22:45:00 through 2026-03-04 22:45:00
Rows:  2,881 test intervals, 100% 15-minute coverage
```

To return to the previous ratio-based validation, set:

```json
"test_split_method": "ratio",
"split_ratio": 0.85
```

## Fast Path: Build Report From Existing Outputs

This works from saved forecast and metric files without retraining the ML models:

```powershell
.venv\Scripts\python -m intraday_power_quant.cli --config configs/jakob-local.json report
```

Outputs:

- `reports/day_ahead_power_trader_project_dashboard.html`
- `reports/day_ahead_power_trader_project_report.md`

## Dashboard Options

The static HTML dashboard needs no server and is the easiest artifact to share. A Streamlit app is also included for local exploration:

```powershell
.venv\Scripts\streamlit.exe run src\intraday_power_quant\streamlit_app.py
```

Use `outputs/model_run` as the results directory in the sidebar for the original full run. Use `outputs/model_run_last30` to inspect the final-month holdout run.

The Streamlit dashboard supports multiple dispatch strategies and day-ahead research checks:

- `Weekly average band`: uses a trailing moving average, normally the last week. Buy/charge when the forecast is `X` below the moving average, and sell/discharge when it is `X` above.
- `Forecast edge`: buy/charge when the 15-minute forecast is sufficiently below the hourly baseline, and sell/discharge when it is sufficiently above.
- `Volatility filtered average`: only trades the moving-average rule when rolling forecast volatility is high.
- `Forecast quantile`: buy/charge below the low forecast quantile and sell/discharge above the high forecast quantile.
- `Mean reversion`: uses a rolling forecast z-score and trades forecast extremes back toward the recent mean.
- `Momentum`: compares the current forecast with the forecast from a chosen lookback window, charging on downward momentum and discharging on upward momentum.
- `Momentum spread`: combines daily spread rank with momentum confirmation, so cheap intervals also need falling momentum and expensive intervals need rising momentum.
- `Channel breakout`: uses recent rolling high/low forecast channels and trades when the forecast breaks outside the channel.
- `Daily spread rank`: ranks forecast prices inside each day, charging in the cheapest forecast intervals and discharging in the most expensive intervals. It can require a minimum daily forecast spread before trading.
- `Ensemble agreement`: trades only when enough ensemble members agree that an interval is cheap or expensive.
- `Predicted best hours`: pairs cheap predicted hours with later expensive predicted hours only when the paper spread covers efficiency and fees.
- `Rolling price optimizer`: uses a discretized daily dynamic program to maximize predicted net cashflow while ending each day empty.
- `Uncertainty-aware optimizer`: penalizes or blocks trades when the available price forecasts disagree.
- `Degradation-aware optimizer`: includes an explicit battery-wear cost per MWh in both optimization and realized PnL.
- `Wind signal`: uses only Energinet DK1 day-ahead wind level and ramp signals to choose actions.
- `Wind-confirmed optimizer`: permits predicted-price optimizer actions only when day-ahead wind conditions confirm them.

The deployed comparison reports setup-specific perfect-foresight benchmarks based on actual test prices:
the battery view optimizes the realized dispatch path, while the prop view optimizes the realized
long/flat/short path including switching costs. Both are labeled as hindsight opportunity ceilings and
are not ranked as tradable strategies.
For each strategy, the site also calculates a no-fee potential counterfactual by setting all modeled
trading fees to zero while retaining battery efficiency and degradation costs. In optimized mode,
both fee-adjusted and no-fee potential are re-optimized on the test period and labeled as hindsight-only.

The `Quant research checks` panel adds:

- `Optimization`: best setting for every strategy by cashflow.
- `Risk`: best setting for every strategy by `cashflow - max drawdown`, plus Calmar, Sharpe-style, Sortino-style, profit-factor, and win-rate metrics.
- `Sharpe`: best setting for every strategy by annualized daily cashflow Sharpe proxy.
- `Walk-forward`: optimize on a rolling historical period, then evaluate on the next unseen period.
- `Costs`: fee/slippage proxy stress test.
- `Robustness`: daily-spread heatmap for nearby parameter settings.
- `Regimes`: performance by high/low daily spread, price level, weekday/weekend, and time of day.
- `Uncertainty`: residual bias, residual std, and post-hoc interval coverage.
- `Decision`: the latest day-ahead action, dispatch size, SOC, and signal reason.

For your example rule, use:

```text
Strategy: Weekly average band
Moving average days: 7
Buy below / sell above by X: 20
```

That means:

```text
Buy/charge when forecast <= last-week moving average - 20
Sell/discharge when forecast >= last-week moving average + 20
```

All strategy simulations use forecast prices for dispatch decisions and realized prices for cashflow.

When you choose a strategy in the sidebar, the dashboard shows only the important parameters for that strategy. Shared battery assumptions remain available for every strategy.

The dashboard also includes `Optimal settings by strategy`. This runs a compact grid search for every strategy, keeps the best cashflow setting for each one, and ranks the winners. It is a research/backtest optimizer over the selected test period, so treat it as a way to compare strategy behavior rather than as proof of future live PnL.

For `Daily spread rank`, the dashboard supports two rule styles:

```text
Percent rank:   charge when daily rank percentile <= X, discharge when rank percentile >= Y
Absolute rank:  charge when daily rank <= X, discharge when daily rank >= Y
```

Daily rank `1` is the cheapest forecast interval of the day. With 15-minute data, a full day usually has 96 ranks.

For confidence filtering, use:

```text
Strategy: Ensemble agreement
Minimum model agreement: 0.60
Limit model disagreement: optional
```

For the thesis-specific edge signal, use:

```text
Strategy: Forecast edge
Edge threshold: 5
Reference forecast: Hourly_Baseline
```

For the mixed volatility/average rule, use:

```text
Strategy: Volatility filtered average
Moving average days: 7
Volatility std days: 7
Minimum rolling std: 30
Buy below / sell above average by X: 0
```

That means:

```text
Only trade when rolling forecast std >= 30
Buy/charge when forecast <= rolling average - X
Sell/discharge when forecast >= rolling average + X
```

For the momentum/spread hybrid, use:

```text
Strategy: Momentum spread
Spread rank rule: Percent rank
Charge when daily rank <= percentile: 0.25
Discharge when daily rank >= percentile: 0.75
Minimum daily forecast spread: 20
Momentum lookback hours: 6
Momentum trigger: 5
Forecast smoothing hours: 1
```

That means:

```text
Buy/charge only when the interval is cheap within the day and momentum is falling by at least 5
Sell/discharge only when the interval is expensive within the day and momentum is rising by at least 5
Skip the day unless forecast max - forecast min >= 20
```

## Full Training Pipeline

```powershell
.venv\Scripts\python -m intraday_power_quant.cli --config configs/jakob-local.json check-data
.venv\Scripts\python -m intraday_power_quant.cli --config configs/jakob-local.json run
```

The full run writes:

- `outputs/model_run/hourly_baseline_minute0_predictions.csv`
- `outputs/model_run/transfer_15min_forecasts_detailed.csv`
- `outputs/model_run/transfer_15min_metrics_detailed.csv`
- `outputs/model_run/run_summary.json`

To explicitly build a final-30-day holdout run:

```powershell
.venv\Scripts\python.exe -m intraday_power_quant.cli --config configs/jakob-local.json run --output-dir outputs\model_run_last30
.venv\Scripts\python.exe -m intraday_power_quant.cli --config configs/jakob-local.json report --results-dir outputs\model_run_last30 --output-html reports\day_ahead_power_trader_project_dashboard_last30.html --output-report reports\day_ahead_power_trader_project_report_last30.md
```

Current final-30-day result:

```text
Best model by MAE: Hourly baseline only
MAE   17.1621
RMSE  23.6360
sMAPE 18.3946
R2     0.4802
```

The default champion column is `TL_Residual_Average`, because the thesis notebook output showed the simple-average residual transfer variant was the strongest result in the broader detailed comparison. You can override it:

```powershell
.venv\Scripts\python -m intraday_power_quant.cli --config configs/jakob-local.json run --champion TL_Residual_Stacked
```

## Ablation

Run the residual model with and without the hourly baseline as an explicit residual feature:

```powershell
.venv\Scripts\python -m intraday_power_quant.cli --config configs/jakob-local.json ablation --output-dir outputs/ablation
```

Current ablation result from the completed local run:

```text
With hourly baseline feature     MAE 17.0951  RMSE 24.6850  R2 0.4911
Without hourly baseline feature  MAE 17.0965  RMSE 23.9013  R2 0.5229
```

## Portfolio Positioning

Suggested title:

> Intraday Power Market Quant Engine: Transfer Learning for 15-Minute DK1 Price Forecasting and Flexibility Valuation

Suggested claim:

> I converted an hourly electricity-price forecaster into a 15-minute forecasting and trading-research engine using transfer learning, residual correction, ensemble models, realistic validation, and a flexibility simulator.
