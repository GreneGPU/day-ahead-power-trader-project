# Day-Ahead Power Trading Project

Transfer-learning 15-minute day-ahead electricity-price forecasting, physical battery optimization, and synthetic proprietary-position research with DK1 imbalance settlement.

Live dashboard: https://greneportfolio.vercel.app/

## Visitor analytics (PostHog)

The public dashboard tracks pageviews, page leaves, and interactions with the
[PostHog JavaScript SDK](https://posthog.com/docs/libraries/js). Tracking runs only
on Vercel production deployments with valid settings. Local development, previews,
and deployments without configuration send no analytics.

To activate:

1. Create a PostHog project and copy its public project token from Project settings
   (`phc_...`, **not** a personal API key).
2. In the Vercel project's **Settings > Environment Variables**, add these for
   **Production**:
   - `POSTHOG_PROJECT_TOKEN`: the public project token.
   - `POSTHOG_HOST`: `https://eu.i.posthog.com` for EU projects, or
     `https://us.i.posthog.com` for US projects.
3. Deploy this version, or redeploy after changing the environment variables.
4. Visit the production dashboard with an ad blocker disabled for the test and
   confirm a `$pageview` in PostHog's live events. Open **Web analytics** for visitor
   counts, pageviews, referring sites, and visit duration. These are visits recorded
   after activation; historical visits cannot be recovered by this integration.

`/api/analytics-config` exposes only the public token and ingestion host. Analytics
loads independently of the dashboard, so blocked requests or PostHog outages do
not stop charts or calculations. Automatic interaction capture, session recordings,
dead-click tracking, and heatmaps are enabled. Session replay masks all input values;
console logs, automatic exceptions, performance capture, and surveys are disabled.
PostHog project settings control recording sampling and canvas capture. The SDK
retains anonymous browser identifiers using its default persistence to count returning
visitors; no login identity is sent. Recordings begin after activation and cannot
reconstruct earlier visits. In Session replay, clear device filters to see both
desktop and mobile visitors.

The dashboard opens from precomputed battery and Prop snapshots, so visiting or refreshing the
site does not rerun the strategy grids. Changing controls keeps the displayed snapshot in place and marks
the settings as pending; `Compare strategies` is the explicit recalculation action. Strategy selection is
instant because each comparison response contains the plotted series for every available strategy.

The live dashboard offers two trading setups. `Physical battery` defaults to 90% round-trip efficiency and a 2026 DK1 distribution-connected fee assumption: 115.41 DKK/MWh while charging and 10.71 DKK/MWh while discharging. `Prop proxy` maps buy/charge signals to long day-ahead positions and sell/discharge signals to short positions, with editable capital, position size, transaction cost, and daily loss limit.

The Prop setup uses one daily position lifecycle. Positions use the next observed day-ahead price move as a synthetic approximation of a pre-delivery close because historical intraday execution prices are unavailable. At the final DK1 interval of every local day, any remaining position is forced flat at the realized imbalance price: a residual long settles on `imbalance price - day-ahead price`, while a residual short settles on `day-ahead price - imbalance price`.

The prop setup is deliberately labeled as a synthetic research proxy. The thesis dataset does not contain historical intraday entry or exit quotes, bid/ask spreads, margin, collateral, or liquidity, so its PnL is not presented as executable Nord Pool spot arbitrage. The end-of-day fallback uses historical Energinet DK1 settlement prices, but realizing that settlement requires physical flexibility or a balance-responsible-party arrangement; activation, metering, collateral, nominations, BRP fees, and market impact are not modeled.

The physical-battery view deliberately preserves the original pre-settlement thesis price series, strategy calculations, and saved results. Imbalance data does not enter that calculation. Prop prices and forecasts are converted from EUR/MWh to DKK/MWh with the interval exchange rates implied by matching Energinet EUR and DKK fields. The dashboard keeps day-ahead prices and forecasts together, and renders the aligned imbalance price in its own chart. The displayed model MAE and RMSE remain in the original EUR/MWh units used to train and evaluate the thesis models.

Its default evaluation mode reserves the final ten complete DK1 calendar days as a chronological holdout: every strategy searches its parameter grid for the highest net cashflow on all earlier available observations, then the dashboard ranks strategies, calculates daily Sharpe, and shows trade logs using only those ten unseen days. A fixed-default mode remains available for comparison.

This repo turns the thesis notebook into a maintainable project:

- hourly source ensemble trained before the 15-minute transition
- minute-0 feature mapping from 15-minute rows into hourly feature space
- hourly baseline forecast merged back to all 15-minute intervals
- residual transfer-learning model
- direct 15-minute benchmark model
- configurable chronological validation split: fixed ratio or final N days
- model-ranking metrics, coverage checks, leakage warnings, and ablation entry point
- forecast-driven battery arbitrage simulator
- asset-free long/short research proxy with capital return, transaction costs, loss limits, and a daily close deadline
- end-of-day fallback settlement using aligned Energinet DK1 prices and forecast-only signals
- separate perfect-foresight benchmarks for battery dispatch and the daily Prop settlement lifecycle
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
|   |-- imbalance_trading.py
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

To refresh the default results bundled with the deployed dashboard after changing data or strategy code:

```powershell
.venv\Scripts\python scripts\build_saved_comparisons.py
```

To replace or extend the aligned Energinet DK1 imbalance extract, import it before rebuilding the snapshots:

```powershell
.venv\Scripts\python scripts\import_imbalance_prices.py "C:\path\to\ImbalancePrice.csv"
.venv\Scripts\python scripts\build_saved_comparisons.py
```

The importer validates DK1 coverage, a complete 15-minute time grid, duplicate timestamps, missing values, EUR/DKK consistency, and exact timestamp alignment with the saved forecast dataset.

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

The deployed comparison reports setup-specific perfect-foresight benchmarks based on actual test prices.
The battery view optimizes the realized dispatch path. The Prop benchmark optimizes each day's realized
long/flat/short path, including switching costs and any final imbalance close. Both are labeled as hindsight opportunity
ceilings and are not ranked as tradable strategies. For each strategy, the site also calculates a zero-cost counterfactual while retaining battery
efficiency and degradation costs where applicable. In optimized mode, both cost-adjusted and zero-cost
potential are re-optimized on the test period and labeled as hindsight-only.

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
- `outputs/model_run/power_forecast_bundle.pkl`
- `outputs/model_run/model_gate.json`

## MLOps on AWS

The repository includes optional MLflow experiment tracking, a serializable forecasting bundle,
quality gates, production monitoring metrics, SageMaker training/inference entry points, conditional
Model Registry registration, batch transform, and CI. See [MLOPS_AWS.md](MLOPS_AWS.md) for setup and commands.

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

## Walk-forward forecast history

The website's backtests run on **walk-forward out-of-sample forecasts** for 13 November 2025 to 4 March 2026 (10,744 quarter-hours in 8 blocks of 14 days), instead of a single 22-day test split. Before each block, the 15-minute residual and direct ensembles are retrained on all 15-minute data strictly earlier than the block (expanding window, first block after 35 days); the hourly source model is trained once on hourly data before 1 October 2025. No interval is predicted by a model that saw it.

```powershell
pip install -e ".[training]"
python -m intraday_power_quant.cli walk-forward --data-dir data/raw --output-dir outputs/walk_forward
python scripts/export_walk_forward.py
python scripts/enrich_deployment_features.py
python scripts/import_imbalance_prices.py --from-api
python scripts/export_formula_features.py <thesis final_15min_day_ahead_safe_modeling_dataset.csv>
python scripts/build_saved_comparisons.py
```

`data/raw` needs the thesis modeling datasets with ISO timestamps (the thesis CSVs are day-first, `dd-mm-yy HH:MM`). Energinet's `Forecasts_Hour` has no DK1 data for 22–24 November 2025; those 72 hours of day-ahead wind/solar features come from the modeling dataset's own forecasts and are flagged in `deployment_data/manifest.json`.

### Choosing the forecast without hindsight

The website's `Prediction` series is the **walk-forward champion**, chosen per block with two rules that only use information available before the block:

* **Nested selection:** the candidate (hourly baseline, TL residual average, TL residual stacking, direct 15-minute) with the lowest MAE on all *earlier* blocks' out-of-sample forecasts; the first block uses the configured champion.
* **Training-CV gate:** a model trained on 15-minute data may only be champion if its time-series cross-validated MAE on that block's *training* data beats the hourly baseline's; otherwise the block falls back to the baseline.

Result: the hourly baseline is champion in all 8 blocks (MAE 18.48 EUR/MWh). The transfer models were worse in the early, data-poor blocks and failed the gate in 7 of 8 blocks. In hindsight TL pure stacking scores best overall (17.85) and the configured TL average 19.81, but no rule using only past information would have selected them here. The gate is conservative (its early CV folds train on little data); a gate based on the last CV folds only is a candidate to validate on future data rather than tune on these results. The ungated transfer forecast remains available as `Transfer_Residual_Prediction`. Per-block champions, gate errors and train-CV vs test errors are written to `outputs/walk_forward/walk_forward_blocks.csv`.

### Benchmark tuning

The benchmark dashboard tunes strategies **walk-forward**: before each 10-day test block, every strategy picks the parameter setting with the best *neighbour-smoothed* P&L over the 20 days before the block (the setting averaged with its adjacent grid settings, so a stable region beats a lone peak) and trades it unchanged on the block. One sweep over the window records each setting's daily P&L, so no fold re-simulates. **OOS kept** reports out-of-sample P&L per day as a share of tuned P&L per day. The default window is the last 40 days (2 blocks). The optimizer loops were moved from per-row pandas indexing to NumPy arrays (identical results, about 9x faster for Prop), and a bug where Prop requests converted the cached price frame to DKK for later battery requests was fixed.

The Strategy Lab defaults to the final 30 complete days and can test the whole history.

## Strategy Lab

The Strategy Lab is on the landing page (`/#lab`; the old `/strategy-lab.html` redirects there), and the benchmark dashboard lives at `/benchmarks.html`. Press one of twelve ready-made strategies, such as long the day's cheapest 20% of forecast intervals and short the priciest 20% (optionally sized up in the extreme 5%), or backtest your own threshold rules against the saved DK1 data. Write a formula (including the per-delivery-day inputs `forecast_rank`, `forecast_z`, `forecast_spread` and `hour`), or choose a forecast price, forecast-minus-baseline spread, forecast change, or uploaded CSV signal. Set lower/upper thresholds and direction, optionally with dynamic sizing (step or scaled up to a maximum multiplier at outer thresholds), then choose physical battery or the long/short Prop proxy. Prices, fees and cashflows in this lab use DKK.

The **Portfolio** panel combines the runs you have made (up to five) into one book. Positions are added up per 15-minute interval with equal, inverse-volatility or manual weights, so opposite trades cancel before costs, and the book is re-simulated with the same Prop proxy rules (next day-ahead move, imbalance close at day end, cost on every MWh change). It reports netted P&L against the sum of stand-alone runs, the trading cost saved by netting, drawdown and net exposure for the chart window, an optional net-exposure cap and portfolio daily loss limit, and an hourly P&L correlation matrix with a diversification ratio. Volatility, correlation and diversification use the whole test period (in-sample), so treat them as illustrative.

Every backtest also reports **risk and execution metrics** for the chart window (win rate, profit factor, annualised daily Sharpe, hourly 95% VaR / expected shortfall, MWh traded and the **break-even trading cost** per MWh at which the edge disappears) and runs **robustness checks** via `POST /api/custom-strategy/robustness`: the same rule on the earlier period before the final 10 days (out of sample with respect to the test window) and a 5×5 threshold-sensitivity grid around your thresholds. A robust edge shows a plateau of similar P&L around the chosen rule rather than a lone peak.

The robustness check also asks **whether the P&L could be chance** (`src/intraday_power_quant/significance.py`): a one-sided t-test on daily P&L (days, not 15-minute intervals, because interval P&L is autocorrelated), a **day-shuffle permutation test** that applies each day's positions to another day's prices (null: the day-specific forecast adds nothing beyond the typical daily price shape), a **within-day timing permutation test**, the number of days needed for 80% power at the current daily Sharpe, and a Bonferroni adjustment for the number of rules tried in the session. On the final 10 days the default cheapest/priciest-20% rule is not significant (daily p ≈ 0.12), and shuffling days keeps most of its P&L, so its edge is mostly the predictable daily price shape.

The CSV template contains the saved timestamps. Keep the `HourUTC,Signal` headers, supply timezone-aware timestamps and numeric signals, and cover every interval in the selected test period. Only use signals available at decision time. Files are sent to the backtest API for calculation; saving rules in your browser does not save the CSV.

Results include cashflow, drawdown, fees, remaining battery energy or Prop equity, a cumulative chart, and an exportable interval log. Battery cashflow excludes the value of remaining energy. Prop results use the existing next-price proxy and daily imbalance settlement. Repeated tuning on the final ten days is exploratory research, not independent validation.

The API accepts declarative rules through `POST /api/custom-strategy`; it does not execute uploaded code. See `/api/docs` for the request schema.

### Formula signals

Choose **Write a formula** in Strategy Lab. For example:

```text
signal = demand + outages - wind - solar
```

`wind` and `solar` use the saved DK1 day-ahead forecasts (MW). `forecast` is the selected price forecast and `baseline` the hourly price baseline (DKK/MWh). `demand` is an alias of the saved `load_fc` forecast (MW). Only `outages` requires a fundamentals CSV with `HourUTC,outages`; the downloadable template leaves missing inputs blank deliberately. Optional `wind` and `solar` columns replace saved inputs for the uploaded series. Every used input must cover the selected test period. Nothing is automatically normalized or filled with zero.

Arithmetic supports +, -, *, /, parentheses and abs, sign, min, max, avg, clamp. Functions operate on values at the same timestamp; avg is not a moving average. Formulas use a bounded arithmetic parser, not Python or JavaScript execution. Configure upper/lower thresholds and direction to convert the signal into actions. Formulas are saved with browser rules; uploaded input files must be loaded again.


### Saved thesis features

The formula editor also accepts all 15 requested lagged features under their exact names: system balance; temperature, humidity, weather wind speed and gas price at lags 96/192/672; and the known-lag gas price daily change and seven-day mean. The expandable feature list inserts variable names into the editor. `load_fc` and its alias `demand` are available without uploads.

`scripts/export_formula_features.py` aligns the source thesis CSV to the 2,116 deployment timestamps, parsing day-first UTC timestamps explicitly. It rejects duplicates, missing coverage and nonfinite values. The exported `deployment_data/formula_features.csv.gz` preserves supplied feature values and lags; its companion manifest records the source filename and SHA-256. No new shifts, interpolation, or normalization are applied. Source units are retained, so scale coefficients appropriately when mixing features.

### Prop replay and current prices

Strategy Lab now runs long/short Prop tests. After running a formula, use Play, Next interval, Restart, the timeline slider, or Show full test. Replay reveals completed intervals and their modeled price-move outcomes; the position shown is the one held during that interval, with the post-settlement position shown separately. Metrics and logs follow the replay cursor.

Up to five runs with identical timestamps can be compared in one browser tab. Each run uses its own formula, thresholds, capital, position size and costs. These are independent full-period tests; switching the inspected run does not switch positions mid-test. A different test period clears the comparison. Strength is signed distance from the threshold midpoint divided by half the threshold gap, oriented toward long/short. +/-100% reaches an entry boundary; it is not a probability or a position-sizing rule.

The separate Latest DK1 day-ahead prices panel reads Energinet DayAheadPrices for today and tomorrow, when published, with a ten-minute cache. These are published auction prices, not live intraday quotes. This panel does not generate live signals from the historical thesis features.
