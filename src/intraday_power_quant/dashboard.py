from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from .config import (
    BatterySettings,
    BestHoursSettings,
    ChannelBreakoutSettings,
    DailySpreadSettings,
    EnsembleAgreementSettings,
    ForecastEdgeSettings,
    MeanReversionSettings,
    MomentumSpreadSettings,
    MomentumSettings,
    VolatilityFilterSettings,
    WeeklyBandSettings,
)
from .evaluation import rank_models
from .optimization import best_parameter_rows, run_strategy_parameter_sweep, simulate_strategy_from_settings
from .research import (
    daily_spread_robustness_grid,
    execution_cost_stress_table,
    forecast_uncertainty_summary,
    latest_decision_table,
    regime_performance_table,
    walk_forward_strategy_optimization,
)
from .trading import (
    BatteryConfig,
    BestHoursConfig,
    ChannelBreakoutConfig,
    DailySpreadConfig,
    EnsembleAgreementConfig,
    ForecastEdgeConfig,
    MeanReversionConfig,
    MomentumSpreadConfig,
    MomentumConfig,
    VolatilityFilterConfig,
    WeeklyBandConfig,
    run_strategy_suite,
    summarize_strategy_result,
)


FORECAST_CANDIDATES = [
    "transfer_15min_forecasts_detailed.csv",
    "transfer_15min_forecasts_detailed.xlsx",
    "transfer_15min_forecasts_tuned_thesis_ready.xlsx",
    "transfer_15min_forecasts_thesis_ready.xlsx",
    "transfer_15min_hourly_forecasts.xlsx",
]

METRICS_CANDIDATES = [
    "transfer_15min_metrics_detailed.csv",
    "transfer_15min_metrics_detailed.xlsx",
    "transfer_15min_metrics_tuned_thesis_ready.xlsx",
    "transfer_15min_metrics_thesis_ready.xlsx",
    "transfer_15min_metrics.xlsx",
]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def find_first_existing(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    tried = "\n".join(str(root / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find any expected result file. Tried:\n{tried}")


def normalize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    rename_map = {
        "Hourly_Baseline_Stacked": "Hourly_Baseline",
        "Transfer_Residual_Prediction": "Prediction",
        "Direct_15min_Stacked_Prediction": "Direct_15min_Prediction",
    }
    output = output.rename(columns={old: new for old, new in rename_map.items() if old in output.columns})
    if "Prediction" not in output.columns:
        for candidate in ["TL_Residual_Average", "TL_Residual_Stacked"]:
            if candidate in output.columns:
                output["Prediction"] = output[candidate]
                break
    if "Direct_15min_Prediction" not in output.columns:
        for candidate in ["Direct_15min_Stacked", "Direct_15min_Average"]:
            if candidate in output.columns:
                output["Direct_15min_Prediction"] = output[candidate]
                break
    if "HourUTC" in output.columns:
        output["HourUTC"] = pd.to_datetime(output["HourUTC"])
    return output


def _format_number(value: float, digits: int = 2) -> str:
    if pd.isna(value) or math.isinf(float(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _records_for_chart(df: pd.DataFrame) -> list[dict[str, object]]:
    columns = ["HourUTC", "Actual_Price", "Hourly_Baseline", "Prediction", "Direct_15min_Prediction"]
    available = [column for column in columns if column in df.columns]
    chart_df = df[available].copy()
    chart_df["HourUTC"] = pd.to_datetime(chart_df["HourUTC"]).dt.strftime("%Y-%m-%d %H:%M")
    for column in available:
        if column != "HourUTC":
            chart_df[column] = chart_df[column].astype(float).round(4)
    return chart_df.to_dict(orient="records")


def _metrics_rows(metrics: pd.DataFrame) -> list[dict[str, object]]:
    keep = [column for column in ["Rank", "Model", "Model_Type", "MAE", "RMSE", "sMAPE", "R2"] if column in metrics.columns]
    table = metrics[keep].copy()
    for column in ["MAE", "RMSE", "sMAPE", "R2"]:
        if column in table.columns:
            table[column] = table[column].astype(float).round(4)
    return table.to_dict(orient="records")


def generate_static_dashboard(
    results_dir: str | Path,
    output_html: str | Path,
    battery_settings: BatterySettings | None = None,
    weekly_band_settings: WeeklyBandSettings | None = None,
    forecast_edge_settings: ForecastEdgeSettings | None = None,
    volatility_filter_settings: VolatilityFilterSettings | None = None,
    mean_reversion_settings: MeanReversionSettings | None = None,
    momentum_settings: MomentumSettings | None = None,
    momentum_spread_settings: MomentumSpreadSettings | None = None,
    channel_breakout_settings: ChannelBreakoutSettings | None = None,
    daily_spread_settings: DailySpreadSettings | None = None,
    best_hours_settings: BestHoursSettings | None = None,
    ensemble_agreement_settings: EnsembleAgreementSettings | None = None,
) -> dict[str, object]:
    root = Path(results_dir)
    forecast_path = find_first_existing(root, FORECAST_CANDIDATES)
    metrics_path = find_first_existing(root, METRICS_CANDIDATES)

    forecasts = normalize_forecast_columns(_read_table(forecast_path))
    metrics = _read_table(metrics_path)
    if "MAE" in metrics.columns:
        metrics = rank_models(metrics, "MAE")

    cfg = battery_settings or BatterySettings()
    weekly_cfg = weekly_band_settings or WeeklyBandSettings()
    edge_cfg = forecast_edge_settings or ForecastEdgeSettings()
    volatility_cfg = volatility_filter_settings or VolatilityFilterSettings()
    mean_cfg = mean_reversion_settings or MeanReversionSettings()
    momentum_cfg = momentum_settings or MomentumSettings()
    momentum_spread_cfg = momentum_spread_settings or MomentumSpreadSettings()
    breakout_cfg = channel_breakout_settings or ChannelBreakoutSettings()
    daily_cfg = daily_spread_settings or DailySpreadSettings()
    best_hours_cfg = best_hours_settings or BestHoursSettings()
    ensemble_cfg = ensemble_agreement_settings or EnsembleAgreementSettings()
    battery_config = BatteryConfig(**cfg.__dict__)
    strategy_results = run_strategy_suite(
        forecasts,
        battery_config=battery_config,
        weekly_band_config=WeeklyBandConfig(**weekly_cfg.__dict__),
        forecast_edge_config=ForecastEdgeConfig(**edge_cfg.__dict__),
        volatility_filter_config=VolatilityFilterConfig(**volatility_cfg.__dict__),
        mean_reversion_config=MeanReversionConfig(**mean_cfg.__dict__),
        breakout_config=ChannelBreakoutConfig(**breakout_cfg.__dict__),
        daily_spread_config=DailySpreadConfig(**daily_cfg.__dict__),
        best_hours_config=BestHoursConfig(**best_hours_cfg.__dict__),
        momentum_config=MomentumConfig(**momentum_cfg.__dict__),
        momentum_spread_config=MomentumSpreadConfig(**momentum_spread_cfg.__dict__),
        ensemble_agreement_config=EnsembleAgreementConfig(**ensemble_cfg.__dict__),
        forecast_col="Prediction",
    )
    strategy_table = pd.DataFrame(
        summarize_strategy_result(name, summary)
        for name, (_, summary) in strategy_results.items()
    ).sort_values("Cashflow", ascending=False)
    best_strategy = strategy_table.iloc[0]
    parameter_sweep = run_strategy_parameter_sweep(
        forecasts,
        battery_config=battery_config,
        forecast_col="Prediction",
    )
    optimization_table = best_parameter_rows(parameter_sweep, ranking_metric="Cashflow")
    risk_adjusted_table = best_parameter_rows(parameter_sweep, ranking_metric="Risk_Adjusted_Score")
    sharpe_table = best_parameter_rows(parameter_sweep, ranking_metric="Daily_Sharpe")
    best_optimized_strategy = optimization_table.iloc[0] if not optimization_table.empty else None
    best_risk_adjusted_strategy = risk_adjusted_table.iloc[0] if not risk_adjusted_table.empty else None
    best_sharpe_strategy = sharpe_table.iloc[0] if not sharpe_table.empty else None

    if best_optimized_strategy is not None:
        best_optimized_sim, _ = simulate_strategy_from_settings(
            str(best_optimized_strategy["Strategy"]),
            forecasts,
            str(best_optimized_strategy["Settings_JSON"]),
            battery_config,
            forecast_col="Prediction",
        )
        cost_stress = execution_cost_stress_table(
            forecasts,
            str(best_optimized_strategy["Strategy"]),
            str(best_optimized_strategy["Settings_JSON"]),
            battery_config,
            forecast_col="Prediction",
        )
        regime_table = regime_performance_table(forecasts, best_optimized_sim)
        latest_decision = latest_decision_table(
            forecasts,
            str(best_optimized_strategy["Strategy"]),
            str(best_optimized_strategy["Settings_JSON"]),
            battery_config,
            forecast_col="Prediction",
        )
    else:
        best_optimized_sim = pd.DataFrame()
        cost_stress = pd.DataFrame()
        regime_table = pd.DataFrame()
        latest_decision = pd.DataFrame()

    walk_forward_table = walk_forward_strategy_optimization(
        forecasts,
        battery_config=battery_config,
        train_days=14,
        test_days=7,
        step_days=7,
        ranking_metric="Risk_Adjusted_Score",
        forecast_col="Prediction",
    )
    robustness_table = daily_spread_robustness_grid(
        forecasts,
        battery_config=battery_config,
        min_daily_spread=20,
        forecast_col="Prediction",
    )
    uncertainty_summary = forecast_uncertainty_summary(forecasts)

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_model = metrics.iloc[0]["Model"] if not metrics.empty and "Model" in metrics.columns else "n/a"
    best_mae = float(metrics.iloc[0]["MAE"]) if not metrics.empty and "MAE" in metrics.columns else float("nan")
    test_start = pd.to_datetime(forecasts["HourUTC"]).min()
    test_end = pd.to_datetime(forecasts["HourUTC"]).max()

    chart_records = _records_for_chart(forecasts)
    cash_frame = pd.DataFrame(
        {"HourUTC": pd.to_datetime(next(iter(strategy_results.values()))[0]["HourUTC"]).dt.strftime("%Y-%m-%d %H:%M")}
    )
    cash_series: list[dict[str, str]] = []
    for idx, (name, (strategy_df, _)) in enumerate(strategy_results.items(), start=1):
        key = f"Strategy_{idx}"
        cash_frame[key] = strategy_df["Cumulative_Cashflow"].astype(float).round(4)
        cash_series.append({"key": key, "label": name})
    cash_records = cash_frame.to_dict(orient="records")
    metric_records = _metrics_rows(metrics)
    strategy_records = strategy_table.round(4).to_dict(orient="records")
    optimization_records = optimization_table.round(4).to_dict(orient="records")
    risk_adjusted_records = risk_adjusted_table.round(4).to_dict(orient="records")
    sharpe_records = sharpe_table.round(4).to_dict(orient="records")
    walk_forward_records = walk_forward_table.to_dict(orient="records") if not walk_forward_table.empty else []
    cost_stress_records = cost_stress.round(4).to_dict(orient="records") if not cost_stress.empty else []
    robustness_records = robustness_table.head(12).round(4).to_dict(orient="records") if not robustness_table.empty else []
    regime_records = regime_table.round(4).to_dict(orient="records") if not regime_table.empty else []
    latest_decision_records = latest_decision.to_dict(orient="records") if not latest_decision.empty else []
    strategy_rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['Strategy']))}</td>"
        f"<td class=\"desc\">{html.escape(str(row['Description']))}</td>"
        f"<td class=\"num\">{_format_number(float(row['Cashflow']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Max_Drawdown']), 2)}</td>"
        f"<td class=\"num\">{int(row['Active_Intervals'])}</td>"
        "</tr>"
        for row in strategy_records
    )
    optimization_rows_html = "\n".join(
        "<tr>"
        f"<td class=\"num\">{int(row['Rank'])}</td>"
        f"<td>{html.escape(str(row['Strategy']))}</td>"
        f"<td class=\"num\">{_format_number(float(row['Cashflow']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Max_Drawdown']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Risk_Adjusted_Score']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Calmar_Ratio']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Daily_Sharpe']), 2)}</td>"
        f"<td class=\"num\">{int(row['Active_Intervals'])}</td>"
        f"<td class=\"num\">{int(row['Evaluations'])}</td>"
        f"<td class=\"settings\">{html.escape(str(row['Settings']))}</td>"
        "</tr>"
        for row in optimization_records
    )
    risk_adjusted_rows_html = "\n".join(
        "<tr>"
        f"<td class=\"num\">{int(row['Rank'])}</td>"
        f"<td>{html.escape(str(row['Strategy']))}</td>"
        f"<td class=\"num\">{_format_number(float(row['Risk_Adjusted_Score']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Cashflow']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Max_Drawdown']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Daily_Sharpe']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Daily_Sortino']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Profit_Factor']), 2)}</td>"
        f"<td class=\"settings\">{html.escape(str(row['Settings']))}</td>"
        "</tr>"
        for row in risk_adjusted_records
    )
    sharpe_rows_html = "\n".join(
        "<tr>"
        f"<td class=\"num\">{int(row['Rank'])}</td>"
        f"<td>{html.escape(str(row['Strategy']))}</td>"
        f"<td class=\"num\">{_format_number(float(row['Daily_Sharpe']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Daily_Sortino']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Cashflow']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Max_Drawdown']), 2)}</td>"
        f"<td class=\"settings\">{html.escape(str(row['Settings']))}</td>"
        "</tr>"
        for row in sharpe_records
    )
    walk_forward_rows_html = "\n".join(
        "<tr>"
        f"<td class=\"num\">{int(row['Window'])}</td>"
        f"<td>{html.escape(str(row['Selected_Strategy']))}</td>"
        f"<td>{html.escape(pd.to_datetime(row['Test_Start']).strftime('%Y-%m-%d'))}</td>"
        f"<td>{html.escape(pd.to_datetime(row['Test_End']).strftime('%Y-%m-%d'))}</td>"
        f"<td class=\"num\">{_format_number(float(row['Train_Risk_Adjusted_Score']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Test_Cashflow']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Test_Max_Drawdown']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Test_Daily_Sharpe']), 2)}</td>"
        f"<td class=\"settings\">{html.escape(str(row['Settings']))}</td>"
        "</tr>"
        for row in walk_forward_records
    )
    cost_stress_rows_html = "\n".join(
        "<tr>"
        f"<td class=\"num\">{_format_number(float(row['Fee_Per_MWh']), 1)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Cashflow']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Max_Drawdown']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Risk_Adjusted_Score']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Daily_Sharpe']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Profit_Factor']), 2)}</td>"
        f"<td class=\"num\">{int(row['Active_Intervals'])}</td>"
        "</tr>"
        for row in cost_stress_records
    )
    robustness_rows_html = "\n".join(
        "<tr>"
        f"<td class=\"num\">{row['Low_Quantile']:.2f}</td>"
        f"<td class=\"num\">{row['High_Quantile']:.2f}</td>"
        f"<td class=\"num\">{_format_number(float(row['Cashflow']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Max_Drawdown']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Risk_Adjusted_Score']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Daily_Sharpe']), 2)}</td>"
        f"<td class=\"num\">{int(row['Active_Intervals'])}</td>"
        "</tr>"
        for row in robustness_records
    )
    regime_rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['Regime_Type']))}</td>"
        f"<td>{html.escape(str(row['Regime']))}</td>"
        f"<td class=\"num\">{_format_number(float(row['Cashflow']), 2)}</td>"
        f"<td class=\"num\">{int(row['Active_Intervals'])}</td>"
        f"<td class=\"num\">{_format_number(float(row['Average_Actual_Price']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Average_Forecast_Error']), 2)}</td>"
        "</tr>"
        for row in regime_records
    )
    latest_decision_rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M'))}</td>"
        f"<td>{html.escape(str(row['Strategy']))}</td>"
        f"<td>{html.escape(str(row['Action']))}</td>"
        f"<td class=\"num\">{_format_number(float(row['Dispatch_MW']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['State_Of_Charge_MWh']), 2)}</td>"
        f"<td class=\"num\">{_format_number(float(row['Forecast_Price']), 2)}</td>"
        f"<td class=\"settings\">{html.escape(str(row['Reason']))}</td>"
        "</tr>"
        for row in latest_decision_records
    )
    best_optimized_name = "n/a"
    best_optimized_cashflow = "n/a"
    best_optimized_settings = "No optimizer rows"
    if best_optimized_strategy is not None:
        best_optimized_name = str(best_optimized_strategy["Strategy"])
        best_optimized_cashflow = _format_number(float(best_optimized_strategy["Cashflow"]), 0)
        best_optimized_settings = str(best_optimized_strategy["Settings"])
    best_risk_name = "n/a"
    best_risk_score = "n/a"
    if best_risk_adjusted_strategy is not None:
        best_risk_name = str(best_risk_adjusted_strategy["Strategy"])
        best_risk_score = _format_number(float(best_risk_adjusted_strategy["Risk_Adjusted_Score"]), 0)
    best_sharpe_name = "n/a"
    best_sharpe_value = "n/a"
    if best_sharpe_strategy is not None:
        best_sharpe_name = str(best_sharpe_strategy["Strategy"])
        best_sharpe_value = _format_number(float(best_sharpe_strategy["Daily_Sharpe"]), 2)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Day-Ahead Power Trading Project Dashboard</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f7f8fa;
      --fg: #15171c;
      --muted: #5f6673;
      --panel: #ffffff;
      --border: #d8dde6;
      --blue: #2f6fdd;
      --green: #16805d;
      --red: #bf3b3b;
      --orange: #b66b18;
      --purple: #7653c9;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111318;
        --fg: #edf0f5;
        --muted: #a4acb8;
        --panel: #191d24;
        --border: #333946;
      }}
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 44px;
    }}
    header {{
      display: flex;
      gap: 18px;
      justify-content: space-between;
      align-items: end;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .card, .chart-wrap, .table-wrap {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .card {{
      padding: 14px 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 24px;
      font-weight: 650;
    }}
    .sub {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin: 0 0 12px;
    }}
    button, label.toggle {{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--fg);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
    }}
    button.active {{
      background: var(--fg);
      color: var(--bg);
    }}
    label.toggle {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .chart-wrap {{
      padding: 16px;
      margin-bottom: 20px;
    }}
    canvas {{
      width: 100%;
      height: 420px;
      display: block;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .table-heading {{
      padding: 14px 12px;
      border-bottom: 1px solid var(--border);
    }}
    .table-heading strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .table-heading p {{
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 11px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    td.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    td.desc, td.settings {{
      min-width: 320px;
      white-space: normal;
      line-height: 1.4;
      color: var(--muted);
    }}
    td.settings {{
      min-width: 380px;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      display: inline-block;
    }}
    @media (max-width: 780px) {{
      header {{
        align-items: start;
        flex-direction: column;
      }}
      .grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      canvas {{
        height: 330px;
      }}
    }}
    @media (max-width: 520px) {{
      main {{
        padding: 18px 12px 30px;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Day-Ahead Power Trading Project</h1>
      <p>15-minute day-ahead DK1 forecast comparison, transfer-learning metrics, and battery dispatch research.</p>
    </div>
    <p>{html.escape(test_start.strftime("%Y-%m-%d %H:%M"))} to {html.escape(test_end.strftime("%Y-%m-%d %H:%M"))}</p>
  </header>

  <section class="grid" aria-label="Summary">
    <article class="card">
      <div class="label">Best MAE model</div>
      <div class="value">{html.escape(str(best_model))}</div>
      <div class="sub">MAE {_format_number(best_mae, 3)}</div>
    </article>
    <article class="card">
      <div class="label">Market data mode</div>
      <div class="value">Day-ahead</div>
      <div class="sub">No intraday order book or fill model assumed</div>
    </article>
    <article class="card">
      <div class="label">Best strategy</div>
      <div class="value">{html.escape(str(best_strategy["Strategy"]))}</div>
      <div class="sub">Cashflow {_format_number(float(best_strategy["Cashflow"]), 0)}</div>
    </article>
    <article class="card">
      <div class="label">Best optimized strategy</div>
      <div class="value">{html.escape(best_optimized_name)}</div>
      <div class="sub">Cashflow {best_optimized_cashflow}; {html.escape(best_optimized_settings)}</div>
    </article>
    <article class="card">
      <div class="label">Best risk-adjusted strategy</div>
      <div class="value">{html.escape(best_risk_name)}</div>
      <div class="sub">PnL minus drawdown {best_risk_score}</div>
    </article>
    <article class="card">
      <div class="label">Best daily Sharpe proxy</div>
      <div class="value">{html.escape(best_sharpe_name)}</div>
      <div class="sub">Annualized daily cashflow Sharpe {best_sharpe_value}</div>
    </article>
    <article class="card">
      <div class="label">Forecast uncertainty</div>
      <div class="value">{uncertainty_summary["Interval_80_Coverage"]:.0%}</div>
      <div class="sub">Post-hoc 80% residual interval coverage</div>
    </article>
    <article class="card">
      <div class="label">Test intervals</div>
      <div class="value">{len(forecasts):,}</div>
      <div class="sub">15-minute observations</div>
    </article>
    <article class="card">
      <div class="label">Weekly moving average</div>
      <div class="value">{weekly_cfg.window_days:g}D / +/-{weekly_cfg.band:g}</div>
      <div class="sub">Buy below MA-{_format_number(float(weekly_cfg.band), 0)}, sell above MA+{_format_number(float(weekly_cfg.band), 0)}</div>
    </article>
    <article class="card">
      <div class="label">Momentum</div>
      <div class="value">{momentum_cfg.lookback_hours:g}H / +/-{momentum_cfg.threshold:g}</div>
      <div class="sub">Smoothed over {momentum_cfg.smoothing_hours:g}H</div>
    </article>
    <article class="card">
      <div class="label">Momentum spread</div>
      <div class="value">{momentum_spread_cfg.lookback_hours:g}H / spread {momentum_spread_cfg.min_daily_spread:g}</div>
      <div class="sub">Momentum +/-{momentum_spread_cfg.momentum_threshold:g}</div>
    </article>
    <article class="card">
      <div class="label">Forecast edge</div>
      <div class="value">+/-{edge_cfg.threshold:g}</div>
      <div class="sub">Against {html.escape(edge_cfg.reference_col)}</div>
    </article>
    <article class="card">
      <div class="label">Volatility filter</div>
      <div class="value">std >= {volatility_cfg.min_volatility:g}</div>
      <div class="sub">{volatility_cfg.average_window_days:g}D MA, {volatility_cfg.volatility_window_days:g}D std</div>
    </article>
    <article class="card">
      <div class="label">Ensemble agreement</div>
      <div class="value">{ensemble_cfg.min_agreement:.0%}</div>
      <div class="sub">Model vote threshold</div>
    </article>
    <article class="card">
      <div class="label">Strategies tested</div>
      <div class="value">{len(strategy_results)}</div>
      <div class="sub">Same battery constraints and realized prices</div>
    </article>
  </section>

  <section class="chart-wrap">
    <div class="toolbar">
      <div>
        <button type="button" data-window="1">1D</button>
        <button type="button" data-window="2">2D</button>
        <button type="button" data-window="7" class="active">7D</button>
        <button type="button" data-window="0">Full</button>
      </div>
      <div class="legend" aria-label="Series toggles">
        <label class="toggle"><input type="checkbox" data-series="actual" checked> <span class="swatch" style="background: var(--blue)"></span> Actual</label>
        <label class="toggle"><input type="checkbox" data-series="prediction" checked> <span class="swatch" style="background: var(--green)"></span> Champion</label>
        <label class="toggle"><input type="checkbox" data-series="baseline" checked> <span class="swatch" style="background: var(--orange)"></span> Hourly baseline</label>
        <label class="toggle"><input type="checkbox" data-series="direct"> <span class="swatch" style="background: var(--purple)"></span> Direct 15-min</label>
      </div>
    </div>
    <canvas id="priceChart" width="1100" height="420" aria-label="Actual and forecast prices over time"></canvas>
  </section>

  <section class="chart-wrap">
    <div class="toolbar">
      <strong>Battery simulation</strong>
      <p>Compares strategy cashflow using forecast prices for dispatch decisions and realized prices for settlement.</p>
    </div>
    <canvas id="cashChart" width="1100" height="320" aria-label="Cumulative battery cashflow over time"></canvas>
  </section>

  <section class="table-wrap" aria-label="Optimized strategy settings">
    <div class="table-heading">
      <strong>Parameter optimization</strong>
      <p>Best cashflow setting found for each strategy in a compact parameter sweep.</p>
    </div>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Strategy</th>
          <th>Best cashflow</th>
          <th>Max drawdown</th>
          <th>Risk score</th>
          <th>Calmar</th>
          <th>Daily Sharpe</th>
          <th>Active intervals</th>
          <th>Runs</th>
          <th>Optimal settings</th>
        </tr>
      </thead>
      <tbody>
        {optimization_rows_html}
      </tbody>
    </table>
  </section>
  <br>

  <section class="table-wrap" aria-label="Risk adjusted optimization">
    <div class="table-heading">
      <strong>Risk-adjusted optimization</strong>
      <p>Ranks settings by cashflow minus max drawdown, so smoother strategies can beat high-PnL fragile settings.</p>
    </div>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Strategy</th>
          <th>Risk score</th>
          <th>Cashflow</th>
          <th>Max drawdown</th>
          <th>Daily Sharpe</th>
          <th>Daily Sortino</th>
          <th>Profit factor</th>
          <th>Settings</th>
        </tr>
      </thead>
      <tbody>
        {risk_adjusted_rows_html}
      </tbody>
    </table>
  </section>
  <br>

  <section class="table-wrap" aria-label="Sharpe optimization">
    <div class="table-heading">
      <strong>Sharpe-style ranking</strong>
      <p>Ranks settings by annualized daily cashflow Sharpe. This is a simulated cashflow proxy, not a live mark-to-market return Sharpe.</p>
    </div>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Strategy</th>
          <th>Daily Sharpe</th>
          <th>Daily Sortino</th>
          <th>Cashflow</th>
          <th>Max drawdown</th>
          <th>Settings</th>
        </tr>
      </thead>
      <tbody>
        {sharpe_rows_html}
      </tbody>
    </table>
  </section>
  <br>

  <section class="table-wrap" aria-label="Walk forward validation">
    <div class="table-heading">
      <strong>Walk-forward validation</strong>
      <p>Optimizes on a rolling train period and applies the selected settings to the next unseen day-ahead period.</p>
    </div>
    <table>
      <thead>
        <tr>
          <th>Window</th>
          <th>Selected strategy</th>
          <th>Test start</th>
          <th>Test end</th>
          <th>Train score</th>
          <th>Test cashflow</th>
          <th>Test drawdown</th>
          <th>Test Sharpe</th>
          <th>Settings</th>
        </tr>
      </thead>
      <tbody>
        {walk_forward_rows_html}
      </tbody>
    </table>
  </section>
  <br>

  <section class="table-wrap" aria-label="Current decision">
    <div class="table-heading">
      <strong>Latest day-ahead decision</strong>
      <p>Applies the best optimized strategy to the latest interval in the forecast file.</p>
    </div>
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Strategy</th>
          <th>Action</th>
          <th>Dispatch MW</th>
          <th>SOC MWh</th>
          <th>Forecast price</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {latest_decision_rows_html}
      </tbody>
    </table>
  </section>
  <br>

  <section class="table-wrap" aria-label="Execution cost stress">
    <div class="table-heading">
      <strong>Execution-cost stress test</strong>
      <p>Uses fee per MWh as a combined transaction-cost and slippage proxy for the best optimized strategy.</p>
    </div>
    <table>
      <thead>
        <tr>
          <th>Fee / MWh</th>
          <th>Cashflow</th>
          <th>Max drawdown</th>
          <th>Risk score</th>
          <th>Daily Sharpe</th>
          <th>Profit factor</th>
          <th>Active intervals</th>
        </tr>
      </thead>
      <tbody>
        {cost_stress_rows_html}
      </tbody>
    </table>
  </section>
  <br>

  <section class="table-wrap" aria-label="Robustness grid">
    <div class="table-heading">
      <strong>Daily-spread robustness grid</strong>
      <p>Top nearby percentile thresholds for the day-ahead daily spread strategy, using a 20 DKK/MWh minimum daily spread.</p>
    </div>
    <table>
      <thead>
        <tr>
          <th>Charge quantile</th>
          <th>Discharge quantile</th>
          <th>Cashflow</th>
          <th>Max drawdown</th>
          <th>Risk score</th>
          <th>Daily Sharpe</th>
          <th>Active intervals</th>
        </tr>
      </thead>
      <tbody>
        {robustness_rows_html}
      </tbody>
    </table>
  </section>
  <br>

  <section class="table-wrap" aria-label="Regime performance">
    <div class="table-heading">
      <strong>Market-regime performance</strong>
      <p>Breaks the best optimized strategy into price, spread, weekday, and time-of-day regimes.</p>
    </div>
    <table>
      <thead>
        <tr>
          <th>Regime type</th>
          <th>Regime</th>
          <th>Cashflow</th>
          <th>Active intervals</th>
          <th>Avg actual price</th>
          <th>Avg forecast error</th>
        </tr>
      </thead>
      <tbody>
        {regime_rows_html}
      </tbody>
    </table>
  </section>
  <br>

  <section class="table-wrap" aria-label="Strategy comparison">
    <table>
      <thead>
        <tr>
          <th>Strategy</th>
          <th>Description</th>
          <th>Cashflow</th>
          <th>Max drawdown</th>
          <th>Active intervals</th>
        </tr>
      </thead>
      <tbody>
        {strategy_rows_html}
      </tbody>
    </table>
  </section>
  <br>

  <section class="table-wrap" aria-label="Model metrics">
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Model</th>
          <th>Type</th>
          <th>MAE</th>
          <th>RMSE</th>
          <th>sMAPE</th>
          <th>R2</th>
        </tr>
      </thead>
      <tbody id="metricsBody"></tbody>
    </table>
  </section>
</main>

<script>
const priceData = {json.dumps(chart_records)};
const cashData = {json.dumps(cash_records)};
const cashSeries = {json.dumps(cash_series)};
const metricRows = {json.dumps(metric_records)};
const colors = {{
  actual: getComputedStyle(document.documentElement).getPropertyValue("--blue").trim(),
  prediction: getComputedStyle(document.documentElement).getPropertyValue("--green").trim(),
  baseline: getComputedStyle(document.documentElement).getPropertyValue("--orange").trim(),
  direct: getComputedStyle(document.documentElement).getPropertyValue("--purple").trim(),
  red: getComputedStyle(document.documentElement).getPropertyValue("--red").trim(),
  axis: getComputedStyle(document.documentElement).getPropertyValue("--muted").trim(),
  fg: getComputedStyle(document.documentElement).getPropertyValue("--fg").trim(),
  border: getComputedStyle(document.documentElement).getPropertyValue("--border").trim()
}};
let windowDays = 7;
const seriesState = {{ actual: true, prediction: true, baseline: true, direct: false }};

function visiblePriceRows() {{
  if (!windowDays) return priceData;
  const last = new Date(priceData[priceData.length - 1].HourUTC.replace(" ", "T"));
  const start = new Date(last.getTime() - windowDays * 24 * 3600 * 1000);
  return priceData.filter(d => new Date(d.HourUTC.replace(" ", "T")) >= start);
}}

function drawLineChart(canvas, rows, series) {{
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * scale);
  canvas.height = Math.round(rect.height * scale);
  ctx.scale(scale, scale);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  const margin = {{ left: 58, right: 18, top: 18, bottom: 42 }};
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const values = [];
  rows.forEach(row => series.forEach(s => {{
    if (s.enabled && Number.isFinite(row[s.key])) values.push(row[s.key]);
  }}));
  if (!rows.length || !values.length) return;
  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  const pad = Math.max((maxY - minY) * 0.08, 1);
  const y0 = minY - pad;
  const y1 = maxY + pad;
  const x = i => margin.left + (rows.length === 1 ? 0 : i / (rows.length - 1)) * innerW;
  const y = v => margin.top + (1 - (v - y0) / (y1 - y0)) * innerH;

  ctx.strokeStyle = colors.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, innerW, innerH);
  ctx.fillStyle = colors.axis;
  ctx.font = "12px Segoe UI, Arial";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {{
    const value = y0 + (i / 4) * (y1 - y0);
    const yy = y(value);
    ctx.strokeStyle = colors.border;
    ctx.beginPath();
    ctx.moveTo(margin.left, yy);
    ctx.lineTo(margin.left + innerW, yy);
    ctx.stroke();
    ctx.fillText(value.toFixed(0), margin.left - 8, yy);
  }}
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const tickCount = Math.min(5, rows.length);
  for (let i = 0; i < tickCount; i++) {{
    const idx = Math.round(i * (rows.length - 1) / Math.max(tickCount - 1, 1));
    ctx.fillText(rows[idx].HourUTC.slice(5), x(idx), margin.top + innerH + 12);
  }}

  series.forEach(s => {{
    if (!s.enabled) return;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 2;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    rows.forEach((row, i) => {{
      const value = row[s.key];
      if (!Number.isFinite(value)) return;
      if (i === 0) ctx.moveTo(x(i), y(value));
      else ctx.lineTo(x(i), y(value));
    }});
    ctx.stroke();
    ctx.setLineDash([]);
  }});
}}

function drawPrice() {{
  const rows = visiblePriceRows();
  drawLineChart(document.getElementById("priceChart"), rows, [
    {{ key: "Actual_Price", enabled: seriesState.actual, color: colors.actual, width: 2.3 }},
    {{ key: "Prediction", enabled: seriesState.prediction, color: colors.prediction, width: 2 }},
    {{ key: "Hourly_Baseline", enabled: seriesState.baseline, color: colors.baseline, width: 1.8, dash: [6, 4] }},
    {{ key: "Direct_15min_Prediction", enabled: seriesState.direct, color: colors.direct, width: 1.8, dash: [2, 4] }}
  ]);
}}

function drawCash() {{
  const rows = cashData;
  const palette = [colors.fg, colors.prediction, colors.baseline, colors.direct, colors.red];
  drawLineChart(document.getElementById("cashChart"), rows, cashSeries.map((s, i) => ({{
    key: s.key,
    enabled: true,
    color: palette[i % palette.length],
    width: 2.1,
    dash: i >= 2 ? [5, 4] : []
  }})));
}}

function renderMetrics() {{
  const body = document.getElementById("metricsBody");
  body.innerHTML = metricRows.map(row => `
    <tr>
      <td class="num">${{row.Rank ?? ""}}</td>
      <td>${{row.Model ?? ""}}</td>
      <td>${{row.Model_Type ?? ""}}</td>
      <td class="num">${{Number(row.MAE).toFixed(4)}}</td>
      <td class="num">${{Number(row.RMSE).toFixed(4)}}</td>
      <td class="num">${{Number(row.sMAPE).toFixed(4)}}</td>
      <td class="num">${{Number(row.R2).toFixed(4)}}</td>
    </tr>`).join("");
}}

document.querySelectorAll("button[data-window]").forEach(button => {{
  button.addEventListener("click", () => {{
    document.querySelectorAll("button[data-window]").forEach(b => b.classList.remove("active"));
    button.classList.add("active");
    windowDays = Number(button.dataset.window);
    drawPrice();
  }});
}});
document.querySelectorAll("input[data-series]").forEach(input => {{
  input.addEventListener("change", () => {{
    seriesState[input.dataset.series] = input.checked;
    drawPrice();
  }});
}});
window.addEventListener("resize", () => {{ drawPrice(); drawCash(); }});
renderMetrics();
drawPrice();
drawCash();
</script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return {
        "forecast_path": str(forecast_path),
        "metrics_path": str(metrics_path),
        "dashboard_path": str(output_path),
        "strategy_table": strategy_table.to_dict(orient="records"),
        "optimization_table": optimization_table.to_dict(orient="records"),
        "risk_adjusted_table": risk_adjusted_table.to_dict(orient="records"),
        "sharpe_table": sharpe_table.to_dict(orient="records"),
        "walk_forward_table": walk_forward_table.to_dict(orient="records"),
        "cost_stress_table": cost_stress.to_dict(orient="records"),
        "robustness_table": robustness_table.to_dict(orient="records"),
        "regime_table": regime_table.to_dict(orient="records"),
        "latest_decision": latest_decision.to_dict(orient="records"),
        "uncertainty_summary": uncertainty_summary,
        "best_strategy": best_strategy.to_dict(),
        "best_optimized_strategy": best_optimized_strategy.to_dict() if best_optimized_strategy is not None else {},
        "best_risk_adjusted_strategy": best_risk_adjusted_strategy.to_dict() if best_risk_adjusted_strategy is not None else {},
        "best_sharpe_strategy": best_sharpe_strategy.to_dict() if best_sharpe_strategy is not None else {},
        "best_model": str(best_model),
        "best_mae": best_mae,
    }


def generate_markdown_report(
    dashboard_result: dict[str, object],
    output_markdown: str | Path,
) -> Path:
    output_path = Path(output_markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_table = dashboard_result["strategy_table"]
    optimization_table = dashboard_result["optimization_table"]
    risk_adjusted_table = dashboard_result["risk_adjusted_table"]
    sharpe_table = dashboard_result["sharpe_table"]
    walk_forward_table = dashboard_result["walk_forward_table"]
    cost_stress_table = dashboard_result["cost_stress_table"]
    robustness_table = dashboard_result["robustness_table"]
    regime_table = dashboard_result["regime_table"]
    latest_decision = dashboard_result["latest_decision"]
    uncertainty = dashboard_result["uncertainty_summary"]
    top_strategy_lines = "\n".join(
        (
            f"- {row['Strategy']}: cashflow {_format_number(float(row['Cashflow']), 2)}, "
            f"max drawdown {_format_number(float(row['Max_Drawdown']), 2)}, "
            f"active intervals {int(row['Active_Intervals'])}"
        )
        for row in strategy_table
    )
    optimization_lines = "\n".join(
        (
            f"- {row['Strategy']}: best cashflow {_format_number(float(row['Cashflow']), 2)}, "
            f"max drawdown {_format_number(float(row['Max_Drawdown']), 2)}, "
            f"risk score {_format_number(float(row['Risk_Adjusted_Score']), 2)}, "
            f"daily Sharpe {_format_number(float(row['Daily_Sharpe']), 2)}, "
            f"runs {int(row['Evaluations'])}, settings `{row['Settings']}`"
        )
        for row in optimization_table
    )
    risk_adjusted_lines = "\n".join(
        (
            f"- {row['Strategy']}: risk score {_format_number(float(row['Risk_Adjusted_Score']), 2)}, "
            f"cashflow {_format_number(float(row['Cashflow']), 2)}, "
            f"max drawdown {_format_number(float(row['Max_Drawdown']), 2)}, "
            f"daily Sharpe {_format_number(float(row['Daily_Sharpe']), 2)}, settings `{row['Settings']}`"
        )
        for row in risk_adjusted_table
    )
    sharpe_lines = "\n".join(
        (
            f"- {row['Strategy']}: daily Sharpe {_format_number(float(row['Daily_Sharpe']), 2)}, "
            f"daily Sortino {_format_number(float(row['Daily_Sortino']), 2)}, "
            f"cashflow {_format_number(float(row['Cashflow']), 2)}, settings `{row['Settings']}`"
        )
        for row in sharpe_table
    )
    walk_forward_lines = "\n".join(
        (
            f"- Window {int(row['Window'])}: selected {row['Selected_Strategy']}, "
            f"test cashflow {_format_number(float(row['Test_Cashflow']), 2)}, "
            f"test max drawdown {_format_number(float(row['Test_Max_Drawdown']), 2)}, "
            f"test daily Sharpe {_format_number(float(row['Test_Daily_Sharpe']), 2)}, "
            f"settings `{row['Settings']}`"
        )
        for row in walk_forward_table
    ) or "- Not enough data for the configured rolling windows."
    cost_lines = "\n".join(
        (
            f"- Fee {_format_number(float(row['Fee_Per_MWh']), 1)} DKK/MWh: "
            f"cashflow {_format_number(float(row['Cashflow']), 2)}, "
            f"risk score {_format_number(float(row['Risk_Adjusted_Score']), 2)}, "
            f"daily Sharpe {_format_number(float(row['Daily_Sharpe']), 2)}"
        )
        for row in cost_stress_table
    )
    robustness_lines = "\n".join(
        (
            f"- charge <= {row['Low_Quantile']:.2f}, discharge >= {row['High_Quantile']:.2f}: "
            f"cashflow {_format_number(float(row['Cashflow']), 2)}, "
            f"risk score {_format_number(float(row['Risk_Adjusted_Score']), 2)}, "
            f"daily Sharpe {_format_number(float(row['Daily_Sharpe']), 2)}"
        )
        for row in robustness_table[:8]
    )
    regime_lines = "\n".join(
        (
            f"- {row['Regime_Type']} / {row['Regime']}: "
            f"cashflow {_format_number(float(row['Cashflow']), 2)}, active intervals {int(row['Active_Intervals'])}"
        )
        for row in regime_table
    )
    latest_decision_line = "- n/a"
    if latest_decision:
        row = latest_decision[0]
        latest_decision_line = (
            f"- {pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')}: "
            f"{row['Strategy']} -> {row['Action']}, dispatch {_format_number(float(row['Dispatch_MW']), 2)} MW, "
            f"forecast {_format_number(float(row['Forecast_Price']), 2)}"
        )
    text = f"""# Day-Ahead Power Trading Project Report

## Current model-run baseline

- Best available model by MAE: {dashboard_result["best_model"]}
- Best MAE: {_format_number(float(dashboard_result["best_mae"]), 4)}
- Forecast source: `{dashboard_result["forecast_path"]}`
- Metrics source: `{dashboard_result["metrics_path"]}`
- Market data mode: day-ahead prices, not intraday order-book transactions

## Strategy simulations

{top_strategy_lines}

## Parameter optimization

Each row is the best setting found for that strategy using the fixed battery assumptions and realized test-period prices.

{optimization_lines}

## Risk-adjusted optimization

Risk score is cashflow minus max drawdown.

{risk_adjusted_lines}

## Sharpe-style ranking

Daily Sharpe is an annualized daily cashflow Sharpe proxy: mean daily simulated cashflow divided by daily cashflow volatility, scaled by sqrt(365). It is not a live mark-to-market return Sharpe.

{sharpe_lines}

## Walk-forward validation

The optimizer trains on rolling historical periods and evaluates the selected strategy on the next unseen period.

{walk_forward_lines}

## Execution-cost stress

Fee per MWh is used as a combined transaction-cost/slippage proxy.

{cost_lines}

## Daily-spread robustness

Top nearby daily-spread percentile settings with minimum daily spread fixed at 20 DKK/MWh.

{robustness_lines}

## Market-regime performance

{regime_lines}

## Forecast uncertainty

- Residual bias: {_format_number(float(uncertainty["Residual_Bias"]), 2)}
- Residual std: {_format_number(float(uncertainty["Residual_Std"]), 2)}
- Post-hoc 80% residual interval coverage: {float(uncertainty["Interval_80_Coverage"]):.1%}
- Latest forecast interval: {_format_number(float(uncertainty["Latest_Interval_Lower"]), 2)} to {_format_number(float(uncertainty["Latest_Interval_Upper"]), 2)}

## Latest day-ahead decision

{latest_decision_line}

## Dashboard

Open `{dashboard_result["dashboard_path"]}` in a browser.
"""
    output_path.write_text(text, encoding="utf-8")
    return output_path
