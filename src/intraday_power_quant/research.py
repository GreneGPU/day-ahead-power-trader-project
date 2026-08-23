from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from .optimization import best_parameter_rows, run_strategy_parameter_sweep, simulate_strategy_from_settings
from .risk import summarize_cashflow_risk
from .trading import BatteryConfig, DailySpreadConfig, simulate_daily_spread_rank_arbitrage


def _sorted_frame(df: pd.DataFrame, time_col: str = "HourUTC") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    return out.sort_values(time_col).reset_index(drop=True)


def walk_forward_strategy_optimization(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    train_days: float = 14.0,
    test_days: float = 7.0,
    step_days: float = 7.0,
    ranking_metric: str = "Risk_Adjusted_Score",
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> pd.DataFrame:
    """Optimize on rolling train windows and evaluate the selected strategy on unseen next windows."""
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days, and step_days must be positive.")

    battery = battery_config or BatteryConfig()
    data = _sorted_frame(df, time_col)
    start = data[time_col].min()
    end = data[time_col].max()
    rows: list[dict[str, object]] = []
    window_id = 1
    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)

    window_start = start
    while window_start + train_delta + test_delta <= end + pd.Timedelta(seconds=1):
        train_start = window_start
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        train = data[(data[time_col] >= train_start) & (data[time_col] < train_end)]
        test = data[(data[time_col] >= test_start) & (data[time_col] < test_end)]

        if not train.empty and not test.empty:
            train_sweep = run_strategy_parameter_sweep(train, battery, time_col, actual_col, forecast_col)
            train_best = best_parameter_rows(train_sweep, ranking_metric=ranking_metric).iloc[0]
            test_sim, test_summary = simulate_strategy_from_settings(
                str(train_best["Strategy"]),
                test,
                str(train_best["Settings_JSON"]),
                battery,
                time_col,
                actual_col,
                forecast_col,
            )
            test_risk = summarize_cashflow_risk(test_sim, time_col)
            rows.append(
                {
                    "Window": window_id,
                    "Train_Start": train_start,
                    "Train_End": train_end,
                    "Test_Start": test_start,
                    "Test_End": test_end,
                    "Selected_Strategy": train_best["Strategy"],
                    "Settings": train_best["Settings"],
                    "Train_Cashflow": float(train_best["Cashflow"]),
                    "Train_Max_Drawdown": float(train_best["Max_Drawdown"]),
                    "Train_Risk_Adjusted_Score": float(train_best["Risk_Adjusted_Score"]),
                    "Test_Cashflow": float(test_summary["total_cashflow"]),
                    "Test_Max_Drawdown": float(test_summary["max_drawdown"]),
                    "Test_Risk_Adjusted_Score": float(test_risk["risk_adjusted_score"]),
                    "Test_Calmar_Ratio": float(test_risk["calmar_ratio"]),
                    "Test_Daily_Sharpe": float(test_risk["daily_sharpe"]),
                    "Test_Daily_Sortino": float(test_risk["daily_sortino"]),
                    "Test_Win_Rate": float(test_risk["win_rate"]),
                    "Test_Active_Intervals": int(test_summary["trades"]),
                }
            )
            window_id += 1

        window_start += step_delta

    return pd.DataFrame(rows)


def execution_cost_stress_table(
    df: pd.DataFrame,
    strategy: str,
    settings_json: str,
    battery_config: BatteryConfig | None = None,
    fees_per_mwh: Iterable[float] = (0.0, 2.5, 5.0, 10.0, 20.0),
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> pd.DataFrame:
    battery = battery_config or BatteryConfig()
    rows: list[dict[str, object]] = []
    for fee in fees_per_mwh:
        fee_battery = replace(battery, fee_per_mwh=float(fee))
        sim, summary = simulate_strategy_from_settings(
            strategy,
            df,
            settings_json,
            fee_battery,
            time_col,
            actual_col,
            forecast_col,
        )
        risk = summarize_cashflow_risk(sim, time_col)
        rows.append(
            {
                "Fee_Per_MWh": float(fee),
                "Strategy": strategy,
                "Cashflow": float(summary["total_cashflow"]),
                "Max_Drawdown": float(summary["max_drawdown"]),
                "Risk_Adjusted_Score": float(risk["risk_adjusted_score"]),
                "Daily_Sharpe": float(risk["daily_sharpe"]),
                "Daily_Sortino": float(risk["daily_sortino"]),
                "Profit_Factor": float(risk["profit_factor"]),
                "Active_Intervals": int(summary["trades"]),
            }
        )
    return pd.DataFrame(rows)


def daily_spread_robustness_grid(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    min_daily_spread: float = 20.0,
    low_quantiles: Iterable[float] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35),
    high_quantiles: Iterable[float] = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> pd.DataFrame:
    battery = battery_config or BatteryConfig()
    rows: list[dict[str, object]] = []
    for low in low_quantiles:
        for high in high_quantiles:
            if low >= high:
                continue
            cfg = DailySpreadConfig(
                low_quantile=float(low),
                high_quantile=float(high),
                rank_mode="percentile",
                min_daily_spread=float(min_daily_spread),
            )
            sim, summary = simulate_daily_spread_rank_arbitrage(df, battery, cfg, time_col, actual_col, forecast_col)
            risk = summarize_cashflow_risk(sim, time_col)
            rows.append(
                {
                    "Low_Quantile": float(low),
                    "High_Quantile": float(high),
                    "Min_Daily_Spread": float(min_daily_spread),
                    "Cashflow": float(summary["total_cashflow"]),
                    "Max_Drawdown": float(summary["max_drawdown"]),
                    "Risk_Adjusted_Score": float(risk["risk_adjusted_score"]),
                    "Daily_Sharpe": float(risk["daily_sharpe"]),
                    "Daily_Sortino": float(risk["daily_sortino"]),
                    "Active_Intervals": int(summary["trades"]),
                }
            )
    return pd.DataFrame(rows).sort_values("Cashflow", ascending=False).reset_index(drop=True)


def regime_performance_table(
    forecasts: pd.DataFrame,
    strategy_sim: pd.DataFrame,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> pd.DataFrame:
    data = _sorted_frame(forecasts[[time_col, actual_col, forecast_col]], time_col)
    sim = strategy_sim[[time_col, "Action", "Cashflow"]].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    data = data.merge(sim, on=time_col, how="left")
    data["Cashflow"] = data["Cashflow"].fillna(0.0)
    data["Action"] = data["Action"].fillna("hold")
    data["Forecast_Error"] = data[actual_col] - data[forecast_col]
    data["Trade_Date"] = data[time_col].dt.date
    data["Daily_Forecast_Spread"] = data.groupby("Trade_Date")[forecast_col].transform(lambda s: s.max() - s.min())
    spread_median = float(data["Daily_Forecast_Spread"].median())
    price_median = float(data[actual_col].median())
    data["Volatility_Regime"] = np.where(
        data["Daily_Forecast_Spread"] >= spread_median,
        "High daily spread",
        "Low daily spread",
    )
    data["Price_Regime"] = np.select(
        [data[actual_col] < 0, data[actual_col] >= price_median],
        ["Negative price", "High price"],
        default="Low price",
    )
    data["Day_Type"] = np.where(data[time_col].dt.dayofweek >= 5, "Weekend", "Weekday")
    hour = data[time_col].dt.hour
    data["Hour_Bucket"] = np.select(
        [hour < 6, hour < 12, hour < 18],
        ["Night", "Morning", "Afternoon"],
        default="Evening",
    )

    rows: list[dict[str, object]] = []
    for regime_type in ["Volatility_Regime", "Price_Regime", "Day_Type", "Hour_Bucket"]:
        for regime, group in data.groupby(regime_type):
            rows.append(
                {
                    "Regime_Type": regime_type.replace("_", " "),
                    "Regime": str(regime),
                    "Cashflow": float(group["Cashflow"].sum()),
                    "Intervals": int(len(group)),
                    "Active_Intervals": int((group["Action"] != "hold").sum()),
                    "Average_Actual_Price": float(group[actual_col].mean()),
                    "Average_Forecast_Error": float(group["Forecast_Error"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["Regime_Type", "Cashflow"], ascending=[True, False])


def forecast_uncertainty_summary(
    forecasts: pd.DataFrame,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> dict[str, object]:
    data = _sorted_frame(forecasts[[time_col, actual_col, forecast_col]], time_col)
    residual = data[actual_col].astype(float) - data[forecast_col].astype(float)
    q10 = float(residual.quantile(0.10))
    q90 = float(residual.quantile(0.90))
    lower = data[forecast_col] + q10
    upper = data[forecast_col] + q90
    coverage = float(((data[actual_col] >= lower) & (data[actual_col] <= upper)).mean())
    latest = data.iloc[-1]
    return {
        "Residual_Bias": float(residual.mean()),
        "Residual_Std": float(residual.std(ddof=0)),
        "Residual_Q10": q10,
        "Residual_Q90": q90,
        "Interval_80_Coverage": coverage,
        "Latest_Timestamp": latest[time_col],
        "Latest_Forecast": float(latest[forecast_col]),
        "Latest_Actual": float(latest[actual_col]),
        "Latest_Interval_Lower": float(latest[forecast_col] + q10),
        "Latest_Interval_Upper": float(latest[forecast_col] + q90),
    }


def latest_decision_table(
    df: pd.DataFrame,
    strategy: str,
    settings_json: str,
    battery_config: BatteryConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> pd.DataFrame:
    battery = battery_config or BatteryConfig()
    sim, _ = simulate_strategy_from_settings(strategy, df, settings_json, battery, time_col, actual_col, forecast_col)
    row = sim.iloc[-1]
    reason_parts = []
    for column in [
        "Forecast_Price",
        "Daily_Forecast_Rank_Pct",
        "Daily_Forecast_Spread",
        "Forecast_Momentum_DKK",
        "Forecast_Z",
        "Rolling_Forecast_Std",
        "Agreement_Charge_Share",
        "Agreement_Discharge_Share",
    ]:
        if column in sim.columns:
            value = row[column]
            if isinstance(value, (int, float, np.floating)):
                reason_parts.append(f"{column}={float(value):.2f}")
    return pd.DataFrame(
        [
            {
                "Timestamp": row[time_col],
                "Strategy": strategy,
                "Action": row["Action"],
                "Dispatch_MW": float(row["Dispatch_MW"]),
                "State_Of_Charge_MWh": float(row["State_Of_Charge_MWh"]),
                "Forecast_Price": float(row["Forecast_Price"]) if "Forecast_Price" in sim.columns else float("nan"),
                "Cashflow_If_Settled": float(row["Cashflow"]),
                "Reason": "; ".join(reason_parts),
            }
        ]
    )
