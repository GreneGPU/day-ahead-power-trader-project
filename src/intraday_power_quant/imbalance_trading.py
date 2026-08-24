from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from intraday_power_quant.prop_trading import PropConfig, _validate_config


def simulate_imbalance_spread_positions(
    signal_simulation: pd.DataFrame,
    config: PropConfig | None = None,
    time_col: str = "HourUTC",
    day_ahead_col: str = "Actual_Price",
    imbalance_col: str = "Imbalance_Price_DKK",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Settle synthetic long/short signals against the same-interval imbalance spread."""

    cfg = config or PropConfig()
    _validate_config(cfg)
    signal_col = "Signal_Action" if "Signal_Action" in signal_simulation.columns else "Action"
    required = [time_col, day_ahead_col, imbalance_col, signal_col]
    missing = [column for column in required if column not in signal_simulation.columns]
    if missing:
        raise KeyError(f"Missing required columns for imbalance simulation: {missing}")

    out = signal_simulation.copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    out = out.sort_values(time_col).reset_index(drop=True)
    signal = out[signal_col].astype(str).str.lower()
    requested_positions = signal.map({"charge": 1, "discharge": -1}).fillna(0).astype(int)
    spread = out[imbalance_col].astype(float) - out[day_ahead_col].astype(float)
    local_dates = out[time_col].dt.tz_convert(cfg.market_timezone).dt.date
    daily_cashflow: dict[object, float] = {}
    rows: list[dict[str, float | int | str]] = []

    for index, requested_position in enumerate(requested_positions):
        date_key = local_dates.iloc[index]
        daily_cashflow.setdefault(date_key, 0.0)
        risk_off = (
            cfg.max_daily_loss_dkk is not None
            and daily_cashflow[date_key] <= -abs(cfg.max_daily_loss_dkk)
        )
        position = 0 if risk_off else int(requested_position)
        traded_mwh = abs(position) * cfg.position_size_mwh
        transaction_cost = traded_mwh * cfg.transaction_cost_dkk_per_mwh
        gross_cashflow = position * float(spread.iloc[index]) * cfg.position_size_mwh
        cashflow = gross_cashflow - transaction_cost
        daily_cashflow[date_key] += cashflow
        action = (
            "risk-off"
            if risk_off
            else "long-spread"
            if position > 0
            else "short-spread"
            if position < 0
            else "flat"
        )
        rows.append(
            {
                "Requested_Position": int(requested_position),
                "Position": position,
                "Position_MWh": position * cfg.position_size_mwh,
                "Day_Ahead_Price_DKK": float(out[day_ahead_col].iloc[index]),
                "Imbalance_Spread_DKK": float(spread.iloc[index]),
                "Price_Change_DKK": float(spread.iloc[index]),
                "Gross_Cashflow": gross_cashflow,
                "Transaction_Cost": transaction_cost,
                "Cashflow": cashflow,
                "Daily_Cashflow": daily_cashflow[date_key],
                "Action": action,
            }
        )

    accounting = pd.DataFrame(rows, index=out.index)
    for column in accounting.columns:
        out[column] = accounting[column]
    out["Dispatch_MW"] = out["Position_MWh"]
    out["State_Of_Charge_MWh"] = np.nan
    out["Cumulative_Cashflow"] = out["Cashflow"].cumsum()
    out["Equity_DKK"] = cfg.initial_capital_dkk + out["Cumulative_Cashflow"]

    cumulative = out["Cumulative_Cashflow"]
    active = out["Position"] != 0
    summary = {
        "total_cashflow": float(out["Cashflow"].sum()),
        "gross_cashflow": float(out["Gross_Cashflow"].sum()),
        "max_drawdown": float((cumulative.cummax() - cumulative).max()),
        "trades": int(active.sum()),
        "position_changes": int(active.sum()),
        "charge_intervals": int((out["Position"] > 0).sum()),
        "discharge_intervals": int((out["Position"] < 0).sum()),
        "long_intervals": int((out["Position"] > 0).sum()),
        "short_intervals": int((out["Position"] < 0).sum()),
        "energy_charged_mwh": 0.0,
        "energy_discharged_mwh": 0.0,
        "total_fee_cost": float(out["Transaction_Cost"].sum()),
        "total_degradation_cost": 0.0,
        "round_trip_efficiency": 1.0,
        "final_soc_mwh": float("nan"),
        "initial_capital_dkk": float(cfg.initial_capital_dkk),
        "ending_equity_dkk": float(out["Equity_DKK"].iloc[-1]),
        "return_pct": float(out["Cashflow"].sum() / cfg.initial_capital_dkk * 100),
    }
    return out, summary


def simulate_imbalance_perfect_foresight(
    df: pd.DataFrame,
    config: PropConfig | None = None,
    time_col: str = "HourUTC",
    day_ahead_col: str = "Actual_Price",
    imbalance_col: str = "Imbalance_Price_DKK",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Choose the hindsight-optimal side of each realized imbalance spread."""

    cfg = config or PropConfig()
    _validate_config(cfg)
    required = [time_col, day_ahead_col, imbalance_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for imbalance benchmark: {missing}")

    frame = df.copy()
    spread = frame[imbalance_col].astype(float) - frame[day_ahead_col].astype(float)
    profitable = spread.abs() > cfg.transaction_cost_dkk_per_mwh
    frame["Forecast_Price"] = frame[forecast_col]
    frame["Signal_Action"] = np.where(
        profitable & (spread > 0),
        "charge",
        np.where(profitable & (spread < 0), "discharge", "hold"),
    )
    out, summary = simulate_imbalance_spread_positions(
        frame,
        replace(cfg, max_daily_loss_dkk=None),
        time_col=time_col,
        day_ahead_col=day_ahead_col,
        imbalance_col=imbalance_col,
    )
    summary["is_hindsight_benchmark"] = True
    return out, summary
