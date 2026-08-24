from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PropConfig:
    """Configuration for the synthetic, asset-free directional trading proxy."""

    initial_capital_dkk: float = 100_000.0
    position_size_mwh: float = 10.0
    transaction_cost_dkk_per_mwh: float = 0.41
    max_daily_loss_dkk: float | None = None
    market_timezone: str = "Europe/Copenhagen"


def _validate_config(config: PropConfig) -> None:
    if config.initial_capital_dkk <= 0:
        raise ValueError("Prop starting capital must be positive.")
    if config.position_size_mwh <= 0:
        raise ValueError("Prop position size must be positive.")
    if config.transaction_cost_dkk_per_mwh < 0:
        raise ValueError("Prop transaction cost must be non-negative.")
    if config.max_daily_loss_dkk is not None and config.max_daily_loss_dkk <= 0:
        raise ValueError("Prop daily loss limit must be positive when enabled.")


def simulate_prop_positions(
    signal_simulation: pd.DataFrame,
    config: PropConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Translate charge/discharge signals into a synthetic long/short price-change proxy.

    A charge signal becomes long, a discharge signal becomes short, and the position
    earns the next observed DK1 price change. This is research accounting, not a
    representation of executable physical day-ahead spot trading.
    """

    cfg = config or PropConfig()
    _validate_config(cfg)
    signal_col = "Signal_Action" if "Signal_Action" in signal_simulation.columns else "Action"
    required = [time_col, actual_col, signal_col]
    missing = [column for column in required if column not in signal_simulation.columns]
    if missing:
        raise KeyError(f"Missing required columns for prop simulation: {missing}")

    out = signal_simulation.copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    out = out.sort_values(time_col).reset_index(drop=True)
    signal = out[signal_col].astype(str).str.lower()
    requested_positions = signal.map({"charge": 1, "discharge": -1}).fillna(0).astype(int)
    if not requested_positions.empty:
        requested_positions.iloc[-1] = 0

    price_change = out[actual_col].astype(float).shift(-1) - out[actual_col].astype(float)
    price_change = price_change.fillna(0.0)
    local_dates = out[time_col].dt.tz_convert(cfg.market_timezone).dt.date
    daily_cashflow: dict[object, float] = {}
    previous_position = 0
    rows: list[dict[str, float | int | str]] = []

    for index, requested_position in enumerate(requested_positions):
        date_key = local_dates.iloc[index]
        daily_cashflow.setdefault(date_key, 0.0)
        risk_off = (
            cfg.max_daily_loss_dkk is not None
            and daily_cashflow[date_key] <= -abs(cfg.max_daily_loss_dkk)
        )
        position = 0 if risk_off else int(requested_position)
        turnover_mwh = abs(position - previous_position) * cfg.position_size_mwh
        transaction_cost = turnover_mwh * cfg.transaction_cost_dkk_per_mwh
        gross_cashflow = position * float(price_change.iloc[index]) * cfg.position_size_mwh
        cashflow = gross_cashflow - transaction_cost
        daily_cashflow[date_key] += cashflow
        action = (
            "risk-off"
            if risk_off
            else "long"
            if position > 0
            else "short"
            if position < 0
            else "exit"
            if previous_position != 0
            else "flat"
        )
        rows.append(
            {
                "Requested_Position": int(requested_position),
                "Position": position,
                "Position_MWh": position * cfg.position_size_mwh,
                "Price_Change_DKK": float(price_change.iloc[index]),
                "Gross_Cashflow": gross_cashflow,
                "Transaction_Cost": transaction_cost,
                "Cashflow": cashflow,
                "Daily_Cashflow": daily_cashflow[date_key],
                "Action": action,
            }
        )
        previous_position = position

    accounting = pd.DataFrame(rows, index=out.index)
    for column in accounting.columns:
        out[column] = accounting[column]
    out["Dispatch_MW"] = out["Position_MWh"]
    out["State_Of_Charge_MWh"] = np.nan
    out["Cumulative_Cashflow"] = out["Cashflow"].cumsum()
    out["Equity_DKK"] = cfg.initial_capital_dkk + out["Cumulative_Cashflow"]

    cumulative = out["Cumulative_Cashflow"]
    active = out["Position"] != 0
    position_changes = out["Position"].ne(out["Position"].shift(fill_value=0))
    summary = {
        "total_cashflow": float(out["Cashflow"].sum()),
        "gross_cashflow": float(out["Gross_Cashflow"].sum()),
        "max_drawdown": float((cumulative.cummax() - cumulative).max()),
        "trades": int(active.sum()),
        "position_changes": int(position_changes.sum()),
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


def simulate_prop_positions_with_eod_imbalance(
    signal_simulation: pd.DataFrame,
    config: PropConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    imbalance_col: str = "Imbalance_Price_DKK",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the Prop proxy and settle positions left open at each local day end.

    Intraday positions earn the next observed day-ahead price move. On the final
    interval of each market day, a remaining position instead settles against the
    realized imbalance spread and is then forced flat before the next day.
    """

    cfg = config or PropConfig()
    _validate_config(cfg)
    signal_col = "Signal_Action" if "Signal_Action" in signal_simulation.columns else "Action"
    required = [time_col, actual_col, imbalance_col, signal_col]
    missing = [column for column in required if column not in signal_simulation.columns]
    if missing:
        raise KeyError(f"Missing required columns for Prop EOD settlement: {missing}")

    out = signal_simulation.copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    out = out.sort_values(time_col).reset_index(drop=True)
    signal = out[signal_col].astype(str).str.lower()
    requested_positions = signal.map({"charge": 1, "discharge": -1}).fillna(0).astype(int)
    actual_price = out[actual_col].astype(float)
    imbalance_price = out[imbalance_col].astype(float)
    next_price_change = (actual_price.shift(-1) - actual_price).fillna(0.0)
    imbalance_spread = imbalance_price - actual_price
    local_dates = out[time_col].dt.tz_convert(cfg.market_timezone).dt.date
    is_day_end = local_dates.ne(local_dates.shift(-1)).fillna(True)
    daily_cashflow: dict[object, float] = {}
    previous_position = 0
    position_change_events = 0
    rows: list[dict[str, float | int | str | bool]] = []

    for index, requested_position in enumerate(requested_positions):
        date_key = local_dates.iloc[index]
        daily_cashflow.setdefault(date_key, 0.0)
        risk_off = (
            cfg.max_daily_loss_dkk is not None
            and daily_cashflow[date_key] <= -abs(cfg.max_daily_loss_dkk)
        )
        position = 0 if risk_off else int(requested_position)
        settles_at_imbalance = bool(is_day_end.iloc[index] and position != 0)
        price_move = (
            float(imbalance_spread.iloc[index])
            if settles_at_imbalance
            else float(next_price_change.iloc[index])
        )
        opening_turnover_mwh = abs(position - previous_position) * cfg.position_size_mwh
        closing_turnover_mwh = cfg.position_size_mwh * abs(position) if settles_at_imbalance else 0.0
        position_change_events += int(position != previous_position) + int(settles_at_imbalance)
        transaction_cost = (
            opening_turnover_mwh + closing_turnover_mwh
        ) * cfg.transaction_cost_dkk_per_mwh
        gross_cashflow = position * price_move * cfg.position_size_mwh
        cashflow = gross_cashflow - transaction_cost
        daily_cashflow[date_key] += cashflow
        position_after_settlement = 0 if bool(is_day_end.iloc[index]) else position
        action = (
            "risk-off"
            if risk_off
            else "imbalance-close-long"
            if settles_at_imbalance and position > 0
            else "imbalance-close-short"
            if settles_at_imbalance and position < 0
            else "long"
            if position > 0
            else "short"
            if position < 0
            else "exit"
            if previous_position != 0
            else "flat"
        )
        rows.append(
            {
                "Requested_Position": int(requested_position),
                "Position": position,
                "Position_MWh": position * cfg.position_size_mwh,
                "Position_After_Settlement": position_after_settlement,
                "Day_Ahead_Price_DKK": float(actual_price.iloc[index]),
                "Imbalance_Spread_DKK": float(imbalance_spread.iloc[index]),
                "Price_Change_DKK": price_move,
                "Gross_Cashflow": gross_cashflow,
                "Transaction_Cost": transaction_cost,
                "Cashflow": cashflow,
                "Daily_Cashflow": daily_cashflow[date_key],
                "Is_Day_End": bool(is_day_end.iloc[index]),
                "EOD_Imbalance_Settlement": settles_at_imbalance,
                "Settlement_Basis": "imbalance" if settles_at_imbalance else "day-ahead proxy",
                "Action": action,
            }
        )
        previous_position = position_after_settlement

    accounting = pd.DataFrame(rows, index=out.index)
    for column in accounting.columns:
        out[column] = accounting[column]
    out["Dispatch_MW"] = out["Position_MWh"]
    out["State_Of_Charge_MWh"] = np.nan
    out["Cumulative_Cashflow"] = out["Cashflow"].cumsum()
    out["Equity_DKK"] = cfg.initial_capital_dkk + out["Cumulative_Cashflow"]

    cumulative = out["Cumulative_Cashflow"]
    active = out["Position"] != 0
    eod_closes = int(out["EOD_Imbalance_Settlement"].sum())
    summary = {
        "total_cashflow": float(out["Cashflow"].sum()),
        "gross_cashflow": float(out["Gross_Cashflow"].sum()),
        "max_drawdown": float((cumulative.cummax() - cumulative).max()),
        "trades": int(active.sum()),
        "position_changes": position_change_events,
        "charge_intervals": int((out["Position"] > 0).sum()),
        "discharge_intervals": int((out["Position"] < 0).sum()),
        "long_intervals": int((out["Position"] > 0).sum()),
        "short_intervals": int((out["Position"] < 0).sum()),
        "eod_imbalance_settlements": eod_closes,
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


def simulate_prop_perfect_foresight(
    df: pd.DataFrame,
    config: PropConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Find the hindsight-optimal long/flat/short path with switching costs."""

    cfg = config or PropConfig()
    _validate_config(cfg)
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for prop perfect foresight: {missing}")

    frame = df.copy()
    frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
    frame = frame.sort_values(time_col).reset_index(drop=True)
    changes = (frame[actual_col].astype(float).shift(-1) - frame[actual_col].astype(float)).fillna(0.0)
    states = (-1, 0, 1)
    scores = {-1: float("-inf"), 0: 0.0, 1: float("-inf")}
    backpointers: list[dict[int, int]] = []

    for index, change in enumerate(changes):
        allowed_states = (0,) if index == len(changes) - 1 else states
        next_scores: dict[int, float] = {}
        previous_for_state: dict[int, int] = {}
        for state in allowed_states:
            candidates = {
                previous: score
                + state * float(change) * cfg.position_size_mwh
                - abs(state - previous)
                * cfg.position_size_mwh
                * cfg.transaction_cost_dkk_per_mwh
                for previous, score in scores.items()
            }
            best_previous = max(candidates, key=candidates.get)
            next_scores[state] = candidates[best_previous]
            previous_for_state[state] = best_previous
        scores = next_scores
        backpointers.append(previous_for_state)

    positions = [0] * len(frame)
    state = 0
    for index in range(len(frame) - 1, -1, -1):
        positions[index] = state
        state = backpointers[index][state]

    signal_frame = frame.copy()
    signal_frame["Forecast_Price"] = signal_frame[forecast_col]
    signal_frame["Signal_Action"] = np.where(
        np.asarray(positions) > 0,
        "charge",
        np.where(np.asarray(positions) < 0, "discharge", "hold"),
    )
    oracle_cfg = replace(cfg, max_daily_loss_dkk=None)
    out, summary = simulate_prop_positions(signal_frame, oracle_cfg, time_col, actual_col)
    summary["is_hindsight_benchmark"] = True
    return out, summary


def simulate_prop_eod_perfect_foresight(
    df: pd.DataFrame,
    config: PropConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    imbalance_col: str = "Imbalance_Price_DKK",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Find the hindsight-optimal daily path under the EOD imbalance rule."""

    cfg = config or PropConfig()
    _validate_config(cfg)
    required = [time_col, actual_col, imbalance_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for Prop EOD benchmark: {missing}")

    frame = df.copy()
    frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
    frame = frame.sort_values(time_col).reset_index(drop=True)
    actual = frame[actual_col].astype(float)
    local_dates = frame[time_col].dt.tz_convert(cfg.market_timezone).dt.date
    is_day_end = local_dates.ne(local_dates.shift(-1)).fillna(True)
    rewards = (actual.shift(-1) - actual).fillna(0.0)
    rewards.loc[is_day_end] = (
        frame.loc[is_day_end, imbalance_col].astype(float) - actual.loc[is_day_end]
    )
    states = (-1, 0, 1)
    positions = np.zeros(len(frame), dtype=int)
    switching_cost = cfg.position_size_mwh * cfg.transaction_cost_dkk_per_mwh

    for _, day_indices in frame.groupby(local_dates, sort=False).groups.items():
        indices = list(day_indices)
        scores = {-1: float("-inf"), 0: 0.0, 1: float("-inf")}
        backpointers: list[dict[int, int]] = []
        for offset, index in enumerate(indices):
            final_interval = offset == len(indices) - 1
            next_scores: dict[int, float] = {}
            previous_for_state: dict[int, int] = {}
            for state in states:
                candidates = {
                    previous: score
                    + state * float(rewards.iloc[index]) * cfg.position_size_mwh
                    - abs(state - previous) * switching_cost
                    - (abs(state) * switching_cost if final_interval else 0.0)
                    for previous, score in scores.items()
                }
                best_previous = max(candidates, key=candidates.get)
                next_scores[state] = candidates[best_previous]
                previous_for_state[state] = best_previous
            scores = next_scores
            backpointers.append(previous_for_state)
        state = max(scores, key=scores.get)
        for offset in range(len(indices) - 1, -1, -1):
            positions[indices[offset]] = state
            state = backpointers[offset][state]

    signal_frame = frame.copy()
    signal_frame["Forecast_Price"] = signal_frame[forecast_col]
    signal_frame["Signal_Action"] = np.where(
        positions > 0,
        "charge",
        np.where(positions < 0, "discharge", "hold"),
    )
    out, summary = simulate_prop_positions_with_eod_imbalance(
        signal_frame,
        replace(cfg, max_daily_loss_dkk=None),
        time_col=time_col,
        actual_col=actual_col,
        imbalance_col=imbalance_col,
    )
    summary["is_hindsight_benchmark"] = True
    return out, summary
