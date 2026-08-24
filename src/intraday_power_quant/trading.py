from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatteryConfig:
    capacity_mwh: float = 100.0
    power_mw: float = 25.0
    initial_soc_mwh: float = 0.0
    charge_efficiency: float = 0.90**0.5
    discharge_efficiency: float = 0.90**0.5
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    fee_per_mwh: float = 0.0
    charge_fee_per_mwh: float = 0.0
    discharge_fee_per_mwh: float = 0.0
    max_daily_loss: float | None = None


@dataclass(frozen=True)
class WeeklyBandConfig:
    window_days: float = 7.0
    band: float = 20.0
    min_history_days: float = 1.0


@dataclass(frozen=True)
class ForecastEdgeConfig:
    threshold: float = 5.0
    reference_col: str = "Hourly_Baseline"


@dataclass(frozen=True)
class VolatilityFilterConfig:
    average_window_days: float = 7.0
    volatility_window_days: float = 7.0
    min_volatility: float = 30.0
    price_band: float = 0.0
    min_history_days: float = 1.0


@dataclass(frozen=True)
class MeanReversionConfig:
    window_days: float = 7.0
    entry_z: float = 1.0
    min_history_days: float = 1.0


@dataclass(frozen=True)
class MomentumConfig:
    lookback_hours: float = 6.0
    threshold: float = 5.0
    smoothing_hours: float = 1.0


@dataclass(frozen=True)
class MomentumSpreadConfig:
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    rank_mode: str = "percentile"
    charge_rank: int = 24
    discharge_rank: int = 73
    min_daily_spread: float = 20.0
    lookback_hours: float = 6.0
    momentum_threshold: float = 5.0
    smoothing_hours: float = 1.0


@dataclass(frozen=True)
class ChannelBreakoutConfig:
    window_days: float = 3.0
    buffer: float = 0.0
    min_history_days: float = 1.0


@dataclass(frozen=True)
class DailySpreadConfig:
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    rank_mode: str = "percentile"
    charge_rank: int = 24
    discharge_rank: int = 73
    min_daily_spread: float = 0.0


@dataclass(frozen=True)
class BestHoursConfig:
    hours_per_day: int = 2
    min_profit_dkk_per_mwh: float = 0.0
    market_timezone: str = "Europe/Copenhagen"


@dataclass(frozen=True)
class EnsembleAgreementConfig:
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    min_agreement: float = 0.60
    max_model_spread: float | None = None


@dataclass(frozen=True)
class RollingOptimizerConfig:
    soc_steps: int = 40
    terminal_soc_mwh: float = 0.0
    market_timezone: str = "Europe/Copenhagen"


@dataclass(frozen=True)
class UncertaintyOptimizerConfig:
    uncertainty_penalty: float = 1.0
    max_model_spread: float | None = 80.0
    soc_steps: int = 40
    terminal_soc_mwh: float = 0.0
    market_timezone: str = "Europe/Copenhagen"


@dataclass(frozen=True)
class DegradationOptimizerConfig:
    degradation_cost_per_mwh: float = 40.0
    soc_steps: int = 40
    terminal_soc_mwh: float = 0.0
    market_timezone: str = "Europe/Copenhagen"


@dataclass(frozen=True)
class WindSignalConfig:
    low_wind_quantile: float = 0.25
    high_wind_quantile: float = 0.75
    ramp_threshold_mw: float = 100.0
    market_timezone: str = "Europe/Copenhagen"


@dataclass(frozen=True)
class WindConfirmedOptimizerConfig:
    low_wind_quantile: float = 0.25
    high_wind_quantile: float = 0.75
    ramp_threshold_mw: float = 100.0
    soc_steps: int = 40
    terminal_soc_mwh: float = 0.0
    market_timezone: str = "Europe/Copenhagen"


PREFERRED_ENSEMBLE_COLUMNS = [
    "TL_Residual_XGB",
    "TL_Residual_LGBM",
    "TL_Residual_CAT",
    "TL_Residual_Average",
    "TL_Residual_Stacked",
]

FALLBACK_ENSEMBLE_COLUMNS = [
    "Direct_15min_XGB",
    "Direct_15min_LGBM",
    "Direct_15min_CAT",
    "Direct_15min_Average",
    "Direct_15min_Stacked",
    "Prediction",
]


STRATEGY_DESCRIPTIONS = {
    "Forecast quantile": (
        "Charges in the lowest forecast-price quantile and discharges in the highest forecast-price "
        "quantile across the test period."
    ),
    "Weekly average band": (
        "Compares each forecast with a trailing moving average, normally the last week. It charges "
        "when the forecast is X DKK/MWh below the average and discharges when it is X above."
    ),
    "Forecast edge": (
        "Compares the 15-minute forecast with the hourly baseline. It charges when the 15-minute "
        "forecast is sufficiently below the hourly baseline and discharges when it is above."
    ),
    "Volatility filtered average": (
        "Trades the moving-average rule only during high-volatility regimes. It charges below the "
        "rolling average and discharges above it when rolling forecast std is high enough."
    ),
    "Mean reversion": (
        "Uses a rolling forecast z-score. It treats very low forecasts versus the recent mean as buy/charge "
        "opportunities and very high forecasts as sell/discharge opportunities."
    ),
    "Momentum": (
        "Compares the current forecast with a smoothed forecast from a chosen lookback. It charges on "
        "downward price momentum and discharges on upward price momentum."
    ),
    "Momentum spread": (
        "Combines daily spread rank with momentum confirmation. It charges cheap daily ranks only when "
        "forecast momentum is falling and discharges expensive ranks only when momentum is rising."
    ),
    "Channel breakout": (
        "Uses recent rolling high and low forecast channels. It charges on downside breaks and discharges "
        "on upside breaks."
    ),
    "Daily spread rank": (
        "Ranks forecast prices within each day and can require a minimum daily forecast spread before "
        "trading the cheapest and most expensive intervals."
    ),
    "Predicted best hours": (
        "Pairs the cheapest predicted DK1 hours with later expensive predicted hours. It only charges "
        "when the predicted sale covers round-trip losses, charge fees, discharge fees, and the selected "
        "minimum paper-profit margin."
    ),
    "Ensemble agreement": (
        "Uses the available ensemble prediction series as a confidence filter. It trades only when enough "
        "model outputs agree that the interval is cheap or expensive."
    ),
    "Rolling price optimizer": (
        "Uses a daily dynamic program to choose the complete predicted-price charge/discharge path while "
        "respecting battery power, state of charge, efficiency, fees, and an empty end-of-day target."
    ),
    "Uncertainty-aware optimizer": (
        "Optimizes the daily battery path using conservative buy and sell prices. Wider disagreement between "
        "the three forecast series makes a trade less attractive or blocks it entirely."
    ),
    "Degradation-aware optimizer": (
        "Runs the daily price optimizer after deducting an explicit battery-wear cost from every MWh charged "
        "and discharged, so marginal cycles are skipped."
    ),
    "Wind signal": (
        "A feature-only benchmark using Energinet DK1 day-ahead onshore and offshore wind forecasts. It charges "
        "during high or sharply rising wind and discharges during low or sharply falling wind; prices do not "
        "choose the action."
    ),
    "Wind-confirmed optimizer": (
        "Uses the predicted-price dynamic optimizer but permits charging only during high/rising forecast wind "
        "and discharging only during low/falling forecast wind."
    ),
}


def forecast_spread_signals(
    df: pd.DataFrame,
    forecast_col: str = "Prediction",
    reference_col: str = "Hourly_Baseline",
    threshold: float = 5.0,
) -> pd.DataFrame:
    output = df.copy()
    output["Forecast_Edge"] = output[forecast_col] - output[reference_col]
    output["Signal"] = np.where(
        output["Forecast_Edge"] >= threshold,
        "high-price",
        np.where(output["Forecast_Edge"] <= -threshold, "low-price", "neutral"),
    )
    return output


def _infer_step_hours(times: pd.Series) -> float:
    ordered = pd.Series(pd.to_datetime(times)).sort_values()
    diffs = ordered.diff().dropna()
    if diffs.empty:
        return 0.25
    return float(diffs.median() / pd.Timedelta(hours=1))


def _rolling_periods(times: pd.Series, window_days: float, min_history_days: float) -> tuple[int, int]:
    if window_days <= 0 or min_history_days <= 0:
        raise ValueError("window_days and min_history_days must be positive.")
    step_hours = _infer_step_hours(times)
    periods_per_day = max(int(round(24 / step_hours)), 1)
    window = max(int(round(window_days * periods_per_day)), 1)
    min_periods = max(int(round(min_history_days * periods_per_day)), 1)
    return window, min_periods


def _periods_from_hours(times: pd.Series, hours: float, minimum: int = 1) -> int:
    if hours < 0:
        raise ValueError("hours must be non-negative.")
    step_hours = _infer_step_hours(times)
    if hours == 0:
        return minimum
    return max(int(round(hours / step_hours)), minimum)


def _available_ensemble_columns(df: pd.DataFrame) -> list[str]:
    preferred = [column for column in PREFERRED_ENSEMBLE_COLUMNS if column in df.columns]
    if preferred:
        return preferred
    return [column for column in FALLBACK_ENSEMBLE_COLUMNS if column in df.columns]


def _simulate_battery_dispatch(
    sim: pd.DataFrame,
    cfg: BatteryConfig,
    time_col: str,
    actual_col: str,
    action_col: str,
    degradation_cost_per_mwh: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not 0 < cfg.charge_efficiency <= 1 or not 0 < cfg.discharge_efficiency <= 1:
        raise ValueError("Charge and discharge efficiencies must be greater than 0 and at most 1.")
    if min(cfg.fee_per_mwh, cfg.charge_fee_per_mwh, cfg.discharge_fee_per_mwh) < 0:
        raise ValueError("Battery fees must be non-negative.")
    if degradation_cost_per_mwh < 0:
        raise ValueError("Battery degradation cost must be non-negative.")
    step_hours = _infer_step_hours(sim[time_col])
    soc = min(max(cfg.initial_soc_mwh, 0.0), cfg.capacity_mwh)
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    daily_cashflow: dict[object, float] = {}

    for row in sim.itertuples(index=False):
        row_dict = row._asdict()
        timestamp = row_dict[time_col]
        actual_price = float(row_dict[actual_col])
        requested_action = str(row_dict[action_col])
        requested_power = row_dict.get("Requested_Power_MW", cfg.power_mw)
        requested_power = (
            cfg.power_mw
            if requested_power is None or pd.isna(requested_power)
            else min(abs(float(requested_power)), cfg.power_mw)
        )
        date_key = timestamp.date()
        daily_cashflow.setdefault(date_key, 0.0)

        action = "hold"
        dispatch_mw = 0.0
        cashflow = 0.0
        blocked_by_risk = (
            cfg.max_daily_loss is not None and daily_cashflow[date_key] <= -abs(cfg.max_daily_loss)
        )

        if not blocked_by_risk and requested_action == "charge" and soc < cfg.capacity_mwh:
            max_charge_mw = (cfg.capacity_mwh - soc) / (step_hours * cfg.charge_efficiency)
            dispatch_mw = -min(requested_power, max_charge_mw)
            energy_mwh = abs(dispatch_mw) * step_hours
            soc += energy_mwh * cfg.charge_efficiency
            cashflow = -energy_mwh * actual_price - energy_mwh * (
                cfg.fee_per_mwh + cfg.charge_fee_per_mwh
                + degradation_cost_per_mwh
            )
            action = "charge"
        elif not blocked_by_risk and requested_action == "discharge" and soc > 0:
            max_discharge_mw = soc * cfg.discharge_efficiency / step_hours
            dispatch_mw = min(requested_power, max_discharge_mw)
            energy_mwh = dispatch_mw * step_hours
            soc -= energy_mwh / cfg.discharge_efficiency
            cashflow = energy_mwh * actual_price - energy_mwh * (
                cfg.fee_per_mwh + cfg.discharge_fee_per_mwh
                + degradation_cost_per_mwh
            )
            action = "discharge"
        elif blocked_by_risk:
            action = "risk-off"

        soc = min(max(soc, 0.0), cfg.capacity_mwh)
        daily_cashflow[date_key] += cashflow
        rows.append(
            {
                **row_dict,
                "Signal_Action": requested_action,
                "Action": action,
                "Dispatch_MW": dispatch_mw,
                "State_Of_Charge_MWh": soc,
                "Cashflow": cashflow,
                "Daily_Cashflow": daily_cashflow[date_key],
            }
        )

    out = pd.DataFrame(rows).drop(columns=[action_col])
    out["Cumulative_Cashflow"] = out["Cashflow"].cumsum()
    total_energy_charged = float((-out.loc[out["Dispatch_MW"] < 0, "Dispatch_MW"] * step_hours).sum())
    total_energy_discharged = float((out.loc[out["Dispatch_MW"] > 0, "Dispatch_MW"] * step_hours).sum())
    total_fee_cost = (
        total_energy_charged * (cfg.fee_per_mwh + cfg.charge_fee_per_mwh)
        + total_energy_discharged * (cfg.fee_per_mwh + cfg.discharge_fee_per_mwh)
    )
    total_degradation_cost = (
        total_energy_charged + total_energy_discharged
    ) * degradation_cost_per_mwh
    summary = {
        "total_cashflow": float(out["Cashflow"].sum()),
        "max_drawdown": float((out["Cumulative_Cashflow"].cummax() - out["Cumulative_Cashflow"]).max()),
        "trades": int((out["Action"] != "hold").sum()),
        "charge_intervals": int((out["Action"] == "charge").sum()),
        "discharge_intervals": int((out["Action"] == "discharge").sum()),
        "energy_charged_mwh": total_energy_charged,
        "energy_discharged_mwh": total_energy_discharged,
        "total_fee_cost": float(total_fee_cost),
        "total_degradation_cost": float(total_degradation_cost),
        "round_trip_efficiency": float(cfg.charge_efficiency * cfg.discharge_efficiency),
        "step_hours": step_hours,
        "final_soc_mwh": float(out["State_Of_Charge_MWh"].iloc[-1]) if not out.empty else float("nan"),
    }
    return out, summary


def simulate_battery_arbitrage(
    df: pd.DataFrame,
    config: BatteryConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    cfg = config or BatteryConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for battery simulation: {missing}")
    if not 0 <= cfg.low_quantile < cfg.high_quantile <= 1:
        raise ValueError("Battery quantiles must satisfy 0 <= low < high <= 1.")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)
    low_threshold = float(sim[forecast_col].quantile(cfg.low_quantile))
    high_threshold = float(sim[forecast_col].quantile(cfg.high_quantile))
    sim["Requested_Action"] = np.where(
        sim[forecast_col] <= low_threshold,
        "charge",
        np.where(sim[forecast_col] >= high_threshold, "discharge", "hold"),
    )

    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(sim, cfg, time_col, actual_col, "Requested_Action")
    summary.update({"low_threshold": low_threshold, "high_threshold": high_threshold})
    return out, summary


def simulate_weekly_average_band_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    band_config: WeeklyBandConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    band = band_config or WeeklyBandConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for weekly-band simulation: {missing}")
    if band.band < 0:
        raise ValueError("band must be non-negative.")
    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)
    window, min_periods = _rolling_periods(sim[time_col], band.window_days, band.min_history_days)

    shifted_forecast = sim[forecast_col].shift(1)
    weekly_average = shifted_forecast.rolling(window=window, min_periods=min_periods).mean()
    fallback_average = shifted_forecast.expanding(min_periods=1).mean()
    sim["Weekly_Average_Forecast"] = weekly_average.combine_first(fallback_average).fillna(sim[forecast_col])
    sim["Buy_Threshold"] = sim["Weekly_Average_Forecast"] - band.band
    sim["Sell_Threshold"] = sim["Weekly_Average_Forecast"] + band.band
    sim["Requested_Action"] = np.where(
        sim[forecast_col] <= sim["Buy_Threshold"],
        "charge",
        np.where(sim[forecast_col] >= sim["Sell_Threshold"], "discharge", "hold"),
    )

    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(sim, battery, time_col, actual_col, "Requested_Action")
    summary.update(
        {
            "band": float(band.band),
            "window_days": float(band.window_days),
            "min_history_days": float(band.min_history_days),
            "average_buy_threshold": float(out["Buy_Threshold"].mean()),
            "average_sell_threshold": float(out["Sell_Threshold"].mean()),
        }
    )
    return out, summary


def simulate_forecast_edge_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    edge_config: ForecastEdgeConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = edge_config or ForecastEdgeConfig()
    required = [time_col, actual_col, forecast_col, cfg.reference_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for forecast-edge simulation: {missing}")
    if cfg.threshold < 0:
        raise ValueError("threshold must be non-negative.")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)
    sim["Forecast_Edge_DKK"] = sim[forecast_col] - sim[cfg.reference_col]
    sim["Buy_Threshold"] = sim[cfg.reference_col] - cfg.threshold
    sim["Sell_Threshold"] = sim[cfg.reference_col] + cfg.threshold
    sim["Requested_Action"] = np.where(
        sim["Forecast_Edge_DKK"] <= -cfg.threshold,
        "charge",
        np.where(sim["Forecast_Edge_DKK"] >= cfg.threshold, "discharge", "hold"),
    )

    sim = sim.rename(columns={forecast_col: "Forecast_Price", cfg.reference_col: "Reference_Price"})
    out, summary = _simulate_battery_dispatch(sim, battery, time_col, actual_col, "Requested_Action")
    summary.update(
        {
            "threshold": float(cfg.threshold),
            "reference_col": cfg.reference_col,
            "average_edge_dkk": float(out["Forecast_Edge_DKK"].mean()),
        }
    )
    return out, summary


def simulate_volatility_filtered_average_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    volatility_config: VolatilityFilterConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = volatility_config or VolatilityFilterConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for volatility-filtered simulation: {missing}")
    if cfg.average_window_days <= 0 or cfg.volatility_window_days <= 0 or cfg.min_history_days <= 0:
        raise ValueError("average_window_days, volatility_window_days, and min_history_days must be positive.")
    if cfg.min_volatility < 0 or cfg.price_band < 0:
        raise ValueError("min_volatility and price_band must be non-negative.")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)
    average_window, average_min_periods = _rolling_periods(
        sim[time_col],
        cfg.average_window_days,
        cfg.min_history_days,
    )
    volatility_window, volatility_min_periods = _rolling_periods(
        sim[time_col],
        cfg.volatility_window_days,
        cfg.min_history_days,
    )

    shifted_forecast = sim[forecast_col].shift(1)
    rolling_average = shifted_forecast.rolling(
        window=average_window,
        min_periods=average_min_periods,
    ).mean()
    rolling_std = shifted_forecast.rolling(
        window=volatility_window,
        min_periods=volatility_min_periods,
    ).std()
    fallback_average = shifted_forecast.expanding(min_periods=1).mean()
    fallback_std = shifted_forecast.expanding(min_periods=2).std()

    sim["Rolling_Average_Forecast"] = rolling_average.combine_first(fallback_average).fillna(sim[forecast_col])
    sim["Rolling_Forecast_Std"] = rolling_std.combine_first(fallback_std).fillna(0.0)
    sim["High_Volatility"] = sim["Rolling_Forecast_Std"] >= cfg.min_volatility
    sim["Buy_Threshold"] = sim["Rolling_Average_Forecast"] - cfg.price_band
    sim["Sell_Threshold"] = sim["Rolling_Average_Forecast"] + cfg.price_band
    sim["Requested_Action"] = np.where(
        sim["High_Volatility"] & (sim[forecast_col] <= sim["Buy_Threshold"]),
        "charge",
        np.where(
            sim["High_Volatility"] & (sim[forecast_col] >= sim["Sell_Threshold"]),
            "discharge",
            "hold",
        ),
    )

    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(sim, battery, time_col, actual_col, "Requested_Action")
    summary.update(
        {
            "average_window_days": float(cfg.average_window_days),
            "volatility_window_days": float(cfg.volatility_window_days),
            "min_volatility": float(cfg.min_volatility),
            "price_band": float(cfg.price_band),
            "min_history_days": float(cfg.min_history_days),
            "high_volatility_intervals": int(out["High_Volatility"].sum()),
            "average_rolling_std": float(out["Rolling_Forecast_Std"].mean()),
        }
    )
    return out, summary


def simulate_mean_reversion_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    mean_reversion_config: MeanReversionConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = mean_reversion_config or MeanReversionConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for mean-reversion simulation: {missing}")
    if cfg.entry_z <= 0:
        raise ValueError("entry_z must be positive.")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)
    window, min_periods = _rolling_periods(sim[time_col], cfg.window_days, cfg.min_history_days)

    shifted_forecast = sim[forecast_col].shift(1)
    rolling_mean = shifted_forecast.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = shifted_forecast.rolling(window=window, min_periods=min_periods).std()
    fallback_mean = shifted_forecast.expanding(min_periods=1).mean()
    fallback_std = shifted_forecast.expanding(min_periods=2).std()
    global_std = float(sim[forecast_col].std()) if len(sim) > 1 else 1.0

    sim["Rolling_Mean_Forecast"] = rolling_mean.combine_first(fallback_mean).fillna(sim[forecast_col])
    sim["Rolling_Std_Forecast"] = (
        rolling_std.combine_first(fallback_std).replace(0, np.nan).fillna(global_std if global_std > 0 else 1.0)
    )
    sim["Forecast_Z"] = (sim[forecast_col] - sim["Rolling_Mean_Forecast"]) / sim["Rolling_Std_Forecast"]
    sim["Requested_Action"] = np.where(
        sim["Forecast_Z"] <= -cfg.entry_z,
        "charge",
        np.where(sim["Forecast_Z"] >= cfg.entry_z, "discharge", "hold"),
    )

    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(sim, battery, time_col, actual_col, "Requested_Action")
    summary.update(
        {
            "entry_z": float(cfg.entry_z),
            "window_days": float(cfg.window_days),
            "min_history_days": float(cfg.min_history_days),
            "average_z": float(out["Forecast_Z"].mean()),
        }
    )
    return out, summary


def simulate_ensemble_agreement_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    ensemble_config: EnsembleAgreementConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = ensemble_config or EnsembleAgreementConfig()
    if not 0 <= cfg.low_quantile < cfg.high_quantile <= 1:
        raise ValueError("Ensemble quantiles must satisfy 0 <= low < high <= 1.")
    if not 0 < cfg.min_agreement <= 1:
        raise ValueError("min_agreement must be between 0 and 1.")
    if cfg.max_model_spread is not None and cfg.max_model_spread < 0:
        raise ValueError("max_model_spread must be non-negative when set.")

    ensemble_cols = _available_ensemble_columns(df)
    if not ensemble_cols:
        raise KeyError("No ensemble prediction columns found for ensemble-agreement simulation.")
    required = [time_col, actual_col] + ensemble_cols
    if forecast_col in df.columns and forecast_col not in required:
        required.append(forecast_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for ensemble-agreement simulation: {missing}")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)
    model_frame = sim[ensemble_cols].astype(float)
    low_thresholds = model_frame.quantile(cfg.low_quantile)
    high_thresholds = model_frame.quantile(cfg.high_quantile)
    charge_votes = model_frame.le(low_thresholds).sum(axis=1)
    discharge_votes = model_frame.ge(high_thresholds).sum(axis=1)
    model_count = len(ensemble_cols)

    sim["Forecast_Price"] = sim[forecast_col] if forecast_col in sim.columns else model_frame.mean(axis=1)
    sim["Agreement_Charge_Share"] = charge_votes / model_count
    sim["Agreement_Discharge_Share"] = discharge_votes / model_count
    sim["Model_Spread_DKK"] = model_frame.max(axis=1) - model_frame.min(axis=1)
    confidence_ok = (
        pd.Series(True, index=sim.index)
        if cfg.max_model_spread is None
        else sim["Model_Spread_DKK"] <= cfg.max_model_spread
    )
    charge_signal = (sim["Agreement_Charge_Share"] >= cfg.min_agreement) & (
        sim["Agreement_Charge_Share"] >= sim["Agreement_Discharge_Share"]
    )
    discharge_signal = (sim["Agreement_Discharge_Share"] >= cfg.min_agreement) & (
        sim["Agreement_Discharge_Share"] > sim["Agreement_Charge_Share"]
    )
    sim["Requested_Action"] = np.where(
        confidence_ok & charge_signal,
        "charge",
        np.where(confidence_ok & discharge_signal, "discharge", "hold"),
    )

    keep_cols = [
        time_col,
        actual_col,
        "Forecast_Price",
        "Agreement_Charge_Share",
        "Agreement_Discharge_Share",
        "Model_Spread_DKK",
        "Requested_Action",
    ]
    out, summary = _simulate_battery_dispatch(sim[keep_cols], battery, time_col, actual_col, "Requested_Action")
    summary.update(
        {
            "low_quantile": float(cfg.low_quantile),
            "high_quantile": float(cfg.high_quantile),
            "min_agreement": float(cfg.min_agreement),
            "max_model_spread": cfg.max_model_spread,
            "model_count": int(model_count),
            "average_model_spread_dkk": float(out["Model_Spread_DKK"].mean()),
        }
    )
    return out, summary


def simulate_momentum_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    momentum_config: MomentumConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = momentum_config or MomentumConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for momentum simulation: {missing}")
    if cfg.lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive.")
    if cfg.threshold < 0:
        raise ValueError("threshold must be non-negative.")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)
    lookback_periods = _periods_from_hours(sim[time_col], cfg.lookback_hours)

    if cfg.smoothing_hours > 0:
        smooth_periods = _periods_from_hours(sim[time_col], cfg.smoothing_hours)
        signal_forecast = sim[forecast_col].rolling(window=smooth_periods, min_periods=1).mean()
    else:
        smooth_periods = 1
        signal_forecast = sim[forecast_col]

    sim["Momentum_Signal_Forecast"] = signal_forecast
    sim["Momentum_Reference_Forecast"] = signal_forecast.shift(lookback_periods).fillna(signal_forecast)
    sim["Forecast_Momentum_DKK"] = sim["Momentum_Signal_Forecast"] - sim["Momentum_Reference_Forecast"]
    sim["Requested_Action"] = np.where(
        sim["Forecast_Momentum_DKK"] <= -cfg.threshold,
        "charge",
        np.where(sim["Forecast_Momentum_DKK"] >= cfg.threshold, "discharge", "hold"),
    )

    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(sim, battery, time_col, actual_col, "Requested_Action")
    summary.update(
        {
            "lookback_hours": float(cfg.lookback_hours),
            "threshold": float(cfg.threshold),
            "smoothing_hours": float(cfg.smoothing_hours),
            "lookback_periods": int(lookback_periods),
            "smoothing_periods": int(smooth_periods),
            "average_momentum_dkk": float(out["Forecast_Momentum_DKK"].mean()),
        }
    )
    return out, summary


def simulate_channel_breakout_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    breakout_config: ChannelBreakoutConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = breakout_config or ChannelBreakoutConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for channel-breakout simulation: {missing}")
    if cfg.buffer < 0:
        raise ValueError("buffer must be non-negative.")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)
    window, min_periods = _rolling_periods(sim[time_col], cfg.window_days, cfg.min_history_days)

    shifted_forecast = sim[forecast_col].shift(1)
    rolling_low = shifted_forecast.rolling(window=window, min_periods=min_periods).min()
    rolling_high = shifted_forecast.rolling(window=window, min_periods=min_periods).max()
    fallback_low = shifted_forecast.expanding(min_periods=1).min()
    fallback_high = shifted_forecast.expanding(min_periods=1).max()

    sim["Rolling_Low_Forecast"] = rolling_low.combine_first(fallback_low).fillna(sim[forecast_col])
    sim["Rolling_High_Forecast"] = rolling_high.combine_first(fallback_high).fillna(sim[forecast_col])
    sim["Buy_Threshold"] = sim["Rolling_Low_Forecast"] - cfg.buffer
    sim["Sell_Threshold"] = sim["Rolling_High_Forecast"] + cfg.buffer
    sim["Requested_Action"] = np.where(
        sim[forecast_col] <= sim["Buy_Threshold"],
        "charge",
        np.where(sim[forecast_col] >= sim["Sell_Threshold"], "discharge", "hold"),
    )

    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(sim, battery, time_col, actual_col, "Requested_Action")
    summary.update(
        {
            "buffer": float(cfg.buffer),
            "window_days": float(cfg.window_days),
            "min_history_days": float(cfg.min_history_days),
        }
    )
    return out, summary


def simulate_daily_spread_rank_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    spread_config: DailySpreadConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = spread_config or DailySpreadConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for daily-spread simulation: {missing}")
    if cfg.min_daily_spread < 0:
        raise ValueError("min_daily_spread must be non-negative.")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)
    sim["Trade_Date"] = sim[time_col].dt.date
    daily_group = sim.groupby("Trade_Date")[forecast_col]
    sim["Daily_Forecast_Rank"] = daily_group.rank(method="first", ascending=True)
    sim["Daily_Forecast_Rank_Pct"] = daily_group.rank(method="first", pct=True, ascending=True)
    sim["Daily_Interval_Count"] = daily_group.transform("count")
    sim["Daily_Forecast_Min"] = daily_group.transform("min")
    sim["Daily_Forecast_Max"] = daily_group.transform("max")
    sim["Daily_Forecast_Spread"] = sim["Daily_Forecast_Max"] - sim["Daily_Forecast_Min"]
    spread_ok = sim["Daily_Forecast_Spread"] >= cfg.min_daily_spread

    rank_mode = cfg.rank_mode.lower().replace("-", "_")
    if rank_mode in {"percentile", "quantile", "pct"}:
        if not 0 <= cfg.low_quantile < cfg.high_quantile <= 1:
            raise ValueError("Daily spread quantiles must satisfy 0 <= low < high <= 1.")
        sim["Requested_Action"] = np.where(
            spread_ok & (sim["Daily_Forecast_Rank_Pct"] <= cfg.low_quantile),
            "charge",
            np.where(spread_ok & (sim["Daily_Forecast_Rank_Pct"] >= cfg.high_quantile), "discharge", "hold"),
        )
        rule_summary = {
            "rank_mode": "percentile",
            "low_quantile": float(cfg.low_quantile),
            "high_quantile": float(cfg.high_quantile),
        }
    elif rank_mode in {"absolute", "rank", "rank_threshold"}:
        if cfg.charge_rank < 1 or cfg.discharge_rank < 1:
            raise ValueError("Daily spread rank thresholds must be positive integers.")
        if cfg.charge_rank >= cfg.discharge_rank:
            raise ValueError("charge_rank must be lower than discharge_rank.")
        sim["Requested_Action"] = np.where(
            spread_ok & (sim["Daily_Forecast_Rank"] <= cfg.charge_rank),
            "charge",
            np.where(spread_ok & (sim["Daily_Forecast_Rank"] >= cfg.discharge_rank), "discharge", "hold"),
        )
        rule_summary = {
            "rank_mode": "absolute",
            "charge_rank": int(cfg.charge_rank),
            "discharge_rank": int(cfg.discharge_rank),
        }
    else:
        raise ValueError("daily spread rank_mode must be 'percentile' or 'absolute'.")

    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(sim, battery, time_col, actual_col, "Requested_Action")
    summary.update(rule_summary)
    summary["min_daily_spread"] = float(cfg.min_daily_spread)
    summary["median_daily_intervals"] = float(sim["Daily_Interval_Count"].median())
    summary["average_daily_spread"] = float(sim["Daily_Forecast_Spread"].mean())
    return out, summary


def simulate_predicted_best_hours_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    best_hours_config: BestHoursConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = best_hours_config or BestHoursConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for predicted-best-hours simulation: {missing}")
    if cfg.hours_per_day < 1:
        raise ValueError("hours_per_day must be a positive integer.")
    if cfg.min_profit_dkk_per_mwh < 0:
        raise ValueError("min_profit_dkk_per_mwh must be non-negative.")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col], utc=True)
    sim = sim.sort_values(time_col).reset_index(drop=True)
    market_time = sim[time_col].dt.tz_convert(cfg.market_timezone)
    sim["Trade_Date"] = market_time.dt.date
    sim["Market_Hour_Key"] = market_time.dt.strftime("%Y-%m-%dT%H%z")
    hourly = (
        sim.groupby(["Trade_Date", "Market_Hour_Key"], sort=False)
        .agg(Hour_Start=(time_col, "min"), Forecast_Hourly=(forecast_col, "mean"))
        .reset_index()
    )

    sim["Requested_Action"] = "hold"
    sim["Predicted_Paper_Margin_DKK_MWh"] = np.nan
    round_trip_efficiency = battery.charge_efficiency * battery.discharge_efficiency
    charge_fee = battery.fee_per_mwh + battery.charge_fee_per_mwh
    discharge_fee = battery.fee_per_mwh + battery.discharge_fee_per_mwh
    selected_margins: list[float] = []

    for _, day_hours in hourly.groupby("Trade_Date", sort=False):
        candidates: list[tuple[float, str, str]] = []
        records = day_hours.to_dict(orient="records")
        for buy in records:
            for sell in records:
                if buy["Hour_Start"] >= sell["Hour_Start"]:
                    continue
                paper_margin = (
                    round_trip_efficiency * (float(sell["Forecast_Hourly"]) - discharge_fee)
                    - (float(buy["Forecast_Hourly"]) + charge_fee)
                )
                if paper_margin > cfg.min_profit_dkk_per_mwh:
                    candidates.append(
                        (paper_margin, str(buy["Market_Hour_Key"]), str(sell["Market_Hour_Key"]))
                    )

        used_hours: set[str] = set()
        selected_pairs = 0
        for paper_margin, buy_hour, sell_hour in sorted(candidates, reverse=True):
            if buy_hour in used_hours or sell_hour in used_hours:
                continue
            buy_mask = sim["Market_Hour_Key"] == buy_hour
            sell_mask = sim["Market_Hour_Key"] == sell_hour
            sim.loc[buy_mask, "Requested_Action"] = "charge"
            sim.loc[sell_mask, "Requested_Action"] = "discharge"
            sim.loc[buy_mask | sell_mask, "Predicted_Paper_Margin_DKK_MWh"] = paper_margin
            used_hours.update([buy_hour, sell_hour])
            selected_margins.append(float(paper_margin))
            selected_pairs += 1
            if selected_pairs >= cfg.hours_per_day:
                break

    sim = sim.drop(columns=["Market_Hour_Key"]).rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(sim, battery, time_col, actual_col, "Requested_Action")
    summary.update(
        {
            "hours_per_day": int(cfg.hours_per_day),
            "min_profit_dkk_per_mwh": float(cfg.min_profit_dkk_per_mwh),
            "profitable_hour_pairs": int(len(selected_margins)),
            "average_predicted_paper_margin": (
                float(np.mean(selected_margins)) if selected_margins else float("nan")
            ),
        }
    )
    return out, summary


def _plan_daily_dispatch_dynamic_program(
    sim: pd.DataFrame,
    battery: BatteryConfig,
    config: RollingOptimizerConfig,
    time_col: str,
    buy_price_col: str,
    sell_price_col: str,
    charge_allowed_col: str | None = None,
    discharge_allowed_col: str | None = None,
    degradation_cost_per_mwh: float = 0.0,
) -> tuple[pd.Series, pd.Series, float]:
    if battery.capacity_mwh <= 0 or battery.power_mw <= 0:
        raise ValueError("Battery capacity and power must be positive.")
    if config.soc_steps < 4:
        raise ValueError("soc_steps must be at least 4.")
    if not 0 <= config.terminal_soc_mwh <= battery.capacity_mwh:
        raise ValueError("terminal_soc_mwh must be within the battery capacity.")

    actions = pd.Series("hold", index=sim.index, dtype="object")
    requested_power = pd.Series(0.0, index=sim.index, dtype="float64")
    market_dates = pd.to_datetime(sim[time_col], utc=True).dt.tz_convert(
        config.market_timezone
    ).dt.date
    step_hours = _infer_step_hours(sim[time_col])
    max_grid_energy = battery.power_mw * step_hours
    total_predicted_value = 0.0
    start_soc = float(battery.initial_soc_mwh)

    for _, day in sim.groupby(market_dates, sort=False):
        states = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, battery.capacity_mwh, config.soc_steps + 1),
                    [start_soc, config.terminal_soc_mwh],
                ]
            )
        )
        state_count = len(states)
        start_index = int(np.argmin(np.abs(states - start_soc)))
        terminal_index = int(np.argmin(np.abs(states - config.terminal_soc_mwh)))
        values = np.full(state_count, -np.inf)
        values[start_index] = 0.0
        predecessors = np.full((len(day), state_count), -1, dtype=int)
        transition_power = np.zeros((len(day), state_count), dtype=float)
        transition_action = np.full((len(day), state_count), "hold", dtype=object)

        for step, (_, row) in enumerate(day.iterrows()):
            next_values = np.full(state_count, -np.inf)
            can_charge = (
                True if charge_allowed_col is None else bool(row[charge_allowed_col])
            )
            can_discharge = (
                True if discharge_allowed_col is None else bool(row[discharge_allowed_col])
            )
            buy_price = float(row[buy_price_col])
            sell_price = float(row[sell_price_col])
            for source_index, source_soc in enumerate(states):
                if not np.isfinite(values[source_index]):
                    continue
                for destination_index, destination_soc in enumerate(states):
                    soc_change = float(destination_soc - source_soc)
                    action = "hold"
                    power_mw = 0.0
                    predicted_cashflow = 0.0
                    if soc_change > 1e-9:
                        if not can_charge:
                            continue
                        grid_energy = soc_change / battery.charge_efficiency
                        if grid_energy > max_grid_energy + 1e-9:
                            continue
                        action = "charge"
                        power_mw = grid_energy / step_hours
                        predicted_cashflow = -grid_energy * (
                            buy_price
                            + battery.fee_per_mwh
                            + battery.charge_fee_per_mwh
                            + degradation_cost_per_mwh
                        )
                    elif soc_change < -1e-9:
                        if not can_discharge:
                            continue
                        grid_energy = -soc_change * battery.discharge_efficiency
                        if grid_energy > max_grid_energy + 1e-9:
                            continue
                        action = "discharge"
                        power_mw = grid_energy / step_hours
                        predicted_cashflow = grid_energy * (
                            sell_price
                            - battery.fee_per_mwh
                            - battery.discharge_fee_per_mwh
                            - degradation_cost_per_mwh
                        )
                    candidate = values[source_index] + predicted_cashflow
                    if candidate > next_values[destination_index] + 1e-9:
                        next_values[destination_index] = candidate
                        predecessors[step, destination_index] = source_index
                        transition_power[step, destination_index] = power_mw
                        transition_action[step, destination_index] = action
            values = next_values

        destination_index = terminal_index
        if not np.isfinite(values[destination_index]):
            feasible = np.flatnonzero(np.isfinite(values))
            if len(feasible) == 0:
                raise ValueError("No feasible dynamic-program path was found.")
            distance = np.abs(states[feasible] - config.terminal_soc_mwh)
            closest = feasible[distance == distance.min()]
            destination_index = int(closest[np.argmax(values[closest])])
        total_predicted_value += float(values[destination_index])
        selected_end_soc = float(states[destination_index])
        day_indices = list(day.index)
        for step in range(len(day_indices) - 1, -1, -1):
            source_index = int(predecessors[step, destination_index])
            if source_index < 0:
                raise ValueError("Dynamic-program path reconstruction failed.")
            row_index = day_indices[step]
            actions.loc[row_index] = str(transition_action[step, destination_index])
            requested_power.loc[row_index] = float(transition_power[step, destination_index])
            destination_index = source_index
        start_soc = selected_end_soc

    return actions, requested_power, total_predicted_value


def simulate_rolling_price_optimizer(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    optimizer_config: RollingOptimizerConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = optimizer_config or RollingOptimizerConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for rolling optimizer: {missing}")
    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col], utc=True)
    sim = sim.sort_values(time_col).reset_index(drop=True)
    actions, power, predicted_value = _plan_daily_dispatch_dynamic_program(
        sim, battery, cfg, time_col, forecast_col, forecast_col
    )
    sim["Requested_Action"] = actions
    sim["Requested_Power_MW"] = power
    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(
        sim, battery, time_col, actual_col, "Requested_Action"
    )
    summary.update(
        {
            "soc_steps": int(cfg.soc_steps),
            "terminal_soc_mwh": float(cfg.terminal_soc_mwh),
            "predicted_optimizer_value": float(predicted_value),
        }
    )
    return out, summary


def simulate_degradation_aware_optimizer(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    optimizer_config: DegradationOptimizerConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = optimizer_config or DegradationOptimizerConfig()
    base_cfg = RollingOptimizerConfig(
        soc_steps=cfg.soc_steps,
        terminal_soc_mwh=cfg.terminal_soc_mwh,
        market_timezone=cfg.market_timezone,
    )
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for degradation optimizer: {missing}")
    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col], utc=True)
    sim = sim.sort_values(time_col).reset_index(drop=True)
    actions, power, predicted_value = _plan_daily_dispatch_dynamic_program(
        sim,
        battery,
        base_cfg,
        time_col,
        forecast_col,
        forecast_col,
        degradation_cost_per_mwh=cfg.degradation_cost_per_mwh,
    )
    sim["Requested_Action"] = actions
    sim["Requested_Power_MW"] = power
    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(
        sim,
        battery,
        time_col,
        actual_col,
        "Requested_Action",
        degradation_cost_per_mwh=cfg.degradation_cost_per_mwh,
    )
    summary.update(
        {
            "soc_steps": int(cfg.soc_steps),
            "terminal_soc_mwh": float(cfg.terminal_soc_mwh),
            "degradation_cost_per_mwh": float(cfg.degradation_cost_per_mwh),
            "predicted_optimizer_value": float(predicted_value),
        }
    )
    return out, summary


def simulate_uncertainty_aware_optimizer(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    optimizer_config: UncertaintyOptimizerConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = optimizer_config or UncertaintyOptimizerConfig()
    if cfg.uncertainty_penalty < 0:
        raise ValueError("uncertainty_penalty must be non-negative.")
    if cfg.max_model_spread is not None and cfg.max_model_spread < 0:
        raise ValueError("max_model_spread must be non-negative or None.")
    model_columns = list(
        dict.fromkeys(
            column
            for column in [forecast_col, "Hourly_Baseline", "Direct_15min_Prediction"]
            if column in df.columns
        )
    )
    required = [time_col, actual_col, forecast_col, *model_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for uncertainty optimizer: {missing}")
    sim = df[list(dict.fromkeys(required))].copy()
    sim[time_col] = pd.to_datetime(sim[time_col], utc=True)
    sim = sim.sort_values(time_col).reset_index(drop=True)
    model_values = sim[model_columns].astype(float)
    sim["Model_Uncertainty_Std"] = model_values.std(axis=1, ddof=0)
    sim["Model_Spread"] = model_values.max(axis=1) - model_values.min(axis=1)
    sim["Conservative_Buy_Price"] = (
        sim[forecast_col] + cfg.uncertainty_penalty * sim["Model_Uncertainty_Std"]
    )
    sim["Conservative_Sell_Price"] = (
        sim[forecast_col] - cfg.uncertainty_penalty * sim["Model_Uncertainty_Std"]
    )
    sim["Uncertainty_Trade_Allowed"] = (
        True if cfg.max_model_spread is None else sim["Model_Spread"] <= cfg.max_model_spread
    )
    base_cfg = RollingOptimizerConfig(
        soc_steps=cfg.soc_steps,
        terminal_soc_mwh=cfg.terminal_soc_mwh,
        market_timezone=cfg.market_timezone,
    )
    actions, power, predicted_value = _plan_daily_dispatch_dynamic_program(
        sim,
        battery,
        base_cfg,
        time_col,
        "Conservative_Buy_Price",
        "Conservative_Sell_Price",
        "Uncertainty_Trade_Allowed",
        "Uncertainty_Trade_Allowed",
    )
    sim["Requested_Action"] = actions
    sim["Requested_Power_MW"] = power
    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(
        sim, battery, time_col, actual_col, "Requested_Action"
    )
    summary.update(
        {
            "uncertainty_penalty": float(cfg.uncertainty_penalty),
            "max_model_spread": (
                float(cfg.max_model_spread) if cfg.max_model_spread is not None else None
            ),
            "model_count": int(len(model_columns)),
            "average_model_spread": float(sim["Model_Spread"].mean()),
            "predicted_optimizer_value": float(predicted_value),
        }
    )
    return out, summary


def _wind_hourly_signals(
    sim: pd.DataFrame,
    cfg: WindSignalConfig | WindConfirmedOptimizerConfig,
    time_col: str,
    wind_col: str,
) -> pd.DataFrame:
    if not 0 <= cfg.low_wind_quantile < cfg.high_wind_quantile <= 1:
        raise ValueError("Wind quantiles must satisfy 0 <= low < high <= 1.")
    if cfg.ramp_threshold_mw < 0:
        raise ValueError("ramp_threshold_mw must be non-negative.")
    market_time = pd.to_datetime(sim[time_col], utc=True).dt.tz_convert(cfg.market_timezone)
    keys = market_time.dt.strftime("%Y-%m-%dT%H%z")
    hourly = pd.DataFrame(
        {
            "Market_Hour_Key": keys,
            "Trade_Date": market_time.dt.date,
            "HourUTC": pd.to_datetime(sim[time_col], utc=True),
            "Wind_DayAhead_MW": sim[wind_col].astype(float),
        }
    ).groupby(["Market_Hour_Key", "Trade_Date"], sort=False, as_index=False).agg(
        HourUTC=("HourUTC", "min"), Wind_DayAhead_MW=("Wind_DayAhead_MW", "mean")
    )
    daily_wind = hourly.groupby("Trade_Date")["Wind_DayAhead_MW"]
    hourly["Wind_Rank_Pct"] = daily_wind.rank(method="average", pct=True)
    hourly["Wind_Ramp_MW"] = daily_wind.diff().fillna(0.0)
    # A zero threshold disables ramp confirmation. Treating zero as a literal
    # comparison would make a flat ramp satisfy charge and discharge together.
    ramp_enabled = cfg.ramp_threshold_mw > 0
    hourly["Wind_Charge_Signal"] = (
        hourly["Wind_Rank_Pct"] >= cfg.high_wind_quantile
    )
    hourly["Wind_Discharge_Signal"] = (
        hourly["Wind_Rank_Pct"] <= cfg.low_wind_quantile
    )
    if ramp_enabled:
        hourly["Wind_Charge_Signal"] |= (
            hourly["Wind_Ramp_MW"] >= cfg.ramp_threshold_mw
        )
        hourly["Wind_Discharge_Signal"] |= (
            hourly["Wind_Ramp_MW"] <= -cfg.ramp_threshold_mw
        )
    return hourly


def simulate_wind_signal_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    wind_config: WindSignalConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
    wind_col: str = "Wind_Total_DayAhead_MW",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = wind_config or WindSignalConfig()
    required = [time_col, actual_col, forecast_col, wind_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for wind-signal simulation: {missing}")
    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col], utc=True)
    sim = sim.sort_values(time_col).reset_index(drop=True)
    market_time = sim[time_col].dt.tz_convert(cfg.market_timezone)
    sim["Market_Hour_Key"] = market_time.dt.strftime("%Y-%m-%dT%H%z")
    hourly = _wind_hourly_signals(sim, cfg, time_col, wind_col)
    sim = sim.merge(
        hourly[
            [
                "Market_Hour_Key",
                "Wind_Rank_Pct",
                "Wind_Ramp_MW",
                "Wind_Charge_Signal",
                "Wind_Discharge_Signal",
            ]
        ],
        on="Market_Hour_Key",
        how="left",
        validate="many_to_one",
    )
    sim["Requested_Action"] = np.where(
        sim["Wind_Charge_Signal"],
        "charge",
        np.where(sim["Wind_Discharge_Signal"], "discharge", "hold"),
    )
    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(
        sim, battery, time_col, actual_col, "Requested_Action"
    )
    summary.update(
        {
            "low_wind_quantile": float(cfg.low_wind_quantile),
            "high_wind_quantile": float(cfg.high_wind_quantile),
            "ramp_threshold_mw": float(cfg.ramp_threshold_mw),
            "average_wind_day_ahead_mw": float(sim[wind_col].mean()),
        }
    )
    return out, summary


def simulate_wind_confirmed_optimizer(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    optimizer_config: WindConfirmedOptimizerConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
    wind_col: str = "Wind_Total_DayAhead_MW",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = optimizer_config or WindConfirmedOptimizerConfig()
    required = [time_col, actual_col, forecast_col, wind_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for wind-confirmed optimizer: {missing}")
    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col], utc=True)
    sim = sim.sort_values(time_col).reset_index(drop=True)
    market_time = sim[time_col].dt.tz_convert(cfg.market_timezone)
    sim["Market_Hour_Key"] = market_time.dt.strftime("%Y-%m-%dT%H%z")
    hourly = _wind_hourly_signals(sim, cfg, time_col, wind_col)
    sim = sim.merge(
        hourly[["Market_Hour_Key", "Wind_Charge_Signal", "Wind_Discharge_Signal"]],
        on="Market_Hour_Key",
        how="left",
        validate="many_to_one",
    )
    base_cfg = RollingOptimizerConfig(
        soc_steps=cfg.soc_steps,
        terminal_soc_mwh=cfg.terminal_soc_mwh,
        market_timezone=cfg.market_timezone,
    )
    actions, power, predicted_value = _plan_daily_dispatch_dynamic_program(
        sim,
        battery,
        base_cfg,
        time_col,
        forecast_col,
        forecast_col,
        "Wind_Charge_Signal",
        "Wind_Discharge_Signal",
    )
    sim["Requested_Action"] = actions
    sim["Requested_Power_MW"] = power
    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(
        sim, battery, time_col, actual_col, "Requested_Action"
    )
    summary.update(
        {
            "low_wind_quantile": float(cfg.low_wind_quantile),
            "high_wind_quantile": float(cfg.high_wind_quantile),
            "ramp_threshold_mw": float(cfg.ramp_threshold_mw),
            "predicted_optimizer_value": float(predicted_value),
        }
    )
    return out, summary


def simulate_perfect_foresight_oracle(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    optimizer_config: RollingOptimizerConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = optimizer_config or RollingOptimizerConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for perfect-foresight oracle: {missing}")
    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col], utc=True)
    sim = sim.sort_values(time_col).reset_index(drop=True)
    actions, power, predicted_value = _plan_daily_dispatch_dynamic_program(
        sim, battery, cfg, time_col, actual_col, actual_col
    )
    sim["Requested_Action"] = actions
    sim["Requested_Power_MW"] = power
    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(
        sim, battery, time_col, actual_col, "Requested_Action"
    )
    summary.update(
        {
            "soc_steps": int(cfg.soc_steps),
            "terminal_soc_mwh": float(cfg.terminal_soc_mwh),
            "predicted_optimizer_value": float(predicted_value),
            "is_hindsight_benchmark": True,
        }
    )
    return out, summary


def simulate_momentum_spread_arbitrage(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    momentum_spread_config: MomentumSpreadConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    cfg = momentum_spread_config or MomentumSpreadConfig()
    required = [time_col, actual_col, forecast_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for momentum-spread simulation: {missing}")
    if cfg.min_daily_spread < 0:
        raise ValueError("min_daily_spread must be non-negative.")
    if cfg.lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive.")
    if cfg.momentum_threshold < 0:
        raise ValueError("momentum_threshold must be non-negative.")

    sim = df[required].copy()
    sim[time_col] = pd.to_datetime(sim[time_col])
    sim = sim.sort_values(time_col).reset_index(drop=True)

    sim["Trade_Date"] = sim[time_col].dt.date
    daily_group = sim.groupby("Trade_Date")[forecast_col]
    sim["Daily_Forecast_Rank"] = daily_group.rank(method="first", ascending=True)
    sim["Daily_Forecast_Rank_Pct"] = daily_group.rank(method="first", pct=True, ascending=True)
    sim["Daily_Interval_Count"] = daily_group.transform("count")
    sim["Daily_Forecast_Min"] = daily_group.transform("min")
    sim["Daily_Forecast_Max"] = daily_group.transform("max")
    sim["Daily_Forecast_Spread"] = sim["Daily_Forecast_Max"] - sim["Daily_Forecast_Min"]
    spread_ok = sim["Daily_Forecast_Spread"] >= cfg.min_daily_spread

    lookback_periods = _periods_from_hours(sim[time_col], cfg.lookback_hours)
    if cfg.smoothing_hours > 0:
        smooth_periods = _periods_from_hours(sim[time_col], cfg.smoothing_hours)
        signal_forecast = sim[forecast_col].rolling(window=smooth_periods, min_periods=1).mean()
    else:
        smooth_periods = 1
        signal_forecast = sim[forecast_col]
    sim["Momentum_Signal_Forecast"] = signal_forecast
    sim["Momentum_Reference_Forecast"] = signal_forecast.shift(lookback_periods).fillna(signal_forecast)
    sim["Forecast_Momentum_DKK"] = sim["Momentum_Signal_Forecast"] - sim["Momentum_Reference_Forecast"]
    charge_momentum = sim["Forecast_Momentum_DKK"] <= -cfg.momentum_threshold
    discharge_momentum = sim["Forecast_Momentum_DKK"] >= cfg.momentum_threshold

    rank_mode = cfg.rank_mode.lower().replace("-", "_")
    if rank_mode in {"percentile", "quantile", "pct"}:
        if not 0 <= cfg.low_quantile < cfg.high_quantile <= 1:
            raise ValueError("Momentum spread quantiles must satisfy 0 <= low < high <= 1.")
        charge_spread = sim["Daily_Forecast_Rank_Pct"] <= cfg.low_quantile
        discharge_spread = sim["Daily_Forecast_Rank_Pct"] >= cfg.high_quantile
        rule_summary = {
            "rank_mode": "percentile",
            "low_quantile": float(cfg.low_quantile),
            "high_quantile": float(cfg.high_quantile),
        }
    elif rank_mode in {"absolute", "rank", "rank_threshold"}:
        if cfg.charge_rank < 1 or cfg.discharge_rank < 1:
            raise ValueError("Momentum spread rank thresholds must be positive integers.")
        if cfg.charge_rank >= cfg.discharge_rank:
            raise ValueError("charge_rank must be lower than discharge_rank.")
        charge_spread = sim["Daily_Forecast_Rank"] <= cfg.charge_rank
        discharge_spread = sim["Daily_Forecast_Rank"] >= cfg.discharge_rank
        rule_summary = {
            "rank_mode": "absolute",
            "charge_rank": int(cfg.charge_rank),
            "discharge_rank": int(cfg.discharge_rank),
        }
    else:
        raise ValueError("momentum spread rank_mode must be 'percentile' or 'absolute'.")

    sim["Spread_Charge_Signal"] = spread_ok & charge_spread
    sim["Spread_Discharge_Signal"] = spread_ok & discharge_spread
    sim["Momentum_Charge_Signal"] = charge_momentum
    sim["Momentum_Discharge_Signal"] = discharge_momentum
    sim["Requested_Action"] = np.where(
        sim["Spread_Charge_Signal"] & sim["Momentum_Charge_Signal"],
        "charge",
        np.where(
            sim["Spread_Discharge_Signal"] & sim["Momentum_Discharge_Signal"],
            "discharge",
            "hold",
        ),
    )

    sim = sim.rename(columns={forecast_col: "Forecast_Price"})
    out, summary = _simulate_battery_dispatch(sim, battery, time_col, actual_col, "Requested_Action")
    summary.update(rule_summary)
    summary.update(
        {
            "min_daily_spread": float(cfg.min_daily_spread),
            "lookback_hours": float(cfg.lookback_hours),
            "momentum_threshold": float(cfg.momentum_threshold),
            "smoothing_hours": float(cfg.smoothing_hours),
            "lookback_periods": int(lookback_periods),
            "smoothing_periods": int(smooth_periods),
            "median_daily_intervals": float(sim["Daily_Interval_Count"].median()),
            "average_daily_spread": float(sim["Daily_Forecast_Spread"].mean()),
            "average_momentum_dkk": float(out["Forecast_Momentum_DKK"].mean()),
        }
    )
    return out, summary


def summarize_strategy_result(name: str, summary: dict[str, float]) -> dict[str, object]:
    return {
        "Strategy": name,
        "Description": STRATEGY_DESCRIPTIONS[name],
        "Cashflow": summary["total_cashflow"],
        "Max_Drawdown": summary["max_drawdown"],
        "Active_Intervals": summary["trades"],
        "Charge_Intervals": summary["charge_intervals"],
        "Discharge_Intervals": summary["discharge_intervals"],
        "Energy_Charged_MWh": summary["energy_charged_mwh"],
        "Energy_Discharged_MWh": summary["energy_discharged_mwh"],
    }


def run_strategy_suite(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    weekly_band_config: WeeklyBandConfig | None = None,
    forecast_edge_config: ForecastEdgeConfig | None = None,
    volatility_filter_config: VolatilityFilterConfig | None = None,
    mean_reversion_config: MeanReversionConfig | None = None,
    breakout_config: ChannelBreakoutConfig | None = None,
    daily_spread_config: DailySpreadConfig | None = None,
    best_hours_config: BestHoursConfig | None = None,
    momentum_config: MomentumConfig | None = None,
    momentum_spread_config: MomentumSpreadConfig | None = None,
    ensemble_agreement_config: EnsembleAgreementConfig | None = None,
    rolling_optimizer_config: RollingOptimizerConfig | None = None,
    uncertainty_optimizer_config: UncertaintyOptimizerConfig | None = None,
    degradation_optimizer_config: DegradationOptimizerConfig | None = None,
    wind_signal_config: WindSignalConfig | None = None,
    wind_confirmed_optimizer_config: WindConfirmedOptimizerConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> dict[str, tuple[pd.DataFrame, dict[str, float]]]:
    battery = battery_config or BatteryConfig()
    suite = {
        "Forecast quantile": simulate_battery_arbitrage(df, battery, time_col, actual_col, forecast_col),
        "Weekly average band": simulate_weekly_average_band_arbitrage(
            df, battery, weekly_band_config, time_col, actual_col, forecast_col
        ),
        "Forecast edge": simulate_forecast_edge_arbitrage(
            df, battery, forecast_edge_config, time_col, actual_col, forecast_col
        ),
        "Volatility filtered average": simulate_volatility_filtered_average_arbitrage(
            df, battery, volatility_filter_config, time_col, actual_col, forecast_col
        ),
        "Daily spread rank": simulate_daily_spread_rank_arbitrage(
            df, battery, daily_spread_config, time_col, actual_col, forecast_col
        ),
        "Predicted best hours": simulate_predicted_best_hours_arbitrage(
            df, battery, best_hours_config, time_col, actual_col, forecast_col
        ),
        "Ensemble agreement": simulate_ensemble_agreement_arbitrage(
            df, battery, ensemble_agreement_config, time_col, actual_col, forecast_col
        ),
        "Mean reversion": simulate_mean_reversion_arbitrage(
            df, battery, mean_reversion_config, time_col, actual_col, forecast_col
        ),
        "Momentum": simulate_momentum_arbitrage(
            df, battery, momentum_config, time_col, actual_col, forecast_col
        ),
        "Momentum spread": simulate_momentum_spread_arbitrage(
            df, battery, momentum_spread_config, time_col, actual_col, forecast_col
        ),
        "Channel breakout": simulate_channel_breakout_arbitrage(
            df, battery, breakout_config, time_col, actual_col, forecast_col
        ),
        "Rolling price optimizer": simulate_rolling_price_optimizer(
            df, battery, rolling_optimizer_config, time_col, actual_col, forecast_col
        ),
        "Uncertainty-aware optimizer": simulate_uncertainty_aware_optimizer(
            df, battery, uncertainty_optimizer_config, time_col, actual_col, forecast_col
        ),
        "Degradation-aware optimizer": simulate_degradation_aware_optimizer(
            df, battery, degradation_optimizer_config, time_col, actual_col, forecast_col
        ),
    }
    if "Wind_Total_DayAhead_MW" in df.columns:
        suite["Wind signal"] = simulate_wind_signal_arbitrage(
            df, battery, wind_signal_config, time_col, actual_col, forecast_col
        )
        suite["Wind-confirmed optimizer"] = simulate_wind_confirmed_optimizer(
            df, battery, wind_confirmed_optimizer_config, time_col, actual_col, forecast_col
        )
    return suite
