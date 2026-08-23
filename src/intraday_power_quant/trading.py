from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatteryConfig:
    capacity_mwh: float = 100.0
    power_mw: float = 25.0
    initial_soc_mwh: float = 50.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    fee_per_mwh: float = 0.0
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
class EnsembleAgreementConfig:
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    min_agreement: float = 0.60
    max_model_spread: float | None = None


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
    "Ensemble agreement": (
        "Uses the transfer-learning ensemble members as a confidence filter. It trades only when enough "
        "models agree that the interval is cheap or expensive."
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
) -> tuple[pd.DataFrame, dict[str, float]]:
    step_hours = _infer_step_hours(sim[time_col])
    soc = min(max(cfg.initial_soc_mwh, 0.0), cfg.capacity_mwh)
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    daily_cashflow: dict[object, float] = {}

    for row in sim.itertuples(index=False):
        row_dict = row._asdict()
        timestamp = row_dict[time_col]
        actual_price = float(row_dict[actual_col])
        requested_action = str(row_dict[action_col])
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
            dispatch_mw = -min(cfg.power_mw, max_charge_mw)
            energy_mwh = abs(dispatch_mw) * step_hours
            soc += energy_mwh * cfg.charge_efficiency
            cashflow = -energy_mwh * actual_price - energy_mwh * cfg.fee_per_mwh
            action = "charge"
        elif not blocked_by_risk and requested_action == "discharge" and soc > 0:
            max_discharge_mw = soc * cfg.discharge_efficiency / step_hours
            dispatch_mw = min(cfg.power_mw, max_discharge_mw)
            energy_mwh = dispatch_mw * step_hours
            soc -= energy_mwh / cfg.discharge_efficiency
            cashflow = energy_mwh * actual_price - energy_mwh * cfg.fee_per_mwh
            action = "discharge"
        elif blocked_by_risk:
            action = "risk-off"

        soc = min(max(soc, 0.0), cfg.capacity_mwh)
        daily_cashflow[date_key] += cashflow
        rows.append(
            {
                **row_dict,
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
    summary = {
        "total_cashflow": float(out["Cashflow"].sum()),
        "max_drawdown": float((out["Cumulative_Cashflow"].cummax() - out["Cumulative_Cashflow"]).max()),
        "trades": int((out["Action"] != "hold").sum()),
        "charge_intervals": int((out["Action"] == "charge").sum()),
        "discharge_intervals": int((out["Action"] == "discharge").sum()),
        "energy_charged_mwh": total_energy_charged,
        "energy_discharged_mwh": total_energy_discharged,
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
    momentum_config: MomentumConfig | None = None,
    momentum_spread_config: MomentumSpreadConfig | None = None,
    ensemble_agreement_config: EnsembleAgreementConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> dict[str, tuple[pd.DataFrame, dict[str, float]]]:
    battery = battery_config or BatteryConfig()
    return {
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
    }
