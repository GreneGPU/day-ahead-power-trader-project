from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
import json

import pandas as pd

from .risk import summarize_cashflow_risk
from .trading import (
    BatteryConfig,
    ChannelBreakoutConfig,
    DailySpreadConfig,
    EnsembleAgreementConfig,
    ForecastEdgeConfig,
    MeanReversionConfig,
    MomentumConfig,
    MomentumSpreadConfig,
    VolatilityFilterConfig,
    WeeklyBandConfig,
    simulate_battery_arbitrage,
    simulate_channel_breakout_arbitrage,
    simulate_daily_spread_rank_arbitrage,
    simulate_ensemble_agreement_arbitrage,
    simulate_forecast_edge_arbitrage,
    simulate_mean_reversion_arbitrage,
    simulate_momentum_arbitrage,
    simulate_momentum_spread_arbitrage,
    simulate_volatility_filtered_average_arbitrage,
    simulate_weekly_average_band_arbitrage,
)


QUANTILE_PAIRS = [(0.10, 0.90), (0.15, 0.85), (0.20, 0.80), (0.25, 0.75), (0.30, 0.70)]
SPREAD_QUANTILE_PAIRS = [(0.15, 0.85), (0.20, 0.80), (0.25, 0.75), (0.30, 0.70)]
ABSOLUTE_RANK_PAIRS = [(12, 85), (18, 79), (24, 73), (32, 65)]


def _format_setting_value(value: object) -> str:
    if value is None:
        return "off"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _settings_text(settings: dict[str, object]) -> str:
    return ", ".join(f"{key}={_format_setting_value(value)}" for key, value in settings.items())


def _json_settings(settings: dict[str, object]) -> str:
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def _summary_row(
    strategy: str,
    settings: dict[str, object],
    sim: pd.DataFrame,
    summary: dict[str, float],
) -> dict[str, object]:
    risk = summarize_cashflow_risk(sim)
    return {
        "Strategy": strategy,
        "Cashflow": float(summary["total_cashflow"]),
        "Max_Drawdown": float(summary["max_drawdown"]),
        "Active_Intervals": int(summary["trades"]),
        "Charge_Intervals": int(summary["charge_intervals"]),
        "Discharge_Intervals": int(summary["discharge_intervals"]),
        "Energy_Charged_MWh": float(summary["energy_charged_mwh"]),
        "Energy_Discharged_MWh": float(summary["energy_discharged_mwh"]),
        "Final_SOC_MWh": float(summary["final_soc_mwh"]),
        "Win_Rate": risk["win_rate"],
        "Profit_Factor": risk["profit_factor"],
        "Calmar_Ratio": risk["calmar_ratio"],
        "Daily_Sharpe": risk["daily_sharpe"],
        "Daily_Sortino": risk["daily_sortino"],
        "Average_Daily_Cashflow": risk["average_daily_cashflow"],
        "Daily_Cashflow_Std": risk["daily_cashflow_std"],
        "Cashflow_Std": risk["cashflow_std"],
        "Downside_Std": risk["downside_std"],
        "Worst_Interval": risk["worst_interval"],
        "Best_Interval": risk["best_interval"],
        "Worst_Day": risk["worst_day"],
        "Best_Day": risk["best_day"],
        "Max_Drawdown_Duration_Intervals": risk["max_drawdown_duration_intervals"],
        "Risk_Adjusted_Score": risk["risk_adjusted_score"],
        "Settings": _settings_text(settings),
        "Settings_JSON": _json_settings(settings),
    }


def _add_result(
    rows: list[dict[str, object]],
    strategy: str,
    settings: dict[str, object],
    simulator: Callable[[], tuple[pd.DataFrame, dict[str, float]]],
) -> None:
    sim, summary = simulator()
    rows.append(_summary_row(strategy, settings, sim, summary))


def best_parameter_rows(sweep: pd.DataFrame, ranking_metric: str = "Cashflow") -> pd.DataFrame:
    if sweep.empty:
        return sweep
    if ranking_metric not in sweep.columns:
        raise KeyError(f"Cannot rank parameter sweep by missing metric: {ranking_metric}")
    ordered = sweep.sort_values([ranking_metric, "Cashflow", "Max_Drawdown"], ascending=[False, False, True]).reset_index(drop=True)
    evaluation_counts = ordered.groupby("Strategy").size().rename("Evaluations")
    best = ordered.groupby("Strategy", as_index=False).head(1).copy()
    best["Evaluations"] = best["Strategy"].map(evaluation_counts).astype(int)
    best = best.sort_values([ranking_metric, "Cashflow", "Max_Drawdown"], ascending=[False, False, True]).reset_index(drop=True)
    best.insert(0, "Rank", range(1, len(best) + 1))
    best.insert(1, "Ranking_Metric", ranking_metric)
    return best


def simulate_strategy_from_settings(
    strategy: str,
    df: pd.DataFrame,
    settings: dict[str, object] | str,
    battery_config: BatteryConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery = battery_config or BatteryConfig()
    payload = json.loads(settings) if isinstance(settings, str) else dict(settings)
    if strategy == "Forecast quantile":
        return simulate_battery_arbitrage(
            df,
            replace(
                battery,
                low_quantile=float(payload["low_quantile"]),
                high_quantile=float(payload["high_quantile"]),
            ),
            time_col,
            actual_col,
            forecast_col,
        )
    if strategy == "Weekly average band":
        return simulate_weekly_average_band_arbitrage(
            df, battery, WeeklyBandConfig(**payload), time_col, actual_col, forecast_col
        )
    if strategy == "Forecast edge":
        return simulate_forecast_edge_arbitrage(
            df, battery, ForecastEdgeConfig(**payload), time_col, actual_col, forecast_col
        )
    if strategy == "Volatility filtered average":
        return simulate_volatility_filtered_average_arbitrage(
            df, battery, VolatilityFilterConfig(**payload), time_col, actual_col, forecast_col
        )
    if strategy == "Mean reversion":
        return simulate_mean_reversion_arbitrage(
            df, battery, MeanReversionConfig(**payload), time_col, actual_col, forecast_col
        )
    if strategy == "Momentum":
        return simulate_momentum_arbitrage(
            df, battery, MomentumConfig(**payload), time_col, actual_col, forecast_col
        )
    if strategy == "Daily spread rank":
        return simulate_daily_spread_rank_arbitrage(
            df, battery, DailySpreadConfig(**payload), time_col, actual_col, forecast_col
        )
    if strategy == "Ensemble agreement":
        return simulate_ensemble_agreement_arbitrage(
            df, battery, EnsembleAgreementConfig(**payload), time_col, actual_col, forecast_col
        )
    if strategy == "Momentum spread":
        return simulate_momentum_spread_arbitrage(
            df, battery, MomentumSpreadConfig(**payload), time_col, actual_col, forecast_col
        )
    if strategy == "Channel breakout":
        return simulate_channel_breakout_arbitrage(
            df, battery, ChannelBreakoutConfig(**payload), time_col, actual_col, forecast_col
        )
    raise KeyError(f"Unknown strategy: {strategy}")


def _iter_weekly_band_grid() -> Iterable[dict[str, object]]:
    for window_days in [3.0, 7.0, 14.0]:
        for band in [0.0, 10.0, 20.0, 30.0, 40.0]:
            yield {"window_days": window_days, "band": band, "min_history_days": 1.0}


def _iter_volatility_grid() -> Iterable[dict[str, object]]:
    for average_window_days in [3.0, 7.0, 14.0]:
        for volatility_window_days in [3.0, 7.0, 14.0]:
            for min_volatility in [15.0, 25.0, 35.0, 45.0]:
                for price_band in [0.0, 10.0, 20.0]:
                    yield {
                        "average_window_days": average_window_days,
                        "volatility_window_days": volatility_window_days,
                        "min_volatility": min_volatility,
                        "price_band": price_band,
                        "min_history_days": 1.0,
                    }


def _iter_daily_spread_grid() -> Iterable[dict[str, object]]:
    for low_quantile, high_quantile in SPREAD_QUANTILE_PAIRS:
        for min_daily_spread in [0.0, 20.0, 40.0, 60.0]:
            yield {
                "low_quantile": low_quantile,
                "high_quantile": high_quantile,
                "rank_mode": "percentile",
                "min_daily_spread": min_daily_spread,
            }
    for charge_rank, discharge_rank in ABSOLUTE_RANK_PAIRS:
        for min_daily_spread in [0.0, 20.0, 40.0, 60.0]:
            yield {
                "rank_mode": "absolute",
                "charge_rank": charge_rank,
                "discharge_rank": discharge_rank,
                "min_daily_spread": min_daily_spread,
            }


def _iter_ensemble_grid() -> Iterable[dict[str, object]]:
    for low_quantile, high_quantile in [(0.20, 0.80), (0.25, 0.75), (0.30, 0.70)]:
        for min_agreement in [0.40, 0.60, 0.80, 1.00]:
            for max_model_spread in [None, 20.0, 40.0, 60.0]:
                yield {
                    "low_quantile": low_quantile,
                    "high_quantile": high_quantile,
                    "min_agreement": min_agreement,
                    "max_model_spread": max_model_spread,
                }


def _iter_momentum_spread_grid() -> Iterable[dict[str, object]]:
    for low_quantile, high_quantile in [(0.15, 0.85), (0.25, 0.75), (0.30, 0.70)]:
        for min_daily_spread in [0.0, 20.0, 40.0]:
            for lookback_hours in [3.0, 6.0, 12.0]:
                for momentum_threshold in [0.0, 5.0, 10.0]:
                    for smoothing_hours in [0.0, 1.0]:
                        yield {
                            "low_quantile": low_quantile,
                            "high_quantile": high_quantile,
                            "rank_mode": "percentile",
                            "min_daily_spread": min_daily_spread,
                            "lookback_hours": lookback_hours,
                            "momentum_threshold": momentum_threshold,
                            "smoothing_hours": smoothing_hours,
                        }


def run_strategy_parameter_sweep(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> pd.DataFrame:
    """Evaluate a compact grid of strategy parameters and return every successful run."""
    battery = battery_config or BatteryConfig()
    rows: list[dict[str, object]] = []

    for low_quantile, high_quantile in QUANTILE_PAIRS:
        settings = {"low_quantile": low_quantile, "high_quantile": high_quantile}
        cfg = replace(battery, low_quantile=low_quantile, high_quantile=high_quantile)
        _add_result(
            rows,
            "Forecast quantile",
            settings,
            lambda cfg=cfg: simulate_battery_arbitrage(df, cfg, time_col, actual_col, forecast_col),
        )

    for settings in _iter_weekly_band_grid():
        cfg = WeeklyBandConfig(**settings)
        _add_result(
            rows,
            "Weekly average band",
            settings,
            lambda cfg=cfg: simulate_weekly_average_band_arbitrage(df, battery, cfg, time_col, actual_col, forecast_col),
        )

    if "Hourly_Baseline" in df.columns:
        for threshold in [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]:
            settings = {"threshold": threshold, "reference_col": "Hourly_Baseline"}
            cfg = ForecastEdgeConfig(**settings)
            _add_result(
                rows,
                "Forecast edge",
                settings,
                lambda cfg=cfg: simulate_forecast_edge_arbitrage(df, battery, cfg, time_col, actual_col, forecast_col),
            )

    for settings in _iter_volatility_grid():
        cfg = VolatilityFilterConfig(**settings)
        _add_result(
            rows,
            "Volatility filtered average",
            settings,
            lambda cfg=cfg: simulate_volatility_filtered_average_arbitrage(
                df, battery, cfg, time_col, actual_col, forecast_col
            ),
        )

    for window_days in [3.0, 7.0, 14.0]:
        for entry_z in [0.5, 1.0, 1.5, 2.0]:
            settings = {"window_days": window_days, "entry_z": entry_z, "min_history_days": 1.0}
            cfg = MeanReversionConfig(**settings)
            _add_result(
                rows,
                "Mean reversion",
                settings,
                lambda cfg=cfg: simulate_mean_reversion_arbitrage(df, battery, cfg, time_col, actual_col, forecast_col),
            )

    for lookback_hours in [1.0, 3.0, 6.0, 12.0, 24.0]:
        for threshold in [0.0, 5.0, 10.0, 20.0]:
            for smoothing_hours in [0.0, 1.0]:
                settings = {
                    "lookback_hours": lookback_hours,
                    "threshold": threshold,
                    "smoothing_hours": smoothing_hours,
                }
                cfg = MomentumConfig(**settings)
                _add_result(
                    rows,
                    "Momentum",
                    settings,
                    lambda cfg=cfg: simulate_momentum_arbitrage(df, battery, cfg, time_col, actual_col, forecast_col),
                )

    for settings in _iter_daily_spread_grid():
        cfg = DailySpreadConfig(**settings)
        _add_result(
            rows,
            "Daily spread rank",
            settings,
            lambda cfg=cfg: simulate_daily_spread_rank_arbitrage(df, battery, cfg, time_col, actual_col, forecast_col),
        )

    for settings in _iter_ensemble_grid():
        cfg = EnsembleAgreementConfig(**settings)
        _add_result(
            rows,
            "Ensemble agreement",
            settings,
            lambda cfg=cfg: simulate_ensemble_agreement_arbitrage(df, battery, cfg, time_col, actual_col, forecast_col),
        )

    for settings in _iter_momentum_spread_grid():
        cfg = MomentumSpreadConfig(**settings)
        _add_result(
            rows,
            "Momentum spread",
            settings,
            lambda cfg=cfg: simulate_momentum_spread_arbitrage(df, battery, cfg, time_col, actual_col, forecast_col),
        )

    for window_days in [1.0, 3.0, 7.0, 14.0]:
        for buffer in [0.0, 5.0, 10.0, 20.0]:
            settings = {"window_days": window_days, "buffer": buffer, "min_history_days": 1.0}
            cfg = ChannelBreakoutConfig(**settings)
            _add_result(
                rows,
                "Channel breakout",
                settings,
                lambda cfg=cfg: simulate_channel_breakout_arbitrage(df, battery, cfg, time_col, actual_col, forecast_col),
            )

    return pd.DataFrame(rows).sort_values(["Cashflow", "Max_Drawdown"], ascending=[False, True]).reset_index(drop=True)


def optimize_strategy_suite(
    df: pd.DataFrame,
    battery_config: BatteryConfig | None = None,
    ranking_metric: str = "Cashflow",
    time_col: str = "HourUTC",
    actual_col: str = "Actual_Price",
    forecast_col: str = "Prediction",
) -> pd.DataFrame:
    """Return the best cashflow setting for each strategy in the sweep."""
    return best_parameter_rows(
        run_strategy_parameter_sweep(
            df,
            battery_config=battery_config,
            time_col=time_col,
            actual_col=actual_col,
            forecast_col=forecast_col,
        ),
        ranking_metric=ranking_metric,
    )
