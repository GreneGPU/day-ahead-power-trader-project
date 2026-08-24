from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from intraday_power_quant.optimization import optimize_strategy_suite, simulate_strategy_from_settings
from intraday_power_quant.imbalance_trading import (
    simulate_imbalance_perfect_foresight,
    simulate_imbalance_spread_positions,
)
from intraday_power_quant.prop_trading import (
    PropConfig,
    simulate_prop_eod_perfect_foresight,
    simulate_prop_positions_with_eod_imbalance,
)
from intraday_power_quant.risk import summarize_cashflow_risk
from intraday_power_quant.trading import (
    BatteryConfig,
    BestHoursConfig,
    ChannelBreakoutConfig,
    DailySpreadConfig,
    DegradationOptimizerConfig,
    EnsembleAgreementConfig,
    ForecastEdgeConfig,
    MeanReversionConfig,
    MomentumConfig,
    MomentumSpreadConfig,
    RollingOptimizerConfig,
    STRATEGY_DESCRIPTIONS,
    UncertaintyOptimizerConfig,
    VolatilityFilterConfig,
    WeeklyBandConfig,
    WindConfirmedOptimizerConfig,
    WindSignalConfig,
    run_strategy_suite,
    simulate_perfect_foresight_oracle,
)


app = FastAPI(
    title="Day-Ahead Power Trading API",
    description="Vercel API for battery strategies and directional positions with configurable delivery settlement.",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "Forecast quantile": {
        "low_quantile": BatteryConfig.low_quantile,
        "high_quantile": BatteryConfig.high_quantile,
    },
    "Weekly average band": asdict(WeeklyBandConfig()),
    "Forecast edge": asdict(ForecastEdgeConfig()),
    "Volatility filtered average": asdict(VolatilityFilterConfig()),
    "Mean reversion": asdict(MeanReversionConfig()),
    "Momentum": asdict(MomentumConfig()),
    "Momentum spread": asdict(MomentumSpreadConfig()),
    "Channel breakout": asdict(ChannelBreakoutConfig()),
    "Daily spread rank": asdict(DailySpreadConfig()),
    "Predicted best hours": asdict(BestHoursConfig()),
    "Ensemble agreement": asdict(EnsembleAgreementConfig()),
    "Rolling price optimizer": asdict(RollingOptimizerConfig()),
    "Uncertainty-aware optimizer": asdict(UncertaintyOptimizerConfig()),
    "Degradation-aware optimizer": asdict(DegradationOptimizerConfig()),
    "Wind signal": asdict(WindSignalConfig()),
    "Wind-confirmed optimizer": asdict(WindConfirmedOptimizerConfig()),
}

FORECAST_COLUMNS = {
    "Prediction": "Transfer residual ensemble",
    "Hourly_Baseline": "Hourly baseline only",
    "Direct_15min_Prediction": "Direct 15-min ensemble",
}

PROP_STRATEGIES = {
    "Forecast quantile",
    "Weekly average band",
    "Forecast edge",
    "Volatility filtered average",
    "Mean reversion",
    "Momentum",
    "Momentum spread",
    "Channel breakout",
    "Daily spread rank",
    "Predicted best hours",
    "Ensemble agreement",
    "Wind signal",
}

SAVED_COMPARISON_FILES = {
    "battery": "default_battery_comparison.json.gz",
    "prop": "default_prop_comparison.json.gz",
    "imbalance": "default_imbalance_comparison.json.gz",
}


class SimulationRequest(BaseModel):
    strategy: str = "Daily spread rank"
    records: list[dict[str, Any]] = Field(min_length=2, max_length=20_000)
    settings: dict[str, Any] | None = None
    battery: dict[str, Any] = Field(default_factory=dict)
    trading_setup: str = "battery"
    prop: dict[str, Any] = Field(default_factory=dict)
    time_col: str = "HourUTC"
    actual_col: str = "Actual_Price"
    forecast_col: str = "Prediction"
    include_intervals: bool = False


class StrategyComparisonRequest(BaseModel):
    forecast_col: str = "Prediction"
    strategy: str | None = None
    days: int | None = Field(default=None, ge=1, le=90)
    optimize: bool = False
    test_days: int = Field(default=10, ge=2, le=30)
    battery: dict[str, Any] = Field(default_factory=dict)
    trading_setup: str = "battery"
    prop: dict[str, Any] = Field(default_factory=dict)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@lru_cache(maxsize=1)
def _load_deployment_results() -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    data_dir = PROJECT_ROOT / "deployment_data"
    forecast_path = data_dir / "predictions.csv.gz"
    if not forecast_path.exists():
        forecast_path = data_dir / "predictions.csv"
    forecasts = pd.read_csv(forecast_path, parse_dates=["HourUTC"])
    metrics = json.loads((data_dir / "model_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    forecasts["HourUTC"] = pd.to_datetime(forecasts["HourUTC"], utc=True)
    imbalance = pd.read_csv(data_dir / "imbalance_prices.csv.gz", parse_dates=["HourUTC"])
    imbalance["HourUTC"] = pd.to_datetime(imbalance["HourUTC"], utc=True)
    forecasts = forecasts.merge(imbalance, on="HourUTC", how="left", validate="one_to_one")
    if forecasts["Imbalance_Price_DKK"].isna().any():
        raise ValueError("Imbalance prices do not cover all deployment predictions.")
    for column in ["Actual_Price", *FORECAST_COLUMNS]:
        forecasts[f"{column}_EUR"] = forecasts[column]
        forecasts[f"{column}_DKK"] = forecasts[column] * forecasts["FX_DKK_per_EUR"]
    forecasts["Actual_Price_DKK"] = forecasts["Spot_Price_DKK"]
    return forecasts.sort_values("HourUTC").reset_index(drop=True), metrics, manifest


def _select_window(frame: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if days is None:
        return frame.copy()
    cutoff = frame["HourUTC"].max() - pd.Timedelta(days=days)
    return frame.loc[frame["HourUTC"] >= cutoff].reset_index(drop=True)


def _split_complete_day_holdout(
    history: pd.DataFrame,
    test_days: int,
    market_timezone: str = "Europe/Copenhagen",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the final complete DK1 calendar days as a chronological holdout."""
    timestamps = pd.to_datetime(history["HourUTC"], utc=True)
    local_timestamps = timestamps.dt.tz_convert(market_timezone)
    complete_dates: list[object] = []

    for local_date in sorted(local_timestamps.dt.date.unique()):
        day_start = pd.Timestamp(local_date).tz_localize(market_timezone)
        day_end = day_start + pd.DateOffset(days=1)
        expected = pd.date_range(day_start, day_end, freq="15min", inclusive="left")
        actual = pd.DatetimeIndex(
            local_timestamps.loc[local_timestamps.dt.date == local_date]
        ).sort_values()
        if actual.equals(expected):
            complete_dates.append(local_date)

    if len(complete_dates) < test_days:
        raise ValueError(
            f"At least {test_days} complete DK1 calendar days are required for optimization."
        )

    holdout_dates = complete_dates[-test_days:]
    expected_dates = pd.date_range(
        pd.Timestamp(holdout_dates[0]),
        pd.Timestamp(holdout_dates[-1]),
        freq="D",
    ).date.tolist()
    if holdout_dates != expected_dates:
        raise ValueError(f"The final {test_days} complete DK1 days must be consecutive.")

    test_start = pd.Timestamp(holdout_dates[0]).tz_localize(market_timezone)
    test_end = pd.Timestamp(holdout_dates[-1]).tz_localize(market_timezone) + pd.DateOffset(days=1)
    train = history.loc[local_timestamps < test_start].copy()
    selected = history.loc[(local_timestamps >= test_start) & (local_timestamps < test_end)].copy()
    if len(train) < 2:
        raise ValueError("At least two earlier intervals are required before the complete-day holdout.")
    return train.reset_index(drop=True), selected.reset_index(drop=True)


@app.get("/api")
def api_root() -> dict[str, Any]:
    return {
        "name": "Day-Ahead Power Trading API",
        "status": "ready",
        "docs": "/api/docs",
        "health": "/api/health",
        "strategies": "/api/strategies",
        "results": "/api/results",
        "compare": "/api/compare",
        "simulate": "/api/simulate",
        "runtime_note": "Forecast serving and strategy simulation run here; model training remains an offline job.",
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "day-ahead-power-trader"}


@app.get("/api/strategies")
def strategies() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": description,
            "default_settings": DEFAULT_SETTINGS[name],
        }
        for name, description in STRATEGY_DESCRIPTIONS.items()
    ]


@app.get("/api/results")
def results(days: int | None = None) -> dict[str, Any]:
    forecasts, metrics, manifest = _load_deployment_results()
    selected = _select_window(forecasts, days)
    return {
        "dataset": {
            **manifest,
            "price_currency": "EUR/MWh for battery; DKK/MWh for Prop",
            "selected_rows": len(selected),
            "selected_start": selected["HourUTC"].min().isoformat(),
            "selected_end": selected["HourUTC"].max().isoformat(),
        },
        "forecast_models": FORECAST_COLUMNS,
        "model_metrics": metrics,
        "prices": _json_records(
            selected[
                [
                    "HourUTC",
                    "Actual_Price",
                    "Prediction",
                    "Hourly_Baseline",
                    "Direct_15min_Prediction",
                    "Actual_Price_DKK",
                    "Prediction_DKK",
                    "Hourly_Baseline_DKK",
                    "Direct_15min_Prediction_DKK",
                    "Imbalance_Price_DKK",
                    "Dominating_Direction",
                ]
            ]
        ),
    }


@lru_cache(maxsize=3)
def _load_saved_comparison(trading_setup: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "deployment_data" / SAVED_COMPARISON_FILES[trading_setup]
    if not path.exists():
        raise FileNotFoundError(f"Saved {trading_setup} comparison is unavailable.")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@app.get("/api/saved-comparison/{trading_setup}")
def saved_comparison(trading_setup: str) -> dict[str, Any]:
    normalized = trading_setup.lower().replace("-", "_")
    if normalized not in SAVED_COMPARISON_FILES:
        raise HTTPException(status_code=422, detail=f"Unknown trading setup: {trading_setup}")
    try:
        return _load_saved_comparison(normalized)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/compare")
def compare_strategies(payload: StrategyComparisonRequest) -> dict[str, Any]:
    if payload.forecast_col not in FORECAST_COLUMNS:
        raise HTTPException(status_code=422, detail=f"Unknown forecast column: {payload.forecast_col}")
    if payload.strategy is not None and payload.strategy not in STRATEGY_DESCRIPTIONS:
        raise HTTPException(status_code=422, detail=f"Unknown strategy: {payload.strategy}")
    trading_setup = payload.trading_setup.lower().replace("-", "_")
    if trading_setup not in {"battery", "prop", "imbalance"}:
        raise HTTPException(status_code=422, detail=f"Unknown trading setup: {payload.trading_setup}")
    is_prop = trading_setup == "prop"
    is_imbalance = trading_setup == "imbalance"
    is_directional = is_prop or is_imbalance
    if is_directional and payload.strategy is not None and payload.strategy not in PROP_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"Strategy is battery-only and unavailable in {trading_setup} mode: {payload.strategy}",
        )
    forecasts, metrics, manifest = _load_deployment_results()
    if is_directional:
        for column in ["Actual_Price", *FORECAST_COLUMNS]:
            forecasts[column] = forecasts[f"{column}_DKK"]
    history = _select_window(forecasts, payload.days)
    selected = history
    train: pd.DataFrame | None = None
    optimization_meta: dict[str, dict[str, Any]] = {}
    try:
        battery = BatteryConfig(**payload.battery)
        prop = PropConfig(**payload.prop)
        no_fee_battery = replace(
            battery,
            fee_per_mwh=0.0,
            charge_fee_per_mwh=0.0,
            discharge_fee_per_mwh=0.0,
        )
        no_fee_prop = replace(prop, transaction_cost_dkk_per_mwh=0.0)

        def prop_transform(
            simulation: pd.DataFrame, _summary: dict[str, float]
        ) -> tuple[pd.DataFrame, dict[str, float]]:
            return simulate_prop_positions_with_eod_imbalance(
                attach_imbalance_settlement(simulation), prop
            )

        def no_fee_prop_transform(
            simulation: pd.DataFrame, _summary: dict[str, float]
        ) -> tuple[pd.DataFrame, dict[str, float]]:
            return simulate_prop_positions_with_eod_imbalance(
                attach_imbalance_settlement(simulation), no_fee_prop
            )

        imbalance_reference = forecasts[
            ["HourUTC", "Imbalance_Price_DKK", "Dominating_Direction"]
        ]

        def attach_imbalance_settlement(simulation: pd.DataFrame) -> pd.DataFrame:
            missing = [
                column
                for column in ("Imbalance_Price_DKK", "Dominating_Direction")
                if column not in simulation.columns
            ]
            if not missing:
                return simulation
            return simulation.merge(
                imbalance_reference[["HourUTC", *missing]],
                on="HourUTC",
                how="left",
                validate="one_to_one",
            )

        def imbalance_transform(
            simulation: pd.DataFrame, _summary: dict[str, float]
        ) -> tuple[pd.DataFrame, dict[str, float]]:
            return simulate_imbalance_spread_positions(
                attach_imbalance_settlement(simulation), prop
            )

        def no_fee_imbalance_transform(
            simulation: pd.DataFrame, _summary: dict[str, float]
        ) -> tuple[pd.DataFrame, dict[str, float]]:
            return simulate_imbalance_spread_positions(
                attach_imbalance_settlement(simulation), no_fee_prop
            )

        result_transform = (
            prop_transform if is_prop else imbalance_transform if is_imbalance else None
        )
        no_fee_result_transform = (
            no_fee_prop_transform
            if is_prop
            else no_fee_imbalance_transform
            if is_imbalance
            else None
        )
        if payload.optimize:
            train, selected = _split_complete_day_holdout(history, payload.test_days)
            optimized = optimize_strategy_suite(
                train,
                battery_config=battery,
                ranking_metric="Cashflow",
                forecast_col=payload.forecast_col,
                result_transform=result_transform,
            )
            test_potential = optimize_strategy_suite(
                selected,
                battery_config=battery,
                ranking_metric="Cashflow",
                forecast_col=payload.forecast_col,
                result_transform=result_transform,
            )
            no_fee_potential = optimize_strategy_suite(
                selected,
                battery_config=battery if is_directional else no_fee_battery,
                ranking_metric="Cashflow",
                forecast_col=payload.forecast_col,
                result_transform=no_fee_result_transform,
            )
            if is_directional:
                optimized = optimized.loc[optimized["Strategy"].isin(PROP_STRATEGIES)].reset_index(drop=True)
                test_potential = test_potential.loc[
                    test_potential["Strategy"].isin(PROP_STRATEGIES)
                ].reset_index(drop=True)
                no_fee_potential = no_fee_potential.loc[
                    no_fee_potential["Strategy"].isin(PROP_STRATEGIES)
                ].reset_index(drop=True)
            test_potential_by_strategy = {
                str(row["Strategy"]): row for row in test_potential.to_dict(orient="records")
            }
            no_fee_potential_by_strategy = {
                str(row["Strategy"]): row
                for row in no_fee_potential.to_dict(orient="records")
            }
            suite: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}
            for optimized_row in optimized.to_dict(orient="records"):
                name = str(optimized_row["Strategy"])
                settings_json = str(optimized_row["Settings_JSON"])
                potential_row = test_potential_by_strategy[name]
                no_fee_potential_row = no_fee_potential_by_strategy[name]
                strategy_result = simulate_strategy_from_settings(
                    name,
                    selected,
                    settings_json,
                    battery_config=battery,
                    forecast_col=payload.forecast_col,
                )
                suite[name] = (
                    result_transform(*strategy_result)
                    if result_transform is not None
                    else strategy_result
                )
                optimization_meta[name] = {
                    "Settings": str(optimized_row["Settings"]),
                    "Settings_JSON": settings_json,
                    "Train_Cashflow": float(optimized_row["Cashflow"]),
                    "Train_Max_Drawdown": float(optimized_row["Max_Drawdown"]),
                    "Test_Potential_Cashflow": float(potential_row["Cashflow"]),
                    "Test_Potential_Max_Drawdown": float(potential_row["Max_Drawdown"]),
                    "Test_Potential_Settings": str(potential_row["Settings"]),
                    "Test_Potential_Settings_JSON": str(potential_row["Settings_JSON"]),
                    "No_Fee_Potential_Cashflow": float(no_fee_potential_row["Cashflow"]),
                    "No_Fee_Potential_Max_Drawdown": float(
                        no_fee_potential_row["Max_Drawdown"]
                    ),
                    "No_Fee_Potential_Settings": str(no_fee_potential_row["Settings"]),
                    "No_Fee_Potential_Settings_JSON": str(
                        no_fee_potential_row["Settings_JSON"]
                    ),
                    "Evaluations": int(optimized_row["Evaluations"]),
                }
        else:
            raw_suite = run_strategy_suite(
                selected,
                battery_config=battery,
                forecast_col=payload.forecast_col,
            )
            raw_no_fee_suite = (
                raw_suite
                if is_directional
                else run_strategy_suite(
                    selected,
                    battery_config=no_fee_battery,
                    forecast_col=payload.forecast_col,
                )
            )
            suite = {
                name: result_transform(*result) if result_transform is not None else result
                for name, result in raw_suite.items()
                if not is_directional or name in PROP_STRATEGIES
            }
            no_fee_suite = {
                name: no_fee_result_transform(*result)
                if no_fee_result_transform is not None
                else result
                for name, result in raw_no_fee_suite.items()
                if not is_directional or name in PROP_STRATEGIES
            }
            for name in suite:
                settings = DEFAULT_SETTINGS[name]
                _, no_fee_summary = no_fee_suite[name]
                optimization_meta[name] = {
                    "Settings": ", ".join(f"{key}={value}" for key, value in settings.items()),
                    "Settings_JSON": json.dumps(settings, sort_keys=True, separators=(",", ":")),
                    "Train_Cashflow": None,
                    "Train_Max_Drawdown": None,
                    "Test_Potential_Cashflow": None,
                    "Test_Potential_Max_Drawdown": None,
                    "Test_Potential_Settings": None,
                    "Test_Potential_Settings_JSON": None,
                    "No_Fee_Potential_Cashflow": float(no_fee_summary["total_cashflow"]),
                    "No_Fee_Potential_Max_Drawdown": float(no_fee_summary["max_drawdown"]),
                    "No_Fee_Potential_Settings": ", ".join(
                        f"{key}={value}" for key, value in settings.items()
                    ),
                    "No_Fee_Potential_Settings_JSON": json.dumps(
                        settings, sort_keys=True, separators=(",", ":")
                    ),
                    "Evaluations": 1,
                }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    comparison: list[dict[str, Any]] = []
    simulations: dict[str, pd.DataFrame] = {}
    for name, (simulation, summary) in suite.items():
        risk = summarize_cashflow_risk(simulation)
        description = STRATEGY_DESCRIPTIONS[name]
        if is_prop:
            description += (
                " In the prop proxy, buy/charge signals map to long positions and "
                "sell/discharge signals map to short positions for the next price move. "
                "Any position still open at the final DK1 interval is forced flat using "
                "that interval's realized imbalance price."
            )
        elif is_imbalance:
            description += (
                " In the unclosed-position scenario, buy/charge signals create long "
                "day-ahead positions and sell/discharge signals create short positions. "
                "They remain open into delivery and settle against the same-interval DK1 "
                "imbalance price."
            )
        if name == "Ensemble agreement" and int(summary.get("model_count", 0)) == 1:
            description += (
                " This deployment contains one compatible ensemble-output series, so its agreement share "
                "is always 100% and the current signal is equivalent to Forecast quantile."
            )
        comparison.append(
            {
                "Strategy": name,
                "Description": description,
                "Cashflow": float(summary["total_cashflow"]),
                "Max_Drawdown": float(summary["max_drawdown"]),
                "Risk_Adjusted_Score": float(risk["risk_adjusted_score"]),
                "Daily_Sharpe": float(risk["daily_sharpe"]),
                "Daily_Sortino": float(risk["daily_sortino"]),
                "Profit_Factor": float(risk["profit_factor"]),
                "Win_Rate": float(risk["win_rate"]),
                "Active_Intervals": int(summary["trades"]),
                "Charge_Intervals": int(summary["charge_intervals"]),
                "Discharge_Intervals": int(summary["discharge_intervals"]),
                "Fee_Cost": float(summary["total_fee_cost"]),
                "Degradation_Cost": float(summary.get("total_degradation_cost", 0.0)),
                "Round_Trip_Efficiency_Pct": float(summary["round_trip_efficiency"] * 100),
                "Final_SOC_MWh": float(summary["final_soc_mwh"]),
                "Return_Pct": float(summary.get("return_pct", float("nan"))),
                "Ending_Equity_DKK": float(summary.get("ending_equity_dkk", float("nan"))),
                "Position_Changes": int(summary.get("position_changes", summary["trades"])),
                **optimization_meta[name],
            }
        )
        simulations[name] = simulation
    comparison.sort(key=lambda row: (row["Cashflow"], -row["Max_Drawdown"]), reverse=True)

    best_name = str(comparison[0]["Strategy"])
    selected_name = payload.strategy or best_name
    best_simulation = simulations[best_name].copy()
    best_simulation["Cumulative_Cashflow"] = best_simulation["Cashflow"].cumsum()
    selected_simulation = simulations[selected_name].copy()
    selected_simulation["Cumulative_Cashflow"] = selected_simulation["Cashflow"].cumsum()
    if is_prop:
        oracle_simulation, oracle_summary = simulate_prop_eod_perfect_foresight(
            selected,
            config=prop,
            forecast_col=payload.forecast_col,
        )
    elif is_imbalance:
        oracle_simulation, oracle_summary = simulate_imbalance_perfect_foresight(
            selected,
            config=prop,
            forecast_col=payload.forecast_col,
        )
    else:
        oracle_simulation, oracle_summary = simulate_perfect_foresight_oracle(
            selected,
            battery_config=battery,
            optimizer_config=RollingOptimizerConfig(
                soc_steps=40,
                terminal_soc_mwh=0.0,
                market_timezone="Europe/Copenhagen",
            ),
            forecast_col=payload.forecast_col,
        )
    oracle_risk = summarize_cashflow_risk(oracle_simulation)
    series_columns = [
        "HourUTC",
        "Action",
        "Actual_Price",
        "Forecast_Price",
        "Dispatch_MW",
        "State_Of_Charge_MWh",
        "Cashflow",
        "Cumulative_Cashflow",
        "Position",
        "Position_MWh",
        "Position_After_Settlement",
        "Price_Change_DKK",
        "Day_Ahead_Price_DKK",
        "Imbalance_Price_DKK",
        "Imbalance_Spread_DKK",
        "Dominating_Direction",
        "Gross_Cashflow",
        "Transaction_Cost",
        "Equity_DKK",
        "Is_Day_End",
        "EOD_Imbalance_Settlement",
        "Settlement_Basis",
    ]
    best_series_columns = [column for column in series_columns if column in best_simulation.columns]
    selected_series_columns = [
        column for column in series_columns if column in selected_simulation.columns
    ]
    strategy_series: dict[str, list[dict[str, Any]]] = {}
    for name, simulation in simulations.items():
        serialized = simulation.copy()
        serialized["Cumulative_Cashflow"] = serialized["Cashflow"].cumsum()
        columns = [column for column in series_columns if column in serialized.columns]
        strategy_series[name] = _json_records(serialized[columns])
    return _clean_json(
        {
            "dataset": {
                **manifest,
                "trading_price_currency": (
                    "DKK/MWh" if is_directional else "Legacy thesis EUR/MWh basis"
                ),
                "history_rows": len(history),
                "history_start": history["HourUTC"].min().isoformat(),
                "history_end": history["HourUTC"].max().isoformat(),
                "selected_rows": len(selected),
                "selected_start": selected["HourUTC"].min().isoformat(),
                "selected_end": selected["HourUTC"].max().isoformat(),
            },
            "evaluation": {
                "mode": "out_of_sample_optimization" if payload.optimize else "fixed_defaults",
                "ranking_metric": "Net cashflow",
                "test_days": payload.test_days if payload.optimize else None,
                "train_rows": len(train) if train is not None else 0,
                "train_start": train["HourUTC"].min().isoformat() if train is not None else None,
                "train_end": train["HourUTC"].max().isoformat() if train is not None else None,
                "test_rows": len(selected),
                "test_start": selected["HourUTC"].min().isoformat(),
                "test_end": selected["HourUTC"].max().isoformat(),
                "daily_observations": int(
                    pd.to_datetime(selected["HourUTC"], utc=True)
                    .dt.tz_convert("Europe/Copenhagen")
                    .dt.date.nunique()
                ),
            },
            "forecast_col": payload.forecast_col,
            "forecast_model": FORECAST_COLUMNS[payload.forecast_col],
            "trading_setup": trading_setup,
            "request": {
                "days": payload.days,
                "optimize": payload.optimize,
                "test_days": payload.test_days,
            },
            "battery": asdict(battery),
            "prop": asdict(prop),
            "perfect_foresight_benchmark": {
                "label": (
                    "Perfect-foresight directional ceiling"
                    if is_prop
                    else "Perfect-foresight unclosed-position settlement ceiling"
                    if is_imbalance
                    else "Perfect-foresight DP ceiling"
                ),
                "Cashflow": float(oracle_summary["total_cashflow"]),
                "Max_Drawdown": float(oracle_summary["max_drawdown"]),
                "Daily_Sharpe": float(oracle_risk["daily_sharpe"]),
                "Active_Intervals": int(oracle_summary["trades"]),
                "Fee_Cost": float(oracle_summary["total_fee_cost"]),
                "Final_SOC_MWh": float(oracle_summary["final_soc_mwh"]),
                "Description": (
                    "Selects the hindsight-optimal long/flat/short path directly from realized "
                    "price changes and end-of-day imbalance spreads, including switching and "
                    "closing costs. It is not tradable."
                    if is_prop
                    else "Selects the hindsight-profitable day-ahead position for each realized "
                    "imbalance settlement with costs included. It is not a tradable strategy."
                    if is_imbalance
                    else "Optimizes directly on realized test prices. This is a hindsight opportunity "
                    "ceiling, not a tradable strategy or forecast result."
                ),
            },
            "model_metrics": metrics,
            "strategies": comparison,
            "best_strategy": best_name,
            "best_strategy_series": _json_records(best_simulation[best_series_columns]),
            "selected_strategy": selected_name,
            "selected_strategy_series": _json_records(
                selected_simulation[selected_series_columns]
            ),
            "strategy_series": strategy_series,
        }
    )


@app.post("/api/simulate")
def simulate(payload: SimulationRequest) -> dict[str, Any]:
    if payload.strategy not in STRATEGY_DESCRIPTIONS:
        raise HTTPException(status_code=422, detail=f"Unknown strategy: {payload.strategy}")
    trading_setup = payload.trading_setup.lower().replace("-", "_")
    if trading_setup not in {"battery", "prop", "imbalance"}:
        raise HTTPException(status_code=422, detail=f"Unknown trading setup: {payload.trading_setup}")
    if trading_setup in {"prop", "imbalance"} and payload.strategy not in PROP_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"Strategy is battery-only and unavailable in {trading_setup} mode: {payload.strategy}",
        )

    frame = pd.DataFrame.from_records(payload.records)
    required = {payload.time_col, payload.actual_col, payload.forecast_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {', '.join(missing)}")

    try:
        frame[payload.time_col] = pd.to_datetime(frame[payload.time_col], errors="raise", utc=True)
        for column in frame.columns:
            if column != payload.time_col:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame = frame.sort_values(payload.time_col).reset_index(drop=True)
        battery = BatteryConfig(**payload.battery)
        settings = payload.settings or DEFAULT_SETTINGS[payload.strategy]
        intervals, summary = simulate_strategy_from_settings(
            payload.strategy,
            frame,
            settings,
            battery_config=battery,
            time_col=payload.time_col,
            actual_col=payload.actual_col,
            forecast_col=payload.forecast_col,
        )
        if trading_setup == "prop":
            intervals, summary = simulate_prop_positions_with_eod_imbalance(
                intervals,
                PropConfig(**payload.prop),
                time_col=payload.time_col,
                actual_col=payload.actual_col,
            )
        elif trading_setup == "imbalance":
            intervals, summary = simulate_imbalance_spread_positions(
                intervals,
                PropConfig(**payload.prop),
                time_col=payload.time_col,
                day_ahead_col=payload.actual_col,
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response: dict[str, Any] = {
        "strategy": payload.strategy,
        "trading_setup": trading_setup,
        "settings": settings,
        "rows_processed": len(frame),
        "summary": summary,
    }
    if payload.include_intervals:
        response["intervals"] = _json_records(intervals)
    return response
