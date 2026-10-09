from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import gzip
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from intraday_power_quant.optimization import (
    optimize_strategy_suite,
    run_strategy_parameter_sweep,
    simulate_strategy_from_settings,
)
from intraday_power_quant.custom_strategy import (
    CustomStrategyRequest,
    prepare_custom_signals,
    requested_actions,
    run_custom_strategy,
)
from intraday_power_quant.latest_prices import latest_prices
from intraday_power_quant.portfolio import book_period_summary, simulate_battery_book, simulate_book
from intraday_power_quant.lab_presets import LAB_NOTE, preset_rule_text, presets_for_setup, run_presets
from intraday_power_quant.significance import significance_tests
from intraday_power_quant.tuning import (
    DailyPnlCapture,
    best_fixed_setting,
    local_dates,
    tune_strategy,
    walk_forward_folds,
)
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
    "Prediction": "Walk-forward champion",
    "Hourly_Baseline": "Hourly baseline only",
    "Direct_15min_Prediction": "Direct 15-min ensemble",
    "Transfer_Residual_Prediction": "TL residual simple average",
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
    tune_days: int = Field(default=20, ge=5, le=60)
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
    fundamentals = pd.read_csv(data_dir / "formula_features.csv.gz")
    fundamentals["HourUTC"] = pd.to_datetime(fundamentals["HourUTC"], utc=True)
    forecasts = forecasts.merge(fundamentals, on="HourUTC", how="left", validate="one_to_one")
    if forecasts[fundamentals.columns.drop("HourUTC")].isna().any().any():
        raise ValueError("Formula features do not cover every prediction timestamp.")
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


EVALUATION_DAYS = {"last_10_days": 10, "last_30_days": 30}


def _complete_dates(history: pd.DataFrame, market_timezone: str = "Europe/Copenhagen") -> list[object]:
    """Copenhagen delivery dates with every quarter-hour present (92/100 on daylight-saving days)."""
    local_timestamps = pd.to_datetime(history["HourUTC"], utc=True).dt.tz_convert(market_timezone)
    complete: list[object] = []
    for local_date in sorted(local_timestamps.dt.date.unique()):
        day_start = pd.Timestamp(local_date).tz_localize(market_timezone)
        expected = pd.date_range(day_start, day_start + pd.DateOffset(days=1), freq="15min", inclusive="left")
        actual = pd.DatetimeIndex(local_timestamps.loc[local_timestamps.dt.date == local_date]).sort_values()
        if actual.equals(expected):
            complete.append(local_date)
    return complete


def _split_complete_day_holdout(
    history: pd.DataFrame,
    test_days: int,
    market_timezone: str = "Europe/Copenhagen",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the final complete DK1 calendar days as a chronological holdout."""
    timestamps = pd.to_datetime(history["HourUTC"], utc=True)
    local_timestamps = timestamps.dt.tz_convert(market_timezone)
    complete_dates = _complete_dates(history, market_timezone)

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


@app.get("/api/analytics-config")
def analytics_config(response: Response) -> dict[str, Any]:
    """Expose only the public browser token, and only on production deployments."""
    response.headers["Cache-Control"] = "no-store"
    token = os.environ.get("POSTHOG_PROJECT_TOKEN", "").strip()
    host = os.environ.get("POSTHOG_HOST", "").strip().rstrip("/")
    if (
        os.environ.get("VERCEL_ENV") != "production"
        or not token.startswith("phc_")
        or host not in {"https://eu.i.posthog.com", "https://us.i.posthog.com"}
    ):
        return {"enabled": False}
    return {"enabled": True, "token": token, "host": host}


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
                    "Transfer_Residual_Prediction",
                    "Actual_Price_DKK",
                    "Prediction_DKK",
                    "Hourly_Baseline_DKK",
                    "Direct_15min_Prediction_DKK",
                    "Transfer_Residual_Prediction_DKK",
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


@app.post("/api/custom-strategy")
def custom_strategy(payload: CustomStrategyRequest) -> dict[str, Any]:
    try:
        history, _, _ = _load_deployment_results()
        frame = prepare_custom_signals(history, payload)
        if payload.evaluation in EVALUATION_DAYS:
            _, frame = _split_complete_day_holdout(frame, EVALUATION_DAYS[payload.evaluation])
        intervals, summary = run_custom_strategy(frame, payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    columns = ["HourUTC", "Custom_Signal", "Signal_Action", "Action", "Actual_Price",
               "Dispatch_MW", "State_Of_Charge_MWh", "Cashflow", "Cumulative_Cashflow"]
    if payload.trading_setup == "prop":
        # Imbalance spread and day-end flags let the browser re-simulate a netted portfolio of runs.
        columns += ["Position", "Position_MWh", "Size_Multiplier", "Position_After_Settlement",
                    "Transaction_Cost", "Equity_DKK", "Settlement_Basis", "Imbalance_Spread_DKK", "Is_Day_End"]
    # Saved model inputs aligned to each interval, for the Strategy Lab factor tiles.
    factor_columns = {"wind": "Wind_Total_DayAhead_MW", "solar": "Solar_DayAhead_MW",
                      "demand": "load_fc", "forecast": f"{payload.forecast_col}_DKK",
                      "baseline": "Hourly_Baseline_DKK", "temperature": "temperature_lag_96",
                      "gas": "gas_price_lag_96", "rank": "Forecast_Daily_Rank"}
    by_time = frame.set_index("HourUTC").reindex(intervals["HourUTC"])
    factors = {name: by_time[column].tolist() for name, column in factor_columns.items() if column in by_time}
    return _clean_json({
        "name": payload.name, "trading_setup": payload.trading_setup,
        "currency": "DKK", "summary": summary,
        "period": {"start": frame["HourUTC"].min().isoformat(),
                   "end": frame["HourUTC"].max().isoformat(), "rows": len(frame)},
        "settings": payload.model_dump(exclude={"signal_records", "fundamental_records"}),
        "intervals": _json_records(intervals[columns]),
        "factors": factors,
    })


def _period_summary(frame: pd.DataFrame, intervals: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    days = pd.to_datetime(frame["HourUTC"], utc=True).dt.tz_convert("Europe/Copenhagen").dt.date.nunique()
    active = intervals["Dispatch_MW"].abs() > 0
    return {
        "start": frame["HourUTC"].min().isoformat(), "end": frame["HourUTC"].max().isoformat(),
        "days": int(days), "rows": len(frame),
        "total_cashflow": summary["total_cashflow"], "pnl_per_day": summary["total_cashflow"] / max(days, 1),
        "max_drawdown": summary["max_drawdown"], "active_intervals": int(active.sum()),
        "win_rate": float((intervals.loc[active, "Cashflow"] > 0).mean()) if active.any() else None,
    }


def _robustness_periods(prepared: pd.DataFrame, evaluation: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """(earlier, test, evaluated, labels): a final-N-day test is compared with everything before it;
    the full period is compared half against half."""
    if evaluation in EVALUATION_DAYS:
        days = EVALUATION_DAYS[evaluation]
        earlier, test = _split_complete_day_holdout(prepared, days)
        return earlier, test, test, ["Earlier period", f"Final {days} days"]
    local_dates = pd.to_datetime(prepared["HourUTC"], utc=True).dt.tz_convert("Europe/Copenhagen").dt.date
    dates = sorted(local_dates.unique())
    second_half = local_dates >= dates[len(dates) // 2]
    return (prepared[~second_half].reset_index(drop=True), prepared[second_half].reset_index(drop=True),
            prepared, ["First half", "Second half"])


@app.post("/api/custom-strategy/robustness")
def custom_strategy_robustness(payload: CustomStrategyRequest) -> dict[str, Any]:
    """Out-of-sample check (earlier period vs the test period) and a threshold sensitivity grid."""
    try:
        history, _, _ = _load_deployment_results()
        prepared = prepare_custom_signals(history, payload)
        earlier, final, evaluated, labels = _robustness_periods(prepared, payload.evaluation)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    periods: dict[str, Any] = {}
    # Lagged inputs can be missing at the very start of the history; drop only those leading rows.
    finite = pd.Series(pd.to_numeric(earlier["Custom_Signal"], errors="coerce")).notna()
    earlier = earlier.loc[finite.idxmax():].reset_index(drop=True) if finite.any() else earlier
    for (name, frame), label in zip({"earlier": earlier, "test": final}.items(), labels):
        try:
            intervals, summary = run_custom_strategy(frame, payload)
            periods[name] = {"label": label, **_period_summary(frame, intervals, summary)}
        except (ValueError, TypeError, KeyError) as exc:
            periods[name] = {"label": label, "error": str(exc)}

    # Sensitivity: shift each threshold by up to ±2 steps (step = an eighth of the threshold gap).
    step = (payload.upper - payload.lower) / 8
    offsets = [-2, -1, 0, 1, 2]
    lowers = [payload.lower + k * step for k in offsets]
    uppers = [payload.upper + k * step for k in offsets]
    grid: list[list[float | None]] = []
    for lower in lowers:
        row: list[float | None] = []
        for upper in uppers:
            if lower >= upper:
                row.append(None)
                continue
            update: dict[str, Any] = {"lower": lower, "upper": upper}
            if payload.sizing != "fixed":
                # Keep the outer sizing thresholds the same distance beyond the shifted entry thresholds.
                update |= {"outer_lower": payload.outer_lower + lower - payload.lower,
                           "outer_upper": payload.outer_upper + upper - payload.upper}
            cell_request = payload.model_copy(update=update)
            cell = evaluated.copy()
            cell["Requested_Action"] = requested_actions(cell["Custom_Signal"], cell_request)
            try:
                row.append(run_custom_strategy(cell, cell_request)[1]["total_cashflow"])
            except (ValueError, TypeError, KeyError):
                row.append(None)
        grid.append(row)
    try:  # battery runs get the daily t-test; the permutation tests need Prop proxy positions
        significance = significance_tests(run_custom_strategy(evaluated, payload)[0])
    except (ValueError, TypeError, KeyError) as exc:
        significance = {"error": str(exc)}
    return _clean_json({
        "periods": periods,
        "grid": {"lowers": lowers, "uppers": uppers, "pnl": grid, "evaluated_rows": len(evaluated),
                 "evaluation": payload.evaluation},
        "significance": significance,
    })


class PortfolioRobustnessRequest(BaseModel):
    runs: list[CustomStrategyRequest] = Field(min_length=1, max_length=5)
    weights: list[float]
    cap_mwh: float = Field(default=0.0, ge=0, le=1e6)
    daily_loss_limit_dkk: float = Field(default=0.0, ge=0, le=1e12)


@app.post("/api/portfolio/robustness")
def portfolio_robustness(payload: PortfolioRobustnessRequest) -> dict[str, Any]:
    """Robustness of a netted book of Strategy Lab runs: earlier vs test period and significance tests."""
    if len(payload.weights) != len(payload.runs) or any(w < 0 for w in payload.weights) or not any(payload.weights):
        raise HTTPException(status_code=422, detail="Give one non-negative weight per run, at least one above 0.")
    setup = payload.runs[0].trading_setup
    if any(run.trading_setup != setup for run in payload.runs):
        raise HTTPException(status_code=422, detail="A book combines runs of one trading setup: all Prop proxy or all Battery.")
    history, _, _ = _load_deployment_results()
    per_period: dict[str, list[pd.DataFrame]] = {"earlier": [], "test": []}
    labels: list[str] = []
    try:
        for run in payload.runs:
            prepared = prepare_custom_signals(history, run)
            earlier, test, _, labels = _robustness_periods(prepared, payload.runs[0].evaluation)
            finite = pd.Series(pd.to_numeric(earlier["Custom_Signal"], errors="coerce")).notna()
            earlier = earlier.loc[finite.idxmax():].reset_index(drop=True) if finite.any() else earlier
            for name, frame in {"earlier": earlier, "test": test}.items():
                per_period[name].append(run_custom_strategy(frame, run)[0])
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cost_rate = payload.runs[0].prop.transaction_cost_dkk_per_mwh
    battery = BatteryConfig(**payload.runs[0].battery.model_dump())
    periods: dict[str, Any] = {}
    test_book = None
    for (name, frames), label in zip(per_period.items(), labels):
        if setup == "battery":  # one shared battery; cap_mwh is read as a power cap in MW
            book = simulate_battery_book(frames, payload.weights, battery, payload.cap_mwh, payload.daily_loss_limit_dkk)
        else:
            book = simulate_book(frames, payload.weights, cost_rate, payload.cap_mwh, payload.daily_loss_limit_dkk)
        periods[name] = {"label": label, **book_period_summary(book)}
        if name == "test":
            test_book = book
    return _clean_json({
        "periods": periods,
        "significance": significance_tests(test_book),
        "members": len(payload.runs),
        "trading_setup": setup,
        "evaluation": payload.runs[0].evaluation,
    })


@app.get("/api/latest-dk1-prices")
def get_latest_dk1_prices(response: Response):
    try:
        data = latest_prices()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Energinet prices are temporarily unavailable. Try again later.") from exc
    response.headers["Cache-Control"] = "public, max-age=60, s-maxage=600"
    return data


def _settings_label(settings_json: str) -> str:
    settings = json.loads(settings_json)
    return ", ".join(f"{key}={'off' if value is None else value}" for key, value in settings.items())


def _stitched_summary(
    simulation: pd.DataFrame, is_directional: bool, battery: BatteryConfig, prop: PropConfig, degradation: float
) -> dict[str, float]:
    """Summary of walk-forward blocks stitched together (each block traded with its own tuned setting)."""
    cashflow = simulation["Cashflow"].astype(float)
    cumulative = cashflow.cumsum()
    total = float(cashflow.sum())
    summary: dict[str, float] = {
        "total_cashflow": total,
        "max_drawdown": float((cumulative.cummax() - cumulative).max()) if len(simulation) else 0.0,
    }
    if is_directional:
        position = simulation["Position"]
        summary.update({
            "trades": int((position != 0).sum()), "charge_intervals": int((position > 0).sum()),
            "discharge_intervals": int((position < 0).sum()),
            "total_fee_cost": float(simulation["Transaction_Cost"].sum()), "total_degradation_cost": 0.0,
            "round_trip_efficiency": 1.0, "final_soc_mwh": float("nan"),
            "return_pct": total / prop.initial_capital_dkk * 100,
            "ending_equity_dkk": prop.initial_capital_dkk + total,
            "position_changes": int(position.ne(position.shift(fill_value=0)).sum()),
        })
        return summary
    dispatch = simulation["Dispatch_MW"].astype(float)
    charged, discharged = float(-dispatch[dispatch < 0].sum() * 0.25), float(dispatch[dispatch > 0].sum() * 0.25)
    summary.update({
        "trades": int((simulation["Action"] != "hold").sum()),
        "charge_intervals": int((simulation["Action"] == "charge").sum()),
        "discharge_intervals": int((simulation["Action"] == "discharge").sum()),
        "total_fee_cost": charged * (battery.fee_per_mwh + battery.charge_fee_per_mwh)
        + discharged * (battery.fee_per_mwh + battery.discharge_fee_per_mwh),
        "total_degradation_cost": degradation,
        "round_trip_efficiency": battery.charge_efficiency * battery.discharge_efficiency,
        "final_soc_mwh": float(simulation["State_Of_Charge_MWh"].iloc[-1]) if len(simulation) else float("nan"),
    })
    return summary


@app.post("/api/compare")
def compare_strategies(payload: StrategyComparisonRequest) -> dict[str, Any]:
    if payload.forecast_col not in FORECAST_COLUMNS:
        raise HTTPException(status_code=422, detail=f"Unknown forecast column: {payload.forecast_col}")
    trading_setup = payload.trading_setup.lower().replace("-", "_")
    if trading_setup not in {"battery", "prop", "imbalance"}:
        raise HTTPException(status_code=422, detail=f"Unknown trading setup: {payload.trading_setup}")
    lab_names = {preset["name"] for preset in presets_for_setup(trading_setup)}
    if payload.strategy is not None and payload.strategy not in STRATEGY_DESCRIPTIONS and payload.strategy not in lab_names:
        raise HTTPException(status_code=422, detail=f"Unknown strategy: {payload.strategy}")
    is_prop = trading_setup == "prop"
    is_imbalance = trading_setup == "imbalance"
    is_directional = is_prop or is_imbalance
    if (is_directional and payload.strategy is not None and payload.strategy not in PROP_STRATEGIES
            and payload.strategy not in lab_names):
        raise HTTPException(
            status_code=422,
            detail=f"Strategy is battery-only and unavailable in {trading_setup} mode: {payload.strategy}",
        )
    forecasts, metrics, manifest = _load_deployment_results()
    if is_directional:
        # Copy first: the loaded frame is cached, and converting it in place leaked DKK prices into
        # later battery (EUR) requests on the same server instance.
        forecasts = forecasts.copy()
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
            # Walk-forward tuning: one sweep over the window records each setting's daily P&L; every test
            # fold then trades the setting with the best neighbour-smoothed P&L over the days before it.
            swept_strategies = PROP_STRATEGIES if is_directional else None
            folds = walk_forward_folds(_complete_dates(history), payload.tune_days, payload.test_days)
            test_dates = [day for _, fold_test in folds for day in fold_test]
            tune_dates = sorted({day for fold_tune, _ in folds for day in fold_tune})
            history_dates = local_dates(history["HourUTC"])
            selected = history[history_dates.isin(test_dates)].reset_index(drop=True)
            train = history[history_dates.isin(tune_dates)].reset_index(drop=True)
            capture = DailyPnlCapture()
            run_strategy_parameter_sweep(
                history,
                battery_config=battery,
                forecast_col=payload.forecast_col,
                result_transform=result_transform,
                strategies=swept_strategies,
                capture=capture,
            )
            if not is_directional:
                no_fee_capture = DailyPnlCapture()
                run_strategy_parameter_sweep(
                    selected,
                    battery_config=no_fee_battery,
                    forecast_col=payload.forecast_col,
                    strategies=swept_strategies,
                    capture=no_fee_capture,
                )
                no_fee_best = {name: best_fixed_setting(entries, test_dates) for name, entries in no_fee_capture.daily.items()}
            else:
                no_fee_best = {name: best_fixed_setting(entries, test_dates) for name, entries in capture.gross_daily.items()}
            suite: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}
            for name, entries in capture.daily.items():
                fold_results = tune_strategy(entries, folds)
                pieces, degradation = [], 0.0
                for fold in fold_results:
                    simulation, sim_summary = simulate_strategy_from_settings(
                        name, history, fold["settings_json"], battery_config=battery, forecast_col=payload.forecast_col,
                    )
                    if result_transform is not None:
                        simulation, sim_summary = result_transform(simulation, sim_summary)
                    piece = simulation[local_dates(simulation["HourUTC"]).isin(fold["test_dates"])]
                    if not is_directional and "Dispatch_MW" in piece:
                        throughput = float(piece["Dispatch_MW"].abs().sum() * 0.25)
                        degradation += throughput * float(json.loads(fold["settings_json"]).get("degradation_cost_per_mwh", 0.0))
                    pieces.append(piece)
                stitched = pd.concat(pieces, ignore_index=True)
                suite[name] = (stitched, _stitched_summary(stitched, is_directional, battery, prop, degradation))
                best_json, best_pnl = best_fixed_setting(entries, test_dates)
                no_fee_json, no_fee_pnl = no_fee_best[name]
                train_total = sum(fold["train_pnl"] for fold in fold_results)
                train_days = len(folds) * payload.tune_days
                train_per_day = train_total / train_days
                test_per_day = suite[name][1]["total_cashflow"] / len(test_dates)
                settings_text = "; ".join(
                    f"block {i + 1}: {_settings_label(fold['settings_json'])}" for i, fold in enumerate(fold_results)
                )
                optimization_meta[name] = {
                    "Settings": settings_text,
                    "Settings_JSON": fold_results[-1]["settings_json"],
                    "Train_Cashflow": float(train_total),
                    "Train_Max_Drawdown": None,
                    "Train_Days": train_days,
                    "OOS_Kept_Pct": float(test_per_day / train_per_day * 100) if train_per_day > 0 else None,
                    "Walk_Forward_Blocks": [
                        {"Tune_Start": str(fold["tune_dates"][0]), "Tune_End": str(fold["tune_dates"][-1]),
                         "Test_Start": str(fold["test_dates"][0]), "Test_End": str(fold["test_dates"][-1]),
                         "Settings": _settings_label(fold["settings_json"]), "Tuned_Cashflow": fold["train_pnl"],
                         "Best_Unsmoothed_Tuned_Cashflow": fold["train_raw_best_pnl"], "Test_Cashflow": fold["test_pnl"]}
                        for fold in fold_results
                    ],
                    "Test_Potential_Cashflow": float(best_pnl),
                    "Test_Potential_Max_Drawdown": None,
                    "Test_Potential_Settings": _settings_label(best_json),
                    "Test_Potential_Settings_JSON": best_json,
                    "No_Fee_Potential_Cashflow": float(no_fee_pnl),
                    "No_Fee_Potential_Max_Drawdown": None,
                    "No_Fee_Potential_Settings": _settings_label(no_fee_json),
                    "No_Fee_Potential_Settings_JSON": no_fee_json,
                    "Evaluations": len(entries),
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
        # Strategy Lab presets: fixed rules, run over the whole window (so lagged signals and the battery's
        # state of charge carry over as for the other strategies) and scored on the same days.
        lab_descriptions: dict[str, str] = {}
        for name, (preset, simulation, no_fee_simulation) in run_presets(
            history, trading_setup, battery, prop, payload.forecast_col
        ).items():
            forecast_col = preset.get("forecast", payload.forecast_col)
            simulation = simulation.merge(
                history[["HourUTC", forecast_col]].rename(columns={forecast_col: "Forecast_Price"}), on="HourUTC", how="left"
            )
            piece = simulation[simulation["HourUTC"].isin(selected["HourUTC"])].reset_index(drop=True)
            no_fee_cash = no_fee_simulation.loc[no_fee_simulation["HourUTC"].isin(selected["HourUTC"]), "Cashflow"].cumsum()
            suite[name] = (piece, _stitched_summary(piece, is_directional, battery, prop, 0.0))
            lab_descriptions[name] = preset["desc"] + LAB_NOTE
            rule = f"{preset_rule_text(preset)} · forecast: {FORECAST_COLUMNS.get(forecast_col, forecast_col)}"
            rule_json = json.dumps(preset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            optimization_meta[name] = {
                "Settings": rule, "Settings_JSON": rule_json,
                "Train_Cashflow": None, "Train_Max_Drawdown": None, "Train_Days": None, "OOS_Kept_Pct": None,
                "Walk_Forward_Blocks": [],
                "Test_Potential_Cashflow": None, "Test_Potential_Max_Drawdown": None,
                "Test_Potential_Settings": None, "Test_Potential_Settings_JSON": None,
                "No_Fee_Potential_Cashflow": float(no_fee_cash.iloc[-1]) if len(no_fee_cash) else 0.0,
                "No_Fee_Potential_Max_Drawdown": float((no_fee_cash.cummax() - no_fee_cash).max()) if len(no_fee_cash) else 0.0,
                "No_Fee_Potential_Settings": rule, "No_Fee_Potential_Settings_JSON": rule_json,
                "Evaluations": 1,
            }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    comparison: list[dict[str, Any]] = []
    simulations: dict[str, pd.DataFrame] = {}
    for name, (simulation, summary) in suite.items():
        risk = summarize_cashflow_risk(simulation)
        description = lab_descriptions.get(name) or STRATEGY_DESCRIPTIONS[name]
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
                "Source": "Strategy Lab" if name in lab_descriptions else "Benchmark",
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
                "tuning": "walk_forward_neighbour_smoothed" if payload.optimize else None,
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
                "tune_days": payload.tune_days,
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
