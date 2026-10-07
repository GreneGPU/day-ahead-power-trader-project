"""Bounded, declarative strategy rules over forecast-time inputs."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .trading import BatteryConfig, _simulate_battery_dispatch
from .prop_trading import PropConfig, simulate_prop_positions_with_eod_imbalance
from .formulas import THESIS_FEATURES, evaluate_formula


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SignalRecord(StrictModel):
    HourUTC: datetime
    Signal: float

    @model_validator(mode="after")
    def timezone_required(self):
        if self.HourUTC.utcoffset() is None:
            raise ValueError("CSV timestamps must include a timezone (use Z for UTC).")
        return self


class CustomBattery(StrictModel):
    capacity_mwh: float = Field(default=100, gt=0, le=100000)
    power_mw: float = Field(default=25, gt=0, le=100000)
    initial_soc_mwh: float = Field(default=0, ge=0)
    charge_efficiency: float = Field(default=0.90**0.5, gt=0, le=1)
    discharge_efficiency: float = Field(default=0.90**0.5, gt=0, le=1)
    charge_fee_per_mwh: float = Field(default=115.41, ge=0, le=100000)
    discharge_fee_per_mwh: float = Field(default=10.71, ge=0, le=100000)

    @model_validator(mode="after")
    def empty_start(self):
        if self.initial_soc_mwh != 0:
            raise ValueError("Custom battery tests start empty, so there is no free initial inventory.")
        return self


class FundamentalRecord(StrictModel):
    HourUTC: datetime
    demand: float | None = None
    outages: float | None = None
    wind: float | None = None
    solar: float | None = None

    @model_validator(mode="after")
    def valid_row(self):
        if self.HourUTC.utcoffset() is None:
            raise ValueError("Fundamentals timestamps must include a timezone.")
        if all(getattr(self, key) is None for key in ("demand", "outages", "wind", "solar")):
            raise ValueError("Each fundamentals row needs at least one numeric input.")
        return self


class CustomProp(StrictModel):
    initial_capital_dkk: float = Field(default=1000000, gt=0, le=1e12)
    position_size_mwh: float = Field(default=10, gt=0, le=100000)
    transaction_cost_dkk_per_mwh: float = Field(default=0.41, ge=0, le=100000)
    max_daily_loss_dkk: float | None = Field(default=None, gt=0, le=1e12)


class CustomStrategyRequest(StrictModel):
    name: str = Field(default="Custom strategy", min_length=1, max_length=80)
    trading_setup: Literal["battery", "prop"] = "battery"
    forecast_col: Literal["Prediction", "Hourly_Baseline", "Direct_15min_Prediction"] = "Prediction"
    signal: Literal["formula", "forecast", "baseline_spread", "forecast_change", "csv"] = "baseline_spread"
    formula: str = Field(default="signal = -wind", min_length=1, max_length=500)
    fundamental_records: list[FundamentalRecord] | None = Field(default=None, max_length=20000)
    lower: float = Field(default=-20, ge=-1e6, le=1e6)
    upper: float = Field(default=20, ge=-1e6, le=1e6)
    direction: Literal["buy_low", "buy_high"] = "buy_low"
    lookback: int = Field(default=4, ge=1, le=672)
    evaluation: Literal["full", "last_10_days"] = "last_10_days"
    battery: CustomBattery = Field(default_factory=CustomBattery)
    prop: CustomProp = Field(default_factory=CustomProp)
    signal_records: list[SignalRecord] | None = Field(default=None, max_length=20000)

    @model_validator(mode="after")
    def consistent_rules(self):
        if self.lower >= self.upper:
            raise ValueError("Lower threshold must be below upper threshold.")
        if self.signal == "csv" and not self.signal_records:
            raise ValueError("Upload a CSV with HourUTC and Signal columns first.")
        if self.signal != "csv" and self.signal_records is not None:
            raise ValueError("Uploaded signals can only be used with the CSV signal source.")
        if self.signal != "formula" and self.fundamental_records is not None:
            raise ValueError("Fundamentals uploads require the formula signal source.")
        return self


def prepare_custom_signals(history: pd.DataFrame, request: CustomStrategyRequest) -> pd.DataFrame:
    frame = history.copy().sort_values("HourUTC").reset_index(drop=True)
    # All custom tests use DKK consistently, including battery settlement and fees.
    frame["Actual_Price"] = frame["Actual_Price_DKK"]
    forecast = frame[f"{request.forecast_col}_DKK"]
    if request.signal == "formula":
        inputs = {"forecast": forecast, "baseline": frame["Hourly_Baseline_DKK"]}
        for name, column in {"wind": "Wind_Total_DayAhead_MW", "solar": "Solar_DayAhead_MW", "demand": "load_fc", **{name: name for name in THESIS_FEATURES}}.items():
            if column in frame:
                inputs[name] = frame[column]
        if request.fundamental_records:
            records = pd.DataFrame([row.model_dump() for row in request.fundamental_records])
            records["HourUTC"] = pd.to_datetime(records["HourUTC"], utc=True)
            if records["HourUTC"].duplicated().any():
                raise ValueError("Fundamentals CSV contains duplicate timestamps.")
            if not records["HourUTC"].isin(frame["HourUTC"]).all():
                raise ValueError("Fundamentals timestamps must match the saved dataset. Download the template.")
            for name in ("demand", "outages", "wind", "solar"):
                if records[name].notna().any():
                    inputs[name] = frame["HourUTC"].map(records.set_index("HourUTC")[name])
        signal = evaluate_formula(request.formula, inputs, frame.index)
    elif request.signal == "forecast":
        signal = forecast
    elif request.signal == "baseline_spread":
        signal = forecast - frame["Hourly_Baseline_DKK"]
    elif request.signal == "forecast_change":
        signal = forecast - forecast.shift(request.lookback)
    else:
        records = pd.DataFrame([row.model_dump() for row in request.signal_records])
        records["HourUTC"] = pd.to_datetime(records["HourUTC"], utc=True)
        if records["HourUTC"].duplicated().any():
            raise ValueError("CSV contains duplicate timestamps.")
        if not records["HourUTC"].isin(frame["HourUTC"]).all():
            raise ValueError("CSV timestamps must match the saved dataset. Download the template.")
        signal = frame["HourUTC"].map(records.set_index("HourUTC")["Signal"])
    frame["Custom_Signal"] = signal
    buy = signal <= request.lower if request.direction == "buy_low" else signal >= request.upper
    sell = signal >= request.upper if request.direction == "buy_low" else signal <= request.lower
    frame["Requested_Action"] = np.select([buy, sell], ["charge", "discharge"], default="hold")
    return frame


def run_custom_strategy(frame: pd.DataFrame, request: CustomStrategyRequest):
    if request.signal == "formula" and not np.isfinite(frame["Custom_Signal"]).all():
        raise ValueError("Formula inputs must cover every interval in the selected test period with finite values.")
    if request.signal == "csv" and frame["Custom_Signal"].isna().any():
        raise ValueError("CSV must supply a signal for every interval in the selected test period.")
    if request.trading_setup == "battery":
        intervals, summary = _simulate_battery_dispatch(
            frame, BatteryConfig(**request.battery.model_dump()),
            "HourUTC", "Actual_Price", "Requested_Action",
        )
    else:
        frame = frame.copy()
        frame["Signal_Action"] = frame["Requested_Action"]
        intervals, summary = simulate_prop_positions_with_eod_imbalance(
            frame, PropConfig(**request.prop.model_dump()),
        )
    cumulative = intervals["Cumulative_Cashflow"]
    # Include starting cashflow of zero in drawdown, including a first-interval loss.
    summary["max_drawdown"] = float((cumulative.cummax().clip(lower=0) - cumulative).max())
    summary["active_intervals"] = int((intervals["Dispatch_MW"].abs() > 0).sum())
    summary["warmup_intervals"] = int(frame["Custom_Signal"].isna().sum())
    return intervals, summary
