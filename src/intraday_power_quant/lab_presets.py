"""The Strategy Lab's ready-made strategies, run as fixed rules in the benchmark comparison.

The presets live in ``public/strategy-presets.json``, which the lab page loads too, so the lab and the
benchmarks always test the same rules. Signals are built exactly as in the lab (DKK forecasts and
thresholds); the trades are then settled with the benchmark's own setup, prices and costs, so a preset
row is comparable with the tuned strategies in the same table. Presets are never tuned.
"""
from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .custom_strategy import CustomStrategyRequest, prepare_custom_signals, run_custom_strategy
from .imbalance_trading import simulate_imbalance_spread_positions
from .prop_trading import PropConfig
from .trading import BatteryConfig

PRESETS_PATH = Path(__file__).resolve().parents[2] / "public" / "strategy-presets.json"
LAB_NOTE = (
    " Strategy Lab preset: a fixed rule, never tuned, so it has no tuned PnL. Its thresholds were set when "
    "the lab was built, from the same price history, so they are not strictly out of sample."
)


@lru_cache(maxsize=1)
def load_presets() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(PRESETS_PATH.read_text(encoding="utf-8")))


def preset_request(preset: dict[str, Any], trading_setup: str, forecast_col: str = "Prediction") -> CustomStrategyRequest:
    """The same request the lab sends when the preset card is pressed (minus the user's form settings)."""
    sized = bool(preset.get("sizing")) and trading_setup == "prop"
    return CustomStrategyRequest(
        name=preset["name"],
        trading_setup="battery" if trading_setup == "battery" else "prop",
        signal=preset.get("signal", "formula"),
        formula=preset.get("formula", "signal = -wind"),
        lookback=preset.get("lookback", 4),
        direction=preset["direction"],
        lower=preset["lower"],
        upper=preset["upper"],
        forecast_col=preset.get("forecast", forecast_col),
        evaluation="full",
        sizing=preset["sizing"] if sized else "fixed",
        **({"outer_lower": preset["outerLower"], "outer_upper": preset["outerUpper"],
            "max_multiplier": preset["maxSize"]} if sized else {}),
    )


def preset_rule_text(preset: dict[str, Any]) -> str:
    low = preset["direction"] == "buy_low"
    signal = preset.get("formula") or f"signal = forecast change over {preset.get('lookback', 4)} × 15 min"
    text = (f"{signal} · long if {'≤' if low else '≥'} {preset['lower' if low else 'upper']}"
            f" · short if {'≥' if low else '≤'} {preset['upper' if low else 'lower']}")
    if preset.get("sizing") == "step":
        text += f" · {preset['maxSize']}× beyond {preset['outerLower']}/{preset['outerUpper']}"
    elif preset.get("sizing") == "scaled":
        text += f" · up to {preset['maxSize']}× at {preset['outerLower']}/{preset['outerUpper']}"
    return text


def presets_for_setup(trading_setup: str) -> list[dict[str, Any]]:
    """Dynamic sizing is a Prop proxy feature, as in the lab; the other setups skip the sizing presets."""
    return [p for p in load_presets() if trading_setup == "prop" or not p.get("sizing")]


def run_preset(
    history: pd.DataFrame,
    preset: dict[str, Any],
    trading_setup: str,
    battery: BatteryConfig,
    prop: PropConfig,
    forecast_col: str = "Prediction",
) -> pd.DataFrame:
    """Interval results of one preset over ``history`` with the benchmark's setup and costs.

    ``history`` must hold the deployment frame's ``*_DKK``/``*_EUR`` price columns. Battery benchmarks are
    settled on the thesis EUR basis, so the battery trades are re-priced in EUR after the DKK signals are made.
    """
    request = preset_request(preset, trading_setup, forecast_col)
    frame = prepare_custom_signals(history, request)
    if trading_setup == "battery":
        frame["Actual_Price"] = frame["Actual_Price_EUR"]
        return run_custom_strategy(frame, request, battery=battery)[0]
    if trading_setup == "imbalance":
        frame = frame.copy()
        frame["Signal_Action"] = frame["Requested_Action"]
        return simulate_imbalance_spread_positions(frame, prop)[0]
    return run_custom_strategy(frame, request, prop=prop)[0]


def run_presets(
    history: pd.DataFrame,
    trading_setup: str,
    battery: BatteryConfig,
    prop: PropConfig,
    forecast_col: str = "Prediction",
) -> dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]:
    """{name: (preset, with costs, without costs)} for every preset that applies to the setup.

    Presets without a forecast of their own use ``forecast_col``, the benchmark's forecast selector.
    """
    no_fee_battery = replace(battery, fee_per_mwh=0.0, charge_fee_per_mwh=0.0, discharge_fee_per_mwh=0.0)
    no_fee_prop = replace(prop, transaction_cost_dkk_per_mwh=0.0)
    return {
        preset["name"]: (
            preset,
            run_preset(history, preset, trading_setup, battery, prop, forecast_col),
            run_preset(history, preset, trading_setup, no_fee_battery, no_fee_prop, forecast_col),
        )
        for preset in presets_for_setup(trading_setup)
    }
