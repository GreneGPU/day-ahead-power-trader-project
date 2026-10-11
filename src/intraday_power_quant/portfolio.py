"""Net several Prop proxy runs into one book, with the same rules as the Strategy Lab's Portfolio panel.

Each interval's net exposure is the weighted sum of the runs' MWh. The book is re-simulated with the Prop
proxy rules: the next day-ahead move, the imbalance spread when a position is open at the day's last
interval, and a trading cost on every change in MWh (plus the close at day end). An optional cap limits net
exposure and an optional daily loss limit flattens the book for the rest of the day.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .trading import BatteryConfig, _simulate_battery_dispatch

REQUIRED = ["HourUTC", "Position_MWh", "Cashflow", "Actual_Price", "Imbalance_Spread_DKK", "Is_Day_End"]


def align_runs(runs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Restrict every run to the timestamps all runs share, in time order."""
    common = set(runs[0]["HourUTC"])
    for run in runs[1:]:
        common &= set(run["HourUTC"])
    return [run[run["HourUTC"].isin(common)].sort_values("HourUTC").reset_index(drop=True) for run in runs]


def day_index(run: pd.DataFrame, utc_days: bool = False) -> np.ndarray:
    """Day number of every interval: Prop days end at the ``Is_Day_End`` row, battery days are UTC dates."""
    if utc_days:
        return pd.factorize(pd.to_datetime(run["HourUTC"], utc=True).dt.strftime("%Y-%m-%d"))[0]
    ends = run["Is_Day_End"].to_numpy(bool)
    return np.concatenate([[0], np.cumsum(ends)[:-1]]).astype(int)


def rolling_weights(runs: list[pd.DataFrame], window: int, tilt: float, cap: float, utc_days: bool = False) -> np.ndarray:
    """Per-interval weights (intervals x runs) that lean towards the runs that earned the recent profit.

    For each day, a run's profit share is its stand-alone P&L over the previous ``window`` days (losses count as
    zero) divided by the sum over runs, scaled so the weights sum to the number of runs. The weight is
    ``(1 - tilt) * 1 + tilt * share``, capped at ``cap`` with the excess handed to the uncapped runs in
    proportion. Days before a full window, and windows where no run made money, use equal weights. Only earlier
    days are used, so the weights could have been known before the day. The rule buys recent winners: it helps
    when winners persist and hurts when leadership rotates.
    """
    runs = align_runs(runs)
    n, days = len(runs), day_index(runs[0], utc_days)
    count = int(days.max()) + 1 if len(days) else 0
    daily = np.zeros((count, n))
    np.add.at(daily, days, np.column_stack([run["Cashflow"].to_numpy(float) for run in runs]))
    per_day = np.ones((count, n))
    for k in range(window, count):
        positive = np.clip(daily[k - window:k].sum(axis=0), 0, None)
        share = positive / positive.sum() * n if positive.sum() > 0 else np.ones(n)
        weights = (1 - tilt) + tilt * share
        capped = np.zeros(n, bool)
        for _ in range(n):
            over = (weights > cap + 1e-12) & ~capped
            if not over.any():
                break
            capped |= over
            weights[capped] = cap
            free, target = weights[~capped].sum(), n - cap * capped.sum()
            if free > 0:
                weights[~capped] *= max(target, 0) / free
        per_day[k] = weights
    return per_day[days]


def simulate_book(runs: list[pd.DataFrame], weights: list[float] | np.ndarray, cost_rate: float,
                  cap_mwh: float = 0.0, daily_loss_limit: float = 0.0) -> pd.DataFrame:
    """``weights`` is one weight per run, or an intervals x runs array (see ``rolling_weights``)."""
    for run in runs:
        missing = [column for column in REQUIRED if column not in run.columns]
        if missing:
            raise KeyError(f"Portfolio runs need Prop proxy columns: {missing}")
    runs = align_runs(runs)
    reference = runs[0]
    n = len(reference)
    weights = np.asarray(weights, float)
    exposure = np.column_stack([run["Position_MWh"].to_numpy(float) for run in runs]) * weights
    standalone = (np.column_stack([run["Cashflow"].to_numpy(float) for run in runs]) * weights).sum(axis=1)
    price = reference["Actual_Price"].to_numpy(float)
    spread = reference["Imbalance_Spread_DKK"].to_numpy(float)
    day_end = reference["Is_Day_End"].to_numpy(bool)

    net_path, cash_path, cost_path = np.zeros(n), np.zeros(n), np.zeros(n)
    capped = risk_off = 0
    previous, day_pnl = 0.0, 0.0
    for t in range(n):
        net = float(exposure[t].sum())
        if cap_mwh > 0 and abs(net) > cap_mwh:
            net, capped = float(np.sign(net) * cap_mwh), capped + 1
        if daily_loss_limit > 0 and day_pnl <= -daily_loss_limit and net != 0:
            net, risk_off = 0.0, risk_off + 1
        move = (spread[t] if net != 0 else 0.0) if day_end[t] else price[t + 1] - price[t]
        cost = (abs(net - previous) + (abs(net) if day_end[t] else 0.0)) * cost_rate
        cash = net * move - cost
        net_path[t], cash_path[t], cost_path[t] = net, cash, cost
        day_pnl += cash
        previous = 0.0 if day_end[t] else net
        if day_end[t]:
            day_pnl = 0.0
    book = reference[["HourUTC", "Actual_Price", "Imbalance_Spread_DKK", "Is_Day_End"]].copy()
    book["Position_MWh"] = net_path
    book["Position"] = np.sign(net_path).astype(int)
    book["Cashflow"] = cash_path
    book["Transaction_Cost"] = cost_path
    book["Standalone_Cashflow"] = standalone
    book.attrs.update(capped=capped, risk_off=risk_off)
    return book


def simulate_battery_book(runs: list[pd.DataFrame], weights: list[float] | np.ndarray, battery: BatteryConfig,
                          cap_mw: float = 0.0, daily_loss_limit: float = 0.0) -> pd.DataFrame:
    """One shared battery for the whole book.

    Each interval the runs' weighted dispatch requests are netted (charging requests are negative MW,
    discharging positive), and the battery executes the net within its power and state-of-charge limits, with
    an optional extra power cap. A daily loss limit idles the battery for the rest of the day while keeping its
    state of charge. Fees are paid on the energy actually moved, so opposite requests that cancel save fees.
    """
    for run in runs:
        if "Dispatch_MW" not in run.columns:
            raise KeyError("Battery portfolio runs need Dispatch_MW.")
    runs = align_runs(runs)
    reference = runs[0]
    weights = np.asarray(weights, float)
    request = (np.column_stack([run["Dispatch_MW"].to_numpy(float) for run in runs]) * weights).sum(axis=1)
    power = np.abs(request)
    if cap_mw > 0:
        power = np.minimum(power, cap_mw)
    frame = reference[["HourUTC", "Actual_Price"]].copy()
    frame["Book_Request"] = np.where(request < 0, "charge", np.where(request > 0, "discharge", "hold"))
    frame["Requested_Power_MW"] = power
    config = replace(battery, max_daily_loss=daily_loss_limit or None)
    book, _ = _simulate_battery_dispatch(frame, config, "HourUTC", "Actual_Price", "Book_Request")
    book["Standalone_Cashflow"] = (np.column_stack([run["Cashflow"].to_numpy(float) for run in runs]) * weights).sum(axis=1)
    book.attrs.update(capped=int((cap_mw > 0) * (np.abs(request) > cap_mw).sum()),
                      risk_off=int((book["Action"] == "risk-off").sum()))
    return book


def book_period_summary(book: pd.DataFrame) -> dict[str, float | int | str | None]:
    days = pd.to_datetime(book["HourUTC"], utc=True).dt.tz_convert("Europe/Copenhagen").dt.date.nunique()
    cumulative = book["Cashflow"].cumsum()
    active = book["Position_MWh"] != 0 if "Position_MWh" in book else book["Dispatch_MW"] != 0
    total = float(book["Cashflow"].sum())
    return {
        "start": pd.Timestamp(book["HourUTC"].min()).isoformat(), "end": pd.Timestamp(book["HourUTC"].max()).isoformat(),
        "days": int(days), "rows": len(book), "total_cashflow": total, "pnl_per_day": total / max(days, 1),
        "standalone_cashflow": float(book["Standalone_Cashflow"].sum()),
        "max_drawdown": float((cumulative.cummax().clip(lower=0) - cumulative).max()) if len(book) else 0.0,
        "win_rate": float((book.loc[active, "Cashflow"] > 0).mean()) if active.any() else None,
        "capped_intervals": int(book.attrs.get("capped", 0)), "risk_off_intervals": int(book.attrs.get("risk_off", 0)),
    }
