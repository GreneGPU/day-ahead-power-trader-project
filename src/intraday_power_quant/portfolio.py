"""Net several Prop proxy runs into one book, with the same rules as the Strategy Lab's Portfolio panel.

Each interval's net exposure is the weighted sum of the runs' MWh. The book is re-simulated with the Prop
proxy rules: the next day-ahead move, the imbalance spread when a position is open at the day's last
interval, and a trading cost on every change in MWh (plus the close at day end). An optional cap limits net
exposure and an optional daily loss limit flattens the book for the rest of the day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["HourUTC", "Position_MWh", "Cashflow", "Actual_Price", "Imbalance_Spread_DKK", "Is_Day_End"]


def align_runs(runs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Restrict every run to the timestamps all runs share, in time order."""
    common = set(runs[0]["HourUTC"])
    for run in runs[1:]:
        common &= set(run["HourUTC"])
    return [run[run["HourUTC"].isin(common)].sort_values("HourUTC").reset_index(drop=True) for run in runs]


def simulate_book(runs: list[pd.DataFrame], weights: list[float], cost_rate: float,
                  cap_mwh: float = 0.0, daily_loss_limit: float = 0.0) -> pd.DataFrame:
    for run in runs:
        missing = [column for column in REQUIRED if column not in run.columns]
        if missing:
            raise KeyError(f"Portfolio runs need Prop proxy columns: {missing}")
    runs = align_runs(runs)
    reference = runs[0]
    n = len(reference)
    exposure = np.column_stack([run["Position_MWh"].to_numpy(float) for run in runs]) * np.asarray(weights, float)
    standalone = sum(w * run["Cashflow"].to_numpy(float) for w, run in zip(weights, runs))
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


def book_period_summary(book: pd.DataFrame) -> dict[str, float | int | str | None]:
    days = pd.to_datetime(book["HourUTC"], utc=True).dt.tz_convert("Europe/Copenhagen").dt.date.nunique()
    cumulative = book["Cashflow"].cumsum()
    active = book["Position_MWh"] != 0
    total = float(book["Cashflow"].sum())
    return {
        "start": pd.Timestamp(book["HourUTC"].min()).isoformat(), "end": pd.Timestamp(book["HourUTC"].max()).isoformat(),
        "days": int(days), "rows": len(book), "total_cashflow": total, "pnl_per_day": total / max(days, 1),
        "standalone_cashflow": float(book["Standalone_Cashflow"].sum()),
        "max_drawdown": float((cumulative.cummax().clip(lower=0) - cumulative).max()) if len(book) else 0.0,
        "win_rate": float((book.loc[active, "Cashflow"] > 0).mean()) if active.any() else None,
        "capped_intervals": int(book.attrs.get("capped", 0)), "risk_off_intervals": int(book.attrs.get("risk_off", 0)),
    }
