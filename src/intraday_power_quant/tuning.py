"""Walk-forward parameter tuning with plateau (neighbour-smoothed) selection.

One parameter sweep over the whole window records every setting's daily P&L. Each test fold then picks,
per strategy, the setting with the best *smoothed* P&L over the tuning days before the fold (its own P&L
averaged with its immediate grid neighbours), and trades it unchanged on the fold. Settings run
continuously over the window, so no fold is ever used to choose its own parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MARKET_TIMEZONE = "Europe/Copenhagen"


def local_dates(times: pd.Series) -> pd.Series:
    return pd.to_datetime(times, utc=True).dt.tz_convert(MARKET_TIMEZONE).dt.date


@dataclass
class DailyPnlCapture:
    """Collects daily P&L (and P&L before trading costs) for every swept setting."""

    daily: dict[str, list[tuple[str, pd.Series]]] = field(default_factory=dict)
    gross_daily: dict[str, list[tuple[str, pd.Series]]] = field(default_factory=dict)

    def __call__(self, strategy: str, settings: dict[str, object], simulation: pd.DataFrame) -> None:
        dates = local_dates(simulation["HourUTC"])
        cash = simulation["Cashflow"].astype(float)
        costs = simulation["Transaction_Cost"].astype(float) if "Transaction_Cost" in simulation else 0.0
        settings_json = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        self.daily.setdefault(strategy, []).append((settings_json, cash.groupby(dates).sum()))
        self.gross_daily.setdefault(strategy, []).append((settings_json, (cash + costs).groupby(dates).sum()))


def grid_neighbours(settings: list[dict[str, object]]) -> list[list[int]]:
    """Indices of settings one grid step away in exactly one numeric parameter, all else equal."""
    numeric = {key for item in settings for key, value in item.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
    levels = {key: sorted({item[key] for item in settings if key in item and item[key] is not None}) for key in numeric}

    def position(item: dict[str, object], key: str) -> int | None:
        value = item.get(key)
        return levels[key].index(value) if value in levels.get(key, []) else None

    neighbours: list[list[int]] = []
    for i, a in enumerate(settings):
        found = []
        for j, b in enumerate(settings):
            if i == j or a.keys() != b.keys():
                continue
            differing = [key for key in a if a[key] != b[key]]
            if len(differing) != 1 or differing[0] not in numeric:
                continue
            pa, pb = position(a, differing[0]), position(b, differing[0])
            if pa is not None and pb is not None and abs(pa - pb) == 1:
                found.append(j)
        neighbours.append(found)
    return neighbours


def smoothed_choice(train_pnl: np.ndarray, neighbours: list[list[int]]) -> int:
    """Index maximizing mean P&L of a setting and its grid neighbours; ties go to the higher own P&L."""
    scores = np.array([np.mean([train_pnl[i], *(train_pnl[j] for j in group)]) for i, group in enumerate(neighbours)])
    order = np.lexsort((-train_pnl, -scores))
    return int(order[0])


def walk_forward_folds(dates: list, tune_days: int, test_days: int) -> list[tuple[list, list]]:
    """(tuning dates, test dates) pairs: consecutive test blocks at the end of ``dates``, each tuned on
    the ``tune_days`` dates immediately before it."""
    folds_count = (len(dates) - tune_days) // test_days
    if folds_count < 1:
        raise ValueError(f"Walk-forward tuning needs at least {tune_days + test_days} complete days in the window.")
    start = len(dates) - folds_count * test_days
    return [(dates[s - tune_days:s], dates[s:s + test_days]) for s in range(start, len(dates), test_days)]


def tune_strategy(entries: list[tuple[str, pd.Series]], folds: list[tuple[list, list]]) -> list[dict[str, object]]:
    """Per fold: chosen settings JSON, its tuning P&L and its out-of-sample P&L."""
    settings = [json.loads(settings_json) for settings_json, _ in entries]
    neighbours = grid_neighbours(settings)
    results = []
    for tune_dates, test_dates in folds:
        train = np.array([float(daily.reindex(tune_dates, fill_value=0.0).sum()) for _, daily in entries])
        choice = smoothed_choice(train, neighbours)
        daily = entries[choice][1]
        results.append({
            "settings_json": entries[choice][0],
            "train_pnl": float(train[choice]),
            "train_raw_best_pnl": float(train.max()),
            "test_pnl": float(daily.reindex(test_dates, fill_value=0.0).sum()),
            "tune_dates": tune_dates, "test_dates": test_dates,
            "neighbours": len(neighbours[choice]),
        })
    return results


def best_fixed_setting(entries: list[tuple[str, pd.Series]], dates: list) -> tuple[str, float]:
    """Hindsight: the single setting with the highest P&L over ``dates``."""
    totals = [float(daily.reindex(dates, fill_value=0.0).sum()) for _, daily in entries]
    best = int(np.argmax(totals))
    return entries[best][0], totals[best]
