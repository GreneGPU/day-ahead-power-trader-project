"""Could a strategy's P&L be chance? Daily t-test plus two permutation tests on the Prop proxy.

The honest unit is the day: 15-minute P&L is autocorrelated and clustered within days, so a
t-test on intervals overstates significance. The permutation tests keep the strategy's exact
exposure profile and only break its link to the prices:

* day shuffle  - each day's position schedule is applied to a different day's prices. Null: the
  day-specific forecast adds nothing beyond the typical daily price shape.
* timing shift - each day's schedule is circularly shifted within the day. Null: timing within
  the day does not matter.

Permutation P&L is gross (before costs), which are small and identical in both cases.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

INTERVALS_PER_DAY = 96
MARKET_TIMEZONE = "Europe/Copenhagen"
_Z_ALPHA_5PCT_ONE_SIDED = 1.6449
_Z_POWER_80PCT = 0.8416


def t_survival(t: float, df: int) -> float:
    """One-sided P(T >= t) for Student's t, by numerical integration (no SciPy dependency)."""
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    if t < 0:
        return 1.0 - t_survival(-t, df)
    grid = np.linspace(t, t + 80.0, 160_001)
    log_norm = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    density = np.exp(log_norm) * (1.0 + grid**2 / df) ** (-(df + 1) / 2)
    step = grid[1] - grid[0]
    return float(min(1.0, max(0.0, step * (density.sum() - (density[0] + density[-1]) / 2))))


def _derangements(rng: np.random.Generator, days: int, count: int) -> np.ndarray:
    """Random permutations in which no day keeps its own schedule (about 37% of draws qualify)."""
    accepted: list[np.ndarray] = []
    while sum(len(batch) for batch in accepted) < count:
        draws = np.argsort(rng.random((count * 3, days)), axis=1)
        accepted.append(draws[(draws != np.arange(days)).all(axis=1)])
    return np.concatenate(accepted)[:count]


def significance_tests(intervals: pd.DataFrame, permutations: int = 2000, seed: int = 7) -> dict | None:
    """p-values using complete Copenhagen delivery days only.

    The daily t-test needs only cashflows, so it also covers battery runs. The permutation tests reuse a run's
    positions on other prices, which needs Prop proxy positions; for battery runs (whose dispatch depends on
    the state of charge) they are left as ``None``.
    """
    if not {"HourUTC", "Cashflow"}.issubset(intervals.columns):
        return None
    has_positions = {"Position_MWh", "Actual_Price", "Imbalance_Spread_DKK"}.issubset(intervals.columns)
    frame = intervals.sort_values("HourUTC").reset_index(drop=True)
    day = pd.to_datetime(frame["HourUTC"], utc=True).dt.tz_convert(MARKET_TIMEZONE).dt.date
    complete = day.groupby(day).transform("size") == INTERVALS_PER_DAY
    frame, day = frame[complete].reset_index(drop=True), day[complete].reset_index(drop=True)
    days = day.nunique()
    if days < 3:
        return {"days": int(days), "error": "At least 3 complete days are needed for significance tests."}

    daily = frame.groupby(day)["Cashflow"].sum()
    mean, sd = float(daily.mean()), float(daily.std(ddof=1))
    t_stat = mean / (sd / math.sqrt(days)) if sd > 0 else (math.inf if mean > 0 else -math.inf if mean < 0 else 0.0)
    daily_sharpe = mean / sd if sd > 0 else None
    days_needed = (math.ceil(((_Z_ALPHA_5PCT_ONE_SIDED + _Z_POWER_80PCT) / daily_sharpe) ** 2)
                   if daily_sharpe and daily_sharpe > 0 else None)

    daily_result = {
        "days": int(days), "mean_daily": mean, "sd_daily": sd, "t_stat": float(t_stat),
        "p_daily": t_survival(float(t_stat), days - 1), "daily_sharpe": daily_sharpe,
        "days_needed_80pct_power": days_needed,
    }
    if not has_positions:
        return {**daily_result, "permutations": 0, "actual_gross": None, "day_shuffle": None, "timing_shift": None}

    exposure = frame["Position_MWh"].to_numpy(float).reshape(days, INTERVALS_PER_DAY)
    price = frame["Actual_Price"].to_numpy(float).reshape(days, INTERVALS_PER_DAY)
    move = np.empty_like(price)
    move[:, :-1] = price[:, 1:] - price[:, :-1]          # next day-ahead move within the day
    move[:, -1] = frame["Imbalance_Spread_DKK"].to_numpy(float).reshape(days, INTERVALS_PER_DAY)[:, -1]
    actual_gross = float((exposure * move).sum())

    rng = np.random.default_rng(seed)
    orders = _derangements(rng, days, permutations)
    offsets = rng.integers(1, INTERVALS_PER_DAY, size=(permutations, days))
    # Batches keep memory bounded (batch x days x 96 values) when the test period is months long.
    batch = max(1, 200_000 // (days * INTERVALS_PER_DAY))
    shuffled, shifted = np.empty(permutations), np.empty(permutations)
    for lo in range(0, permutations, batch):
        hi = min(permutations, lo + batch)
        shuffled[lo:hi] = np.einsum("pdi,di->p", exposure[orders[lo:hi]], move)
        columns = (np.arange(INTERVALS_PER_DAY)[None, None, :] - offsets[lo:hi, :, None]) % INTERVALS_PER_DAY
        rolled = np.take_along_axis(np.broadcast_to(exposure, (hi - lo, days, INTERVALS_PER_DAY)), columns, axis=2)
        shifted[lo:hi] = np.einsum("pdi,di->p", rolled, move)

    def permutation_p(null: np.ndarray) -> float:
        return float((1 + (null >= actual_gross).sum()) / (1 + len(null)))

    return {
        **daily_result,
        "permutations": permutations,
        "actual_gross": actual_gross,
        "day_shuffle": {"null_mean": float(shuffled.mean()), "p": permutation_p(shuffled)},
        "timing_shift": {"null_mean": float(shifted.mean()), "p": permutation_p(shifted)},
    }
