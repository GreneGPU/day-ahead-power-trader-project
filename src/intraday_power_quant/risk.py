from __future__ import annotations

import math

import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf") if numerator > 0 else float("nan")
    return float(numerator / denominator)


def _safe_scaled_ratio(numerator: float, denominator: float, scale: float = 1.0) -> float:
    if denominator <= 0 or pd.isna(denominator):
        return float("nan")
    return float(numerator / denominator * scale)


def max_drawdown_duration(cumulative_cashflow: pd.Series) -> int:
    high_water = cumulative_cashflow.cummax()
    below_peak = cumulative_cashflow < high_water
    longest = 0
    current = 0
    for is_below in below_peak:
        if bool(is_below):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def summarize_cashflow_risk(
    sim: pd.DataFrame,
    time_col: str = "HourUTC",
    market_timezone: str = "Europe/Copenhagen",
) -> dict[str, float]:
    if sim.empty or "Cashflow" not in sim.columns:
        return {
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "calmar_ratio": float("nan"),
            "daily_sharpe": float("nan"),
            "daily_sortino": float("nan"),
            "average_daily_cashflow": float("nan"),
            "daily_cashflow_std": float("nan"),
            "cashflow_std": float("nan"),
            "downside_std": float("nan"),
            "worst_interval": float("nan"),
            "best_interval": float("nan"),
            "worst_day": float("nan"),
            "best_day": float("nan"),
            "max_drawdown_duration_intervals": 0,
            "risk_adjusted_score": float("nan"),
        }

    cashflow = sim["Cashflow"].astype(float)
    cumulative = cashflow.cumsum()
    total = float(cashflow.sum())
    max_drawdown = float((cumulative.cummax() - cumulative).max())
    active_cashflow = cashflow[cashflow != 0]
    gains = float(active_cashflow[active_cashflow > 0].sum())
    losses = float(active_cashflow[active_cashflow < 0].sum())
    win_rate = _safe_ratio(float((active_cashflow > 0).sum()), float(len(active_cashflow)))
    profit_factor = _safe_ratio(gains, abs(losses))
    downside = cashflow[cashflow < 0]

    if time_col in sim.columns:
        dates = pd.to_datetime(sim[time_col], utc=True).dt.tz_convert(market_timezone).dt.date
        daily_cashflow = cashflow.groupby(dates).sum()
    else:
        daily_cashflow = pd.Series([total])
    daily_mean = float(daily_cashflow.mean())
    daily_std = float(daily_cashflow.std(ddof=1)) if len(daily_cashflow) > 1 else float("nan")
    daily_downside = daily_cashflow[daily_cashflow < 0]
    daily_downside_std = (
        float(daily_downside.std(ddof=1)) if len(daily_downside) > 1 else float("nan")
    )
    annual_scale = math.sqrt(365)

    return {
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "calmar_ratio": _safe_ratio(total, max_drawdown),
        "daily_sharpe": _safe_scaled_ratio(daily_mean, daily_std, annual_scale),
        "daily_sortino": _safe_scaled_ratio(daily_mean, daily_downside_std, annual_scale),
        "average_daily_cashflow": daily_mean,
        "daily_cashflow_std": daily_std,
        "cashflow_std": float(cashflow.std(ddof=0)),
        "downside_std": float(downside.std(ddof=0)) if not downside.empty else 0.0,
        "worst_interval": float(cashflow.min()),
        "best_interval": float(cashflow.max()),
        "worst_day": float(daily_cashflow.min()),
        "best_day": float(daily_cashflow.max()),
        "max_drawdown_duration_intervals": max_drawdown_duration(cumulative),
        "risk_adjusted_score": float(total - max_drawdown),
    }


def clean_metric(value: float) -> float:
    if isinstance(value, (int, float)) and math.isinf(float(value)):
        return float("nan")
    return value
