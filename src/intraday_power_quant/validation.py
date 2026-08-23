from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def chronological_split_mask(
    df: pd.DataFrame,
    time_col: str,
    split_ratio: float,
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must be between 0 and 1.")
    ordered = df.sort_values(time_col).reset_index(drop=True)
    split_index = int(len(ordered) * split_ratio)
    split_index = min(max(split_index, 1), len(ordered) - 1)
    split_time = ordered.loc[split_index, time_col]
    train_mask = df[time_col] < split_time
    test_mask = df[time_col] >= split_time
    return train_mask, test_mask, split_time


def last_n_days_split_mask(
    df: pd.DataFrame,
    time_col: str,
    test_period_days: int | float,
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    if test_period_days <= 0:
        raise ValueError("test_period_days must be positive.")
    times = pd.to_datetime(df[time_col])
    last_time = times.max()
    split_time = last_time - pd.Timedelta(days=float(test_period_days))
    train_mask = times < split_time
    test_mask = times >= split_time
    if not train_mask.any():
        raise ValueError("The requested test period leaves no training rows.")
    if not test_mask.any():
        raise ValueError("The requested test period leaves no test rows.")
    return train_mask, test_mask, split_time


def make_train_test_split_mask(
    df: pd.DataFrame,
    time_col: str,
    method: str,
    split_ratio: float,
    test_period_days: int | float,
) -> tuple[pd.Series, pd.Series, pd.Timestamp, str]:
    normalized_method = method.lower().replace("-", "_")
    if normalized_method == "ratio":
        train_mask, test_mask, split_time = chronological_split_mask(df, time_col, split_ratio)
        label = f"ratio:{split_ratio:.4f}"
    elif normalized_method in {"last_n_days", "last_days", "last_month"}:
        train_mask, test_mask, split_time = last_n_days_split_mask(df, time_col, test_period_days)
        label = f"last_n_days:{test_period_days:g}"
    else:
        raise ValueError("test_split_method must be 'ratio' or 'last_n_days'.")
    return train_mask, test_mask, split_time, label


def make_walk_forward_windows(
    times: pd.Series,
    train_days: int = 90,
    test_days: int = 14,
    step_days: int = 14,
) -> list[WalkForwardWindow]:
    ordered = pd.Series(pd.to_datetime(times)).dropna().sort_values()
    if ordered.empty:
        return []
    windows: list[WalkForwardWindow] = []
    cursor = ordered.min() + pd.Timedelta(days=train_days)
    last_time = ordered.max()
    while cursor + pd.Timedelta(days=test_days) <= last_time:
        train_start = cursor - pd.Timedelta(days=train_days)
        train_end = cursor - pd.Timedelta(minutes=15)
        test_start = cursor
        test_end = cursor + pd.Timedelta(days=test_days) - pd.Timedelta(minutes=15)
        windows.append(WalkForwardWindow(train_start, train_end, test_start, test_end))
        cursor += pd.Timedelta(days=step_days)
    return windows


def add_season_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    output = df.copy()
    month = pd.to_datetime(output[time_col]).dt.month
    output["Season"] = month.map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )
    return output


def leakage_warnings(
    feature_cols: list[str],
    target: str,
    time_col: str,
    allowed_target_lag_pattern: str = r"(lag|roll|mean|std|known)",
) -> list[str]:
    warnings: list[str] = []
    lower_target = target.lower()
    allowed_pattern = re.compile(allowed_target_lag_pattern, re.IGNORECASE)

    for column in feature_cols:
        lower_col = column.lower()
        if lower_col == lower_target:
            warnings.append(f"Feature equals target: {column}")
        if lower_col == time_col.lower():
            warnings.append(f"Feature equals timestamp: {column}")
        if lower_target in lower_col and not allowed_pattern.search(lower_col):
            warnings.append(f"Potential target-derived feature without lag marker: {column}")
        if any(token in lower_col for token in ["future", "next", "lead", "t_plus", "t+1"]):
            warnings.append(f"Potential future-looking feature: {column}")

    return warnings
