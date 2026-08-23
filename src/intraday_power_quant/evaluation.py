from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def calculate_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: actual={actual.shape}, predicted={predicted.shape}")
    if actual.size == 0:
        raise ValueError("Cannot calculate metrics for empty arrays.")

    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - np.sum(error**2) / denominator) if denominator != 0 else float("nan")
    epsilon = 1e-6
    smape = float(100 * np.mean(2 * np.abs(error) / (np.abs(actual) + np.abs(predicted) + epsilon)))
    mape = float(100 * np.mean(np.abs(error) / np.maximum(np.abs(actual), epsilon)))

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "R2": r2}


def calculate_15min_coverage(times: Iterable[object]) -> dict[str, object]:
    values = pd.Series(pd.to_datetime(list(times))).dropna().sort_values().reset_index(drop=True)
    if values.empty:
        return {
            "Test_Expected_Rows": 0,
            "Test_Observed_Rows": 0,
            "Test_Coverage_Pct": 0.0,
            "Test_Continuous_15min": False,
        }

    expected_index = pd.date_range(start=values.min(), end=values.max(), freq="15min")
    expected_rows = len(expected_index)
    observed_rows = len(values.drop_duplicates())
    coverage_pct = observed_rows / expected_rows * 100 if expected_rows else 0.0
    return {
        "Test_Expected_Rows": int(expected_rows),
        "Test_Observed_Rows": int(observed_rows),
        "Test_Coverage_Pct": float(coverage_pct),
        "Test_Continuous_15min": bool(observed_rows == expected_rows),
    }


def build_metrics_table(
    results: pd.DataFrame,
    actual_col: str,
    metric_specs: list[tuple[str, str, str, int, bool | None]],
    metadata: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_type, prediction_column, number_features, uses_baseline in metric_specs:
        if prediction_column not in results.columns:
            continue
        rows.append(
            {
                "Model": model_name,
                "Model_Type": model_type,
                "Prediction_Column": prediction_column,
                "Number_Features": number_features,
                "Uses_Hourly_Baseline_As_Feature": uses_baseline,
                **metadata,
                **calculate_metrics(results[actual_col], results[prediction_column]),
            }
        )
    return pd.DataFrame(rows)


def rank_models(metrics: pd.DataFrame, primary_metric: str = "MAE") -> pd.DataFrame:
    if primary_metric not in metrics.columns:
        raise KeyError(f"Missing metric column: {primary_metric}")
    ranked = metrics.copy()
    ranked["Rank"] = ranked[primary_metric].rank(method="dense", ascending=True).astype(int)
    return ranked.sort_values(["Rank", primary_metric]).reset_index(drop=True)


def error_by_time_bucket(
    results: pd.DataFrame,
    actual_col: str,
    prediction_col: str,
    time_col: str = "HourUTC",
    bucket: str = "hour",
) -> pd.DataFrame:
    df = results.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["Absolute_Error"] = (df[actual_col] - df[prediction_col]).abs()
    if bucket == "hour":
        group_key = df[time_col].dt.hour
        label = "Hour"
    elif bucket == "date":
        group_key = df[time_col].dt.date
        label = "Date"
    elif bucket == "month":
        group_key = df[time_col].dt.to_period("M").astype(str)
        label = "Month"
    else:
        raise ValueError("bucket must be one of: hour, date, month")

    return (
        df.groupby(group_key, observed=True)["Absolute_Error"]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
        .rename(columns={time_col: label, "index": label, "mean": "MAE"})
    )

