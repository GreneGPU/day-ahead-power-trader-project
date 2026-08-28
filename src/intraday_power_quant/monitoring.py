from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_monitoring_report(
    frame: pd.DataFrame,
    actual_col: str = "Actual_Price",
    prediction_col: str = "Prediction",
    baseline_col: str = "Hourly_Baseline",
    time_col: str = "HourUTC",
) -> dict[str, Any]:
    required = [actual_col, prediction_col]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing monitoring columns: {missing}")
    valid = frame.dropna(subset=required).copy()
    error = valid[actual_col] - valid[prediction_col]
    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "scored_rows": int(len(valid)),
        "prediction_coverage_pct": float(100 * len(valid) / len(frame)) if len(frame) else 0.0,
        "mae": float(error.abs().mean()) if len(valid) else None,
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(valid) else None,
        "mean_error": float(error.mean()) if len(valid) else None,
    }
    if baseline_col in valid:
        baseline_mae = float((valid[actual_col] - valid[baseline_col]).abs().mean())
        report["baseline_mae"] = baseline_mae
        report["mae_improvement_pct"] = (
            float(100 * (baseline_mae - report["mae"]) / baseline_mae) if baseline_mae else None
        )
    if time_col in valid and len(valid):
        timestamps = pd.to_datetime(valid[time_col], utc=True, errors="coerce")
        report["start"] = timestamps.min().isoformat()
        report["end"] = timestamps.max().isoformat()
    return report


def write_monitoring_report(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
