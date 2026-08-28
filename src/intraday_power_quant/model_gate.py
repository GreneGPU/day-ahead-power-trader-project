from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelGateSettings:
    min_coverage_pct: float = 99.0
    max_baseline_mae_regression_pct: float = 0.0
    require_continuous_15min: bool = True
    require_no_leakage_warnings: bool = True


def evaluate_model_gate(
    metrics: pd.DataFrame,
    candidate_prediction_column: str,
    coverage: dict[str, Any],
    leakage_warnings: list[str],
    settings: ModelGateSettings | None = None,
    current_champion_mae: float | None = None,
) -> dict[str, Any]:
    cfg = settings or ModelGateSettings()
    reasons: list[str] = []

    candidate_rows = metrics.loc[metrics["Prediction_Column"] == candidate_prediction_column]
    baseline_rows = metrics.loc[metrics["Prediction_Column"] == "Hourly_Baseline"]
    if candidate_rows.empty:
        reasons.append(f"Candidate metric row not found: {candidate_prediction_column}")
        candidate_mae = math.nan
    else:
        candidate_mae = float(candidate_rows.iloc[0]["MAE"])
        if not math.isfinite(candidate_mae):
            reasons.append("Candidate MAE is not finite")

    baseline_mae = math.nan if baseline_rows.empty else float(baseline_rows.iloc[0]["MAE"])
    if not math.isfinite(baseline_mae):
        reasons.append("Hourly baseline MAE is unavailable")
    elif math.isfinite(candidate_mae):
        allowed_mae = baseline_mae * (1 + cfg.max_baseline_mae_regression_pct / 100)
        if candidate_mae > allowed_mae:
            reasons.append(
                f"Candidate MAE {candidate_mae:.6f} exceeds allowed baseline MAE {allowed_mae:.6f}"
            )

    coverage_pct = float(coverage.get("Test_Coverage_Pct", 0.0))
    if coverage_pct < cfg.min_coverage_pct:
        reasons.append(
            f"Coverage {coverage_pct:.3f}% is below {cfg.min_coverage_pct:.3f}%"
        )
    continuous = bool(coverage.get("Test_Continuous_15min", False))
    if cfg.require_continuous_15min and not continuous:
        reasons.append("Test timestamps are not a continuous 15-minute sequence")
    if cfg.require_no_leakage_warnings and leakage_warnings:
        reasons.append(f"Leakage warnings present: {len(leakage_warnings)}")
    if current_champion_mae is not None and math.isfinite(candidate_mae):
        if candidate_mae > current_champion_mae:
            reasons.append(
                f"Candidate MAE {candidate_mae:.6f} exceeds current champion MAE "
                f"{current_champion_mae:.6f}"
            )

    return {
        "passed": not reasons,
        "candidate_prediction_column": candidate_prediction_column,
        "candidate_mae": candidate_mae,
        "hourly_baseline_mae": baseline_mae,
        "current_champion_mae": current_champion_mae,
        "coverage_pct": coverage_pct,
        "continuous_15min": continuous,
        "leakage_warning_count": len(leakage_warnings),
        "settings": asdict(cfg),
        "reasons": reasons,
    }


def write_model_gate(result: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return output_path
