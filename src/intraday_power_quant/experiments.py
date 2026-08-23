from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .transfer import run_transfer_learning


def run_baseline_feature_ablation(config: ProjectConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for use_baseline in [False, True]:
        output_dir = Path(config.output_dir) / (
            "with_hourly_baseline_feature" if use_baseline else "without_hourly_baseline_feature"
        )
        run_config = replace(
            config,
            output_dir=output_dir,
            use_hourly_baseline_as_residual_feature=use_baseline,
        )
        summary = run_transfer_learning(run_config)
        metrics_path = Path(summary["outputs"]["metrics"].get("csv"))
        metrics = pd.read_csv(metrics_path)
        best = metrics.sort_values("MAE").iloc[0]
        rows.append(
            {
                "Uses_Hourly_Baseline_As_Feature": use_baseline,
                "Best_Model": best["Model"],
                "Best_MAE": best["MAE"],
                "Best_RMSE": best["RMSE"],
                "Best_R2": best["R2"],
                "Output_Dir": str(output_dir),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison_path = Path(config.output_dir) / "baseline_feature_ablation.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    return comparison

