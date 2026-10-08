"""Walk-forward (expanding window) out-of-sample forecasts for the whole 15-minute period.

The single-split run only produces out-of-sample predictions for its final test block. Here the 15-minute
residual and direct ensembles are retrained before every block on all 15-minute data strictly earlier than
that block, so every predicted interval is out of sample. The hourly source model is trained once on hourly
data before the transfer start, which is already earlier than every 15-minute interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .evaluation import build_metrics_table, calculate_15min_coverage
from .models import predict_stacked_ensemble, train_stacked_ensemble
from .transfer import _write_frame, prepare_transfer_inputs


def walk_forward_blocks(times: pd.Series, min_train_days: float, block_days: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Consecutive [start, end) test blocks; the first starts after ``min_train_days`` of history."""
    first = pd.Timestamp(times.min()).normalize() + pd.Timedelta(days=min_train_days)
    last = pd.Timestamp(times.max())
    blocks, cursor = [], first
    while cursor <= last:
        blocks.append((cursor, cursor + pd.Timedelta(days=block_days)))
        cursor += pd.Timedelta(days=block_days)
    return blocks


def run_walk_forward(config: ProjectConfig, min_train_days: float = 35, block_days: float = 14) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = prepare_transfer_inputs(config)
    frame: pd.DataFrame = inputs["df_15_transfer"]
    time_col, target = config.time_col, config.target
    feature_cols_15: list[str] = inputs["feature_cols_15"]
    residual_cols = feature_cols_15 + (["Hourly_Baseline"] if config.use_hourly_baseline_as_residual_feature else [])

    pieces, folds = [], []
    blocks = walk_forward_blocks(frame[time_col], min_train_days, block_days)
    for number, (start, end) in enumerate(blocks, start=1):
        train, test = frame[frame[time_col] < start], frame[(frame[time_col] >= start) & (frame[time_col] < end)]
        if test.empty:
            continue
        print(f"\nWalk-forward block {number}/{len(blocks)}: train {len(train)} rows to {train[time_col].max()}, "
              f"test {len(test)} rows {test[time_col].min()} -> {test[time_col].max()}")
        residual_model = train_stacked_ensemble(train[residual_cols].reset_index(drop=True),
                                                train["residual_15min"].reset_index(drop=True), n_splits=config.n_splits)
        direct_model = train_stacked_ensemble(train[feature_cols_15].reset_index(drop=True),
                                              train[target].reset_index(drop=True), n_splits=config.n_splits)
        residual = predict_stacked_ensemble(residual_model, test[residual_cols].reset_index(drop=True))
        direct = predict_stacked_ensemble(direct_model, test[feature_cols_15].reset_index(drop=True))
        baseline = test["Hourly_Baseline"].to_numpy()
        piece = pd.DataFrame({
            time_col: test[time_col].to_numpy(), "Actual_Price": test[target].to_numpy(), "Hourly_Baseline": baseline,
            "Direct_15min_XGB": direct["xgb"], "Direct_15min_LGBM": direct["lgb"], "Direct_15min_CAT": direct["cat"],
            "Direct_15min_Average": direct["average"], "Direct_15min_Stacked": direct["stacked"],
            "TL_Residual_XGB": baseline + residual["xgb"], "TL_Residual_LGBM": baseline + residual["lgb"],
            "TL_Residual_CAT": baseline + residual["cat"], "TL_Residual_Average": baseline + residual["average"],
            "TL_Residual_Stacked": baseline + residual["stacked"],
            "Walk_Forward_Block": number, "Train_End": train[time_col].max(), "Train_Rows": len(train),
        })
        pieces.append(piece)
        folds.append({"Block": number, "Train_Start": train[time_col].min(), "Train_End": train[time_col].max(),
                      "Train_Rows": len(train), "Test_Start": test[time_col].min(), "Test_End": test[time_col].max(),
                      "Test_Rows": len(test)})

    results = pd.concat(pieces, ignore_index=True)
    champion = config.champion_prediction_column if config.champion_prediction_column in results else "TL_Residual_Stacked"
    results["Prediction"] = results[champion]
    results["Direct_15min_Prediction"] = results["Direct_15min_Stacked"]
    results["Error"] = results["Actual_Price"] - results["Prediction"]
    results["Absolute_Error"] = results["Error"].abs()

    fold_table = pd.DataFrame(folds)
    for column, label in [("Hourly_Baseline", "Baseline"), ("Prediction", "Transfer"), ("Direct_15min_Prediction", "Direct")]:
        fold_table[f"{label}_MAE"] = results.groupby("Walk_Forward_Block").apply(
            lambda block, c=column: (block["Actual_Price"] - block[c]).abs().mean(), include_groups=False).to_numpy()

    coverage = calculate_15min_coverage(results[time_col])
    metadata = {"Test_Start": results[time_col].min(), "Test_End": results[time_col].max(), "Test_Rows": len(results),
                "Test_Split_Method": f"walk_forward:{block_days:g}d blocks, expanding window, first after {min_train_days:g}d",
                **coverage}
    n_direct, n_residual = len(feature_cols_15), len(residual_cols)
    metric_specs = [
        ("Hourly baseline only", "Hourly source baseline", "Hourly_Baseline", 1, None),
        ("Direct 15-min ensemble", "15-min scratch", "Direct_15min_Prediction", n_direct, None),
        ("Transfer residual ensemble", "Transfer learning", "Prediction", n_residual, config.use_hourly_baseline_as_residual_feature),
        ("TL residual pure stacking", "Transfer learning", "TL_Residual_Stacked", n_residual, config.use_hourly_baseline_as_residual_feature),
    ]
    metrics = build_metrics_table(results, "Actual_Price", metric_specs, metadata)

    outputs = {
        "forecasts": _write_frame(results, output_dir / "walk_forward_forecasts"),
        "metrics": _write_frame(metrics, output_dir / "walk_forward_metrics"),
        "folds": _write_frame(fold_table, output_dir / "walk_forward_blocks"),
    }
    summary = {
        "method": metadata["Test_Split_Method"], "champion_prediction_column": champion,
        "blocks": len(folds), "rows": len(results), "test_start": str(metadata["Test_Start"]),
        "test_end": str(metadata["Test_End"]), "coverage": coverage,
        "hourly_source_train_end": str(inputs["df_hourly_source_train"][time_col].max()),
        "source_paths": {key: str(value) for key, value in inputs["source_paths"].items()},
        "leakage_warnings": inputs["warnings"], "outputs": outputs,
    }
    (output_dir / "walk_forward_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nWalk-forward MAE by block:")
    print(fold_table[["Block", "Test_Start", "Train_Rows", "Baseline_MAE", "Transfer_MAE", "Direct_MAE"]].to_string(index=False))
    return summary
