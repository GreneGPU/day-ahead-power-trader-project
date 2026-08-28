from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import (
    check_columns,
    data_quality_summary,
    drop_model_na,
    infer_feature_columns,
    load_market_data,
    map_15min_to_hourly_feature_space,
    prepare_time_series_frame,
    transferable_hourly_features,
)
from .evaluation import build_metrics_table, calculate_15min_coverage
from .mlflow_tracking import MLflowRunSettings, log_training_run
from .model_bundle import ForecastModelBundle
from .model_gate import ModelGateSettings, evaluate_model_gate, write_model_gate
from .models import base_model_names, optional_model_status, predict_stacked_ensemble, train_stacked_ensemble
from .validation import leakage_warnings, make_train_test_split_mask


def _write_frame(df: pd.DataFrame, output_base: Path) -> dict[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_base.with_suffix(".csv")
    xlsx_path = output_base.with_suffix(".xlsx")
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
        return {"csv": str(csv_path), "xlsx": str(xlsx_path)}
    except Exception as exc:
        return {"csv": str(csv_path), "xlsx_error": str(exc)}


def run_transfer_learning(config: ProjectConfig) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_hourly, df_15, source_paths = load_market_data(
        config.data_dir,
        config.hourly_candidates,
        config.min15_candidates,
    )

    df_hourly = prepare_time_series_frame(df_hourly, config.time_col)
    df_15 = prepare_time_series_frame(df_15, config.time_col)

    hourly_feature_cols = infer_feature_columns(df_hourly, config.time_col, config.target)
    feature_cols_15 = infer_feature_columns(df_15, config.time_col, config.target)
    check_columns(df_hourly, hourly_feature_cols, config.time_col, config.target, "hourly")
    check_columns(df_15, feature_cols_15, config.time_col, config.target, "15-min")

    df_hourly_model = drop_model_na(df_hourly, hourly_feature_cols, config.target)
    df_15_model = drop_model_na(df_15, feature_cols_15, config.target)
    transferable_features, excluded_features = transferable_hourly_features(hourly_feature_cols, df_15_model)

    warnings = leakage_warnings(feature_cols_15, config.target, config.time_col)
    transfer_start = pd.Timestamp(config.transfer_start)
    df_hourly_source_train = df_hourly_model[df_hourly_model[config.time_col] < transfer_start].copy()
    if len(df_hourly_source_train) < 1000:
        raise ValueError("Not enough hourly source training data before the transfer start date.")

    print("Source files:")
    print(f"  hourly: {source_paths['hourly']}")
    print(f"  15-min: {source_paths['min15']}")
    print("Optional model packages:", optional_model_status())
    print("Base models:", base_model_names())
    print(f"Transferable hourly features: {len(transferable_features)}")
    if excluded_features:
        print(f"Excluded hourly features: {excluded_features}")
    if warnings:
        print("Leakage check warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    X_hourly_train = df_hourly_source_train[transferable_features].reset_index(drop=True)
    y_hourly_train = df_hourly_source_train[config.target].reset_index(drop=True)

    print("\nTraining hourly source model")
    hourly_source_model = train_stacked_ensemble(X_hourly_train, y_hourly_train, n_splits=config.n_splits)

    df_15_hourly_rows = df_15_model[df_15_model[config.time_col].dt.minute == 0].copy()
    df_15_hourly_rows = df_15_hourly_rows.sort_values(config.time_col).reset_index(drop=True)
    X_15_hourly_as_hourly = map_15min_to_hourly_feature_space(df_15_hourly_rows, transferable_features)
    hourly_baseline_preds = predict_stacked_ensemble(hourly_source_model, X_15_hourly_as_hourly)

    hourly_baseline_df = pd.DataFrame(
        {
            config.time_col: df_15_hourly_rows[config.time_col].values,
            "Hourly_Baseline": hourly_baseline_preds["stacked"],
            "Hourly_XGB_Prediction": hourly_baseline_preds["xgb"],
            "Hourly_LGBM_Prediction": hourly_baseline_preds["lgb"],
            "Hourly_CAT_Prediction": hourly_baseline_preds["cat"],
            "Hourly_Average_Prediction": hourly_baseline_preds["average"],
            "Hourly_Stacked_Prediction": hourly_baseline_preds["stacked"],
            "Actual_Minute0_Price": df_15_hourly_rows[config.target].values,
        }
    )
    hourly_baseline_df["Error"] = hourly_baseline_df["Actual_Minute0_Price"] - hourly_baseline_df["Hourly_Baseline"]
    hourly_baseline_df["Absolute_Error"] = hourly_baseline_df["Error"].abs()

    df_15_transfer = df_15_model.copy()
    df_15_transfer["HourUTC_floor"] = df_15_transfer[config.time_col].dt.floor("h")
    df_15_transfer = df_15_transfer.merge(
        hourly_baseline_df[
            [
                config.time_col,
                "Hourly_Baseline",
                "Hourly_XGB_Prediction",
                "Hourly_LGBM_Prediction",
                "Hourly_CAT_Prediction",
                "Hourly_Average_Prediction",
                "Hourly_Stacked_Prediction",
            ]
        ],
        left_on="HourUTC_floor",
        right_on=config.time_col,
        how="left",
        suffixes=("", "_hourly"),
    )
    df_15_transfer = df_15_transfer.drop(columns=[f"{config.time_col}_hourly"])
    df_15_transfer = df_15_transfer.dropna(subset=["Hourly_Baseline"]).reset_index(drop=True)
    df_15_transfer["residual_15min"] = df_15_transfer[config.target] - df_15_transfer["Hourly_Baseline"]
    df_15_transfer = df_15_transfer.sort_values(config.time_col).reset_index(drop=True)

    train_15_mask, test_15_mask, split_time, split_label = make_train_test_split_mask(
        df_15_transfer,
        config.time_col,
        config.test_split_method,
        config.split_ratio,
        config.test_period_days,
    )
    print(f"\n15-min adaptation split: {split_label}")
    print(f"Split time: {split_time}")

    transfer_feature_cols = feature_cols_15.copy()
    if config.use_hourly_baseline_as_residual_feature:
        transfer_feature_cols = transfer_feature_cols + ["Hourly_Baseline"]

    X_15_train = df_15_transfer.loc[train_15_mask, transfer_feature_cols].reset_index(drop=True)
    y_15_train_residual = df_15_transfer.loc[train_15_mask, "residual_15min"].reset_index(drop=True)
    X_15_test = df_15_transfer.loc[test_15_mask, transfer_feature_cols].reset_index(drop=True)
    y_15_test_actual = df_15_transfer.loc[test_15_mask, config.target].reset_index(drop=True)

    print("\nTraining transfer residual model")
    residual_model = train_stacked_ensemble(X_15_train, y_15_train_residual, n_splits=config.n_splits)
    residual_preds = predict_stacked_ensemble(residual_model, X_15_test)

    baseline_test_prediction = df_15_transfer.loc[test_15_mask, "Hourly_Baseline"].values
    X_15_direct_train = df_15_transfer.loc[train_15_mask, feature_cols_15].reset_index(drop=True)
    y_15_direct_train = df_15_transfer.loc[train_15_mask, config.target].reset_index(drop=True)
    X_15_direct_test = df_15_transfer.loc[test_15_mask, feature_cols_15].reset_index(drop=True)

    print("\nTraining direct 15-min model")
    direct_15min_model = train_stacked_ensemble(X_15_direct_train, y_15_direct_train, n_splits=config.n_splits)
    direct_15min_preds = predict_stacked_ensemble(direct_15min_model, X_15_direct_test)

    results = pd.DataFrame(
        {
            config.time_col: df_15_transfer.loc[test_15_mask, config.time_col].values,
            "Actual_Price": y_15_test_actual.values,
            "Hourly_Baseline": baseline_test_prediction,
            "Direct_15min_XGB": direct_15min_preds["xgb"],
            "Direct_15min_LGBM": direct_15min_preds["lgb"],
            "Direct_15min_CAT": direct_15min_preds["cat"],
            "Direct_15min_Average": direct_15min_preds["average"],
            "Direct_15min_Stacked": direct_15min_preds["stacked"],
            "TL_Residual_XGB": baseline_test_prediction + residual_preds["xgb"],
            "TL_Residual_LGBM": baseline_test_prediction + residual_preds["lgb"],
            "TL_Residual_CAT": baseline_test_prediction + residual_preds["cat"],
            "TL_Residual_Average": baseline_test_prediction + residual_preds["average"],
            "TL_Residual_Stacked": baseline_test_prediction + residual_preds["stacked"],
        }
    )

    champion_column = config.champion_prediction_column
    if champion_column not in results.columns:
        champion_column = "TL_Residual_Stacked"
    results["Prediction"] = results[champion_column]
    results["Direct_15min_Prediction"] = results["Direct_15min_Stacked"]
    results["Error"] = results["Actual_Price"] - results["Prediction"]
    results["Absolute_Error"] = results["Error"].abs()
    results["Direct_Error"] = results["Actual_Price"] - results["Direct_15min_Prediction"]
    results["Direct_Absolute_Error"] = results["Direct_Error"].abs()

    train_start = df_15_transfer.loc[train_15_mask, config.time_col].min()
    train_end = df_15_transfer.loc[train_15_mask, config.time_col].max()
    test_start = df_15_transfer.loc[test_15_mask, config.time_col].min()
    test_end = df_15_transfer.loc[test_15_mask, config.time_col].max()
    coverage = calculate_15min_coverage(df_15_transfer.loc[test_15_mask, config.time_col])

    metadata = {
        "Train_Start": train_start,
        "Train_End": train_end,
        "Test_Start": test_start,
        "Test_End": test_end,
        "Train_Rows": int(train_15_mask.sum()),
        "Test_Rows": int(test_15_mask.sum()),
        "Split_Time": split_time,
        "Test_Split_Method": config.test_split_method,
        "Test_Period_Days": config.test_period_days,
        "Hourly_Source_Train_Start": df_hourly_source_train[config.time_col].min(),
        "Hourly_Source_Train_End": df_hourly_source_train[config.time_col].max(),
        "Hourly_Source_Number_Features": len(transferable_features),
        "Tuned_Configuration_Source": "Hourly randomized-search Autumn setup",
        **coverage,
    }

    metric_specs = [
        ("Hourly baseline only", "Hourly source baseline", "Hourly_Baseline", 1, None),
        ("Direct 15-min XGBoost", "15-min scratch", "Direct_15min_XGB", len(feature_cols_15), None),
        ("Direct 15-min LightGBM", "15-min scratch", "Direct_15min_LGBM", len(feature_cols_15), None),
        ("Direct 15-min CatBoost", "15-min scratch", "Direct_15min_CAT", len(feature_cols_15), None),
        ("Direct 15-min simple average", "15-min scratch", "Direct_15min_Average", len(feature_cols_15), None),
        ("Direct 15-min pure stacking", "15-min scratch", "Direct_15min_Stacked", len(feature_cols_15), None),
        ("TL residual XGBoost", "Transfer learning", "TL_Residual_XGB", len(transfer_feature_cols), config.use_hourly_baseline_as_residual_feature),
        ("TL residual LightGBM", "Transfer learning", "TL_Residual_LGBM", len(transfer_feature_cols), config.use_hourly_baseline_as_residual_feature),
        ("TL residual CatBoost", "Transfer learning", "TL_Residual_CAT", len(transfer_feature_cols), config.use_hourly_baseline_as_residual_feature),
        ("TL residual simple average", "Transfer learning", "TL_Residual_Average", len(transfer_feature_cols), config.use_hourly_baseline_as_residual_feature),
        ("TL residual pure stacking", "Transfer learning", "TL_Residual_Stacked", len(transfer_feature_cols), config.use_hourly_baseline_as_residual_feature),
    ]
    metrics = build_metrics_table(results, "Actual_Price", metric_specs, metadata)

    bundle = ForecastModelBundle(
        hourly_model=hourly_source_model,
        residual_model=residual_model,
        direct_model=direct_15min_model,
        hourly_feature_cols=transferable_features,
        residual_feature_cols=transfer_feature_cols,
        direct_feature_cols=feature_cols_15,
        champion_prediction_column=champion_column,
        time_col=config.time_col,
        metadata={
            "project_name": config.project_name,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "split_method": config.test_split_method,
            "source_paths": {key: str(value) for key, value in source_paths.items()},
        },
    )
    bundle_path = bundle.save(output_dir / "power_forecast_bundle.pkl")
    gate = evaluate_model_gate(
        metrics=metrics,
        candidate_prediction_column=champion_column,
        coverage=coverage,
        leakage_warnings=warnings,
        settings=ModelGateSettings(
            min_coverage_pct=config.model_gate_min_coverage_pct,
            max_baseline_mae_regression_pct=config.model_gate_max_baseline_mae_regression_pct,
            require_continuous_15min=config.model_gate_require_continuous_15min,
            require_no_leakage_warnings=config.model_gate_require_no_leakage_warnings,
        ),
    )
    gate_path = write_model_gate(gate, output_dir / "model_gate.json")

    outputs = {
        "hourly_baseline": _write_frame(hourly_baseline_df, output_dir / "hourly_baseline_minute0_predictions"),
        "forecasts": _write_frame(results, output_dir / "transfer_15min_forecasts_detailed"),
        "metrics": _write_frame(metrics, output_dir / "transfer_15min_metrics_detailed"),
        "model_bundle": {
            "pickle": str(bundle_path),
            "metadata": str(bundle_path.with_suffix(".metadata.json")),
        },
        "model_gate": str(gate_path),
    }

    summary = {
        "project_name": config.project_name,
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "output_dir": str(output_dir),
        "data_quality": {
            "hourly": data_quality_summary(df_hourly, config.time_col, config.target),
            "min15": data_quality_summary(df_15, config.time_col, config.target),
        },
        "model_packages": optional_model_status(),
        "base_models": base_model_names(),
        "transferable_hourly_features": len(transferable_features),
        "excluded_hourly_features": excluded_features,
        "leakage_warnings": warnings,
        "champion_prediction_column": champion_column,
        "model_gate": gate,
        "validation_split": {
            "method": config.test_split_method,
            "label": split_label,
            "split_ratio": config.split_ratio,
            "test_period_days": config.test_period_days,
            "split_time": split_time,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_rows": int(train_15_mask.sum()),
            "test_rows": int(test_15_mask.sum()),
            "coverage": coverage,
        },
        "outputs": outputs,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if config.mlflow_enabled:
        registered_model_name = config.mlflow_registered_model_name if gate["passed"] else None
        summary["mlflow"] = log_training_run(
            settings=MLflowRunSettings(
                tracking_uri=config.mlflow_tracking_uri,
                experiment_name=config.mlflow_experiment_name,
                registered_model_name=registered_model_name,
                run_name=f"{config.project_name}-{split_label}",
            ),
            bundle=bundle,
            config_params=config.__dict__,
            metrics=metrics,
            artifact_paths=[
                summary_path,
                bundle_path,
                bundle_path.with_suffix(".metadata.json"),
                gate_path,
                outputs["metrics"]["csv"],
                outputs["forecasts"]["csv"],
            ],
        )
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    print("\nRun summary:")
    print(summary_path)
    return summary
