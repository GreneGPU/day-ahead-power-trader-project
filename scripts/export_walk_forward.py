"""Export walk-forward out-of-sample forecasts as the website's deployment data.

Replaces predictions.csv.gz and the project-holdout rows of model_metrics.json; the provided thesis
benchmark rows (SARIMAX, LSTM, CNN-LSTM) are kept unchanged. Run the enrichment, imbalance and formula
feature scripts afterwards so every deployment file covers the new period.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_COLUMNS = ["HourUTC", "Actual_Price", "Hourly_Baseline", "Prediction", "Direct_15min_Prediction",
                    "Transfer_Residual_Prediction"]
METRIC_FIELDS = ["Model", "Model_Type", "Metric_Source", "Number_Features", "Test_Start", "Test_End", "Test_Rows",
                 "Test_Coverage_Pct", "MAE", "RMSE", "sMAPE", "R2"]
WEBSITE_MODELS = ["Walk-forward champion", "Hourly baseline only", "Direct 15-min ensemble",
                  "TL residual simple average", "TL residual pure stacking"]


def export(results_dir: Path, output_dir: Path) -> None:
    forecasts = pd.read_csv(results_dir / "walk_forward_forecasts.csv", parse_dates=["HourUTC"])
    forecasts["HourUTC"] = pd.to_datetime(forecasts["HourUTC"], utc=True)  # the modeling data is UTC without offset
    forecasts = forecasts.sort_values("HourUTC").reset_index(drop=True)
    # The previously configured (ungated) transfer champion stays available as its own forecast series.
    forecasts["Transfer_Residual_Prediction"] = forecasts["Static_Champion_Prediction"]
    if forecasts[FORECAST_COLUMNS].isna().any().any() or forecasts["HourUTC"].duplicated().any():
        raise ValueError("Walk-forward forecasts must be complete and unique per interval.")

    metrics = pd.read_csv(results_dir / "walk_forward_metrics.csv")
    metrics = metrics[metrics["Model"].isin(WEBSITE_MODELS)].copy()
    metrics["Metric_Source"] = "Walk-forward out-of-sample"
    for column in ("Test_Start", "Test_End"):
        metrics[column] = pd.to_datetime(metrics[column], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    existing = json.loads((output_dir / "model_metrics.json").read_text(encoding="utf-8"))
    benchmarks = [row for row in existing if row.get("Metric_Source") == "Provided model comparison"]
    metrics["order"] = metrics["Model"].map(WEBSITE_MODELS.index)
    project_rows = metrics.sort_values("order")[METRIC_FIELDS].to_dict(orient="records")

    output_dir.mkdir(parents=True, exist_ok=True)
    deployment = forecasts[FORECAST_COLUMNS]
    # Keep Energinet feature columns already enriched for the same timestamps (avoids re-downloading them).
    existing_path = output_dir / "predictions.csv.gz"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        extra = [c for c in existing.columns if c not in FORECAST_COLUMNS]
        existing["HourUTC"] = pd.to_datetime(existing["HourUTC"], utc=True)
        if extra and existing["HourUTC"].equals(deployment["HourUTC"]):
            deployment = deployment.join(existing[extra])
    deployment.to_csv(output_dir / "predictions.csv.gz", index=False,
                                       compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
                                       date_format="%Y-%m-%dT%H:%M:%SZ")
    (output_dir / "model_metrics.json").write_text(json.dumps(project_rows + benchmarks, indent=2, default=str), encoding="utf-8")
    summary = json.loads((results_dir / "walk_forward_summary.json").read_text(encoding="utf-8"))
    # Update rather than replace: keeps provenance written by the enrichment and imbalance scripts.
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest |= {
        "source_forecast_file": "walk_forward_forecasts.csv",
        "source_metrics_file": "walk_forward_metrics.csv",
        "forecast_method": summary["method"],
        "walk_forward_blocks": summary["blocks"],
        "walk_forward_champions": summary.get("champions"),
        "configured_champion": summary.get("configured_champion"),
        "rows": len(forecasts),
        "start": forecasts["HourUTC"].min().isoformat(),
        "end": forecasts["HourUTC"].max().isoformat(),
        "forecast_columns": FORECAST_COLUMNS[2:],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(forecasts)} intervals {manifest['start']} -> {manifest['end']} "
          f"({summary['blocks']} walk-forward blocks) and {len(project_rows)} + {len(benchmarks)} metric rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "outputs" / "walk_forward")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "deployment_data")
    args = parser.parse_args()
    export(args.results_dir, args.output_dir)
