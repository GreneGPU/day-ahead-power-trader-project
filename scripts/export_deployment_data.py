from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FORECAST_COLUMNS = [
    "HourUTC",
    "Actual_Price",
    "Hourly_Baseline",
    "Prediction",
    "Direct_15min_Prediction",
]

METRIC_COLUMNS = [
    "Model",
    "Model_Type",
    "Number_Features",
    "Test_Start",
    "Test_End",
    "Test_Rows",
    "Test_Coverage_Pct",
    "MAE",
    "RMSE",
    "sMAPE",
    "R2",
]


def _read_excel(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_excel(path)


def _require_columns(frame: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def export_deployment_data(forecasts_path: Path, metrics_path: Path, output_dir: Path) -> None:
    forecasts = _read_excel(forecasts_path)
    metrics = _read_excel(metrics_path)
    _require_columns(forecasts, FORECAST_COLUMNS, forecasts_path)
    _require_columns(metrics, METRIC_COLUMNS, metrics_path)

    deployment_forecasts = forecasts[FORECAST_COLUMNS].copy()
    deployment_forecasts["HourUTC"] = pd.to_datetime(
        deployment_forecasts["HourUTC"], errors="raise", utc=True
    )
    deployment_forecasts = deployment_forecasts.sort_values("HourUTC").reset_index(drop=True)
    if deployment_forecasts.empty:
        raise ValueError("The forecast workbook contains no rows.")
    if deployment_forecasts[FORECAST_COLUMNS].isna().any().any():
        raise ValueError("The deployment forecast columns contain missing values.")

    deployment_metrics = metrics[METRIC_COLUMNS].copy()
    for column in ("Test_Start", "Test_End"):
        deployment_metrics[column] = pd.to_datetime(
            deployment_metrics[column], errors="raise", utc=True
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output_dir.mkdir(parents=True, exist_ok=True)
    deployment_forecasts.to_csv(
        output_dir / "predictions.csv.gz",
        index=False,
        compression="gzip",
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    (output_dir / "model_metrics.json").write_text(
        json.dumps(deployment_metrics.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    manifest = {
        "source_forecast_file": forecasts_path.name,
        "source_metrics_file": metrics_path.name,
        "rows": len(deployment_forecasts),
        "start": deployment_forecasts["HourUTC"].min().isoformat(),
        "end": deployment_forecasts["HourUTC"].max().isoformat(),
        "forecast_columns": FORECAST_COLUMNS[2:],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export sanitized prediction results for the Vercel dashboard."
    )
    parser.add_argument("forecasts", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("deployment_data"))
    args = parser.parse_args()
    export_deployment_data(args.forecasts, args.metrics, args.output_dir)


if __name__ == "__main__":
    main()
