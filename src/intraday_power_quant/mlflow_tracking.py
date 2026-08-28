from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .model_bundle import ForecastModelBundle

try:
    import mlflow.pyfunc as _mlflow_pyfunc

    _PythonModelBase = _mlflow_pyfunc.PythonModel
except ModuleNotFoundError:  # MLflow is an optional dependency.
    _PythonModelBase = object


class MLflowPowerForecastModel(_PythonModelBase):
    def __init__(self, bundle: ForecastModelBundle):
        self.bundle = bundle

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context, params
        return self.bundle.predict_components(model_input)


@dataclass(frozen=True)
class MLflowRunSettings:
    tracking_uri: str | None = None
    experiment_name: str = "power-trader-dk1-15min"
    registered_model_name: str | None = None
    run_name: str | None = None


def _metric_values(metrics: pd.DataFrame, champion_column: str) -> dict[str, float]:
    logged: dict[str, float] = {}
    for label, prediction_column in {
        "candidate": champion_column,
        "hourly_baseline": "Hourly_Baseline",
        "direct_stacked": "Direct_15min_Stacked",
    }.items():
        rows = metrics.loc[metrics["Prediction_Column"] == prediction_column]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for metric in ["MAE", "RMSE", "MAPE", "sMAPE", "R2", "Test_Coverage_Pct"]:
            if metric in row and pd.notna(row[metric]):
                logged[f"{label}_{metric.lower()}"] = float(row[metric])
    return logged


def log_training_run(
    settings: MLflowRunSettings,
    bundle: ForecastModelBundle,
    config_params: dict[str, Any],
    metrics: pd.DataFrame,
    artifact_paths: list[str | Path],
) -> dict[str, str]:
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'MLflow tracking is enabled. Install the project with the "mlops" extra.'
        ) from exc

    if settings.tracking_uri:
        mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name=settings.run_name) as run:
        safe_params = {
            key: value
            for key, value in config_params.items()
            if value is not None and isinstance(value, (str, int, float, bool))
        }
        mlflow.log_params(safe_params)
        mlflow.log_metrics(_metric_values(metrics, bundle.champion_prediction_column))
        for artifact_path in artifact_paths:
            path = Path(artifact_path)
            if path.exists() and path.is_file():
                mlflow.log_artifact(str(path), artifact_path="run_outputs")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MLflowPowerForecastModel(bundle),
            registered_model_name=settings.registered_model_name,
        )
        return {
            "run_id": run.info.run_id,
            "artifact_uri": run.info.artifact_uri,
            "model_uri": model_info.model_uri,
        }
