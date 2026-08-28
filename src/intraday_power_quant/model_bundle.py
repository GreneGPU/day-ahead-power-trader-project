from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from .models import predict_stacked_ensemble


MODEL_FORMAT_VERSION = 1


@dataclass
class ForecastModelBundle:
    """Serializable end-to-end bundle for prepared 15-minute forecast features.

    The scoring frame may contain a precomputed ``Hourly_Baseline`` column. If
    it does not, hourly features must be supplied as ``hourly__<feature>`` so
    the bundled hourly source model can create the baseline first.
    """

    hourly_model: dict[str, Any]
    residual_model: dict[str, Any]
    direct_model: dict[str, Any]
    hourly_feature_cols: list[str]
    residual_feature_cols: list[str]
    direct_feature_cols: list[str]
    champion_prediction_column: str = "TL_Residual_Average"
    time_col: str = "HourUTC"
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: int = MODEL_FORMAT_VERSION

    def _hourly_baseline(self, frame: pd.DataFrame) -> pd.Series:
        if "Hourly_Baseline" in frame.columns:
            return frame["Hourly_Baseline"].astype(float).reset_index(drop=True)

        prefixed = [f"hourly__{column}" for column in self.hourly_feature_cols]
        missing = [column for column in prefixed if column not in frame.columns]
        if missing:
            raise ValueError(
                "Scoring input needs Hourly_Baseline or prefixed hourly features. "
                f"Missing: {missing}"
            )
        hourly = frame[prefixed].copy()
        hourly.columns = self.hourly_feature_cols
        predictions = predict_stacked_ensemble(self.hourly_model, hourly)
        return pd.Series(predictions["stacked"], name="Hourly_Baseline")

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing {label} features: {missing}")

    def predict_components(self, model_input: pd.DataFrame) -> pd.DataFrame:
        frame = model_input.reset_index(drop=True).copy()
        baseline = self._hourly_baseline(frame)

        residual_columns_without_baseline = [
            column for column in self.residual_feature_cols if column != "Hourly_Baseline"
        ]
        self._require_columns(frame, residual_columns_without_baseline, "residual-model")
        self._require_columns(frame, self.direct_feature_cols, "direct-model")

        residual_features = frame[residual_columns_without_baseline].copy()
        if "Hourly_Baseline" in self.residual_feature_cols:
            residual_features["Hourly_Baseline"] = baseline.values
        residual_features = residual_features[self.residual_feature_cols]
        direct_features = frame[self.direct_feature_cols].copy()

        residual = predict_stacked_ensemble(self.residual_model, residual_features)
        direct = predict_stacked_ensemble(self.direct_model, direct_features)
        output = pd.DataFrame(
            {
                "Hourly_Baseline": baseline.values,
                "Direct_15min_XGB": direct["xgb"],
                "Direct_15min_LGBM": direct["lgb"],
                "Direct_15min_CAT": direct["cat"],
                "Direct_15min_Average": direct["average"],
                "Direct_15min_Stacked": direct["stacked"],
                "TL_Residual_XGB": baseline.values + residual["xgb"],
                "TL_Residual_LGBM": baseline.values + residual["lgb"],
                "TL_Residual_CAT": baseline.values + residual["cat"],
                "TL_Residual_Average": baseline.values + residual["average"],
                "TL_Residual_Stacked": baseline.values + residual["stacked"],
            }
        )
        champion = self.champion_prediction_column
        if champion not in output.columns:
            champion = "TL_Residual_Stacked"
        output["Prediction"] = output[champion]
        output["Direct_15min_Prediction"] = output["Direct_15min_Stacked"]
        if self.time_col in frame.columns:
            output.insert(0, self.time_col, frame[self.time_col].values)
        return output

    def predict(self, model_input: pd.DataFrame) -> pd.Series:
        return self.predict_components(model_input)["Prediction"]

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "champion_prediction_column": self.champion_prediction_column,
            "time_col": self.time_col,
            "hourly_feature_cols": self.hourly_feature_cols,
            "residual_feature_cols": self.residual_feature_cols,
            "direct_feature_cols": self.direct_feature_cols,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metadata_path = output_path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(self.metadata_payload(), indent=2, default=str), encoding="utf-8"
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ForecastModelBundle":
        """Load a trusted model artifact created by :meth:`save`."""

        with Path(path).open("rb") as handle:
            bundle = pickle.load(handle)  # noqa: S301 - model artifacts are trusted inputs.
        if not isinstance(bundle, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(bundle).__name__}")
        if bundle.format_version != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format {bundle.format_version}; expected {MODEL_FORMAT_VERSION}."
            )
        return bundle
