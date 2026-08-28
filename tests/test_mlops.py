from __future__ import annotations

import json

import numpy as np
import pandas as pd

from intraday_power_quant.model_bundle import ForecastModelBundle
from intraday_power_quant.model_gate import evaluate_model_gate
from intraday_power_quant.monitoring import build_monitoring_report
from sagemaker_jobs.inference import input_fn, output_fn


class ConstantEstimator:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def _ensemble(value: float):
    return {
        "xgb": ConstantEstimator(value),
        "lgb": ConstantEstimator(value),
        "cat": ConstantEstimator(value),
        "meta": ConstantEstimator(value),
    }


def test_bundle_round_trip_and_inference(tmp_path):
    bundle = ForecastModelBundle(
        hourly_model=_ensemble(10),
        residual_model=_ensemble(2),
        direct_model=_ensemble(11),
        hourly_feature_cols=["hour"],
        residual_feature_cols=["feature", "Hourly_Baseline"],
        direct_feature_cols=["feature"],
    )
    path = bundle.save(tmp_path / "model.pkl")
    loaded = ForecastModelBundle.load(path)
    prediction = loaded.predict_components(pd.DataFrame({"feature": [1], "Hourly_Baseline": [10]}))
    assert prediction.loc[0, "Prediction"] == 12
    assert prediction.loc[0, "Direct_15min_Prediction"] == 11
    assert json.loads(output_fn(prediction, "application/json"))[0]["Prediction"] == 12
    assert input_fn('{"feature": 1, "Hourly_Baseline": 10}', "application/json").shape == (1, 2)


def test_model_gate_and_monitoring():
    metrics = pd.DataFrame(
        {"Prediction_Column": ["Hourly_Baseline", "TL_Residual_Average"], "MAE": [4.0, 3.0]}
    )
    gate = evaluate_model_gate(
        metrics,
        "TL_Residual_Average",
        {"Test_Coverage_Pct": 100, "Test_Continuous_15min": True},
        [],
    )
    assert gate["passed"] is True
    report = build_monitoring_report(
        pd.DataFrame({"Actual_Price": [10, 14], "Prediction": [11, 12], "Hourly_Baseline": [8, 10]})
    )
    assert report["mae"] == 1.5
    assert report["mae_improvement_pct"] == 50.0
