from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from intraday_power_quant.model_bundle import ForecastModelBundle  # noqa: E402


def model_fn(model_dir: str) -> ForecastModelBundle:
    return ForecastModelBundle.load(Path(model_dir) / "power_forecast_bundle.pkl")


def input_fn(request_body: str | bytes, content_type: str) -> pd.DataFrame:
    body = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    if media_type in {"application/json", "application/jsonlines", "application/x-jsonlines"}:
        try:
            payload: Any = json.loads(body)
            records = payload["instances"] if isinstance(payload, dict) and "instances" in payload else payload
            if isinstance(records, dict):
                records = [records]
            return pd.DataFrame(records)
        except json.JSONDecodeError:
            return pd.read_json(StringIO(body), lines=True)
    if media_type in {"text/csv", "application/csv"}:
        return pd.read_csv(StringIO(body))
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(model_input: pd.DataFrame, model: ForecastModelBundle) -> pd.DataFrame:
    return model.predict_components(model_input)


def output_fn(prediction: pd.DataFrame, accept: str) -> str:
    media_type = accept.split(";", maxsplit=1)[0].strip().lower()
    if media_type in {"application/json", "application/jsonlines", "application/x-jsonlines"}:
        return prediction.to_json(orient="records", date_format="iso")
    if media_type in {"text/csv", "application/csv", "*/*"}:
        return prediction.to_csv(index=False)
    raise ValueError(f"Unsupported accept type: {accept}")
