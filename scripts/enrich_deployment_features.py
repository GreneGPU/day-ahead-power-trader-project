from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "deployment_data" / "predictions.csv.gz"
MANIFEST_PATH = PROJECT_ROOT / "deployment_data" / "manifest.json"
ENERGINET_API = "https://api.energidataservice.dk/dataset/Forecasts_Hour"


def _fetch_energinet_records(start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, object]]:
    params = urlencode(
        {
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "end": end.strftime("%Y-%m-%dT%H:%M"),
            "filter": json.dumps({"PriceArea": "DK1"}, separators=(",", ":")),
            "limit": 10_000,
        }
    )
    with urlopen(f"{ENERGINET_API}?{params}", timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    return list(payload["records"])


def main() -> None:
    forecasts = pd.read_csv(DATA_PATH)
    forecasts["HourUTC"] = pd.to_datetime(forecasts["HourUTC"], utc=True)
    hour_key = forecasts["HourUTC"].dt.floor("h")
    start = hour_key.min()
    end = hour_key.max() + pd.Timedelta(hours=1)

    # The public API interprets bare request timestamps in Danish local time.
    # Request a UTC-hour buffer on both sides, then join by the explicit HourUTC field.
    records = _fetch_energinet_records(start - pd.Timedelta(hours=1), end + pd.Timedelta(hours=1))
    features = pd.DataFrame.from_records(records)
    features["HourUTC"] = pd.to_datetime(features["HourUTC"], utc=True)
    hourly = features.pivot_table(
        index="HourUTC",
        columns="ForecastType",
        values=["ForecastDayAhead", "ForecastIntraday"],
        aggfunc="last",
    )
    hourly.columns = [f"{value}_{kind}" for value, kind in hourly.columns]
    hourly = hourly.reset_index().rename(
        columns={
            "ForecastDayAhead_Offshore Wind": "Wind_Offshore_DayAhead_MW",
            "ForecastDayAhead_Onshore Wind": "Wind_Onshore_DayAhead_MW",
            "ForecastDayAhead_Solar": "Solar_DayAhead_MW",
            "ForecastIntraday_Offshore Wind": "Wind_Offshore_Intraday_MW",
            "ForecastIntraday_Onshore Wind": "Wind_Onshore_Intraday_MW",
        }
    )
    hourly["Wind_Total_DayAhead_MW"] = (
        hourly["Wind_Offshore_DayAhead_MW"] + hourly["Wind_Onshore_DayAhead_MW"]
    )
    hourly["Wind_Total_Intraday_MW"] = (
        hourly["Wind_Offshore_Intraday_MW"] + hourly["Wind_Onshore_Intraday_MW"]
    )
    hourly["Wind_Intraday_Revision_MW"] = (
        hourly["Wind_Total_Intraday_MW"] - hourly["Wind_Total_DayAhead_MW"]
    )

    feature_columns = [
        "Wind_Offshore_DayAhead_MW",
        "Wind_Onshore_DayAhead_MW",
        "Wind_Total_DayAhead_MW",
        "Solar_DayAhead_MW",
        "Wind_Total_Intraday_MW",
        "Wind_Intraday_Revision_MW",
    ]
    enriched = forecasts.drop(columns=[c for c in feature_columns if c in forecasts.columns])
    enriched["Feature_HourUTC"] = hour_key
    enriched = enriched.merge(
        hourly[["HourUTC", *feature_columns]],
        left_on="Feature_HourUTC",
        right_on="HourUTC",
        how="left",
        suffixes=("", "_feature"),
        validate="many_to_one",
    )
    enriched = enriched.drop(columns=["Feature_HourUTC", "HourUTC_feature"])
    missing = enriched[feature_columns].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(f"Energinet feature coverage is incomplete: {missing.to_dict()}")

    enriched.to_csv(
        DATA_PATH,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["feature_columns"] = feature_columns
    manifest["feature_source"] = "Energinet Forecasts_Hour, DK1"
    manifest["feature_source_url"] = (
        "https://www.energidataservice.dk/tso-electricity/Forecasts_Hour"
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Enriched {len(enriched)} intervals with {len(feature_columns)} Energinet features.")


if __name__ == "__main__":
    main()
