from __future__ import annotations

import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "deployment_data" / "predictions.csv.gz"
MANIFEST_PATH = PROJECT_ROOT / "deployment_data" / "manifest.json"
ENERGINET_API = "https://api.energidataservice.dk/dataset/Forecasts_Hour"
THESIS_15MIN = PROJECT_ROOT / "data" / "raw" / "final_15min_day_ahead_safe_modeling_dataset.csv"


def _fetch_energinet_records(start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, object]]:
    params = urlencode(
        {
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "end": end.strftime("%Y-%m-%dT%H:%M"),
            "filter": json.dumps({"PriceArea": "DK1"}, separators=(",", ":")),
            "limit": 10_000,
        }
    )
    # The public API rate-limits bursts (HTTP 429); back off and retry a few times.
    for attempt in range(6):
        try:
            with urlopen(f"{ENERGINET_API}?{params}", timeout=60) as response:  # noqa: S310
                payload = json.load(response)
            time.sleep(2)
            return list(payload["records"])
        except HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def main() -> None:
    forecasts = pd.read_csv(DATA_PATH)
    forecasts["HourUTC"] = pd.to_datetime(forecasts["HourUTC"], utc=True)
    hour_key = forecasts["HourUTC"].dt.floor("h")
    start = hour_key.min()
    end = hour_key.max() + pd.Timedelta(hours=1)

    # The public API interprets bare request timestamps in Danish local time.
    # Request a UTC-hour buffer on both sides, then join by the explicit HourUTC field.
    # Fetch in 20-day chunks: one request is capped at 10,000 records (about 5 forecast types per hour).
    records, cursor, stop = [], start - pd.Timedelta(hours=1), end + pd.Timedelta(hours=1)
    while cursor < stop:
        chunk_end = min(cursor + pd.Timedelta(days=20), stop)
        records.extend(_fetch_energinet_records(cursor, chunk_end))
        cursor = chunk_end
    features = pd.DataFrame.from_records(records).drop_duplicates(["HourUTC", "PriceArea", "ForecastType"], keep="last")
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

    # Energinet's Forecasts_Hour has holes (e.g. DK1 2025-11-22 to 11-24). Fill only the day-ahead columns
    # for those hours from the modeling dataset's own wind/solar forecasts, averaged to the hour; the unused
    # intraday columns stay empty rather than invented.
    gap = enriched["Wind_Total_DayAhead_MW"].isna()
    filled_hours = 0
    if gap.any():
        thesis = pd.read_csv(THESIS_15MIN, usecols=["HourUTC", "wind_offshore_fc", "wind_onshore_fc", "solar_fc"])
        thesis["HourUTC"] = pd.to_datetime(thesis["HourUTC"], utc=True)
        hourly_thesis = thesis.groupby(thesis["HourUTC"].dt.floor("h"))[["wind_offshore_fc", "wind_onshore_fc", "solar_fc"]].mean()
        keys = enriched.loc[gap, "HourUTC"].dt.floor("h")
        fill = hourly_thesis.reindex(keys).to_numpy()
        enriched.loc[gap, "Wind_Offshore_DayAhead_MW"] = fill[:, 0]
        enriched.loc[gap, "Wind_Onshore_DayAhead_MW"] = fill[:, 1]
        enriched.loc[gap, "Solar_DayAhead_MW"] = fill[:, 2]
        enriched.loc[gap, "Wind_Total_DayAhead_MW"] = fill[:, 0] + fill[:, 1]
        filled_hours = int(keys.nunique())
    day_ahead = ["Wind_Offshore_DayAhead_MW", "Wind_Onshore_DayAhead_MW", "Wind_Total_DayAhead_MW", "Solar_DayAhead_MW"]
    missing = enriched[day_ahead].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(f"Day-ahead feature coverage is incomplete: {missing.to_dict()}")

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
    manifest["feature_gap_filled_hours"] = filled_hours
    manifest["feature_gap_fill_source"] = (
        "Modeling dataset wind/solar day-ahead forecasts (hourly mean of 15-min values); intraday columns left empty"
        if filled_hours else None
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Enriched {len(enriched)} intervals with {len(feature_columns)} Energinet features.")


if __name__ == "__main__":
    main()
