from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Downloads" / "ImbalancePrice (1).csv"

PRICE_PAIRS = (
    ("ImbalancePriceEUR", "ImbalancePriceDKK"),
    ("aFRRVWAUpEUR", "aFRRVWAUpDKK"),
    ("aFRRVWADownEUR", "aFRRVWADownDKK"),
    ("mFRRMarginalPriceUpEUR", "mFRRMarginalPriceUpDKK"),
    ("mFRRMarginalPriceDownEUR", "mFRRMarginalPriceDownDKK"),
)


def import_imbalance_prices(source: Path, output: Path) -> pd.DataFrame:
    raw = pd.read_csv(source, sep=";", decimal=",")
    raw["TimeUTC"] = pd.to_datetime(raw["TimeUTC"], utc=True, errors="raise")
    if set(raw["PriceArea"].unique()) != {"DK1"}:
        raise ValueError("The imbalance extract must contain DK1 only.")
    if raw.duplicated(["TimeUTC", "PriceArea"]).any():
        raise ValueError("The imbalance extract contains duplicate DK1 timestamps.")

    raw = raw.sort_values("TimeUTC").reset_index(drop=True)
    expected = pd.date_range(raw["TimeUTC"].min(), raw["TimeUTC"].max(), freq="15min")
    missing = expected.difference(raw["TimeUTC"])
    if len(missing):
        raise ValueError(f"The imbalance extract is missing {len(missing)} quarter-hours.")

    rates = []
    for eur_col, dkk_col in PRICE_PAIRS:
        eur = pd.to_numeric(raw[eur_col], errors="raise")
        dkk = pd.to_numeric(raw[dkk_col], errors="raise")
        rates.append((dkk / eur).where(eur.abs() > 0.01))
    fx = pd.concat(rates, axis=1).median(axis=1)
    if fx.isna().any() or not fx.between(7.0, 8.0).all():
        raise ValueError("Could not derive a valid EUR/DKK rate for every interval.")
    raw["FX_DKK_per_EUR"] = fx

    forecast_path = PROJECT_ROOT / "deployment_data" / "predictions.csv.gz"
    forecasts = pd.read_csv(forecast_path, usecols=["HourUTC", "Actual_Price"])
    forecasts["HourUTC"] = pd.to_datetime(forecasts["HourUTC"], utc=True)
    selected = forecasts.merge(raw, left_on="HourUTC", right_on="TimeUTC", how="left", validate="one_to_one")
    if selected["ImbalancePriceDKK"].isna().any():
        raise ValueError("The imbalance extract does not cover every saved prediction timestamp.")
    if not np.allclose(selected["Actual_Price"], selected["SpotPriceEUR"], atol=0.001):
        raise ValueError("Saved actual prices do not match the extract's EUR spot prices.")

    selected["Spot_Price_DKK"] = selected["SpotPriceEUR"] * selected["FX_DKK_per_EUR"]
    output_frame = selected[
        [
            "HourUTC",
            "SpotPriceEUR",
            "Spot_Price_DKK",
            "FX_DKK_per_EUR",
            "ImbalancePriceEUR",
            "ImbalancePriceDKK",
            "DominatingDirection",
            "SatisfiedDemand",
            "aFRRUpMW",
            "aFRRDownMW",
            "mFRRMarginalPriceUpDKK",
            "mFRRMarginalPriceDownDKK",
        ]
    ].rename(
        columns={
            "SpotPriceEUR": "Spot_Price_EUR",
            "ImbalancePriceEUR": "Imbalance_Price_EUR",
            "ImbalancePriceDKK": "Imbalance_Price_DKK",
            "DominatingDirection": "Dominating_Direction",
            "SatisfiedDemand": "Satisfied_Demand_MW",
            "aFRRUpMW": "aFRR_Up_MW",
            "aFRRDownMW": "aFRR_Down_MW",
            "mFRRMarginalPriceUpDKK": "mFRR_Marginal_Price_Up_DKK",
            "mFRRMarginalPriceDownDKK": "mFRR_Marginal_Price_Down_DKK",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output, index=False, compression="gzip")
    return output_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and import DK1 imbalance prices.")
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "deployment_data" / "imbalance_prices.csv.gz",
    )
    args = parser.parse_args()
    frame = import_imbalance_prices(args.source, args.output)
    print(
        f"Wrote {args.output} with {len(frame):,} matched intervals from "
        f"{frame['HourUTC'].min()} through {frame['HourUTC'].max()}."
    )


if __name__ == "__main__":
    main()
