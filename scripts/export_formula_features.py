"""Export already-lagged thesis features for the exact website timestamps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_power_quant.formulas import THESIS_FEATURES


def export(source: Path, output: Path):
    features = pd.read_csv(source, usecols=["HourUTC", *THESIS_FEATURES])
    # The thesis CSV is day-first (e.g. 04-03-26), with UTC delivery times.
    features["HourUTC"] = pd.to_datetime(features["HourUTC"], format="%d-%m-%y %H:%M", utc=True)
    times = pd.read_csv(output / "predictions.csv.gz", usecols=["HourUTC"])
    times["HourUTC"] = pd.to_datetime(times["HourUTC"], utc=True)
    aligned = times.merge(features, on="HourUTC", how="left", validate="one_to_one")
    if not np.isfinite(aligned[THESIS_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Source features must cover every website timestamp with finite values.")
    aligned.to_csv(output / "formula_features.csv.gz", index=False,
                   compression={"method": "gzip", "mtime": 0}, date_format="%Y-%m-%dT%H:%M:%SZ")
    (output / "formula_features_manifest.json").write_text(json.dumps({
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(aligned), "columns": THESIS_FEATURES,
        "timestamp_format": "%d-%m-%y %H:%M (UTC)",
        "lag_policy": "Preserve supplied lagged values; no recomputation, filling or extra shifting.",
    }, indent=2), encoding="utf-8")
    print(f"Exported {len(aligned)} timestamps and {len(THESIS_FEATURES)} features.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("deployment_data"))
    args = parser.parse_args()
    export(args.source, args.output)
