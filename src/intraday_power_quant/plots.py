from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_forecast_window(
    results: pd.DataFrame,
    output_path: str | Path,
    days_back: int | None = 7,
    prediction_col: str = "Prediction",
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install matplotlib to generate PNG plots.") from exc

    df = results.copy()
    df["HourUTC"] = pd.to_datetime(df["HourUTC"])
    if days_back is not None:
        df = df[df["HourUTC"] >= df["HourUTC"].max() - pd.Timedelta(days=days_back)].copy()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 6))
    plt.plot(df["HourUTC"], df["Actual_Price"], label="Actual")
    if "Hourly_Baseline" in df.columns:
        plt.plot(df["HourUTC"], df["Hourly_Baseline"], label="Hourly baseline", linestyle="--")
    plt.plot(df["HourUTC"], df[prediction_col], label=prediction_col)
    if "Direct_15min_Prediction" in df.columns:
        plt.plot(df["HourUTC"], df["Direct_15min_Prediction"], label="Direct 15-min", linestyle=":")
    plt.title("15-minute DK1 price forecast")
    plt.xlabel("Time")
    plt.ylabel("DK1 price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()
    return output

