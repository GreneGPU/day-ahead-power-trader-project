from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def find_input_file(data_dir: str | Path, candidates: Iterable[str]) -> Path:
    root = Path(data_dir)
    attempted: list[Path] = []
    for name in dedupe_preserve_order(candidates):
        candidate = root / name
        attempted.append(candidate)
        if candidate.exists():
            return candidate
    attempted_text = "\n".join(f"- {path}" for path in attempted)
    raise FileNotFoundError(f"No candidate input file found. Tried:\n{attempted_text}")


def read_dataset(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    raise ValueError(f"Unsupported file type: {input_path}")


def load_market_data(
    data_dir: str | Path,
    hourly_candidates: Iterable[str],
    min15_candidates: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    hourly_path = find_input_file(data_dir, hourly_candidates)
    min15_path = find_input_file(data_dir, min15_candidates)
    return read_dataset(hourly_path), read_dataset(min15_path), {
        "hourly": hourly_path,
        "min15": min15_path,
    }


def prepare_time_series_frame(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    prepared = df.copy()
    if time_col not in prepared.columns:
        raise KeyError(f"Missing time column: {time_col}")
    prepared[time_col] = pd.to_datetime(prepared[time_col])
    prepared = prepared.sort_values(time_col).reset_index(drop=True)
    return prepared


def infer_feature_columns(df: pd.DataFrame, time_col: str, target: str) -> list[str]:
    forbidden = {time_col, target}
    return [column for column in df.columns if column not in forbidden]


def check_columns(df: pd.DataFrame, feature_cols: list[str], time_col: str, target: str, name: str) -> None:
    forbidden = [column for column in [time_col, target] if column in feature_cols]
    if forbidden:
        raise ValueError(f"{name}: forbidden columns used as features: {forbidden}")
    missing = [column for column in feature_cols if column not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing feature columns: {missing}")


def drop_model_na(df: pd.DataFrame, feature_cols: list[str], target: str) -> pd.DataFrame:
    return df.dropna(subset=[target] + feature_cols).reset_index(drop=True)


def rename_15min_columns_to_hourly_space(df_15_input: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df_15_input.columns:
        new_col = col
        new_col = new_col.replace("_lag_96", "_lag_24")
        new_col = new_col.replace("_lag_192", "_lag_48")
        new_col = new_col.replace("_lag_672", "_lag_168")
        new_col = new_col.replace("known_lag96", "known_lag24")
        new_col = new_col.replace("known_lag192", "known_lag48")
        new_col = new_col.replace("known_lag672", "known_lag168")
        new_col = new_col.replace("roll_mean_96_safe", "roll_mean_24_safe")
        new_col = new_col.replace("roll_std_96_safe", "roll_std_24_safe")
        new_col = new_col.replace("roll_mean_672_safe", "roll_mean_168_safe")
        new_col = new_col.replace("roll_std_672_safe", "roll_std_168_safe")
        rename_map[col] = new_col
    return df_15_input.rename(columns=rename_map)


def transferable_hourly_features(
    hourly_feature_cols: list[str],
    df_15_model: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    preview = rename_15min_columns_to_hourly_space(df_15_model)
    transferable = [column for column in hourly_feature_cols if column in preview.columns]
    excluded = [column for column in hourly_feature_cols if column not in transferable]
    return transferable, excluded


def map_15min_to_hourly_feature_space(
    df_15_input: pd.DataFrame,
    hourly_features: list[str],
) -> pd.DataFrame:
    mapped = rename_15min_columns_to_hourly_space(df_15_input.copy())
    missing = [column for column in hourly_features if column not in mapped.columns]
    if missing:
        raise ValueError(f"Missing mapped hourly features: {missing}")
    return mapped[hourly_features].copy()


def data_quality_summary(df: pd.DataFrame, time_col: str, target: str) -> dict[str, object]:
    time_values = pd.to_datetime(df[time_col])
    duplicate_times = int(time_values.duplicated().sum())
    missing_target = int(df[target].isna().sum()) if target in df.columns else None
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "start": time_values.min(),
        "end": time_values.max(),
        "duplicate_timestamps": duplicate_times,
        "missing_target": missing_target,
        "monotonic_time": bool(time_values.is_monotonic_increasing),
    }

