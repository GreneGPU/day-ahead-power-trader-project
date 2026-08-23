from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any


DEFAULT_HOURLY_CANDIDATES = [
    "final_day_ahead_safe_modeling_dataset_hourly.csv",
    "final_day_ahead_safe_modeling_dataset.csv",
    "final_day_ahead_safe_modeling_dataset.xlsx",
]

DEFAULT_MIN15_CANDIDATES = [
    "final_15min_day_ahead_safe_modeling_dataset.csv",
    "final_15min_day_ahead_safe_modeling_dataset.xlsx",
]


@dataclass(frozen=True)
class BatterySettings:
    capacity_mwh: float = 100.0
    power_mw: float = 25.0
    initial_soc_mwh: float = 50.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    fee_per_mwh: float = 0.0
    max_daily_loss: float | None = None


@dataclass(frozen=True)
class WeeklyBandSettings:
    window_days: float = 7.0
    band: float = 20.0
    min_history_days: float = 1.0


@dataclass(frozen=True)
class ForecastEdgeSettings:
    threshold: float = 5.0
    reference_col: str = "Hourly_Baseline"


@dataclass(frozen=True)
class VolatilityFilterSettings:
    average_window_days: float = 7.0
    volatility_window_days: float = 7.0
    min_volatility: float = 30.0
    price_band: float = 0.0
    min_history_days: float = 1.0


@dataclass(frozen=True)
class MeanReversionSettings:
    window_days: float = 7.0
    entry_z: float = 1.0
    min_history_days: float = 1.0


@dataclass(frozen=True)
class MomentumSettings:
    lookback_hours: float = 6.0
    threshold: float = 5.0
    smoothing_hours: float = 1.0


@dataclass(frozen=True)
class MomentumSpreadSettings:
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    rank_mode: str = "percentile"
    charge_rank: int = 24
    discharge_rank: int = 73
    min_daily_spread: float = 20.0
    lookback_hours: float = 6.0
    momentum_threshold: float = 5.0
    smoothing_hours: float = 1.0


@dataclass(frozen=True)
class ChannelBreakoutSettings:
    window_days: float = 3.0
    buffer: float = 0.0
    min_history_days: float = 1.0


@dataclass(frozen=True)
class DailySpreadSettings:
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    rank_mode: str = "percentile"
    charge_rank: int = 24
    discharge_rank: int = 73
    min_daily_spread: float = 0.0


@dataclass(frozen=True)
class EnsembleAgreementSettings:
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    min_agreement: float = 0.60
    max_model_spread: float | None = None


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str = "Day-Ahead Power Trader Project"
    data_dir: Path = Path("data/raw")
    reference_results_dir: Path = Path("outputs")
    output_dir: Path = Path("outputs/model_run")
    hourly_candidates: list[str] = field(default_factory=lambda: DEFAULT_HOURLY_CANDIDATES.copy())
    min15_candidates: list[str] = field(default_factory=lambda: DEFAULT_MIN15_CANDIDATES.copy())
    target: str = "price_DK1"
    time_col: str = "HourUTC"
    transfer_start: str = "2025-10-01 00:00:00"
    split_ratio: float = 0.85
    test_split_method: str = "ratio"
    test_period_days: int = 30
    n_splits: int = 5
    use_hourly_baseline_as_residual_feature: bool = True
    champion_prediction_column: str = "TL_Residual_Average"
    battery: BatterySettings = field(default_factory=BatterySettings)
    weekly_band: WeeklyBandSettings = field(default_factory=WeeklyBandSettings)
    forecast_edge: ForecastEdgeSettings = field(default_factory=ForecastEdgeSettings)
    volatility_filter: VolatilityFilterSettings = field(default_factory=VolatilityFilterSettings)
    mean_reversion: MeanReversionSettings = field(default_factory=MeanReversionSettings)
    momentum: MomentumSettings = field(default_factory=MomentumSettings)
    momentum_spread: MomentumSpreadSettings = field(default_factory=MomentumSpreadSettings)
    channel_breakout: ChannelBreakoutSettings = field(default_factory=ChannelBreakoutSettings)
    daily_spread: DailySpreadSettings = field(default_factory=DailySpreadSettings)
    ensemble_agreement: EnsembleAgreementSettings = field(default_factory=EnsembleAgreementSettings)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ProjectConfig":
        payload = dict(data)
        if "battery" in payload and isinstance(payload["battery"], dict):
            payload["battery"] = BatterySettings(**payload["battery"])
        if "weekly_band" in payload and isinstance(payload["weekly_band"], dict):
            payload["weekly_band"] = WeeklyBandSettings(**payload["weekly_band"])
        if "forecast_edge" in payload and isinstance(payload["forecast_edge"], dict):
            payload["forecast_edge"] = ForecastEdgeSettings(**payload["forecast_edge"])
        if "volatility_filter" in payload and isinstance(payload["volatility_filter"], dict):
            payload["volatility_filter"] = VolatilityFilterSettings(**payload["volatility_filter"])
        if "mean_reversion" in payload and isinstance(payload["mean_reversion"], dict):
            payload["mean_reversion"] = MeanReversionSettings(**payload["mean_reversion"])
        if "momentum" in payload and isinstance(payload["momentum"], dict):
            payload["momentum"] = MomentumSettings(**payload["momentum"])
        if "momentum_spread" in payload and isinstance(payload["momentum_spread"], dict):
            payload["momentum_spread"] = MomentumSpreadSettings(**payload["momentum_spread"])
        if "channel_breakout" in payload and isinstance(payload["channel_breakout"], dict):
            payload["channel_breakout"] = ChannelBreakoutSettings(**payload["channel_breakout"])
        if "daily_spread" in payload and isinstance(payload["daily_spread"], dict):
            payload["daily_spread"] = DailySpreadSettings(**payload["daily_spread"])
        if "ensemble_agreement" in payload and isinstance(payload["ensemble_agreement"], dict):
            payload["ensemble_agreement"] = EnsembleAgreementSettings(**payload["ensemble_agreement"])
        for key in ["data_dir", "reference_results_dir", "output_dir"]:
            if key in payload:
                payload[key] = Path(payload[key])
        return cls(**payload)

    def with_overrides(self, **kwargs: Any) -> "ProjectConfig":
        normalized = dict(kwargs)
        for key in ["data_dir", "reference_results_dir", "output_dir"]:
            if key in normalized and normalized[key] is not None:
                normalized[key] = Path(normalized[key])
        return replace(self, **{k: v for k, v in normalized.items() if v is not None})


def load_config(path: str | Path | None = None) -> ProjectConfig:
    if path is None:
        return ProjectConfig()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return ProjectConfig.from_mapping(json.load(handle))


def write_config(config: ProjectConfig, path: str | Path) -> None:
    payload = {
        "project_name": config.project_name,
        "data_dir": str(config.data_dir),
        "reference_results_dir": str(config.reference_results_dir),
        "output_dir": str(config.output_dir),
        "hourly_candidates": config.hourly_candidates,
        "min15_candidates": config.min15_candidates,
        "target": config.target,
        "time_col": config.time_col,
        "transfer_start": config.transfer_start,
        "split_ratio": config.split_ratio,
        "test_split_method": config.test_split_method,
        "test_period_days": config.test_period_days,
        "n_splits": config.n_splits,
        "use_hourly_baseline_as_residual_feature": config.use_hourly_baseline_as_residual_feature,
        "champion_prediction_column": config.champion_prediction_column,
        "battery": config.battery.__dict__,
        "weekly_band": config.weekly_band.__dict__,
        "forecast_edge": config.forecast_edge.__dict__,
        "volatility_filter": config.volatility_filter.__dict__,
        "mean_reversion": config.mean_reversion.__dict__,
        "momentum": config.momentum.__dict__,
        "momentum_spread": config.momentum_spread.__dict__,
        "channel_breakout": config.channel_breakout.__dict__,
        "daily_spread": config.daily_spread.__dict__,
        "ensemble_agreement": config.ensemble_agreement.__dict__,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
