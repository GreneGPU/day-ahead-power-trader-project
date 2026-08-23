from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intraday_power_quant.data import rename_15min_columns_to_hourly_space
from intraday_power_quant.evaluation import calculate_15min_coverage, calculate_metrics
from intraday_power_quant.optimization import optimize_strategy_suite
from intraday_power_quant.research import (
    daily_spread_robustness_grid,
    execution_cost_stress_table,
    forecast_uncertainty_summary,
    latest_decision_table,
    regime_performance_table,
    walk_forward_strategy_optimization,
)
from intraday_power_quant.trading import (
    BatteryConfig,
    ChannelBreakoutConfig,
    DailySpreadConfig,
    EnsembleAgreementConfig,
    ForecastEdgeConfig,
    MeanReversionConfig,
    MomentumConfig,
    MomentumSpreadConfig,
    VolatilityFilterConfig,
    WeeklyBandConfig,
    forecast_spread_signals,
    run_strategy_suite,
    simulate_battery_arbitrage,
    simulate_channel_breakout_arbitrage,
    simulate_daily_spread_rank_arbitrage,
    simulate_ensemble_agreement_arbitrage,
    simulate_forecast_edge_arbitrage,
    simulate_mean_reversion_arbitrage,
    simulate_momentum_arbitrage,
    simulate_momentum_spread_arbitrage,
    simulate_volatility_filtered_average_arbitrage,
    simulate_weekly_average_band_arbitrage,
)
from intraday_power_quant.validation import (
    leakage_warnings,
    make_train_test_split_mask,
    make_walk_forward_windows,
)


def main() -> None:
    metrics = calculate_metrics([1, 2, 3], [1, 2, 4])
    assert round(metrics["MAE"], 6) == round(1 / 3, 6)

    coverage = calculate_15min_coverage(pd.date_range("2026-01-01", periods=4, freq="15min"))
    assert coverage["Test_Continuous_15min"] is True
    assert coverage["Test_Observed_Rows"] == 4

    mapped = rename_15min_columns_to_hourly_space(pd.DataFrame(columns=["price_lag_96", "roll_mean_672_safe"]))
    assert "price_lag_24" in mapped.columns
    assert "roll_mean_168_safe" in mapped.columns

    warnings = leakage_warnings(["price_DK1", "future_price", "price_DK1_lag_96"], "price_DK1", "HourUTC")
    assert len(warnings) >= 2

    rows = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=16, freq="15min"),
            "Actual_Price": [20, 18, 15, 12, 25, 35, 55, 80, 76, 70, 40, 25, 18, 14, 60, 88],
            "Prediction": [19, 17, 14, 13, 24, 32, 58, 82, 75, 69, 41, 24, 16, 15, 59, 85],
            "Hourly_Baseline": [30] * 16,
            "TL_Residual_XGB": [18, 17, 15, 12, 25, 33, 59, 81, 74, 70, 42, 25, 17, 16, 60, 84],
            "TL_Residual_LGBM": [20, 16, 14, 14, 23, 31, 57, 83, 76, 68, 40, 23, 15, 14, 58, 86],
            "TL_Residual_CAT": [19, 18, 13, 13, 24, 32, 58, 82, 75, 69, 41, 24, 16, 15, 59, 85],
        }
    )
    signals = forecast_spread_signals(rows, threshold=20)
    assert set(signals["Signal"]).issubset({"high-price", "low-price", "neutral"})

    sim, summary = simulate_battery_arbitrage(rows, BatteryConfig(capacity_mwh=10, power_mw=5))
    assert len(sim) == len(rows)
    assert "total_cashflow" in summary

    weekly_sim, weekly_summary = simulate_weekly_average_band_arbitrage(
        rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        WeeklyBandConfig(window_days=1, band=10, min_history_days=0.25),
    )
    assert len(weekly_sim) == len(rows)
    assert "Weekly_Average_Forecast" in weekly_sim.columns
    assert weekly_summary["band"] == 10

    edge_sim, edge_summary = simulate_forecast_edge_arbitrage(
        rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        ForecastEdgeConfig(threshold=10),
    )
    assert "Forecast_Edge_DKK" in edge_sim.columns
    assert edge_summary["reference_col"] == "Hourly_Baseline"

    volatility_sim, volatility_summary = simulate_volatility_filtered_average_arbitrage(
        rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        VolatilityFilterConfig(
            average_window_days=1,
            volatility_window_days=1,
            min_volatility=10,
            price_band=0,
            min_history_days=0.02,
        ),
    )
    assert "Rolling_Forecast_Std" in volatility_sim.columns
    assert volatility_summary["min_volatility"] == 10

    mean_sim, _ = simulate_mean_reversion_arbitrage(
        rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        MeanReversionConfig(window_days=1, entry_z=0.5, min_history_days=0.25),
    )
    assert "Forecast_Z" in mean_sim.columns

    momentum_sim, momentum_summary = simulate_momentum_arbitrage(
        rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        MomentumConfig(lookback_hours=1, threshold=1, smoothing_hours=0),
    )
    assert "Forecast_Momentum_DKK" in momentum_sim.columns
    assert momentum_summary["lookback_hours"] == 1

    momentum_spread_sim, momentum_spread_summary = simulate_momentum_spread_arbitrage(
        rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        MomentumSpreadConfig(
            rank_mode="absolute",
            charge_rank=2,
            discharge_rank=12,
            min_daily_spread=20,
            lookback_hours=1,
            momentum_threshold=1,
            smoothing_hours=0,
        ),
    )
    assert "Spread_Charge_Signal" in momentum_spread_sim.columns
    assert momentum_spread_summary["momentum_threshold"] == 1

    breakout_sim, _ = simulate_channel_breakout_arbitrage(
        rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        ChannelBreakoutConfig(window_days=1, buffer=0, min_history_days=0.25),
    )
    assert "Rolling_High_Forecast" in breakout_sim.columns

    daily_sim, _ = simulate_daily_spread_rank_arbitrage(
        rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        DailySpreadConfig(rank_mode="absolute", charge_rank=2, discharge_rank=12, min_daily_spread=20),
    )
    assert "Daily_Forecast_Rank_Pct" in daily_sim.columns
    assert "Daily_Forecast_Spread" in daily_sim.columns

    ensemble_sim, ensemble_summary = simulate_ensemble_agreement_arbitrage(
        rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        EnsembleAgreementConfig(low_quantile=0.25, high_quantile=0.75, min_agreement=2 / 3),
    )
    assert "Agreement_Charge_Share" in ensemble_sim.columns
    assert ensemble_summary["model_count"] >= 3

    suite = run_strategy_suite(rows, BatteryConfig(capacity_mwh=10, power_mw=5))
    assert len(suite) == 10

    optimized = optimize_strategy_suite(rows, BatteryConfig(capacity_mwh=10, power_mw=5))
    assert len(optimized) == 10
    assert "Settings" in optimized.columns

    best = optimized.iloc[0]
    cost_stress = execution_cost_stress_table(
        rows,
        best["Strategy"],
        best["Settings_JSON"],
        BatteryConfig(capacity_mwh=10, power_mw=5),
    )
    assert not cost_stress.empty
    uncertainty = forecast_uncertainty_summary(rows)
    assert 0 <= uncertainty["Interval_80_Coverage"] <= 1
    decision = latest_decision_table(
        rows,
        best["Strategy"],
        best["Settings_JSON"],
        BatteryConfig(capacity_mwh=10, power_mw=5),
    )
    assert len(decision) == 1
    regimes = regime_performance_table(rows, daily_sim)
    assert not regimes.empty
    robustness = daily_spread_robustness_grid(rows, BatteryConfig(capacity_mwh=10, power_mw=5))
    assert not robustness.empty

    wf_rows = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=8 * 96, freq="15min"),
            "Actual_Price": [20, 18, 15, 12, 25, 35, 55, 80, 76, 70, 40, 25] * 64,
            "Prediction": [19, 17, 14, 13, 24, 32, 58, 82, 75, 69, 41, 24] * 64,
            "Hourly_Baseline": [30] * (8 * 96),
        }
    )
    walk_forward = walk_forward_strategy_optimization(
        wf_rows,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        train_days=3,
        test_days=2,
        step_days=2,
    )
    assert not walk_forward.empty

    train_mask, test_mask, split_time, split_label = make_train_test_split_mask(
        rows,
        "HourUTC",
        method="last_n_days",
        split_ratio=0.85,
        test_period_days=0.05,
    )
    assert train_mask.any()
    assert test_mask.any()
    assert split_label.startswith("last_n_days")
    assert pd.Timestamp(split_time) < rows["HourUTC"].max()

    windows = make_walk_forward_windows(rows["HourUTC"], train_days=1, test_days=1, step_days=1)
    assert isinstance(windows, list)
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
