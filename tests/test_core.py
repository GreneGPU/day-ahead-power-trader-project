from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intraday_power_quant.evaluation import calculate_metrics
from intraday_power_quant.optimization import optimize_strategy_suite, run_strategy_parameter_sweep
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
from intraday_power_quant.validation import make_train_test_split_mask


def test_metrics_are_exact_for_perfect_forecast():
    metrics = calculate_metrics([10, 20, 30], [10, 20, 30])
    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["R2"] == 1


def test_battery_simulation_returns_one_row_per_interval():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=8, freq="15min"),
            "Actual_Price": [10, 12, 15, 80, 90, 70, 20, 18],
            "Prediction": [10, 11, 16, 78, 88, 72, 22, 19],
        }
    )
    sim, summary = simulate_battery_arbitrage(frame, BatteryConfig(capacity_mwh=10, power_mw=5))
    assert len(sim) == 8
    assert summary["trades"] > 0


def test_weekly_average_band_strategy_adds_thresholds():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=16, freq="15min"),
            "Actual_Price": [10, 12, 15, 80, 90, 70, 20, 18, 10, 12, 15, 80, 90, 70, 20, 18],
            "Prediction": [10, 11, 16, 78, 88, 72, 22, 19, 10, 11, 16, 78, 88, 72, 22, 19],
        }
    )
    sim, summary = simulate_weekly_average_band_arbitrage(
        frame,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        WeeklyBandConfig(window_days=1, band=10, min_history_days=0.25),
    )
    assert "Buy_Threshold" in sim.columns
    assert "Sell_Threshold" in sim.columns
    assert summary["trades"] > 0


def test_strategy_suite_contains_common_strategy_families():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=32, freq="15min"),
            "Actual_Price": list(range(32)),
            "Prediction": list(range(16)) + list(range(16, 0, -1)),
            "Hourly_Baseline": [16] * 32,
        }
    )
    suite = run_strategy_suite(
        frame,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        mean_reversion_config=MeanReversionConfig(window_days=1, entry_z=0.5, min_history_days=0.25),
        momentum_config=MomentumConfig(lookback_hours=1, threshold=1, smoothing_hours=0),
        breakout_config=ChannelBreakoutConfig(window_days=1, buffer=0, min_history_days=0.25),
        daily_spread_config=DailySpreadConfig(low_quantile=0.25, high_quantile=0.75),
    )
    assert set(suite) == {
        "Forecast quantile",
        "Weekly average band",
        "Forecast edge",
        "Volatility filtered average",
        "Ensemble agreement",
        "Mean reversion",
        "Momentum",
        "Momentum spread",
        "Channel breakout",
        "Daily spread rank",
    }
    assert "Forecast_Z" in simulate_mean_reversion_arbitrage(frame)[0].columns
    assert "Forecast_Momentum_DKK" in simulate_momentum_arbitrage(frame)[0].columns
    assert "Spread_Charge_Signal" in simulate_momentum_spread_arbitrage(frame)[0].columns
    assert "Forecast_Edge_DKK" in simulate_forecast_edge_arbitrage(frame)[0].columns
    assert "Rolling_Forecast_Std" in simulate_volatility_filtered_average_arbitrage(frame)[0].columns
    assert "Agreement_Charge_Share" in simulate_ensemble_agreement_arbitrage(frame)[0].columns
    assert "Rolling_Low_Forecast" in simulate_channel_breakout_arbitrage(frame)[0].columns
    assert "Daily_Forecast_Rank" in simulate_daily_spread_rank_arbitrage(frame)[0].columns


def test_optimizer_returns_best_setting_per_strategy():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=40, freq="15min"),
            "Actual_Price": [20, 18, 15, 12, 25, 35, 55, 80, 76, 70] * 4,
            "Prediction": [19, 17, 14, 13, 24, 32, 58, 82, 75, 69] * 4,
            "Hourly_Baseline": [30] * 40,
            "TL_Residual_XGB": [18, 17, 15, 12, 25, 33, 59, 81, 74, 70] * 4,
            "TL_Residual_LGBM": [20, 16, 14, 14, 23, 31, 57, 83, 76, 68] * 4,
            "TL_Residual_CAT": [19, 18, 13, 13, 24, 32, 58, 82, 75, 69] * 4,
        }
    )
    sweep = run_strategy_parameter_sweep(frame, BatteryConfig(capacity_mwh=10, power_mw=5))
    optimized = optimize_strategy_suite(frame, BatteryConfig(capacity_mwh=10, power_mw=5))
    assert not sweep.empty
    assert set(["Rank", "Strategy", "Cashflow", "Evaluations", "Settings"]).issubset(optimized.columns)
    assert optimized["Strategy"].is_unique
    assert len(optimized) == 10
    assert optimized.iloc[0]["Cashflow"] >= optimized.iloc[-1]["Cashflow"]


def test_research_tables_support_day_ahead_validation_panels():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=10 * 96, freq="15min"),
            "Actual_Price": ([20, 18, 15, 12, 25, 35, 55, 80, 76, 70, 40, 25] * 80),
            "Prediction": ([19, 17, 14, 13, 24, 32, 58, 82, 75, 69, 41, 24] * 80),
            "Hourly_Baseline": [30] * (10 * 96),
        }
    )
    battery = BatteryConfig(capacity_mwh=10, power_mw=5)
    optimized = optimize_strategy_suite(frame, battery)
    best = optimized.iloc[0]
    cost = execution_cost_stress_table(frame, best["Strategy"], best["Settings_JSON"], battery)
    walk_forward = walk_forward_strategy_optimization(
        frame,
        battery,
        train_days=4,
        test_days=2,
        step_days=2,
    )
    robustness = daily_spread_robustness_grid(frame, battery)
    sim, _ = simulate_daily_spread_rank_arbitrage(frame, battery)
    regimes = regime_performance_table(frame, sim)
    uncertainty = forecast_uncertainty_summary(frame)
    decision = latest_decision_table(frame, best["Strategy"], best["Settings_JSON"], battery)
    assert not cost.empty
    assert not walk_forward.empty
    assert not robustness.empty
    assert not regimes.empty
    assert 0 <= uncertainty["Interval_80_Coverage"] <= 1
    assert decision.iloc[0]["Action"] in {"charge", "discharge", "hold", "risk-off"}


def test_daily_spread_supports_absolute_rank_thresholds():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=12, freq="15min"),
            "Actual_Price": list(range(12)),
            "Prediction": list(range(12)),
        }
    )
    sim, summary = simulate_daily_spread_rank_arbitrage(
        frame,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        DailySpreadConfig(rank_mode="absolute", charge_rank=2, discharge_rank=10),
    )
    assert "Daily_Forecast_Rank_Pct" in sim.columns
    assert summary["rank_mode"] == "absolute"
    assert summary["charge_rank"] == 2


def test_momentum_spread_requires_rank_and_momentum_confirmation():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=24, freq="15min"),
            "Actual_Price": [40, 35, 30, 25, 20, 15, 10, 12, 18, 25, 35, 50, 65, 80, 95, 90, 75, 60, 45, 30, 20, 15, 12, 10],
            "Prediction": [40, 35, 30, 25, 20, 15, 10, 12, 18, 25, 35, 50, 65, 80, 95, 90, 75, 60, 45, 30, 20, 15, 12, 10],
        }
    )
    sim, summary = simulate_momentum_spread_arbitrage(
        frame,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        MomentumSpreadConfig(
            rank_mode="absolute",
            charge_rank=8,
            discharge_rank=17,
            min_daily_spread=20,
            lookback_hours=1,
            momentum_threshold=1,
            smoothing_hours=0,
        ),
    )
    assert "Forecast_Momentum_DKK" in sim.columns
    assert "Spread_Discharge_Signal" in sim.columns
    assert summary["rank_mode"] == "absolute"
    assert summary["momentum_threshold"] == 1


def test_forecast_edge_strategy_uses_hourly_baseline_reference():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=6, freq="15min"),
            "Actual_Price": [10, 20, 30, 40, 50, 60],
            "Prediction": [5, 12, 30, 48, 70, 65],
            "Hourly_Baseline": [20, 20, 30, 40, 50, 50],
        }
    )
    sim, summary = simulate_forecast_edge_arbitrage(
        frame,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        ForecastEdgeConfig(threshold=10),
    )
    assert "Reference_Price" in sim.columns
    assert summary["threshold"] == 10
    assert set(sim["Action"]).issubset({"charge", "discharge", "hold"})


def test_volatility_filtered_average_requires_high_std():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=16, freq="15min"),
            "Actual_Price": [20, 25, 30, 35, 100, 95, 15, 10, 80, 85, 90, 12, 18, 22, 75, 78],
            "Prediction": [20, 25, 30, 35, 100, 95, 15, 10, 80, 85, 90, 12, 18, 22, 75, 78],
        }
    )
    sim, summary = simulate_volatility_filtered_average_arbitrage(
        frame,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        VolatilityFilterConfig(
            average_window_days=1,
            volatility_window_days=1,
            min_volatility=10,
            price_band=0,
            min_history_days=0.02,
        ),
    )
    assert "High_Volatility" in sim.columns
    assert "Rolling_Average_Forecast" in sim.columns
    assert summary["high_volatility_intervals"] > 0


def test_ensemble_agreement_strategy_counts_model_votes():
    frame = pd.DataFrame(
        {
            "HourUTC": pd.date_range("2026-01-01", periods=8, freq="15min"),
            "Actual_Price": [10, 12, 14, 70, 75, 80, 20, 18],
            "Prediction": [10, 12, 14, 70, 75, 80, 20, 18],
            "TL_Residual_XGB": [9, 11, 15, 72, 76, 79, 22, 19],
            "TL_Residual_LGBM": [10, 12, 14, 71, 77, 78, 21, 18],
            "TL_Residual_CAT": [8, 13, 16, 70, 78, 81, 20, 17],
        }
    )
    sim, summary = simulate_ensemble_agreement_arbitrage(
        frame,
        BatteryConfig(capacity_mwh=10, power_mw=5),
        EnsembleAgreementConfig(low_quantile=0.25, high_quantile=0.75, min_agreement=2 / 3),
    )
    assert "Agreement_Charge_Share" in sim.columns
    assert summary["model_count"] == 3


def test_last_n_days_split_uses_end_of_data():
    frame = pd.DataFrame({"HourUTC": pd.date_range("2026-01-01", periods=10 * 96, freq="15min")})
    train_mask, test_mask, split_time, label = make_train_test_split_mask(
        frame,
        "HourUTC",
        method="last_n_days",
        split_ratio=0.85,
        test_period_days=3,
    )
    assert label == "last_n_days:3"
    assert train_mask.sum() > 0
    assert test_mask.sum() > 0
    assert frame.loc[test_mask, "HourUTC"].min() >= split_time
