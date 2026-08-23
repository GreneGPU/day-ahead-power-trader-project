from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from intraday_power_quant.dashboard import (
    FORECAST_CANDIDATES,
    METRICS_CANDIDATES,
    _read_table,
    find_first_existing,
    normalize_forecast_columns,
)
from intraday_power_quant.evaluation import rank_models
from intraday_power_quant.optimization import best_parameter_rows, run_strategy_parameter_sweep, simulate_strategy_from_settings
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
    STRATEGY_DESCRIPTIONS,
    VolatilityFilterConfig,
    WeeklyBandConfig,
    run_strategy_suite,
    summarize_strategy_result,
)


DEFAULT_BATTERY = BatteryConfig()
DEFAULT_WEEKLY = WeeklyBandConfig()
DEFAULT_FORECAST_EDGE = ForecastEdgeConfig()
DEFAULT_VOLATILITY_FILTER = VolatilityFilterConfig()
DEFAULT_MEAN_REVERSION = MeanReversionConfig()
DEFAULT_MOMENTUM = MomentumConfig()
DEFAULT_MOMENTUM_SPREAD = MomentumSpreadConfig()
DEFAULT_BREAKOUT = ChannelBreakoutConfig()
DEFAULT_DAILY_SPREAD = DailySpreadConfig()
DEFAULT_ENSEMBLE = EnsembleAgreementConfig()
DEFAULT_RESULTS_DIR = "outputs/model_run_last30" if Path("outputs/model_run_last30").exists() else "outputs/model_run"


st.set_page_config(page_title="Day-Ahead Power Trader Project", layout="wide")


@st.cache_data(show_spinner=False)
def load_results(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(results_dir)
    forecasts = normalize_forecast_columns(_read_table(find_first_existing(root, FORECAST_CANDIDATES)))
    metrics = _read_table(find_first_existing(root, METRICS_CANDIDATES))
    if "MAE" in metrics.columns:
        metrics = rank_models(metrics, "MAE")
    return forecasts, metrics


@st.cache_data(show_spinner=False)
def research_results(forecasts: pd.DataFrame, battery_payload: dict[str, float | None]) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
]:
    battery = BatteryConfig(**battery_payload)
    sweep = run_strategy_parameter_sweep(
        forecasts,
        battery_config=battery,
        forecast_col="Prediction",
    )
    cash_best = best_parameter_rows(sweep, ranking_metric="Cashflow")
    risk_best = best_parameter_rows(sweep, ranking_metric="Risk_Adjusted_Score")
    sharpe_best = best_parameter_rows(sweep, ranking_metric="Daily_Sharpe")
    if cash_best.empty:
        return cash_best, risk_best, sharpe_best, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()

    best_row = cash_best.iloc[0]
    best_sim, _ = simulate_strategy_from_settings(
        str(best_row["Strategy"]),
        forecasts,
        str(best_row["Settings_JSON"]),
        battery,
        forecast_col="Prediction",
    )
    walk_forward = walk_forward_strategy_optimization(
        forecasts,
        battery_config=battery,
        train_days=14,
        test_days=7,
        step_days=7,
        ranking_metric="Risk_Adjusted_Score",
        forecast_col="Prediction",
    )
    cost_stress = execution_cost_stress_table(
        forecasts,
        str(best_row["Strategy"]),
        str(best_row["Settings_JSON"]),
        battery,
        forecast_col="Prediction",
    )
    robustness = daily_spread_robustness_grid(
        forecasts,
        battery_config=battery,
        min_daily_spread=20,
        forecast_col="Prediction",
    )
    regimes = regime_performance_table(forecasts, best_sim)
    uncertainty = forecast_uncertainty_summary(forecasts)
    decision = latest_decision_table(
        forecasts,
        str(best_row["Strategy"]),
        str(best_row["Settings_JSON"]),
        battery,
        forecast_col="Prediction",
    )
    return cash_best, risk_best, sharpe_best, walk_forward, cost_stress, robustness, regimes, uncertainty, decision


st.title("Day-Ahead Power Trader Project")
st.caption("15-minute day-ahead forecast comparison, transfer-learning metrics, and flexibility simulation.")

with st.sidebar:
    results_dir = st.text_input("Results directory", value=DEFAULT_RESULTS_DIR)
    st.divider()
    strategy = st.selectbox(
        "Strategy",
        list(STRATEGY_DESCRIPTIONS),
    )
    st.caption(STRATEGY_DESCRIPTIONS[strategy])
    st.divider()
    st.subheader("Battery")
    capacity = st.number_input("Capacity MWh", min_value=1.0, value=100.0, step=10.0)
    power = st.number_input("Power MW", min_value=1.0, value=25.0, step=5.0)
    initial_soc = st.number_input("Initial SOC MWh", min_value=0.0, value=50.0, step=10.0)
    charge_efficiency = st.slider("Charge efficiency", 0.50, 1.00, 0.95, 0.01)
    discharge_efficiency = st.slider("Discharge efficiency", 0.50, 1.00, 0.95, 0.01)
    fee = st.number_input("Fee per MWh", min_value=0.0, value=0.0, step=1.0)
    st.divider()
    st.subheader("Strategy parameters")

    low_q = DEFAULT_BATTERY.low_quantile
    high_q = DEFAULT_BATTERY.high_quantile
    weekly_cfg = DEFAULT_WEEKLY
    edge_cfg = DEFAULT_FORECAST_EDGE
    volatility_cfg = DEFAULT_VOLATILITY_FILTER
    mean_cfg = DEFAULT_MEAN_REVERSION
    momentum_cfg = DEFAULT_MOMENTUM
    momentum_spread_cfg = DEFAULT_MOMENTUM_SPREAD
    breakout_cfg = DEFAULT_BREAKOUT
    daily_cfg = DEFAULT_DAILY_SPREAD
    ensemble_cfg = DEFAULT_ENSEMBLE
    selected_parameters: dict[str, float | str] = {}

    if strategy == "Forecast quantile":
        low_q = st.slider("Charge below forecast quantile", 0.0, 0.9, DEFAULT_BATTERY.low_quantile, 0.05)
        high_q_min = max(0.1, min(round(low_q + 0.05, 2), 1.0))
        high_q = st.slider(
            "Discharge above forecast quantile",
            high_q_min,
            1.0,
            max(DEFAULT_BATTERY.high_quantile, high_q_min),
            0.05,
        )
        selected_parameters = {
            "Charge below quantile": low_q,
            "Discharge above quantile": high_q,
        }
        st.caption("Higher separation means fewer trades; narrower separation means more cycling.")
    elif strategy == "Weekly average band":
        weekly_cfg = WeeklyBandConfig(
            window_days=st.number_input("Moving average days", min_value=1.0, value=7.0, step=1.0),
            band=st.number_input("Buy below / sell above by X, DKK/MWh", min_value=0.0, value=20.0, step=5.0),
            min_history_days=st.number_input("Minimum history days", min_value=0.25, value=1.0, step=0.25),
        )
        selected_parameters = {
            "Moving average days": weekly_cfg.window_days,
            "Buy when forecast <= moving average - X": weekly_cfg.band,
            "Sell when forecast >= moving average + X": weekly_cfg.band,
            "Minimum history days": weekly_cfg.min_history_days,
        }
        st.caption("Example: 7 days and X=20 buys below the last-week moving average minus 20 and sells above plus 20.")
    elif strategy == "Forecast edge":
        edge_cfg = ForecastEdgeConfig(
            threshold=st.number_input("Edge threshold, DKK/MWh", min_value=0.0, value=5.0, step=1.0),
            reference_col="Hourly_Baseline",
        )
        selected_parameters = {
            "Reference forecast": edge_cfg.reference_col,
            "Buy when 15-min forecast <= reference - X": edge_cfg.threshold,
            "Sell when 15-min forecast >= reference + X": edge_cfg.threshold,
        }
        st.caption("This directly tests whether the 15-minute transfer forecast adds tradable edge versus the hourly baseline.")
    elif strategy == "Volatility filtered average":
        volatility_cfg = VolatilityFilterConfig(
            average_window_days=st.number_input(
                "Moving average days",
                min_value=1.0,
                value=DEFAULT_VOLATILITY_FILTER.average_window_days,
                step=1.0,
            ),
            volatility_window_days=st.number_input(
                "Volatility std days",
                min_value=1.0,
                value=DEFAULT_VOLATILITY_FILTER.volatility_window_days,
                step=1.0,
            ),
            min_volatility=st.number_input(
                "Minimum rolling std, DKK/MWh",
                min_value=0.0,
                value=DEFAULT_VOLATILITY_FILTER.min_volatility,
                step=5.0,
            ),
            price_band=st.number_input(
                "Buy below / sell above average by X, DKK/MWh",
                min_value=0.0,
                value=DEFAULT_VOLATILITY_FILTER.price_band,
                step=5.0,
            ),
            min_history_days=st.number_input(
                "Minimum history days",
                min_value=0.25,
                value=DEFAULT_VOLATILITY_FILTER.min_history_days,
                step=0.25,
            ),
        )
        selected_parameters = {
            "Moving average days": volatility_cfg.average_window_days,
            "Volatility std days": volatility_cfg.volatility_window_days,
            "Minimum rolling std": volatility_cfg.min_volatility,
            "Buy when forecast <= average - X": volatility_cfg.price_band,
            "Sell when forecast >= average + X": volatility_cfg.price_band,
            "Minimum history days": volatility_cfg.min_history_days,
        }
        st.caption("This only trades when rolling forecast volatility is high enough.")
    elif strategy == "Mean reversion":
        mean_cfg = MeanReversionConfig(
            window_days=st.number_input("Rolling mean/std days", min_value=1.0, value=7.0, step=1.0),
            entry_z=st.number_input("Entry z-score", min_value=0.1, value=1.0, step=0.1),
            min_history_days=st.number_input("Minimum history days", min_value=0.25, value=1.0, step=0.25),
        )
        selected_parameters = {
            "Rolling mean/std days": mean_cfg.window_days,
            "Entry z-score": mean_cfg.entry_z,
            "Minimum history days": mean_cfg.min_history_days,
        }
        st.caption("Higher z-score requires more extreme forecasts before dispatch.")
    elif strategy == "Momentum":
        momentum_cfg = MomentumConfig(
            lookback_hours=st.number_input("Momentum lookback hours", min_value=0.25, value=6.0, step=0.25),
            threshold=st.number_input("Momentum trigger, DKK/MWh", min_value=0.0, value=5.0, step=1.0),
            smoothing_hours=st.number_input("Forecast smoothing hours", min_value=0.0, value=1.0, step=0.25),
        )
        selected_parameters = {
            "Momentum lookback hours": momentum_cfg.lookback_hours,
            "Momentum trigger DKK/MWh": momentum_cfg.threshold,
            "Forecast smoothing hours": momentum_cfg.smoothing_hours,
        }
        st.caption("Positive momentum discharges; negative momentum charges. A larger trigger trades less often.")
    elif strategy == "Momentum spread":
        momentum_spread_rank_mode = st.radio(
            "Spread rank rule",
            ["Percent rank", "Absolute rank"],
            horizontal=True,
        )
        if momentum_spread_rank_mode == "Percent rank":
            spread_low_q = st.slider(
                "Charge when daily rank <= percentile",
                0.0,
                0.9,
                DEFAULT_MOMENTUM_SPREAD.low_quantile,
                0.05,
            )
            spread_high_q_min = max(0.1, min(round(spread_low_q + 0.05, 2), 1.0))
            spread_high_q = st.slider(
                "Discharge when daily rank >= percentile",
                spread_high_q_min,
                1.0,
                max(DEFAULT_MOMENTUM_SPREAD.high_quantile, spread_high_q_min),
                0.05,
            )
            spread_rank_kwargs = {
                "low_quantile": spread_low_q,
                "high_quantile": spread_high_q,
                "rank_mode": "percentile",
            }
            selected_parameters = {
                "Spread rank rule": "Percent rank",
                "Charge when rank percentile <=": spread_low_q,
                "Discharge when rank percentile >=": spread_high_q,
            }
        else:
            spread_charge_rank = st.number_input(
                "Charge when daily rank <=",
                min_value=1,
                max_value=199,
                value=DEFAULT_MOMENTUM_SPREAD.charge_rank,
                step=1,
            )
            spread_discharge_rank = st.number_input(
                "Discharge when daily rank >=",
                min_value=spread_charge_rank + 1,
                max_value=200,
                value=max(DEFAULT_MOMENTUM_SPREAD.discharge_rank, spread_charge_rank + 1),
                step=1,
            )
            spread_rank_kwargs = {
                "rank_mode": "absolute",
                "charge_rank": int(spread_charge_rank),
                "discharge_rank": int(spread_discharge_rank),
            }
            selected_parameters = {
                "Spread rank rule": "Absolute rank",
                "Charge when daily rank <=": int(spread_charge_rank),
                "Discharge when daily rank >=": int(spread_discharge_rank),
            }

        momentum_spread_cfg = MomentumSpreadConfig(
            **spread_rank_kwargs,
            min_daily_spread=st.number_input(
                "Minimum daily forecast spread, DKK/MWh",
                min_value=0.0,
                value=DEFAULT_MOMENTUM_SPREAD.min_daily_spread,
                step=5.0,
            ),
            lookback_hours=st.number_input(
                "Momentum lookback hours",
                min_value=0.25,
                value=DEFAULT_MOMENTUM_SPREAD.lookback_hours,
                step=0.25,
            ),
            momentum_threshold=st.number_input(
                "Momentum trigger, DKK/MWh",
                min_value=0.0,
                value=DEFAULT_MOMENTUM_SPREAD.momentum_threshold,
                step=1.0,
            ),
            smoothing_hours=st.number_input(
                "Forecast smoothing hours",
                min_value=0.0,
                value=DEFAULT_MOMENTUM_SPREAD.smoothing_hours,
                step=0.25,
            ),
        )
        selected_parameters.update(
            {
                "Minimum daily forecast spread": momentum_spread_cfg.min_daily_spread,
                "Momentum lookback hours": momentum_spread_cfg.lookback_hours,
                "Momentum trigger DKK/MWh": momentum_spread_cfg.momentum_threshold,
                "Forecast smoothing hours": momentum_spread_cfg.smoothing_hours,
            }
        )
        st.caption("Requires daily spread and momentum to agree before dispatching.")
    elif strategy == "Channel breakout":
        breakout_cfg = ChannelBreakoutConfig(
            window_days=st.number_input("Rolling channel days", min_value=1.0, value=3.0, step=1.0),
            buffer=st.number_input("Breakout buffer, DKK/MWh", min_value=0.0, value=0.0, step=5.0),
            min_history_days=st.number_input("Minimum history days", min_value=0.25, value=1.0, step=0.25),
        )
        selected_parameters = {
            "Rolling channel days": breakout_cfg.window_days,
            "Breakout buffer": breakout_cfg.buffer,
            "Minimum history days": breakout_cfg.min_history_days,
        }
        st.caption("A positive buffer filters out small breaks beyond the recent forecast range.")
    elif strategy == "Daily spread rank":
        daily_rank_mode_label = st.radio(
            "Daily rank rule",
            ["Percent rank", "Absolute rank"],
            horizontal=True,
        )
        if daily_rank_mode_label == "Percent rank":
            daily_low_q = st.slider(
                "Charge when daily rank <= percentile",
                0.0,
                0.9,
                DEFAULT_DAILY_SPREAD.low_quantile,
                0.05,
            )
            daily_high_q_min = max(0.1, min(round(daily_low_q + 0.05, 2), 1.0))
            daily_high_q = st.slider(
                "Discharge when daily rank >= percentile",
                daily_high_q_min,
                1.0,
                max(DEFAULT_DAILY_SPREAD.high_quantile, daily_high_q_min),
                0.05,
            )
            daily_cfg = DailySpreadConfig(
                low_quantile=daily_low_q,
                high_quantile=daily_high_q,
                rank_mode="percentile",
                min_daily_spread=st.number_input(
                    "Minimum daily forecast spread, DKK/MWh",
                    min_value=0.0,
                    value=DEFAULT_DAILY_SPREAD.min_daily_spread,
                    step=5.0,
                ),
            )
            selected_parameters = {
                "Daily rank rule": "Percent rank",
                "Charge when rank percentile <=": daily_cfg.low_quantile,
                "Discharge when rank percentile >=": daily_cfg.high_quantile,
                "Minimum daily forecast spread": daily_cfg.min_daily_spread,
            }
        else:
            charge_rank = st.number_input(
                "Charge when daily rank <=",
                min_value=1,
                max_value=199,
                value=DEFAULT_DAILY_SPREAD.charge_rank,
                step=1,
            )
            discharge_rank = st.number_input(
                "Discharge when daily rank >=",
                min_value=charge_rank + 1,
                max_value=200,
                value=max(DEFAULT_DAILY_SPREAD.discharge_rank, charge_rank + 1),
                step=1,
            )
            daily_cfg = DailySpreadConfig(
                rank_mode="absolute",
                charge_rank=int(charge_rank),
                discharge_rank=int(discharge_rank),
                min_daily_spread=st.number_input(
                    "Minimum daily forecast spread, DKK/MWh",
                    min_value=0.0,
                    value=DEFAULT_DAILY_SPREAD.min_daily_spread,
                    step=5.0,
                ),
            )
            selected_parameters = {
                "Daily rank rule": "Absolute rank",
                "Charge when daily rank <=": daily_cfg.charge_rank,
                "Discharge when daily rank >=": daily_cfg.discharge_rank,
                "Minimum daily forecast spread": daily_cfg.min_daily_spread,
            }
        st.caption("Daily rank 1 is the cheapest forecast interval of the day; larger ranks are pricier.")
    elif strategy == "Ensemble agreement":
        ensemble_low_q = st.slider(
            "Cheap threshold quantile",
            0.0,
            0.9,
            DEFAULT_ENSEMBLE.low_quantile,
            0.05,
        )
        ensemble_high_q_min = max(0.1, min(round(ensemble_low_q + 0.05, 2), 1.0))
        ensemble_high_q = st.slider(
            "Expensive threshold quantile",
            ensemble_high_q_min,
            1.0,
            max(DEFAULT_ENSEMBLE.high_quantile, ensemble_high_q_min),
            0.05,
        )
        min_agreement = st.slider(
            "Minimum model agreement",
            0.20,
            1.00,
            DEFAULT_ENSEMBLE.min_agreement,
            0.05,
        )
        limit_spread = st.checkbox("Limit model disagreement")
        max_model_spread = None
        if limit_spread:
            max_model_spread = st.number_input("Max model spread, DKK/MWh", min_value=0.0, value=20.0, step=5.0)
        ensemble_cfg = EnsembleAgreementConfig(
            low_quantile=ensemble_low_q,
            high_quantile=ensemble_high_q,
            min_agreement=min_agreement,
            max_model_spread=max_model_spread,
        )
        selected_parameters = {
            "Cheap threshold quantile": ensemble_cfg.low_quantile,
            "Expensive threshold quantile": ensemble_cfg.high_quantile,
            "Minimum model agreement": ensemble_cfg.min_agreement,
            "Max model spread": "Off" if ensemble_cfg.max_model_spread is None else ensemble_cfg.max_model_spread,
        }
        st.caption("Uses the available TL ensemble members first; it falls back to direct models if TL columns are absent.")

try:
    forecasts, metrics = load_results(results_dir)
except Exception as exc:
    st.error(str(exc))
    st.stop()

cfg = BatteryConfig(
    capacity_mwh=capacity,
    power_mw=power,
    initial_soc_mwh=initial_soc,
    charge_efficiency=charge_efficiency,
    discharge_efficiency=discharge_efficiency,
    low_quantile=low_q,
    high_quantile=high_q,
    fee_per_mwh=fee,
)
strategy_results = run_strategy_suite(
    forecasts,
    battery_config=cfg,
    weekly_band_config=weekly_cfg,
    forecast_edge_config=edge_cfg,
    volatility_filter_config=volatility_cfg,
    mean_reversion_config=mean_cfg,
    breakout_config=breakout_cfg,
    daily_spread_config=daily_cfg,
    momentum_config=momentum_cfg,
    momentum_spread_config=momentum_spread_cfg,
    ensemble_agreement_config=ensemble_cfg,
)
battery_df, battery_summary = strategy_results[strategy]

best_model = metrics.iloc[0]["Model"] if not metrics.empty and "Model" in metrics.columns else "n/a"
best_mae = metrics.iloc[0]["MAE"] if not metrics.empty and "MAE" in metrics.columns else None

top = st.columns(5)
top[0].metric("Best model", str(best_model), f"MAE {best_mae:.4f}" if best_mae is not None else None)
top[1].metric("Test intervals", f"{len(forecasts):,}")
top[2].metric("Strategy cashflow", f"{battery_summary['total_cashflow']:,.0f}")
top[3].metric("Max drawdown", f"{battery_summary['max_drawdown']:,.0f}")
top[4].metric("Market mode", "Day-ahead")

with st.expander("Selected strategy settings", expanded=True):
    st.write(STRATEGY_DESCRIPTIONS[strategy])
    settings_rows = [
        {"Setting": "Battery capacity MWh", "Value": capacity},
        {"Setting": "Battery power MW", "Value": power},
        {"Setting": "Initial SOC MWh", "Value": initial_soc},
        {"Setting": "Charge efficiency", "Value": charge_efficiency},
        {"Setting": "Discharge efficiency", "Value": discharge_efficiency},
        {"Setting": "Fee per MWh", "Value": fee},
    ]
    settings_rows.extend({"Setting": key, "Value": value} for key, value in selected_parameters.items())
    st.dataframe(pd.DataFrame(settings_rows), width="stretch", hide_index=True)

series_options = [
    column
    for column in ["Actual_Price", "Prediction", "Hourly_Baseline", "Direct_15min_Prediction"]
    if column in forecasts.columns
]
selected_series = st.multiselect("Price series", series_options, default=series_options[:3])

plot_df = forecasts[["HourUTC"] + selected_series].melt("HourUTC", var_name="Series", value_name="Price")
fig = px.line(plot_df, x="HourUTC", y="Price", color="Series", title="15-minute DK1 day-ahead prices and forecasts")
st.plotly_chart(fig, width="stretch")

cash_fig = px.line(
    battery_df,
    x="HourUTC",
    y="Cumulative_Cashflow",
    title=f"{strategy} cumulative cashflow",
)
st.plotly_chart(cash_fig, width="stretch")

st.subheader("Strategy comparison")
st.caption("The selected strategy uses the edited sidebar parameters; other strategies use their defaults.")
st.dataframe(
    pd.DataFrame(
        summarize_strategy_result(name, summary)
        for name, (_, summary) in strategy_results.items()
    ).sort_values("Cashflow", ascending=False),
    width="stretch",
    hide_index=True,
)

with st.expander("Quant research checks", expanded=True):
    st.caption("Day-ahead research checks: parameter optimization, risk ranking, walk-forward validation, cost stress, robustness, regimes, uncertainty, and latest decision.")
    run_optimization = st.checkbox("Run research checks", value=True)
    if run_optimization:
        with st.spinner("Testing strategy grids and validation windows..."):
            (
                optimization_table,
                risk_adjusted_table,
                sharpe_table,
                walk_forward_table,
                cost_stress_table,
                robustness_table,
                regime_table,
                uncertainty,
                decision_table,
            ) = research_results(forecasts, cfg.__dict__)
        if optimization_table.empty:
            st.warning("No optimization rows were produced for the available forecast columns.")
        else:
            best_row = optimization_table.iloc[0]
            risk_row = risk_adjusted_table.iloc[0]
            sharpe_row = sharpe_table.iloc[0]
            opt_top = st.columns(5)
            opt_top[0].metric("Best optimized strategy", str(best_row["Strategy"]))
            opt_top[1].metric("Best cashflow", f"{best_row['Cashflow']:,.0f}")
            opt_top[2].metric("Best risk-adjusted", str(risk_row["Strategy"]))
            opt_top[3].metric("Best daily Sharpe", f"{sharpe_row['Daily_Sharpe']:,.2f}")
            opt_top[4].metric("80% interval coverage", f"{uncertainty['Interval_80_Coverage']:.1%}")

            tabs = st.tabs([
                "Optimization",
                "Risk",
                "Sharpe",
                "Walk-forward",
                "Costs",
                "Robustness",
                "Regimes",
                "Uncertainty",
                "Decision",
            ])
            with tabs[0]:
                display_columns = [
                    "Rank",
                    "Strategy",
                    "Cashflow",
                    "Max_Drawdown",
                    "Risk_Adjusted_Score",
                    "Calmar_Ratio",
                    "Daily_Sharpe",
                    "Daily_Sortino",
                    "Active_Intervals",
                    "Evaluations",
                    "Settings",
                ]
                st.dataframe(optimization_table[display_columns], width="stretch", hide_index=True)
            with tabs[1]:
                risk_columns = [
                    "Rank",
                    "Strategy",
                    "Risk_Adjusted_Score",
                    "Cashflow",
                    "Max_Drawdown",
                    "Daily_Sharpe",
                    "Daily_Sortino",
                    "Profit_Factor",
                    "Win_Rate",
                    "Worst_Day",
                    "Settings",
                ]
                st.dataframe(risk_adjusted_table[risk_columns], width="stretch", hide_index=True)
            with tabs[2]:
                sharpe_columns = [
                    "Rank",
                    "Strategy",
                    "Daily_Sharpe",
                    "Daily_Sortino",
                    "Cashflow",
                    "Max_Drawdown",
                    "Profit_Factor",
                    "Settings",
                ]
                st.caption("Daily Sharpe is an annualized daily cashflow Sharpe proxy, not a live mark-to-market return Sharpe.")
                st.dataframe(sharpe_table[sharpe_columns], width="stretch", hide_index=True)
            with tabs[3]:
                st.dataframe(walk_forward_table, width="stretch", hide_index=True)
            with tabs[4]:
                st.dataframe(cost_stress_table, width="stretch", hide_index=True)
                if not cost_stress_table.empty:
                    cost_fig = px.line(cost_stress_table, x="Fee_Per_MWh", y="Cashflow", markers=True, title="Cashflow sensitivity to fee/slippage proxy")
                    st.plotly_chart(cost_fig, width="stretch")
            with tabs[5]:
                st.dataframe(robustness_table.head(20), width="stretch", hide_index=True)
                if not robustness_table.empty:
                    heatmap = robustness_table.pivot_table(
                        index="Low_Quantile",
                        columns="High_Quantile",
                        values="Cashflow",
                        aggfunc="mean",
                    )
                    heat_fig = px.imshow(
                        heatmap,
                        text_auto=".0f",
                        aspect="auto",
                        title="Daily spread robustness heatmap: cashflow by charge/discharge percentile",
                    )
                    st.plotly_chart(heat_fig, width="stretch")
            with tabs[6]:
                st.dataframe(regime_table, width="stretch", hide_index=True)
            with tabs[7]:
                uncertainty_rows = pd.DataFrame(
                    [{"Metric": key, "Value": value} for key, value in uncertainty.items()]
                )
                st.dataframe(uncertainty_rows, width="stretch", hide_index=True)
            with tabs[8]:
                st.dataframe(decision_table, width="stretch", hide_index=True)

st.subheader("Model ranking")
metric_columns = [column for column in ["Rank", "Model", "Model_Type", "MAE", "RMSE", "sMAPE", "R2"] if column in metrics.columns]
st.dataframe(metrics[metric_columns], width="stretch", hide_index=True)

st.subheader("Dispatch log")
st.dataframe(battery_df.tail(200), width="stretch", hide_index=True)
