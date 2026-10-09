from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

from fastapi.testclient import TestClient

from api.index import app


client = TestClient(app)


def _records() -> list[dict[str, object]]:
    start = datetime(2026, 8, 23, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    for index in range(96):
        hour = index / 4
        forecast = 410 + 38 * math.sin((hour - 7) * math.pi / 12)
        rows.append(
            {
                "HourUTC": (start + timedelta(minutes=15 * index)).isoformat(),
                "Prediction": forecast,
                "Actual_Price": forecast + 5 * math.sin(index),
            }
        )
    return rows


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_daily_spread_simulation() -> None:
    response = client.post(
        "/api/simulate",
        json={"strategy": "Daily spread rank", "records": _records()},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["rows_processed"] == 96
    assert payload["summary"]["trades"] > 0


def test_simulation_rejects_missing_columns() -> None:
    response = client.post(
        "/api/simulate",
        json={"records": [{"HourUTC": "2026-08-23T00:00:00Z"}, {"HourUTC": "2026-08-23T00:15:00Z"}]},
    )
    assert response.status_code == 422
    assert "Missing required columns" in response.json()["detail"]


def test_deployment_results_use_real_predictions() -> None:
    response = client.get("/api/results")
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"]["rows"] == 10744  # walk-forward out-of-sample history
    assert len(payload["prices"]) == 10744
    assert len(payload["model_metrics"]) == 12  # 6 walk-forward rows (incl. the candidate rule) + 6 thesis benchmarks
    assert payload["model_metrics"][0]["Model"] == "Walk-forward champion"
    sarimax_tl = next(
        row
        for row in payload["model_metrics"]
        if row["Model"] == "SARIMAX — TL 15-minute EPF"
    )
    assert sarimax_tl["MAE"] == 18.66
    assert sarimax_tl["Number_Features"] == 76
    assert all(
        row["Number_Features"] == 75
        for row in payload["model_metrics"]
        if row["Model_Type"] == "Direct 15-minute EPF benchmark"
    )
    # First walk-forward interval, 13 Nov 2025 00:00 UTC: thesis price 25.370001 EUR, Energinet imbalance 284 DKK.
    assert payload["prices"][0]["HourUTC"].startswith("2025-11-13T00:00")
    assert payload["prices"][0]["Actual_Price"] == 25.370001
    assert math.isclose(payload["prices"][0]["Actual_Price_DKK"], 189.457796, abs_tol=1e-6)
    assert payload["prices"][0]["Imbalance_Price_DKK"] == 284.0
    assert payload["dataset"]["price_currency"] == "EUR/MWh for battery; DKK/MWh for Prop"


def test_strategy_comparison_uses_all_strategy_families() -> None:
    response = client.post(
        "/api/compare",
        json={
            "forecast_col": "Prediction",
            "days": 7,
            "battery": {
                "capacity_mwh": 100,
                "power_mw": 25,
                "initial_soc_mwh": 50,
                "charge_efficiency": 0.90**0.5,
                "discharge_efficiency": 0.90**0.5,
                "charge_fee_per_mwh": 115.41,
                "discharge_fee_per_mwh": 10.71,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"]["selected_rows"] > 600
    assert len(_benchmark_rows(payload)) == 16
    assert any(row["Strategy"] == "Predicted best hours" for row in payload["strategies"])
    assert any(row["Strategy"] == "Wind signal" for row in payload["strategies"])
    assert any(row["Strategy"] == "Rolling price optimizer" for row in payload["strategies"])
    assert payload["perfect_foresight_benchmark"]["Cashflow"] is not None
    assert payload["perfect_foresight_benchmark"]["Final_SOC_MWh"] == 0
    assert payload["best_strategy"] == payload["strategies"][0]["Strategy"]
    assert len(payload["best_strategy_series"]) == payload["dataset"]["selected_rows"]
    assert set(payload["strategy_series"]) == {
        row["Strategy"] for row in payload["strategies"]
    }
    assert all(
        len(series) == payload["dataset"]["selected_rows"]
        for series in payload["strategy_series"].values()
    )
    assert payload["strategies"][0]["Fee_Cost"] > 0
    assert math.isclose(payload["strategies"][0]["Round_Trip_Efficiency_Pct"], 90)
    daily_spread = next(row for row in payload["strategies"] if row["Strategy"] == "Daily spread rank")
    assert math.isclose(
        daily_spread["No_Fee_Potential_Cashflow"],
        daily_spread["Cashflow"] + daily_spread["Fee_Cost"],
        rel_tol=1e-9,
        abs_tol=1e-6,
    )
    assert daily_spread["No_Fee_Potential_Settings"]
    ensemble = next(row for row in payload["strategies"] if row["Strategy"] == "Ensemble agreement")
    assert "one compatible ensemble-output series" in ensemble["Description"]


def test_strategy_comparison_rejects_unknown_forecast() -> None:
    response = client.post("/api/compare", json={"forecast_col": "Unknown"})
    assert response.status_code == 422


def test_strategy_comparison_returns_requested_strategy_trade_log() -> None:
    response = client.post(
        "/api/compare",
        json={
            "forecast_col": "Prediction",
            "strategy": "Daily spread rank",
            "days": 7,
            "battery": {"capacity_mwh": 100, "power_mw": 25, "initial_soc_mwh": 50},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_strategy"] == "Daily spread rank"
    assert len(payload["selected_strategy_series"]) == payload["dataset"]["selected_rows"]
    trades = [
        row for row in payload["selected_strategy_series"] if row["Action"] in {"charge", "discharge"}
    ]
    selected_summary = next(
        row for row in payload["strategies"] if row["Strategy"] == "Daily spread rank"
    )
    assert len(trades) == selected_summary["Active_Intervals"]
    assert {
        "HourUTC",
        "Action",
        "Actual_Price",
        "Forecast_Price",
        "Dispatch_MW",
        "State_Of_Charge_MWh",
        "Cashflow",
        "Cumulative_Cashflow",
    }.issubset(trades[0])


def test_strategy_comparison_rejects_unknown_strategy() -> None:
    response = client.post("/api/compare", json={"strategy": "Unknown"})
    assert response.status_code == 422


def test_prop_comparison_returns_directional_accounting() -> None:
    response = client.post(
        "/api/compare",
        json={
            "forecast_col": "Prediction",
            "strategy": "Daily spread rank",
            "days": 7,
            "trading_setup": "prop",
            "prop": {
                "initial_capital_dkk": 100_000,
                "position_size_mwh": 10,
                "transaction_cost_dkk_per_mwh": 0.41,
                "max_daily_loss_dkk": 5_000,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["trading_setup"] == "prop"
    assert len(_benchmark_rows(payload)) == 12
    selected = next(
        row for row in payload["strategies"] if row["Strategy"] == "Daily spread rank"
    )
    assert selected["Return_Pct"] is not None
    assert selected["Degradation_Cost"] == 0
    day_ends = [
        row for row in payload["selected_strategy_series"] if row["Is_Day_End"]
    ]
    assert day_ends
    assert all(row["Position_After_Settlement"] == 0 for row in day_ends)
    assert any(row["EOD_Imbalance_Settlement"] for row in day_ends)
    assert any(
        row["Action"] in {"long", "short"} for row in payload["selected_strategy_series"]
    )
    assert {
        "Imbalance_Price_DKK",
        "Imbalance_Spread_DKK",
        "Position_After_Settlement",
        "EOD_Imbalance_Settlement",
        "Settlement_Basis",
    }.issubset(payload["selected_strategy_series"][0])
    assert payload["perfect_foresight_benchmark"]["label"] == (
        "Perfect-foresight directional ceiling"
    )


def test_prop_comparison_rejects_battery_only_optimizer() -> None:
    response = client.post(
        "/api/compare",
        json={"trading_setup": "prop", "strategy": "Rolling price optimizer"},
    )
    assert response.status_code == 422
    assert "battery-only" in response.json()["detail"]


def test_imbalance_comparison_returns_spread_accounting() -> None:
    response = client.post(
        "/api/compare",
        json={
            "forecast_col": "Prediction",
            "strategy": "Daily spread rank",
            "days": 7,
            "trading_setup": "imbalance",
            "prop": {
                "initial_capital_dkk": 100_000,
                "position_size_mwh": 10,
                "transaction_cost_dkk_per_mwh": 0.41,
                "max_daily_loss_dkk": 5_000,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["trading_setup"] == "imbalance"
    assert len(_benchmark_rows(payload)) == 12
    assert payload["perfect_foresight_benchmark"]["label"] == (
        "Perfect-foresight unclosed-position settlement ceiling"
    )
    selected = payload["selected_strategy_series"]
    assert any(row["Action"] in {"long-spread", "short-spread"} for row in selected)
    assert {
        "Day_Ahead_Price_DKK",
        "Imbalance_Price_DKK",
        "Imbalance_Spread_DKK",
        "Dominating_Direction",
        "Transaction_Cost",
    }.issubset(selected[0])


def _benchmark_rows(payload: dict) -> list[dict]:
    """Tuned benchmark strategies, without the Strategy Lab presets that run alongside them."""
    return [row for row in payload["strategies"] if row["Source"] == "Benchmark"]


def test_saved_comparisons_are_available_without_recalculation() -> None:
    for setup, expected_count, lab_count in (("battery", 16, 10), ("prop", 12, 12), ("imbalance", 12, 10)):
        response = client.get(f"/api/saved-comparison/{setup}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["saved_result"] is True
        assert payload["trading_setup"] == setup
        assert payload["evaluation"]["test_days"] == 10
        assert payload["evaluation"]["daily_observations"] == 20  # 2 walk-forward blocks of 10 days
        assert len(_benchmark_rows(payload)) == expected_count
        assert len(payload["strategies"]) == expected_count + lab_count  # plus the Strategy Lab presets
        assert payload["request"]["days"] == 40  # matches the dashboard's default history window
        if setup == "battery":
            assert math.isclose(_benchmark_rows(payload)[0]["Cashflow"], 4281.855638, rel_tol=1e-9)
        assert all(len(row["Walk_Forward_Blocks"]) == 2 for row in _benchmark_rows(payload))
        assert set(payload["strategy_series"]) == {
            row["Strategy"] for row in payload["strategies"]
        }


def test_strategy_optimization_is_walk_forward_and_never_tunes_on_its_test_block() -> None:
    response = client.post(
        "/api/compare",
        json={
            "forecast_col": "Prediction",
            "strategy": "Momentum",
            "days": 40,
            "optimize": True,
            "test_days": 10,
            "tune_days": 20,
            "battery": {
                "capacity_mwh": 100,
                "power_mw": 25,
                "initial_soc_mwh": 50,
                "charge_efficiency": 0.90**0.5,
                "discharge_efficiency": 0.90**0.5,
                "charge_fee_per_mwh": 115.41,
                "discharge_fee_per_mwh": 10.71,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    evaluation = payload["evaluation"]
    assert evaluation["mode"] == "out_of_sample_optimization"
    assert evaluation["tuning"] == "walk_forward_neighbour_smoothed"
    assert evaluation["test_days"] == 10 and evaluation["daily_observations"] == 20
    assert evaluation["test_rows"] == 20 * 96
    assert len(payload["selected_strategy_series"]) == evaluation["test_rows"]
    momentum = next(row for row in payload["strategies"] if row["Strategy"] == "Momentum")
    blocks = momentum["Walk_Forward_Blocks"]
    assert len(blocks) == 2 and momentum["Evaluations"] > 1
    for block in blocks:
        assert block["Tune_End"] < block["Test_Start"]  # tuning days end before the block they choose for
    assert blocks[0]["Test_End"] < blocks[1]["Test_Start"]
    # The stitched out-of-sample series is exactly the per-block results.
    assert math.isclose(momentum["Cashflow"], sum(block["Test_Cashflow"] for block in blocks), abs_tol=1e-6)
    assert momentum["Train_Days"] == 40 and momentum["Train_Cashflow"] is not None
    assert momentum["Test_Potential_Settings"] and momentum["No_Fee_Potential_Settings"]
    assert "block 1:" in momentum["Settings"] and "lookback_hours" in momentum["Settings"]


def test_prop_requests_do_not_change_cached_battery_prices() -> None:
    from api.index import _load_deployment_results

    before = float(_load_deployment_results()[0]["Actual_Price"].iloc[0])
    assert client.post("/api/compare", json={"trading_setup": "prop", "days": 3}).status_code == 200
    assert float(_load_deployment_results()[0]["Actual_Price"].iloc[0]) == before