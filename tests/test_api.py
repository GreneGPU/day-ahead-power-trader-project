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
    assert payload["dataset"]["rows"] == 2116
    assert len(payload["prices"]) == 2116
    assert len(payload["model_metrics"]) == 3
    assert payload["prices"][0]["Actual_Price"] == 106.029999


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
    assert len(payload["strategies"]) == 16
    assert any(row["Strategy"] == "Predicted best hours" for row in payload["strategies"])
    assert any(row["Strategy"] == "Wind signal" for row in payload["strategies"])
    assert any(row["Strategy"] == "Rolling price optimizer" for row in payload["strategies"])
    assert payload["perfect_foresight_benchmark"]["Cashflow"] is not None
    assert payload["perfect_foresight_benchmark"]["Final_SOC_MWh"] == 0
    assert payload["best_strategy"] == payload["strategies"][0]["Strategy"]
    assert len(payload["best_strategy_series"]) == payload["dataset"]["selected_rows"]
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
    assert len(payload["strategies"]) == 12
    selected = next(
        row for row in payload["strategies"] if row["Strategy"] == "Daily spread rank"
    )
    assert selected["Return_Pct"] is not None
    assert selected["Degradation_Cost"] == 0
    assert payload["selected_strategy_series"][-1]["Position"] == 0
    assert any(
        row["Action"] in {"long", "short"} for row in payload["selected_strategy_series"]
    )
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


def test_strategy_optimization_uses_earlier_train_and_later_unseen_test_data() -> None:
    response = client.post(
        "/api/compare",
        json={
            "forecast_col": "Prediction",
            "strategy": "Momentum",
            "days": 7,
            "optimize": True,
            "test_days": 6,
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
    assert evaluation["train_rows"] + evaluation["test_rows"] == payload["dataset"]["history_rows"]
    assert evaluation["train_end"] < evaluation["test_start"]
    assert evaluation["test_days"] == 6
    assert evaluation["daily_observations"] == 6
    assert evaluation["test_rows"] == 6 * 96
    assert len(payload["selected_strategy_series"]) == evaluation["test_rows"]
    momentum = next(row for row in payload["strategies"] if row["Strategy"] == "Momentum")
    assert momentum["Evaluations"] > 1
    assert momentum["Train_Cashflow"] is not None
    assert momentum["Test_Potential_Cashflow"] is not None
    assert momentum["Test_Potential_Cashflow"] >= momentum["Cashflow"]
    assert momentum["Test_Potential_Settings"]
    assert momentum["No_Fee_Potential_Cashflow"] is not None
    assert momentum["No_Fee_Potential_Cashflow"] >= momentum["Test_Potential_Cashflow"]
    assert momentum["No_Fee_Potential_Settings"]
    assert "lookback_hours" in momentum["Settings"]
