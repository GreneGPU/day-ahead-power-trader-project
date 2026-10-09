import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.index import _load_deployment_results, app
from intraday_power_quant.custom_strategy import prepare_custom_signals, run_custom_strategy
from intraday_power_quant.lab_presets import load_presets, preset_request, presets_for_setup, run_preset
from intraday_power_quant.prop_trading import PropConfig
from intraday_power_quant.trading import BatteryConfig

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]


def test_presets_are_valid_lab_requests_and_the_lab_loads_the_same_file():
    presets = load_presets()
    assert len(presets) == 12 and len({p["name"] for p in presets}) == 12
    for preset in presets:
        for setup in ("prop", "battery"):
            preset_request(preset, setup)  # validates against the lab's request schema
    script = (ROOT / "public" / "strategy-lab.js").read_text(encoding="utf-8")
    assert "fetch('/strategy-presets.json')" in script and not re.search(r"const presets\s*=\s*\[\s*\{", script)
    assert len(presets_for_setup("battery")) == len(presets_for_setup("imbalance")) == 10  # no sizing presets


def test_prop_preset_matches_the_lab_backtest_with_the_same_costs():
    history, _, _ = _load_deployment_results()
    window = history.tail(96 * 12).reset_index(drop=True)
    preset = next(p for p in load_presets() if p["name"] == "Forecast momentum")
    request = preset_request(preset, "prop")
    lab = run_custom_strategy(prepare_custom_signals(window, request), request)[0]
    prop = PropConfig(**request.prop.model_dump())
    benchmark = run_preset(window, preset, "prop", BatteryConfig(), prop)
    assert benchmark["Cashflow"].sum() == pytest.approx(lab["Cashflow"].sum())


@pytest.mark.parametrize("setup", ["prop", "battery", "imbalance"])
def test_compare_includes_lab_presets_as_fixed_rules(setup):
    response = client.post("/api/compare", json={"trading_setup": setup, "days": 6})
    assert response.status_code == 200, response.text
    data = response.json()
    lab = [row for row in data["strategies"] if row["Source"] == "Strategy Lab"]
    assert {row["Strategy"] for row in lab} == {p["name"] for p in presets_for_setup(setup)}
    assert all(row["Train_Cashflow"] is None and row["Evaluations"] == 1 for row in lab)
    assert all(row["Strategy"] in data["strategy_series"] for row in lab)
    assert sum(row["Source"] == "Benchmark" for row in data["strategies"]) > 0


def test_a_lab_preset_can_be_selected_and_sizing_presets_are_prop_only():
    response = client.post("/api/compare", json={"trading_setup": "prop", "days": 6, "strategy": "Forecast momentum"})
    assert response.status_code == 200, response.text
    assert response.json()["selected_strategy"] == "Forecast momentum"
    sized = next(p["name"] for p in load_presets() if p.get("sizing"))
    assert client.post("/api/compare", json={"trading_setup": "battery", "days": 6, "strategy": sized}).status_code == 422
