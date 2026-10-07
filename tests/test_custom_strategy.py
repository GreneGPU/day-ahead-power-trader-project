import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.index import app
from intraday_power_quant.custom_strategy import (
    CustomStrategyRequest, prepare_custom_signals, run_custom_strategy,
)


client = TestClient(app)


def history():
    return pd.DataFrame({
        "HourUTC": pd.date_range("2026-08-01T00:00:00Z", periods=4, freq="15min"),
        "Actual_Price_DKK": [100., 120., 150., 130.],
        "Prediction_DKK": [80., 100., 120., 100.],
        "Hourly_Baseline_DKK": [100.] * 4,
        "Imbalance_Price_DKK": [110., 130., 160., 140.],
    })


def test_threshold_boundaries_and_battery_accounting():
    request = CustomStrategyRequest(battery={"capacity_mwh": 1, "power_mw": 4,
        "charge_efficiency": 1, "discharge_efficiency": 1,
        "charge_fee_per_mwh": 2, "discharge_fee_per_mwh": 3})
    frame = prepare_custom_signals(history(), request)
    assert frame.Requested_Action.tolist() == ["charge", "hold", "discharge", "hold"]
    result, summary = run_custom_strategy(frame, request)
    assert result.State_Of_Charge_MWh.tolist() == [1, 1, 0, 0]
    assert result.Cashflow.tolist() == [-102, 0, 147, 0]
    assert summary["total_cashflow"] == 45
    assert summary["max_drawdown"] == 102  # Includes initial outlay.
    assert summary["total_fee_cost"] == 5


def test_forecast_change_is_causal_and_warmup_holds():
    request = CustomStrategyRequest(signal="forecast_change", lookback=2)
    frame = history()
    before = prepare_custom_signals(frame, request)
    frame.loc[3, "Prediction_DKK"] = 1e6
    frame["Actual_Price_DKK"] = -999
    after = prepare_custom_signals(frame, request)
    pd.testing.assert_series_equal(before.Custom_Signal.iloc[:3], after.Custom_Signal.iloc[:3])
    assert before.Requested_Action.iloc[:2].tolist() == ["hold", "hold"]
    assert before.Custom_Signal.iloc[2] == 40


def test_prop_signals_do_not_depend_on_battery_inventory():
    request = CustomStrategyRequest(trading_setup="prop", direction="buy_high",
        prop={"initial_capital_dkk": 1000, "position_size_mwh": 1,
              "transaction_cost_dkk_per_mwh": 2})
    result, summary = run_custom_strategy(prepare_custom_signals(history(), request), request)
    assert result.Position.tolist() == [-1, 0, 1, 0]
    assert summary["total_fee_cost"] == 8
    assert summary["total_cashflow"] == -48
    assert summary["ending_equity_dkk"] == 952


def test_prop_last_interval_closes_with_costs():
    request = CustomStrategyRequest(trading_setup="prop", signal="forecast", lower=200, upper=300,
        prop={"position_size_mwh": 1, "transaction_cost_dkk_per_mwh": 2})
    result, summary = run_custom_strategy(prepare_custom_signals(history(), request), request)
    assert result.Position_After_Settlement.iloc[-1] == 0
    assert result.EOD_Imbalance_Settlement.iloc[-1]
    assert summary["total_fee_cost"] == 4
    assert summary["total_cashflow"] == 36


@pytest.mark.parametrize("payload", [
    {"lower": 20, "upper": 20}, {"signal": "actual_price"},
    {"lookback": 0}, {"battery": {"capacity_mwh": -1}},
    {"battery": {"initial_soc_mwh": 10}}, {"prop": {"position_size_mwh": 0}},
    {"signal": "csv"}, {"code": "import os"},
])
def test_invalid_requests_are_rejected(payload):
    assert client.post("/api/custom-strategy", json=payload).status_code == 422


def test_csv_joins_by_timestamp_and_rejects_missing_and_duplicate_rows():
    frame = history()
    rows = [{"HourUTC": timestamp.isoformat(), "Signal": signal}
            for timestamp, signal in zip(frame.HourUTC, [-20, 0, 20, 0])]
    request = CustomStrategyRequest(signal="csv", signal_records=list(reversed(rows)))
    prepared = prepare_custom_signals(frame, request)
    assert prepared.Custom_Signal.tolist() == [-20, 0, 20, 0]
    missing = request.model_copy(update={"signal_records": request.signal_records[:-1]})
    with pytest.raises(ValueError, match="every interval"):
        run_custom_strategy(prepare_custom_signals(frame, missing), missing)
    duplicate = request.model_copy(update={"signal_records": request.signal_records * 2})
    with pytest.raises(ValueError, match="duplicate"):
        prepare_custom_signals(frame, duplicate)


def test_csv_rejects_nonfinite_values_and_timezone_free_dates():
    for signal, stamp in [(float("nan"), "2026-08-01T00:00:00Z"), (0, "2026-08-01T00:00:00")]:
        with pytest.raises(ValueError):
            CustomStrategyRequest(signal="csv", signal_records=[{"HourUTC": stamp, "Signal": signal}])


@pytest.mark.parametrize("setup", ["battery", "prop"])
def test_saved_data_endpoint_returns_a_complete_finite_backtest(setup):
    response = client.post("/api/custom-strategy", json={"trading_setup": setup})
    assert response.status_code == 200
    data = response.json()
    assert data["currency"] == "DKK"
    assert data["period"]["rows"] == 960
    assert len(data["intervals"]) == 960
    assert np.isfinite(data["summary"]["total_cashflow"])
    assert data["intervals"][-1]["Cumulative_Cashflow"] == pytest.approx(data["summary"]["total_cashflow"])


def test_fundamental_formula_exact_values_and_direction():
    frame = history()
    frame['Wind_Total_DayAhead_MW'] = [20., 30., 40., 50.]
    frame['Solar_DayAhead_MW'] = [5., 10., 15., 20.]
    rows = [{'HourUTC': timestamp.isoformat(), 'demand': demand, 'outages': 10}
            for timestamp, demand in zip(frame.HourUTC, [100., 80., 60., 40.])]
    request = CustomStrategyRequest(signal='formula', formula='signal = demand + outages - wind - solar',
        fundamental_records=rows[::-1], lower=0, upper=50, direction='buy_high')
    result = prepare_custom_signals(frame, request)
    assert result.Custom_Signal.tolist() == [85., 50., 15., -20.]
    assert result.Requested_Action.tolist() == ['charge', 'charge', 'hold', 'discharge']


@pytest.mark.parametrize('formula', [
    '__import__("os")', 'wind.__class__', 'wind[0]', '[wind for x in wind]',
    'wind ** 9999', 'open("x")', 'lambda: 1', 'actual_price', 'wind / 0',
    'clamp(wind, 3, 1)', '1e999', 'max()', 'wind; solar',
])
def test_formula_rejects_code_and_invalid_math(formula):
    from intraday_power_quant.formulas import evaluate_formula
    with pytest.raises(ValueError):
        evaluate_formula(formula, {'wind': pd.Series([1., 2.])}, pd.RangeIndex(2))


def test_formula_functions_and_precedence():
    from intraday_power_quant.formulas import evaluate_formula
    values = {'wind': pd.Series([-4., 2., 8.])}
    output = evaluate_formula('signal = clamp(wind, -2, 3) + abs(wind) * 2 - avg(2,4) + sign(wind)', values, pd.RangeIndex(3))
    assert output.tolist() == [2., 4., 17.]
    assert evaluate_formula('min(wind, 0) + max(wind, 0)', values, pd.RangeIndex(3)).tolist() == [-4.,2.,8.]


def test_formula_missing_inputs_are_not_fabricated():
    response = client.post('/api/custom-strategy', json={'signal':'formula', 'formula':'demand + outages - wind - solar'})
    assert response.status_code == 422
    assert "Missing 'outages'" in response.json()['detail']


def test_formula_partial_upload_fails_selected_period():
    frame = history()
    request = CustomStrategyRequest(signal='formula', formula='demand', fundamental_records=[
        {'HourUTC':frame.HourUTC.iloc[0].isoformat(), 'demand':100}])
    prepared = prepare_custom_signals(frame, request)
    with pytest.raises(ValueError, match='cover every interval'):
        run_custom_strategy(prepared, request)
    # Coverage is required only over the chosen test period.
    run_custom_strategy(prepared.iloc[:1], request)


def test_formula_endpoint_saved_wind_and_solar():
    response = client.post('/api/custom-strategy', json={'signal':'formula', 'formula':'signal = -wind - solar', 'trading_setup':'prop'})
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data['intervals']) == 960
    assert all(row['Custom_Signal'] <= 0 for row in data['intervals'])
    assert 'fundamental_records' not in data['settings']


@pytest.mark.parametrize('feature', __import__('intraday_power_quant.formulas', fromlist=['THESIS_FEATURES']).THESIS_FEATURES)
def test_saved_thesis_feature_formula_matches_source(feature):
    from api.index import _load_deployment_results
    frame, _, _ = _load_deployment_results()
    response = client.post('/api/custom-strategy', json={
        'signal': 'formula', 'formula': f'signal = {feature}', 'evaluation': 'full'})
    assert response.status_code == 200, response.text
    np.testing.assert_allclose([row['Custom_Signal'] for row in response.json()['intervals']], frame[feature])


def test_demand_alias_and_residual_demand_without_upload():
    from api.index import _load_deployment_results
    frame, _, _ = _load_deployment_results()
    request = CustomStrategyRequest(signal='formula', formula='demand - wind - solar')
    prepared = prepare_custom_signals(frame, request)
    np.testing.assert_allclose(prepared.Custom_Signal, frame.load_fc - frame.Wind_Total_DayAhead_MW - frame.Solar_DayAhead_MW)
    assert not prepared.Custom_Signal.isna().any()


def test_prop_replay_fields_reconcile_to_summary():
    response = client.post('/api/custom-strategy', json={'trading_setup':'prop'})
    assert response.status_code == 200
    data = response.json()
    rows = data['intervals']
    assert sum(r['Transaction_Cost'] for r in rows) == pytest.approx(data['summary']['total_fee_cost'])
    assert rows[-1]['Equity_DKK'] == pytest.approx(data['summary']['ending_equity_dkk'])
    assert rows[-1]['Position_After_Settlement'] == 0
    assert all(r['Position'] in (-1,0,1) for r in rows)


def test_replay_factors_align_with_intervals():
    from api.index import _load_deployment_results
    frame, _, _ = _load_deployment_results()
    response = client.post('/api/custom-strategy', json={'signal':'formula', 'formula':'signal = demand - wind - solar', 'trading_setup':'prop'})
    assert response.status_code == 200, response.text
    data = response.json()
    assert {'wind', 'solar', 'demand', 'forecast', 'baseline'} <= set(data['factors'])
    assert all(len(series) == len(data['intervals']) for series in data['factors'].values())
    by_time = frame.set_index('HourUTC')
    for i in (0, len(data['intervals']) - 1):
        stamp = pd.Timestamp(data['intervals'][i]['HourUTC'])
        assert data['factors']['wind'][i] == pytest.approx(by_time.loc[stamp, 'Wind_Total_DayAhead_MW'])
        assert data['factors']['demand'][i] - data['factors']['wind'][i] - data['factors']['solar'][i] == pytest.approx(data['intervals'][i]['Custom_Signal'])


def test_latest_prices_service_failure_is_explained(monkeypatch):
    import api.index as api_module
    def fail():
        raise OSError('Unavailable')
    monkeypatch.setattr(api_module, 'latest_prices', fail)
    response=client.get('/api/latest-dk1-prices')
    assert response.status_code == 502
    assert 'temporarily unavailable' in response.json()['detail']
