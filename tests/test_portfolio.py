import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.index import _load_deployment_results, _split_complete_day_holdout, app
from intraday_power_quant.custom_strategy import CustomStrategyRequest, prepare_custom_signals, run_custom_strategy
from intraday_power_quant.portfolio import simulate_battery_book, simulate_book
from intraday_power_quant.trading import BatteryConfig

client = TestClient(app)
RANK = {'signal': 'formula', 'formula': 'signal = forecast_rank', 'direction': 'buy_low',
        'lower': 20, 'upper': 80, 'trading_setup': 'prop'}
BATTERY = {**RANK, 'trading_setup': 'battery'}


def run_intervals(**overrides):
    request = CustomStrategyRequest(**{**RANK, **overrides})
    history, _, _ = _load_deployment_results()
    _, test = _split_complete_day_holdout(prepare_custom_signals(history, request), 10)
    return run_custom_strategy(test, request)[0]


def test_one_run_book_reproduces_the_run():
    run = run_intervals()
    book = simulate_book([run], [1.0], cost_rate=0.41)
    np.testing.assert_allclose(book['Cashflow'], run['Cashflow'], atol=1e-9)
    np.testing.assert_allclose(book['Transaction_Cost'], run['Transaction_Cost'], atol=1e-9)


def test_duplicate_halves_equal_one_run_and_mirror_images_net_to_zero():
    run = run_intervals()
    halves = simulate_book([run, run], [0.5, 0.5], cost_rate=0.41)
    np.testing.assert_allclose(halves['Cashflow'], run['Cashflow'], atol=1e-9)
    mirror = run_intervals(direction='buy_high')  # long where the first is short and vice versa
    flat = simulate_book([run, mirror], [1.0, 1.0], cost_rate=0.41)
    assert (flat['Position_MWh'] == 0).all() and flat['Cashflow'].abs().max() == 0
    assert flat['Standalone_Cashflow'].sum() == pytest.approx(run['Cashflow'].sum() + mirror['Cashflow'].sum())


def test_portfolio_robustness_endpoint_matches_the_backtest_for_a_single_run():
    backtest = client.post('/api/custom-strategy', json=RANK).json()
    response = client.post('/api/portfolio/robustness', json={'runs': [RANK], 'weights': [1.0]})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['periods']['test']['label'] == 'Final 10 days' and data['periods']['earlier']['label'] == 'Earlier period'
    assert data['periods']['test']['total_cashflow'] == pytest.approx(backtest['summary']['total_cashflow'])
    assert data['significance']['days'] == 10


@pytest.mark.parametrize('body', [
    {'runs': [RANK], 'weights': [1.0, 1.0]},
    {'runs': [RANK], 'weights': [0.0]},
    {'runs': [RANK, BATTERY], 'weights': [1.0, 1.0]},
])
def test_portfolio_robustness_validation(body):
    assert client.post('/api/portfolio/robustness', json=body).status_code == 422


def test_one_run_battery_book_reproduces_the_run():
    run = run_intervals(trading_setup='battery')
    request = CustomStrategyRequest(**BATTERY)
    book = simulate_battery_book([run], [1.0], BatteryConfig(**request.battery.model_dump()))
    np.testing.assert_allclose(book['Dispatch_MW'], run['Dispatch_MW'], atol=1e-9)
    np.testing.assert_allclose(book['Cashflow'], run['Cashflow'], atol=1e-9)
    assert book['Standalone_Cashflow'].sum() == pytest.approx(run['Cashflow'].sum())


def test_opposite_battery_runs_net_to_an_idle_battery():
    run = run_intervals(trading_setup='battery')
    mirror = run_intervals(trading_setup='battery', direction='buy_high')
    config = BatteryConfig(**CustomStrategyRequest(**BATTERY).battery.model_dump())
    book = simulate_battery_book([run, run], [0.5, 0.5], config)
    np.testing.assert_allclose(book['Cashflow'], run['Cashflow'], atol=1e-9)
    capped = simulate_battery_book([run], [1.0], config, cap_mw=0.5)
    assert capped['Dispatch_MW'].abs().max() <= 0.5 + 1e-12 and capped.attrs['capped'] > 0
    netted = simulate_battery_book([run, mirror], [1.0, 1.0], config)
    # Where the two runs ask for opposite moves of the same size the shared battery does nothing.
    cancel = (run['Dispatch_MW'] + mirror['Dispatch_MW']).abs() < 1e-12
    assert (netted.loc[cancel, 'Dispatch_MW'] == 0).all()


def test_battery_portfolio_endpoint_matches_the_backtest_and_runs_only_the_t_test():
    backtest = client.post('/api/custom-strategy', json=BATTERY).json()
    response = client.post('/api/portfolio/robustness', json={'runs': [BATTERY], 'weights': [1.0]})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['trading_setup'] == 'battery'
    assert data['periods']['test']['total_cashflow'] == pytest.approx(backtest['summary']['total_cashflow'])
    assert data['significance']['days'] == 10 and data['significance']['day_shuffle'] is None
