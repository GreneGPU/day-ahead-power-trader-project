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


def _five_day_runs():
    """Three Prop runs over the final 10 days (rank rule, its mirror, and a z-score rule)."""
    return [run_intervals(), run_intervals(direction='buy_high'), run_intervals(formula='signal = forecast_z', lower=-1, upper=1)]


def test_rolling_weights_use_only_earlier_days_and_respect_the_cap():
    from intraday_power_quant.portfolio import day_index, rolling_weights
    runs = _five_day_runs()
    weights = rolling_weights(runs, window=3, tilt=1.0, cap=2.0)
    days = day_index(runs[0])
    assert weights.shape == (len(runs[0]), 3) and days.max() == 9
    assert (weights[days < 3] == 1).all()                       # equal until a full window exists
    assert weights.max() <= 2.0 + 1e-9 and (weights >= 0).all()
    assert (weights.sum(axis=1) <= 3 + 1e-9).all()
    # Changing the last day's results must not change any weight: only earlier days are used.
    altered = [run.copy() for run in runs]
    altered[0].loc[days == 9, 'Cashflow'] += 1e6
    np.testing.assert_allclose(rolling_weights(altered, 3, 1.0, 2.0), weights)
    # The first run and its mirror image cannot both have made money in a window, so full tilt drops one of
    # them, unless no run made money and the day falls back to equal weights.
    later = weights[days >= 3]
    assert ((later[:, 0] == 0) | (later[:, 1] == 0) | (later == 1).all(axis=1)).all()
    assert (later != 1).any()


def test_rolling_book_with_no_tilt_equals_the_equal_weight_book():
    from intraday_power_quant.portfolio import rolling_weights
    runs = _five_day_runs()
    equal = simulate_book(runs, [1.0, 1.0, 1.0], cost_rate=0.41)
    untilted = simulate_book(runs, rolling_weights(runs, 3, 0.0, 2.0), cost_rate=0.41)
    np.testing.assert_allclose(untilted['Cashflow'], equal['Cashflow'], atol=1e-9)
    tilted = simulate_book(runs, rolling_weights(runs, 3, 0.5, 2.0), cost_rate=0.41)
    assert not np.allclose(tilted['Cashflow'], equal['Cashflow'])


@pytest.mark.parametrize('setup', ['prop', 'battery'])
def test_portfolio_endpoint_supports_rolling_weights(setup):
    runs = [{**RANK, 'trading_setup': setup}, {**RANK, 'trading_setup': setup, 'formula': 'signal = forecast_z', 'lower': -1, 'upper': 1}]
    fixed = client.post('/api/portfolio/robustness', json={'runs': runs, 'weights': [1.0, 1.0]}).json()
    body = {'runs': runs, 'weights': [1.0, 1.0], 'weighting': 'rolling', 'rolling_window_days': 3, 'rolling_tilt': 0.5, 'rolling_cap': 2}
    response = client.post('/api/portfolio/robustness', json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['weighting'] == 'rolling'
    for period in ('earlier', 'test'):
        assert data['periods'][period]['equal_weight_cashflow'] == pytest.approx(fixed['periods'][period]['total_cashflow'])
    untilted = client.post('/api/portfolio/robustness', json={**body, 'rolling_tilt': 0}).json()
    assert untilted['periods']['test']['total_cashflow'] == pytest.approx(fixed['periods']['test']['total_cashflow'])
    assert client.post('/api/portfolio/robustness', json={**body, 'rolling_cap': 0.5}).status_code == 422
