import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.index import app
from intraday_power_quant.significance import significance_tests, t_survival

client = TestClient(app)


@pytest.mark.parametrize('t, df, expected', [(0.0, 10, 0.5), (1.812, 10, 0.05), (2.228, 10, 0.025),
                                             (1.729, 19, 0.05), (-1.812, 10, 0.95)])
def test_t_survival_matches_critical_values(t, df, expected):
    assert t_survival(t, df) == pytest.approx(expected, abs=5e-4)


def synthetic_run(days: int, skilled: bool, seed: int = 3) -> pd.DataFrame:
    """Complete Copenhagen days (January, UTC+1) with random prices and either perfect or random positions."""
    rng = np.random.default_rng(seed)
    n = days * 96
    price = 500 + np.cumsum(rng.normal(0, 20, n))
    spread = rng.normal(0, 20, n)
    move = np.append(price[1:] - price[:-1], 0.0)
    day_end = (np.arange(n) % 96) == 95
    move[day_end] = spread[day_end]
    exposure = 10 * (np.sign(move) if skilled else rng.choice([-1.0, 0.0, 1.0], n))
    return pd.DataFrame({'HourUTC': pd.date_range('2026-01-04 23:00', periods=n, freq='15min', tz='UTC'),
                         'Actual_Price': price, 'Imbalance_Spread_DKK': spread,
                         'Position_MWh': exposure, 'Cashflow': exposure * move})


def test_perfect_foresight_is_significant_on_every_test():
    result = significance_tests(synthetic_run(12, skilled=True), permutations=500)
    assert result['days'] == 12 and result['p_daily'] < 1e-6
    assert result['actual_gross'] == pytest.approx(synthetic_run(12, skilled=True)['Cashflow'].sum())
    assert result['day_shuffle']['p'] == pytest.approx(1 / 501)
    assert result['timing_shift']['p'] == pytest.approx(1 / 501)
    assert result['days_needed_80pct_power'] <= 3


def test_random_positions_are_not_significant():
    result = significance_tests(synthetic_run(12, skilled=False), permutations=500)
    assert result['day_shuffle']['p'] > 0.05 and result['timing_shift']['p'] > 0.05


def test_too_few_complete_days_is_reported():
    assert 'error' in significance_tests(synthetic_run(2, skilled=True))


def test_robustness_reports_significance_for_the_test_period():
    payload = {'signal': 'formula', 'formula': 'signal = forecast_rank', 'direction': 'buy_low',
               'lower': 20, 'upper': 80, 'trading_setup': 'prop'}
    significance = client.post('/api/custom-strategy/robustness', json=payload).json()['significance']
    assert significance['days'] == 10
    # Matches the offline analysis: mean 4,319 DKK/day, sd 10,630, one-sided daily p of about 0.115.
    assert significance['mean_daily'] == pytest.approx(4318.68, abs=1)
    assert significance['p_daily'] == pytest.approx(0.115, abs=0.005)
    # Most of the edge is the typical daily price shape, so shuffling days keeps most of the P&L.
    assert 0.15 < significance['day_shuffle']['p'] < 0.6
    assert significance['day_shuffle']['null_mean'] > 0.5 * significance['actual_gross']
