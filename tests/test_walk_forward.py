import json
from pathlib import Path

import pandas as pd
import pytest

from intraday_power_quant.walk_forward import walk_forward_blocks

ROOT = Path(__file__).resolve().parents[1]


def test_walk_forward_blocks_are_contiguous_and_start_after_training_history():
    # First usable modeling row (the very first 15-min row has missing lags and is dropped).
    times = pd.Series(pd.date_range('2025-10-09 00:00', '2026-03-04 22:45', freq='15min'))
    blocks = walk_forward_blocks(times, min_train_days=35, block_days=14)
    assert blocks[0][0] == pd.Timestamp('2025-11-13')  # 35 days after the first calendar day
    assert all(end == next_start for (_, end), (next_start, _) in zip(blocks, blocks[1:]))
    assert all(end - start == pd.Timedelta(days=14) for start, end in blocks)
    assert blocks[-1][0] <= times.max() < blocks[-1][1]


def test_deployment_predictions_match_the_walk_forward_manifest():
    manifest = json.loads((ROOT / 'deployment_data' / 'manifest.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(ROOT / 'deployment_data' / 'predictions.csv.gz', parse_dates=['HourUTC'])
    assert manifest['forecast_method'].startswith('walk_forward')
    assert len(predictions) == manifest['rows'] == manifest['imbalance_rows']
    assert predictions['HourUTC'].is_monotonic_increasing and not predictions['HourUTC'].duplicated().any()
    # Every prediction lies after the first training block, i.e. no interval was in its own model's training data.
    assert predictions['HourUTC'].min() >= pd.Timestamp('2025-11-13', tz='UTC')


def test_cv_gate_compares_out_of_fold_error_with_the_baseline_on_the_same_rows():
    import numpy as np
    from intraday_power_quant.walk_forward import cv_gate

    target = np.array([10.0, -10.0, 5.0, -5.0])
    good = cv_gate(np.array([np.nan, -8.0, 4.0, -4.0]), target, np.zeros(4))   # first row has no OOF value
    assert good["passes"] and good["baseline_cv_mae"] == pytest.approx((10 + 5 + 5) / 3)
    bad = cv_gate(np.array([np.nan, 8.0, -4.0, 4.0]), target, np.zeros(4))
    assert not bad["passes"]


def test_champion_uses_only_earlier_blocks_and_respects_the_gate():
    from intraday_power_quant.walk_forward import choose_champion

    earlier = pd.DataFrame({"Actual_Price": [100.0, 110.0], "Hourly_Baseline": [90.0, 100.0],
                            "TL_Residual_Average": [99.0, 109.0], "TL_Residual_Stacked": [80.0, 90.0],
                            "Direct_15min_Stacked": [70.0, 70.0]})
    both_pass = {"residual": True, "direct": True}
    assert choose_champion(None, "TL_Residual_Average", both_pass)[0] == "TL_Residual_Average"
    assert choose_champion(earlier, "TL_Residual_Average", both_pass)[0] == "TL_Residual_Average"
    # The best earlier model is residual-based, but its gate failed on this block's training data.
    champion, reason = choose_champion(earlier, "TL_Residual_Average", {"residual": False, "direct": True})
    assert champion == "Hourly_Baseline" and "gate" in reason


def test_last_folds_mask_matches_sklearn_time_series_split():
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    from intraday_power_quant.walk_forward import last_folds_mask

    for rows in (12, 3360, 12764):
        folds = list(TimeSeriesSplit(n_splits=5).split(np.zeros(rows)))
        expected = np.zeros(rows, dtype=bool)
        for _, validation in folds[-2:]:
            expected[validation] = True
        assert (last_folds_mask(rows, 5, 2) == expected).all()
