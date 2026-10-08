import json
from pathlib import Path

import pandas as pd

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
