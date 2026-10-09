import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from intraday_power_quant.tuning import (
    best_fixed_setting,
    grid_neighbours,
    smoothed_choice,
    tune_strategy,
    walk_forward_folds,
)


def test_grid_neighbours_are_one_numeric_step_apart_with_categories_fixed():
    settings = [
        {"band": 0.0, "window": 3.0, "mode": "a"},
        {"band": 10.0, "window": 3.0, "mode": "a"},
        {"band": 20.0, "window": 3.0, "mode": "a"},
        {"band": 10.0, "window": 7.0, "mode": "a"},
        {"band": 10.0, "window": 3.0, "mode": "b"},   # different category: never a neighbour
        {"band": 10.0, "window": 3.0, "spread": None, "mode": "a"},  # different keys: never a neighbour
    ]
    neighbours = grid_neighbours(settings)
    assert sorted(neighbours[1]) == [0, 2, 3]
    assert neighbours[0] == [1]          # 0 -> 20 is two steps
    assert neighbours[4] == [] and neighbours[5] == []


def test_smoothed_choice_prefers_a_stable_plateau_over_a_lone_peak():
    # Setting 0 is a lone peak among poor neighbours; setting 3 sits in a consistently good region.
    pnl = np.array([100.0, -50.0, 40.0, 60.0, 55.0])
    neighbours = [[1], [0, 2], [1, 3], [2, 4], [3]]
    assert int(np.argmax(pnl)) == 0
    # Any setting inside the good region (3 or its edge neighbour 4) is fine; the lone peak is not.
    assert smoothed_choice(pnl, neighbours) in {3, 4}


def test_walk_forward_folds_tune_on_the_days_just_before_each_block():
    dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    folds = walk_forward_folds(dates, tune_days=20, test_days=10)
    assert len(folds) == 2
    for tune, test in folds:
        assert len(tune) == 20 and len(test) == 10 and tune[-1] + dt.timedelta(days=1) == test[0]
    assert folds[-1][1][-1] == dates[-1]
    with pytest.raises(ValueError):
        walk_forward_folds(dates[:25], tune_days=20, test_days=10)


def test_tuning_never_uses_the_block_it_is_tested_on():
    dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(30)]
    tune, test = dates[:20], dates[20:]
    steady = pd.Series(10.0, index=dates)
    # "Lucky" loses during tuning but is spectacular on the test days.
    lucky = pd.Series([-5.0] * 20 + [1000.0] * 10, index=dates)
    entries = [(json.dumps({"k": 1.0}), steady), (json.dumps({"k": 2.0}), lucky)]
    [fold] = tune_strategy(entries, [(tune, test)])
    assert json.loads(fold["settings_json"]) == {"k": 1.0}
    assert fold["train_pnl"] == 200.0 and fold["test_pnl"] == 100.0
    # Hindsight on the test days would have picked the lucky setting.
    assert best_fixed_setting(entries, test) == (json.dumps({"k": 2.0}), 10000.0)
