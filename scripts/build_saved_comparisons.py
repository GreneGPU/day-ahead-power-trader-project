from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.index import StrategyComparisonRequest, compare_strategies


DEFAULT_BATTERY = {
    "capacity_mwh": 100,
    "power_mw": 25,
    "initial_soc_mwh": 0,
    "charge_efficiency": 0.90**0.5,
    "discharge_efficiency": 0.90**0.5,
    "charge_fee_per_mwh": 115.41,
    "discharge_fee_per_mwh": 10.71,
}

DEFAULT_PROP = {
    "initial_capital_dkk": 100_000,
    "position_size_mwh": 10,
    "transaction_cost_dkk_per_mwh": 0.41,
    "max_daily_loss_dkk": 5_000,
}


def build_snapshot(trading_setup: str) -> dict[str, object]:
    result = compare_strategies(
        StrategyComparisonRequest(
            forecast_col="Prediction",
            optimize=True,
            test_days=10,
            trading_setup=trading_setup,
            battery=DEFAULT_BATTERY,
            prop=DEFAULT_PROP,
        )
    )
    result["saved_result"] = True
    result["saved_at"] = datetime.now(timezone.utc).isoformat()
    return result


def main() -> None:
    output_dir = PROJECT_ROOT / "deployment_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    available = ("battery", "prop", "imbalance")
    requested = tuple(sys.argv[1:]) or available
    unknown = [setup for setup in requested if setup not in available]
    if unknown:
        raise ValueError(f"Unknown trading setup(s): {', '.join(unknown)}")
    for setup in requested:
        output_path = output_dir / f"default_{setup}_comparison.json.gz"
        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(build_snapshot(setup), handle, separators=(",", ":"), allow_nan=False)
        print(f"Wrote {output_path.relative_to(PROJECT_ROOT)} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
