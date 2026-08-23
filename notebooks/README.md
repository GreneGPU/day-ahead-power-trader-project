# Notebooks

Keep exploratory notebooks here. The production path is the Python package in `src/intraday_power_quant`.

Recommended workflow:

1. Use notebooks for discovery and plots.
2. Move durable logic into `src/`.
3. Run `python -m intraday_power_quant.cli run --config configs/jakob-local.json`.
4. Keep generated metrics and forecasts under `outputs/`.

