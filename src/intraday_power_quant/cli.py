from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .dashboard import generate_markdown_report, generate_static_dashboard
from .data import data_quality_summary, load_market_data, prepare_time_series_frame
from .experiments import run_baseline_feature_ablation
from .models import optional_model_status
from .transfer import run_transfer_learning


def _load_with_overrides(args: argparse.Namespace):
    config = load_config(args.config)
    return config.with_overrides(
        data_dir=getattr(args, "data_dir", None),
        reference_results_dir=getattr(args, "results_dir", None),
        output_dir=getattr(args, "output_dir", None),
        champion_prediction_column=getattr(args, "champion", None),
    )


def cmd_check_data(args: argparse.Namespace) -> None:
    config = _load_with_overrides(args)
    hourly, min15, paths = load_market_data(config.data_dir, config.hourly_candidates, config.min15_candidates)
    hourly = prepare_time_series_frame(hourly, config.time_col)
    min15 = prepare_time_series_frame(min15, config.time_col)
    print("Resolved files:")
    print(f"  hourly: {paths['hourly']}")
    print(f"  15-min: {paths['min15']}")
    print("Hourly summary:", data_quality_summary(hourly, config.time_col, config.target))
    print("15-min summary:", data_quality_summary(min15, config.time_col, config.target))
    print("Model package status:", optional_model_status())


def cmd_run(args: argparse.Namespace) -> None:
    config = _load_with_overrides(args)
    summary = run_transfer_learning(config)
    print("Wrote outputs to:", summary["output_dir"])


def cmd_report(args: argparse.Namespace) -> None:
    config = _load_with_overrides(args)
    results_dir = Path(args.results_dir) if args.results_dir else Path(config.reference_results_dir)
    output_html = (
        Path(args.output_html)
        if args.output_html
        else Path("reports/day_ahead_power_trader_project_dashboard.html")
    )
    output_md = (
        Path(args.output_report)
        if args.output_report
        else Path("reports/day_ahead_power_trader_project_report.md")
    )
    result = generate_static_dashboard(
        results_dir=results_dir,
        output_html=output_html,
        battery_settings=config.battery,
        weekly_band_settings=config.weekly_band,
        forecast_edge_settings=config.forecast_edge,
        volatility_filter_settings=config.volatility_filter,
        mean_reversion_settings=config.mean_reversion,
        momentum_settings=config.momentum,
        momentum_spread_settings=config.momentum_spread,
        channel_breakout_settings=config.channel_breakout,
        daily_spread_settings=config.daily_spread,
        ensemble_agreement_settings=config.ensemble_agreement,
    )
    report_path = generate_markdown_report(result, output_md)
    print("Dashboard:", result["dashboard_path"])
    print("Report:", report_path)


def cmd_ablation(args: argparse.Namespace) -> None:
    config = _load_with_overrides(args)
    comparison = run_baseline_feature_ablation(config)
    print(comparison.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day-ahead power trader research toolkit.")
    parser.add_argument("--config", default="configs/default.json", help="Path to a JSON config file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    check_data = subparsers.add_parser("check-data", help="Resolve source files and print data-quality summaries.")
    check_data.add_argument("--data-dir")
    check_data.set_defaults(func=cmd_check_data)

    run = subparsers.add_parser("run", help="Run the full transfer-learning training pipeline.")
    run.add_argument("--data-dir")
    run.add_argument("--output-dir")
    run.add_argument("--champion")
    run.set_defaults(func=cmd_run)

    report = subparsers.add_parser("report", help="Build a static dashboard/report from saved forecast outputs.")
    report.add_argument("--results-dir")
    report.add_argument("--output-html")
    report.add_argument("--output-report")
    report.set_defaults(func=cmd_report)

    ablation = subparsers.add_parser("ablation", help="Run the hourly-baseline residual-feature ablation.")
    ablation.add_argument("--data-dir")
    ablation.add_argument("--output-dir")
    ablation.add_argument("--champion")
    ablation.set_defaults(func=cmd_ablation)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
