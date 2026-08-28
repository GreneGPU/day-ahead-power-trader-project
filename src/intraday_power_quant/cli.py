from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .dashboard import generate_markdown_report, generate_static_dashboard
from .data import data_quality_summary, load_market_data, prepare_time_series_frame
from .experiments import run_baseline_feature_ablation
from .model_bundle import ForecastModelBundle
from .model_gate import ModelGateSettings, evaluate_model_gate, write_model_gate
from .monitoring import build_monitoring_report, write_monitoring_report
from .models import optional_model_status
from .transfer import run_transfer_learning


def _load_with_overrides(args: argparse.Namespace):
    config = load_config(args.config)
    return config.with_overrides(
        data_dir=getattr(args, "data_dir", None),
        reference_results_dir=getattr(args, "results_dir", None),
        output_dir=getattr(args, "output_dir", None),
        champion_prediction_column=getattr(args, "champion", None),
        mlflow_enabled=getattr(args, "enable_mlflow", None),
        mlflow_tracking_uri=getattr(args, "mlflow_tracking_uri", None),
        mlflow_experiment_name=getattr(args, "mlflow_experiment", None),
        mlflow_registered_model_name=getattr(args, "mlflow_register_model", None),
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


def cmd_score(args: argparse.Namespace) -> None:
    bundle = ForecastModelBundle.load(args.model_bundle)
    input_path = Path(args.input)
    frame = pd.read_json(input_path, lines=True) if args.json_lines else pd.read_csv(input_path)
    predictions = bundle.predict_components(frame)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.json_lines:
        predictions.to_json(output_path, orient="records", lines=True, date_format="iso")
    else:
        predictions.to_csv(output_path, index=False)
    print("Predictions:", output_path)


def cmd_gate(args: argparse.Namespace) -> None:
    metrics = pd.read_csv(args.metrics)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    validation = summary.get("validation_split", {})
    result = evaluate_model_gate(
        metrics=metrics,
        candidate_prediction_column=args.candidate or summary["champion_prediction_column"],
        coverage=validation.get("coverage", {}),
        leakage_warnings=summary.get("leakage_warnings", []),
        settings=ModelGateSettings(
            min_coverage_pct=args.min_coverage,
            max_baseline_mae_regression_pct=args.max_baseline_regression,
            require_continuous_15min=not args.allow_gaps,
        ),
        current_champion_mae=args.current_champion_mae,
    )
    output_path = write_model_gate(result, args.output)
    print(json.dumps(result, indent=2))
    print("Model gate:", output_path)


def cmd_monitor(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.forecasts)
    report = build_monitoring_report(frame)
    output_path = write_monitoring_report(report, args.output)
    print(json.dumps(report, indent=2))
    print("Monitoring report:", output_path)


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
    run.add_argument("--enable-mlflow", action="store_true", default=None)
    run.add_argument("--mlflow-tracking-uri")
    run.add_argument("--mlflow-experiment")
    run.add_argument("--mlflow-register-model")
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

    score = subparsers.add_parser("score", help="Generate forecasts from a saved model bundle.")
    score.add_argument("--model-bundle", required=True)
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--json-lines", action="store_true")
    score.set_defaults(func=cmd_score)

    gate = subparsers.add_parser("gate", help="Evaluate model promotion rules from saved outputs.")
    gate.add_argument("--metrics", required=True)
    gate.add_argument("--summary", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--candidate")
    gate.add_argument("--current-champion-mae", type=float)
    gate.add_argument("--min-coverage", type=float, default=99.0)
    gate.add_argument("--max-baseline-regression", type=float, default=0.0)
    gate.add_argument("--allow-gaps", action="store_true")
    gate.set_defaults(func=cmd_gate)

    monitor = subparsers.add_parser("monitor", help="Calculate production forecast quality metrics.")
    monitor.add_argument("--forecasts", required=True)
    monitor.add_argument("--output", default="outputs/monitoring/latest.json")
    monitor.set_defaults(func=cmd_monitor)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
