from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from intraday_power_quant.config import load_config  # noqa: E402
from intraday_power_quant.transfer import run_transfer_learning  # noqa: E402


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SageMaker entry point for the power forecast model.")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--enable-mlflow", default=os.getenv("POWER_TRADER_ENABLE_MLFLOW", "false"))
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--mlflow-experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "power-trader-dk1-15min")
    )
    parser.add_argument("--mlflow-register-model", default=os.getenv("MLFLOW_REGISTERED_MODEL_NAME"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args, _ = build_parser().parse_known_args(argv)
    train_dir = Path(os.getenv("SM_CHANNEL_TRAIN", "data/raw"))
    config_dir = Path(os.getenv("SM_CHANNEL_CONFIG", "configs"))
    if args.config_file:
        config_path = Path(args.config_file)
    else:
        config_path = config_dir / "default.json"
        if not config_path.exists():
            candidates = sorted(config_dir.glob("*.json"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected default.json or exactly one JSON config in {config_dir}"
                )
            config_path = candidates[0]
    output_dir = Path(os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    model_dir = Path(os.getenv("SM_MODEL_DIR", "/opt/ml/model"))

    config = load_config(config_path).with_overrides(
        data_dir=train_dir,
        output_dir=output_dir,
        mlflow_enabled=_as_bool(args.enable_mlflow),
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment,
        mlflow_registered_model_name=args.mlflow_register_model,
    )
    summary = run_transfer_learning(config)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_candidates = [
        summary["outputs"]["model_bundle"]["pickle"],
        summary["outputs"]["model_bundle"]["metadata"],
        summary["outputs"]["model_gate"],
        summary["summary_path"],
        summary["outputs"]["metrics"]["csv"],
    ]
    for artifact in artifact_candidates:
        source = Path(artifact)
        shutil.copy2(source, model_dir / source.name)
    print("SageMaker model artifacts:", model_dir)


if __name__ == "__main__":
    main()
