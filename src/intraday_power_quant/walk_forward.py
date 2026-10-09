"""Walk-forward (expanding window) out-of-sample forecasts for the whole 15-minute period.

The single-split run only produces out-of-sample predictions for its final test block. Here the 15-minute
residual and direct ensembles are retrained before every block on all 15-minute data strictly earlier than
that block, so every predicted interval is out of sample. The hourly source model is trained once on hourly
data before the transfer start, which is already earlier than every 15-minute interval.

Each block's champion is chosen without hindsight:

* nested selection - the candidate with the lowest MAE on all *earlier* blocks' out-of-sample forecasts
  (the first block, with no history, uses the configured champion);
* per-block gate - a model trained on 15-minute data (residual or direct) may only be champion if its
  time-series cross-validated MAE on that block's *training* data beats the hourly baseline's; otherwise the
  block falls back to the hourly baseline.

A candidate rule is evaluated alongside without being adopted: the same gate measured on the *last* CV folds
only, which are trained on most of the data and so judge the final model less pessimistically. Its decisions
and forecasts are recorded so it can be validated on future data before it replaces the official rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .evaluation import build_metrics_table, calculate_15min_coverage
from .models import predict_stacked_ensemble, train_stacked_ensemble
from .transfer import _write_frame, prepare_transfer_inputs

CANDIDATES = ["Hourly_Baseline", "TL_Residual_Average", "TL_Residual_Stacked", "Direct_15min_Stacked"]
CANDIDATE_GATE_LAST_FOLDS = 2
FAMILY = {"TL_Residual_Average": "residual", "TL_Residual_Stacked": "residual", "Direct_15min_Stacked": "direct"}


def walk_forward_blocks(times: pd.Series, min_train_days: float, block_days: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Consecutive [start, end) test blocks; the first starts after ``min_train_days`` of history."""
    first = pd.Timestamp(times.min()).normalize() + pd.Timedelta(days=min_train_days)
    last = pd.Timestamp(times.max())
    blocks, cursor = [], first
    while cursor <= last:
        blocks.append((cursor, cursor + pd.Timedelta(days=block_days)))
        cursor += pd.Timedelta(days=block_days)
    return blocks


def last_folds_mask(rows: int, n_splits: int, last_folds: int) -> np.ndarray:
    """Rows validated in the last ``last_folds`` of a TimeSeriesSplit (equal, consecutive validation chunks)."""
    fold_size = rows // (n_splits + 1)
    mask = np.zeros(rows, dtype=bool)
    mask[rows - last_folds * fold_size:] = True
    return mask


def cv_gate(oof_prediction: np.ndarray, target: np.ndarray, baseline_target: np.ndarray,
            rows: np.ndarray | None = None) -> dict[str, float | bool]:
    """Compare a model's out-of-fold MAE with the baseline's on the same training rows.

    For the residual family ``target`` is the residual and ``baseline_target`` is zero (the baseline's
    residual prediction); for the direct family ``target`` is the price and ``baseline_target`` the
    hourly baseline.
    """
    valid = ~np.isnan(oof_prediction)
    if rows is not None:
        valid &= rows
    model_mae = float(np.abs(target[valid] - oof_prediction[valid]).mean())
    baseline_mae = float(np.abs(target[valid] - baseline_target[valid]).mean())
    return {"model_cv_mae": model_mae, "baseline_cv_mae": baseline_mae, "passes": model_mae < baseline_mae}


def choose_champion(history: pd.DataFrame | None, default: str, gates: dict[str, bool]) -> tuple[str, str]:
    """(champion column, reason) from earlier blocks' OOS errors, then the per-block gate."""
    if history is None or history.empty:
        choice, reason = default, "first block: configured champion"
    else:
        maes = {c: float((history["Actual_Price"] - history[c]).abs().mean()) for c in CANDIDATES}
        choice = min(maes, key=maes.get)
        reason = f"lowest MAE on earlier blocks ({maes[choice]:.2f})"
    family = FAMILY.get(choice)
    if family is not None and not gates[family]:
        return "Hourly_Baseline", f"{reason}; {family} model failed its training CV gate"
    return choice, reason


def run_walk_forward(config: ProjectConfig, min_train_days: float = 35, block_days: float = 14) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = prepare_transfer_inputs(config)
    frame: pd.DataFrame = inputs["df_15_transfer"]
    time_col, target = config.time_col, config.target
    feature_cols_15: list[str] = inputs["feature_cols_15"]
    residual_cols = feature_cols_15 + (["Hourly_Baseline"] if config.use_hourly_baseline_as_residual_feature else [])
    default_champion = config.champion_prediction_column if config.champion_prediction_column in CANDIDATES else "TL_Residual_Stacked"

    pieces, folds = [], []
    blocks = walk_forward_blocks(frame[time_col], min_train_days, block_days)
    for number, (start, end) in enumerate(blocks, start=1):
        train, test = frame[frame[time_col] < start], frame[(frame[time_col] >= start) & (frame[time_col] < end)]
        if test.empty:
            continue
        print(f"\nWalk-forward block {number}/{len(blocks)}: train {len(train)} rows to {train[time_col].max()}, "
              f"test {len(test)} rows {test[time_col].min()} -> {test[time_col].max()}")
        residual_model = train_stacked_ensemble(train[residual_cols].reset_index(drop=True),
                                                train["residual_15min"].reset_index(drop=True), n_splits=config.n_splits)
        direct_model = train_stacked_ensemble(train[feature_cols_15].reset_index(drop=True),
                                              train[target].reset_index(drop=True), n_splits=config.n_splits)
        train_residual = train["residual_15min"].to_numpy(float)
        train_price, train_baseline = train[target].to_numpy(float), train["Hourly_Baseline"].to_numpy(float)
        gates = {
            "residual": cv_gate(residual_model["oof_average"], train_residual, np.zeros_like(train_residual)),
            "direct": cv_gate(direct_model["oof_average"], train_price, train_baseline),
        }
        last_rows = last_folds_mask(len(train), config.n_splits, CANDIDATE_GATE_LAST_FOLDS)
        candidate_gates = {
            "residual": cv_gate(residual_model["oof_average"], train_residual, np.zeros_like(train_residual), last_rows),
            "direct": cv_gate(direct_model["oof_average"], train_price, train_baseline, last_rows),
        }
        residual = predict_stacked_ensemble(residual_model, test[residual_cols].reset_index(drop=True))
        direct = predict_stacked_ensemble(direct_model, test[feature_cols_15].reset_index(drop=True))
        baseline = test["Hourly_Baseline"].to_numpy()
        piece = pd.DataFrame({
            time_col: test[time_col].to_numpy(), "Actual_Price": test[target].to_numpy(), "Hourly_Baseline": baseline,
            "Direct_15min_XGB": direct["xgb"], "Direct_15min_LGBM": direct["lgb"], "Direct_15min_CAT": direct["cat"],
            "Direct_15min_Average": direct["average"], "Direct_15min_Stacked": direct["stacked"],
            "TL_Residual_XGB": baseline + residual["xgb"], "TL_Residual_LGBM": baseline + residual["lgb"],
            "TL_Residual_CAT": baseline + residual["cat"], "TL_Residual_Average": baseline + residual["average"],
            "TL_Residual_Stacked": baseline + residual["stacked"],
            "Walk_Forward_Block": number, "Train_End": train[time_col].max(), "Train_Rows": len(train),
        })
        earlier = pd.concat(pieces, ignore_index=True) if pieces else None
        champion, reason = choose_champion(earlier, default_champion, {k: bool(v["passes"]) for k, v in gates.items()})
        piece["Prediction"] = piece[champion]
        piece["Champion"] = champion
        candidate, candidate_reason = choose_champion(earlier, default_champion,
                                                      {k: bool(v["passes"]) for k, v in candidate_gates.items()})
        piece["Candidate_Prediction"] = piece[candidate]
        piece["Candidate_Champion"] = candidate
        print(f"Block {number} champion: {champion} ({reason}); residual CV MAE {gates['residual']['model_cv_mae']:.2f} "
              f"vs baseline {gates['residual']['baseline_cv_mae']:.2f}")
        pieces.append(piece)
        folds.append({"Block": number, "Train_Start": train[time_col].min(), "Train_End": train[time_col].max(),
                      "Train_Rows": len(train), "Test_Start": test[time_col].min(), "Test_End": test[time_col].max(),
                      "Test_Rows": len(test), "Champion": champion, "Champion_Reason": reason,
                      "Residual_CV_MAE": gates["residual"]["model_cv_mae"],
                      "Residual_Gate_Baseline_CV_MAE": gates["residual"]["baseline_cv_mae"],
                      "Residual_Gate_Passed": gates["residual"]["passes"],
                      "Direct_CV_MAE": gates["direct"]["model_cv_mae"],
                      "Direct_Gate_Baseline_CV_MAE": gates["direct"]["baseline_cv_mae"],
                      "Direct_Gate_Passed": gates["direct"]["passes"],
                      "Candidate_Champion": candidate, "Candidate_Reason": candidate_reason,
                      "Residual_LastCV_MAE": candidate_gates["residual"]["model_cv_mae"],
                      "Residual_LastCV_Baseline_MAE": candidate_gates["residual"]["baseline_cv_mae"],
                      "Residual_LastCV_Gate_Passed": candidate_gates["residual"]["passes"],
                      "Direct_LastCV_Gate_Passed": candidate_gates["direct"]["passes"]})

    results = pd.concat(pieces, ignore_index=True)
    results["Direct_15min_Prediction"] = results["Direct_15min_Stacked"]
    results["Static_Champion_Prediction"] = results[default_champion]
    results["Error"] = results["Actual_Price"] - results["Prediction"]
    results["Absolute_Error"] = results["Error"].abs()

    fold_table = pd.DataFrame(folds)
    by_block = results.groupby("Walk_Forward_Block")
    for column, label in [("Hourly_Baseline", "Baseline"), ("Prediction", "Champion"), ("Candidate_Prediction", "Candidate"),
                          ("TL_Residual_Average", "TL_Average"),
                          ("TL_Residual_Stacked", "TL_Stacked"), ("Direct_15min_Prediction", "Direct")]:
        fold_table[f"{label}_MAE"] = by_block.apply(
            lambda block, c=column: (block["Actual_Price"] - block[c]).abs().mean(), include_groups=False).to_numpy()
    # Train-vs-test gap of the champion family: cross-validated error on the training data vs error on the block.
    fold_table["Residual_CV_to_Test_Gap"] = fold_table["TL_Average_MAE"] - fold_table["Residual_CV_MAE"]

    coverage = calculate_15min_coverage(results[time_col])
    metadata = {"Test_Start": results[time_col].min(), "Test_End": results[time_col].max(), "Test_Rows": len(results),
                "Test_Split_Method": (f"walk_forward:{block_days:g}d blocks, expanding window, first after {min_train_days:g}d; "
                                      "champion by earlier-block MAE with a training-CV gate"),
                **coverage}
    n_direct, n_residual = len(feature_cols_15), len(residual_cols)
    uses_baseline = config.use_hourly_baseline_as_residual_feature
    metric_specs = [
        ("Walk-forward champion", "Gated nested selection", "Prediction", n_residual, uses_baseline),
        ("Candidate: last-CV-folds gate", "Candidate rule (not adopted)", "Candidate_Prediction", n_residual, uses_baseline),
        ("Hourly baseline only", "Hourly source baseline", "Hourly_Baseline", 1, None),
        ("Direct 15-min ensemble", "15-min scratch", "Direct_15min_Prediction", n_direct, None),
        ("TL residual simple average", "Transfer learning", "TL_Residual_Average", n_residual, uses_baseline),
        ("TL residual pure stacking", "Transfer learning", "TL_Residual_Stacked", n_residual, uses_baseline),
    ]
    metrics = build_metrics_table(results, "Actual_Price", metric_specs, metadata)

    outputs = {
        "forecasts": _write_frame(results, output_dir / "walk_forward_forecasts"),
        "metrics": _write_frame(metrics, output_dir / "walk_forward_metrics"),
        "folds": _write_frame(fold_table, output_dir / "walk_forward_blocks"),
    }
    summary = {
        "method": metadata["Test_Split_Method"], "configured_champion": default_champion,
        "champions": fold_table["Champion"].tolist(),
        "candidate_rule": f"training-CV gate on the last {CANDIDATE_GATE_LAST_FOLDS} CV folds only (not adopted)",
        "candidate_champions": fold_table["Candidate_Champion"].tolist(),
        "blocks": len(folds), "rows": len(results), "test_start": str(metadata["Test_Start"]),
        "test_end": str(metadata["Test_End"]), "coverage": coverage,
        "hourly_source_train_end": str(inputs["df_hourly_source_train"][time_col].max()),
        "source_paths": {key: str(value) for key, value in inputs["source_paths"].items()},
        "leakage_warnings": inputs["warnings"], "outputs": outputs,
    }
    (output_dir / "walk_forward_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nWalk-forward blocks:")
    print(fold_table[["Block", "Champion", "Residual_Gate_Passed", "Candidate_Champion", "Residual_LastCV_Gate_Passed",
                      "Baseline_MAE", "Champion_MAE", "Candidate_MAE", "TL_Average_MAE", "TL_Stacked_MAE",
                      "Direct_MAE"]].round(2).to_string(index=False))
    return summary
