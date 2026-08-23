from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _import_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The ML pipeline requires scikit-learn. Install requirements.txt before running model training."
        ) from exc

    return {
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "Ridge": Ridge,
        "TimeSeriesSplit": TimeSeriesSplit,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
    }


def optional_model_status() -> dict[str, bool]:
    import importlib.util

    return {
        "xgboost": importlib.util.find_spec("xgboost") is not None,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "catboost": importlib.util.find_spec("catboost") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
    }


def make_xgb(random_state: int = 42) -> Any:
    sklearn = _import_sklearn()
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=1.0,
            colsample_bytree=1.0,
            min_child_weight=5,
            reg_lambda=0.5,
            random_state=random_state,
            n_jobs=-1,
        )
    except ModuleNotFoundError:
        return sklearn["HistGradientBoostingRegressor"](
            max_iter=800,
            learning_rate=0.03,
            max_leaf_nodes=31,
            random_state=random_state,
        )


def make_lgb(random_state: int = 42) -> Any:
    sklearn = _import_sklearn()
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=63,
            subsample=1.0,
            colsample_bytree=0.9,
            min_child_samples=10,
            reg_lambda=0.0,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
    except ModuleNotFoundError:
        return sklearn["RandomForestRegressor"](
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )


def make_cat(random_state: int = 42) -> Any:
    sklearn = _import_sklearn()
    try:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=800,
            learning_rate=0.10,
            depth=4,
            l2_leaf_reg=1,
            loss_function="RMSE",
            verbose=0,
            random_seed=random_state,
        )
    except ModuleNotFoundError:
        return sklearn["ExtraTreesRegressor"](
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )


def base_model_names() -> dict[str, str]:
    status = optional_model_status()
    return {
        "xgb": "Tuned XGBoost" if status["xgboost"] else "HistGradientBoosting fallback",
        "lgb": "Tuned LightGBM" if status["lightgbm"] else "RandomForest fallback",
        "cat": "Tuned CatBoost" if status["catboost"] else "ExtraTrees fallback",
    }


def train_stacked_ensemble(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5) -> dict[str, Any]:
    sklearn = _import_sklearn()
    tscv = sklearn["TimeSeriesSplit"](n_splits=n_splits)
    n_train = len(X_train)
    oof_preds = np.full((n_train, 3), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        print(f"Fold {fold}: train={len(tr_idx)}, validation={len(val_idx)}")
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]

        model_xgb = make_xgb()
        model_lgb = make_lgb()
        model_cat = make_cat()

        model_xgb.fit(X_tr, y_tr)
        model_lgb.fit(X_tr, y_tr)
        model_cat.fit(X_tr, y_tr)

        oof_preds[val_idx, 0] = model_xgb.predict(X_val)
        oof_preds[val_idx, 1] = model_lgb.predict(X_val)
        oof_preds[val_idx, 2] = model_cat.predict(X_val)

    valid_oof_mask = ~np.isnan(oof_preds).any(axis=1)
    X_meta_train = pd.DataFrame(oof_preds[valid_oof_mask], columns=["xgb_pred", "lgb_pred", "cat_pred"])
    y_meta_train = y_train.loc[valid_oof_mask].reset_index(drop=True)
    meta_model = sklearn["make_pipeline"](sklearn["StandardScaler"](), sklearn["Ridge"](alpha=1.0))
    meta_model.fit(X_meta_train, y_meta_train)

    final_xgb = make_xgb()
    final_lgb = make_lgb()
    final_cat = make_cat()
    final_xgb.fit(X_train, y_train)
    final_lgb.fit(X_train, y_train)
    final_cat.fit(X_train, y_train)

    return {
        "xgb": final_xgb,
        "lgb": final_lgb,
        "cat": final_cat,
        "meta": meta_model,
        "feature_cols": list(X_train.columns),
    }


def predict_stacked_ensemble(model_bundle: dict[str, Any], X: pd.DataFrame) -> dict[str, np.ndarray]:
    xgb_pred = model_bundle["xgb"].predict(X)
    lgb_pred = model_bundle["lgb"].predict(X)
    cat_pred = model_bundle["cat"].predict(X)
    X_meta = pd.DataFrame({"xgb_pred": xgb_pred, "lgb_pred": lgb_pred, "cat_pred": cat_pred})
    stacked_pred = model_bundle["meta"].predict(X_meta)
    average_pred = (xgb_pred + lgb_pred + cat_pred) / 3
    return {
        "stacked": stacked_pred,
        "average": average_pred,
        "xgb": xgb_pred,
        "lgb": lgb_pred,
        "cat": cat_pred,
    }

