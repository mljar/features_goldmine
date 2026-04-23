from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def _prepare_for_lgbm(X):
    if not isinstance(X, pd.DataFrame):
        return X
    X_out = X.copy()
    for col in X_out.columns:
        if not pd.api.types.is_numeric_dtype(X_out[col]):
            X_out[col] = X_out[col].astype("category")
    return X_out


def train_fast_lgbm(X, y, task: str, random_state: int = 42):
    common = {
        "n_estimators": 120,
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "random_state": random_state,
        "n_jobs": 1,
        "verbosity": -1,
    }

    if task == "regression":
        model = lgb.LGBMRegressor(objective="regression", **common)
    elif task == "binary":
        model = lgb.LGBMClassifier(objective="binary", **common)
    elif task == "multiclass":
        num_classes = int(len(set(y)))
        model = lgb.LGBMClassifier(objective="multiclass", num_class=num_classes, **common)
    else:
        raise ValueError(f"Unsupported task: {task}")

    X_fit = _prepare_for_lgbm(X)
    model.fit(X_fit, y)
    booster_dump = model.booster_.dump_model()
    return model, booster_dump
