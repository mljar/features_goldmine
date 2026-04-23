from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ..records import CandidateFeature

MAX_CATEGORICAL_COLUMNS = 30
MAX_CATEGORIES_PER_COLUMN = 500
OOF_SPLITS = 5
OOF_SMOOTHING = 20.0


def _safe_col(name: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "col"


def _get_categorical_columns(X: pd.DataFrame) -> list[str]:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if len(cat_cols) > MAX_CATEGORICAL_COLUMNS:
        return cat_cols[:MAX_CATEGORICAL_COLUMNS]
    return cat_cols


def build_frequency_candidates(X: pd.DataFrame) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    cols = _get_categorical_columns(X)
    if not cols:
        return pd.DataFrame(index=X.index), []

    frames: list[pd.Series] = []
    recs: list[CandidateFeature] = []

    for col in cols:
        s = X[col]
        value_counts = s.value_counts(dropna=False)
        if len(value_counts) > MAX_CATEGORIES_PER_COLUMN:
            continue
        freqs = s.astype("object").map(value_counts / max(1, len(s))).fillna(0.0).astype(float)
        name = f"{_safe_col(col)}_freq"
        frames.append(freqs.rename(name))
        recs.append(
            CandidateFeature(
                name=name,
                source_columns=[col],
                strategy="categorical_frequency",
                formula_name="frequency",
                feature_type="categorical_frequency",
                metadata={
                    "column": col,
                    "mapping": (value_counts / max(1, len(s))).to_dict(),
                    "fallback": 0.0,
                },
            )
        )

    if not frames:
        return pd.DataFrame(index=X.index), []
    return pd.concat(frames, axis=1), recs


def _smoothed_mapping(target: pd.Series, cats: pd.Series, global_mean: float) -> pd.Series:
    grp = pd.DataFrame({"cat": cats, "y": target}).groupby("cat", dropna=False, observed=False)["y"].agg(["mean", "count"])
    return (grp["mean"] * grp["count"] + global_mean * OOF_SMOOTHING) / (grp["count"] + OOF_SMOOTHING)


def _oof_encode_one_target(cats: pd.Series, target: pd.Series, random_state: int, stratify: pd.Series | None = None):
    n_splits = min(OOF_SPLITS, len(cats))
    if stratify is not None:
        min_class_count = int(pd.Series(stratify).value_counts(dropna=False).min())
        n_splits = min(n_splits, min_class_count)
    if n_splits < 2:
        global_mean = float(target.mean())
        return pd.Series(global_mean, index=cats.index), {"mapping": {}, "global": global_mean}

    if stratify is not None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = cv.split(cats, stratify)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = cv.split(cats)

    oof = pd.Series(index=cats.index, dtype=float)
    for tr_idx, va_idx in splits:
        cat_tr = cats.iloc[tr_idx]
        y_tr = target.iloc[tr_idx]
        cat_va = cats.iloc[va_idx]
        global_mean = float(y_tr.mean())
        mapping = _smoothed_mapping(y_tr, cat_tr, global_mean)
        encoded = cat_va.astype("object").map(mapping)
        oof.iloc[va_idx] = encoded.fillna(global_mean).to_numpy(dtype=float)

    global_full = float(target.mean())
    full_mapping = _smoothed_mapping(target, cats, global_full)
    return oof.fillna(global_full), {"mapping": full_mapping.to_dict(), "global": global_full}


def build_oof_target_candidates(
    X: pd.DataFrame,
    y,
    task: str,
    random_state: int,
) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    cols = _get_categorical_columns(X)
    if not cols:
        return pd.DataFrame(index=X.index), []

    y_s = pd.Series(y, index=X.index)
    frames: list[pd.Series] = []
    recs: list[CandidateFeature] = []

    if task in {"regression", "binary"}:
        if task == "binary":
            classes = y_s.dropna().unique().tolist()
            if len(classes) != 2:
                return pd.DataFrame(index=X.index), []
            pos = sorted(classes)[-1]
            target = (y_s == pos).astype(float)
            stratify = y_s
            suffix = "te"
        else:
            target = y_s.astype(float)
            stratify = None
            suffix = "te"

        for col in cols:
            s = X[col]
            if s.nunique(dropna=False) > MAX_CATEGORIES_PER_COLUMN:
                continue
            oof, full = _oof_encode_one_target(s, target, random_state=random_state, stratify=stratify)
            name = f"{_safe_col(col)}_{suffix}"
            frames.append(oof.rename(name))
            recs.append(
                CandidateFeature(
                    name=name,
                    source_columns=[col],
                    strategy="categorical_oof_target",
                    formula_name="oof_target_mean",
                    feature_type="categorical_oof_target",
                    metadata={
                        "column": col,
                        "mapping": full["mapping"],
                        "fallback": float(full["global"]),
                        "task": task,
                    },
                )
            )

    elif task == "multiclass":
        classes = sorted(y_s.dropna().unique().tolist())
        for col in cols:
            s = X[col]
            if s.nunique(dropna=False) > MAX_CATEGORIES_PER_COLUMN:
                continue
            for c in classes:
                target = (y_s == c).astype(float)
                oof, full = _oof_encode_one_target(s, target, random_state=random_state + int(hash(str(c)) % 1000), stratify=y_s)
                name = f"{_safe_col(col)}_te_class_{_safe_col(c)}"
                frames.append(oof.rename(name))
                recs.append(
                    CandidateFeature(
                        name=name,
                        source_columns=[col],
                        strategy="categorical_oof_target",
                        formula_name="oof_target_mean_one_vs_rest",
                        feature_type="categorical_oof_target",
                        metadata={
                            "column": col,
                            "class": c,
                            "mapping": full["mapping"],
                            "fallback": float(full["global"]),
                            "task": task,
                        },
                    )
                )

    if not frames:
        return pd.DataFrame(index=X.index), []
    return pd.concat(frames, axis=1), recs
