from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..records import CandidateFeature

MAX_CAT_COLUMNS = 6
MAX_NUMERIC_ANCHORS = 6
MAX_CAT_CARDINALITY = 40
SMOOTHING = 20.0
EPS = 1e-9


def _safe_col(name: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "col"


def _select_categorical_columns(X: pd.DataFrame) -> list[str]:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    selected: list[str] = []
    for col in cat_cols:
        if X[col].nunique(dropna=False) <= MAX_CAT_CARDINALITY:
            selected.append(col)
        if len(selected) >= MAX_CAT_COLUMNS:
            break
    return selected


def _select_numeric_anchors(X: pd.DataFrame, raw_importance_order: list[str] | None = None) -> list[str]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []

    anchors: list[str] = []
    if raw_importance_order:
        for col in raw_importance_order:
            if col in numeric_cols and col not in anchors:
                anchors.append(col)
            if len(anchors) >= MAX_NUMERIC_ANCHORS:
                break

    if len(anchors) < MAX_NUMERIC_ANCHORS:
        by_var = (
            X[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .var(numeric_only=True)
            .sort_values(ascending=False)
            .index.tolist()
        )
        for col in by_var:
            if col not in anchors:
                anchors.append(col)
            if len(anchors) >= MAX_NUMERIC_ANCHORS:
                break
    return anchors[:MAX_NUMERIC_ANCHORS]


def _smoothed_group_stats(cat_s: pd.Series, num_s: pd.Series) -> tuple[dict, dict, dict]:
    num_clean = pd.to_numeric(num_s, errors="coerce")
    num_clean = num_clean.where(np.isfinite(num_clean), np.nan)
    num_clean = num_clean.fillna(num_clean.median())
    base = pd.DataFrame({"cat": cat_s.astype("object"), "num": num_clean})
    grp = base.groupby("cat", dropna=False)["num"].agg(["mean", "std", "count", "sum"])

    global_mean = float(base["num"].mean())
    global_std = float(base["num"].std())
    if not np.isfinite(global_std) or global_std < EPS:
        global_std = 1.0

    counts = grp["count"].astype(float)
    sm_mean = (grp["sum"] + SMOOTHING * global_mean) / (counts + SMOOTHING)
    sm_var = ((counts * (grp["std"].fillna(0.0) ** 2)) + SMOOTHING * (global_std**2)) / (counts + SMOOTHING)
    sm_std = np.sqrt(np.clip(sm_var, EPS, None))
    log_count = np.log1p(counts)

    return sm_mean.to_dict(), sm_std.to_dict(), log_count.to_dict()


def build_categorical_group_deviation_candidates(
    X: pd.DataFrame,
    raw_importance_order: list[str] | None = None,
) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    cat_cols = _select_categorical_columns(X)
    num_anchors = _select_numeric_anchors(X, raw_importance_order=raw_importance_order)
    if not cat_cols or not num_anchors:
        return pd.DataFrame(index=X.index), []

    frames: list[pd.Series] = []
    recs: list[CandidateFeature] = []

    for cat in cat_cols:
        cat_s = X[cat].astype("object")
        count_map = cat_s.value_counts(dropna=False)
        log_count_map = np.log1p(count_map.astype(float)).to_dict()
        log_count_fallback = float(np.log1p(max(1.0, float(count_map.mean()) if len(count_map) else 1.0)))

        count_name = f"gde_{_safe_col(cat)}_log_count"
        count_feat = cat_s.map(log_count_map).fillna(log_count_fallback).astype(float)
        frames.append(count_feat.rename(count_name))
        recs.append(
            CandidateFeature(
                name=count_name,
                source_columns=[cat],
                strategy="categorical_group_deviation",
                formula_name="log_count",
                feature_type="categorical_group_deviation",
                metadata={
                    "column_cat": cat,
                    "column_num": None,
                    "mapping_count": log_count_map,
                    "fallback_count": log_count_fallback,
                },
            )
        )

        for num in num_anchors:
            num_s = X[num]
            mean_map, std_map, _ = _smoothed_group_stats(cat_s, num_s)

            global_mean = float(pd.to_numeric(num_s, errors="coerce").replace([np.inf, -np.inf], np.nan).mean())
            global_std = float(pd.to_numeric(num_s, errors="coerce").replace([np.inf, -np.inf], np.nan).std())
            if not np.isfinite(global_mean):
                global_mean = 0.0
            if not np.isfinite(global_std) or global_std < EPS:
                global_std = 1.0

            mapped_mean = cat_s.map(mean_map).fillna(global_mean).astype(float)
            mapped_std = cat_s.map(std_map).fillna(global_std).astype(float)
            num_clean = pd.to_numeric(num_s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(global_mean)

            delta_name = f"gde_{_safe_col(num)}_minus_{_safe_col(cat)}_mean"
            z_name = f"gde_{_safe_col(num)}_z_{_safe_col(cat)}"

            delta = (num_clean - mapped_mean).astype(float)
            z = (delta / (mapped_std + EPS)).astype(float)

            frames.append(delta.rename(delta_name))
            recs.append(
                CandidateFeature(
                    name=delta_name,
                    source_columns=[num, cat],
                    strategy="categorical_group_deviation",
                    formula_name="minus_group_mean",
                    feature_type="categorical_group_deviation",
                    metadata={
                        "column_cat": cat,
                        "column_num": num,
                        "mapping_mean": mean_map,
                        "mapping_std": std_map,
                        "fallback_mean": global_mean,
                        "fallback_std": global_std,
                    },
                )
            )

            frames.append(z.rename(z_name))
            recs.append(
                CandidateFeature(
                    name=z_name,
                    source_columns=[num, cat],
                    strategy="categorical_group_deviation",
                    formula_name="group_z",
                    feature_type="categorical_group_deviation",
                    metadata={
                        "column_cat": cat,
                        "column_num": num,
                        "mapping_mean": mean_map,
                        "mapping_std": std_map,
                        "fallback_mean": global_mean,
                        "fallback_std": global_std,
                    },
                )
            )

    if not frames:
        return pd.DataFrame(index=X.index), []
    return pd.concat(frames, axis=1), recs
