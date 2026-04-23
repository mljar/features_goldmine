from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..records import CandidateFeature

MAX_CAT_COLUMNS = 6
MAX_NUMERIC_ANCHORS = 6
MAX_CAT_CARDINALITY = 40
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


def build_categorical_prototype_candidates(
    X: pd.DataFrame,
    raw_importance_order: list[str] | None = None,
) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    cat_cols = _select_categorical_columns(X)
    anchors = _select_numeric_anchors(X, raw_importance_order=raw_importance_order)
    if not cat_cols or len(anchors) < 2:
        return pd.DataFrame(index=X.index), []

    num = X[anchors].replace([np.inf, -np.inf], np.nan)
    global_mean = num.mean(axis=0).fillna(0.0)
    num = num.fillna(global_mean)

    frames: list[pd.Series] = []
    recs: list[CandidateFeature] = []

    for cat in cat_cols:
        cat_s = X[cat].astype("object")
        grouped = num.groupby(cat_s, dropna=False).mean()
        proto_map = {k: row.to_dict() for k, row in grouped.iterrows()}

        proto_rows = pd.DataFrame(
            [proto_map.get(v, global_mean.to_dict()) for v in cat_s],
            index=X.index,
            columns=anchors,
        ).astype(float)

        delta = num - proto_rows
        l2_name = f"cproto_{_safe_col(cat)}_l2"
        zabs_name = f"cproto_{_safe_col(cat)}_zabs_mean"

        l2 = np.sqrt(np.sum(np.square(delta.to_numpy(dtype=float)), axis=1))
        proto_std = grouped.std(axis=0).reindex(anchors).fillna(1.0).replace(0.0, 1.0)
        z = delta.divide(proto_std + EPS, axis=1)
        zabs_mean = np.abs(z.to_numpy(dtype=float)).mean(axis=1)

        frames.append(pd.Series(l2, index=X.index, name=l2_name))
        recs.append(
            CandidateFeature(
                name=l2_name,
                source_columns=[cat, *anchors],
                strategy="categorical_prototypes",
                formula_name="prototype_l2",
                feature_type="categorical_prototype",
                metadata={
                    "column_cat": cat,
                    "anchor_columns": anchors,
                    "prototype_map": proto_map,
                    "global_prototype": global_mean.to_dict(),
                },
            )
        )

        frames.append(pd.Series(zabs_mean, index=X.index, name=zabs_name))
        recs.append(
            CandidateFeature(
                name=zabs_name,
                source_columns=[cat, *anchors],
                strategy="categorical_prototypes",
                formula_name="prototype_zabs_mean",
                feature_type="categorical_prototype",
                metadata={
                    "column_cat": cat,
                    "anchor_columns": anchors,
                    "prototype_map": proto_map,
                    "global_prototype": global_mean.to_dict(),
                    "anchor_std": proto_std.to_dict(),
                },
            )
        )

    if not frames:
        return pd.DataFrame(index=X.index), []
    return pd.concat(frames, axis=1), recs

