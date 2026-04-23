from __future__ import annotations

import hashlib
import itertools
import re

import pandas as pd

from ..records import CandidateFeature

MAX_CAT_COLUMNS = 8
MAX_CAT_CARDINALITY = 80
MAX_PAIRS = 8
HASH_BUCKETS = 64


def _safe_col(name: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "col"


def _stable_bucket(text: str, n_buckets: int) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h, 16) % max(1, int(n_buckets))


def _select_categorical_columns(X: pd.DataFrame) -> list[str]:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    selected: list[str] = []
    for col in cat_cols:
        if X[col].nunique(dropna=False) <= MAX_CAT_CARDINALITY:
            selected.append(col)
        if len(selected) >= MAX_CAT_COLUMNS:
            break
    return selected


def build_categorical_hash_cross_candidates(X: pd.DataFrame) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    cat_cols = _select_categorical_columns(X)
    if len(cat_cols) < 2:
        return pd.DataFrame(index=X.index), []

    pairs = list(itertools.combinations(cat_cols, 2))[:MAX_PAIRS]
    frames: list[pd.Series] = []
    recs: list[CandidateFeature] = []

    for c1, c2 in pairs:
        s1 = X[c1].astype("object")
        s1 = s1.where(s1.notna(), "__nan__")
        s2 = X[c2].astype("object")
        s2 = s2.where(s2.notna(), "__nan__")
        tokens = s1.astype(str) + "||" + s2.astype(str)
        buckets = tokens.map(lambda t: _stable_bucket(t, HASH_BUCKETS))
        bucket_freq = (buckets.value_counts(dropna=False) / max(1, len(buckets))).to_dict()
        feat = buckets.map(bucket_freq).fillna(0.0).astype(float)
        name = f"hcross_{_safe_col(c1)}_{_safe_col(c2)}_freq"
        frames.append(feat.rename(name))
        recs.append(
            CandidateFeature(
                name=name,
                source_columns=[c1, c2],
                strategy="categorical_hash_cross",
                formula_name="hash_bucket_frequency",
                feature_type="categorical_hash_cross",
                metadata={
                    "column_1": c1,
                    "column_2": c2,
                    "n_buckets": HASH_BUCKETS,
                    "bucket_freq": bucket_freq,
                    "fallback": 0.0,
                },
            )
        )

    if not frames:
        return pd.DataFrame(index=X.index), []
    return pd.concat(frames, axis=1), recs
