from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from ..mining.lgbm_trainer import train_fast_lgbm
from ..records import CandidateFeature

MAX_SUPPORT_ROWS = 50_000
MIN_ROWS = 120
MIN_NUMERIC_FEATURES = 2
K_VALUES = (5, 15)
MAX_ANCHORS = 6
TOP_RAW_IMPORTANCE = 5
TOP_INTERACTION_ADD = 2
EPS = 1e-12


def _safe_col(name: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "col"


def _select_numeric_columns(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=[np.number]).columns.tolist()


def _preprocess_fit(X_num: pd.DataFrame) -> dict:
    med = X_num.median(axis=0).fillna(0.0)
    X_fill = X_num.fillna(med)
    mean = X_fill.mean(axis=0).to_numpy(dtype=float)
    std = X_fill.std(axis=0).to_numpy(dtype=float)
    std = np.where(np.abs(std) < 1e-12, 1.0, std)
    return {
        "medians": med.to_dict(),
        "mean": mean,
        "std": std,
        "columns": X_num.columns.tolist(),
    }


def _preprocess_apply(X: pd.DataFrame, prep: dict) -> np.ndarray:
    cols = prep["columns"]
    med = pd.Series(prep["medians"])
    arr = X[cols].fillna(med).to_numpy(dtype=float)
    arr = (arr - np.asarray(prep["mean"], dtype=float)) / np.asarray(prep["std"], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _maybe_subsample(idx: np.ndarray, max_size: int, random_state: int) -> np.ndarray:
    if len(idx) <= max_size:
        return idx
    rs = np.random.RandomState(random_state)
    out = rs.choice(idx, size=max_size, replace=False)
    out.sort()
    return out


def _choose_anchor_features(
    X_num: pd.DataFrame,
    y,
    task: str,
    random_state: int,
    ranked_pairs: list[dict] | None,
) -> list[str]:
    model, _ = train_fast_lgbm(X_num, y, task=task, random_state=random_state)
    gains = model.booster_.feature_importance(importance_type="gain")
    names = model.booster_.feature_name()

    raw_ranked = sorted(zip(names, gains), key=lambda t: float(t[1]), reverse=True)
    anchors = [n for n, _ in raw_ranked[:TOP_RAW_IMPORTANCE] if n in X_num.columns]

    if ranked_pairs:
        for row in ranked_pairs[:40]:
            a, b = row["pair"]
            for f in (a, b):
                if f in X_num.columns and f not in anchors:
                    anchors.append(f)
                if len(anchors) >= TOP_RAW_IMPORTANCE + TOP_INTERACTION_ADD:
                    break
            if len(anchors) >= TOP_RAW_IMPORTANCE + TOP_INTERACTION_ADD:
                break

    if len(anchors) < MAX_ANCHORS:
        variances = X_num.var(numeric_only=True).sort_values(ascending=False).index.tolist()
        for f in variances:
            if f not in anchors:
                anchors.append(f)
            if len(anchors) >= MAX_ANCHORS:
                break

    return anchors[:MAX_ANCHORS]


def _compute_compact_features(
    X_query: np.ndarray,
    X_support: np.ndarray,
    nn: NearestNeighbors,
    anchor_names: list[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    d, ind = nn.kneighbors(X_query, return_distance=True)

    for k in K_VALUES:
        kk = min(k, d.shape[1])
        if kk < 1:
            continue
        idx = ind[:, :kk]
        nbr = X_support[idx]  # (n_query, kk, n_features)
        nbr_mean = nbr.mean(axis=1)
        nbr_std = nbr.std(axis=1)
        delta = X_query - nbr_mean
        z = delta / np.clip(nbr_std, EPS, None)

        for j, anchor in enumerate(anchor_names):
            safe = _safe_col(anchor)
            out[f"ctx_raw_{safe}_delta_k{k}"] = delta[:, j]
            out[f"ctx_raw_{safe}_z_k{k}"] = z[:, j]

        abs_z = np.abs(z)
        out[f"ctx_raw_z_abs_mean_k{k}"] = abs_z.mean(axis=1)
        out[f"ctx_raw_z_abs_max_k{k}"] = abs_z.max(axis=1)
        out[f"ctx_raw_mean_vec_l2_k{k}"] = np.sqrt(np.sum(delta**2, axis=1))

    return out


def compute_context_features_from_state(X: pd.DataFrame, state: dict) -> pd.DataFrame:
    prep = state["preprocess"]
    X_std = _preprocess_apply(X, prep)
    X_support = np.asarray(state["support_X"], dtype=float)
    if len(X_support) == 0:
        return pd.DataFrame(index=X.index)

    nn = NearestNeighbors(n_neighbors=min(max(K_VALUES), len(X_support)), metric="euclidean")
    nn.fit(X_support)

    feats = _compute_compact_features(X_std, X_support, nn, state["anchor_names"])
    out = pd.DataFrame(feats, index=X.index)

    names = state.get("feature_names")
    if names is not None:
        missing = [n for n in names if n not in out.columns]
        for n in missing:
            out[n] = 0.0
        out = out[names]

    return out


def build_context_knn_candidates(
    X: pd.DataFrame,
    y,
    task: str,
    random_state: int,
    ranked_pairs: list[dict] | None = None,
) -> tuple[pd.DataFrame, list[CandidateFeature], dict | None]:
    num_cols = _select_numeric_columns(X)
    if len(num_cols) < MIN_NUMERIC_FEATURES or len(X) < MIN_ROWS:
        return pd.DataFrame(index=X.index), [], None

    X_num = X[num_cols].copy()
    anchor_names = _choose_anchor_features(
        X_num=X_num,
        y=y,
        task=task,
        random_state=random_state,
        ranked_pairs=ranked_pairs,
    )
    if len(anchor_names) < 2:
        return pd.DataFrame(index=X.index), [], None

    X_anchor = X_num[anchor_names]
    prep = _preprocess_fit(X_anchor)
    X_std = _preprocess_apply(X_anchor, prep)

    y_s = pd.Series(y, index=X.index)
    n_splits = min(5, len(X))
    if task in {"binary", "multiclass"}:
        min_class = int(y_s.value_counts(dropna=False).min())
        n_splits = min(n_splits, min_class)
    if n_splits < 2:
        return pd.DataFrame(index=X.index), [], None

    if task in {"binary", "multiclass"}:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = cv.split(X_std, y_s)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = cv.split(X_std)

    feature_names = []
    for k in K_VALUES:
        for a in anchor_names:
            safe = _safe_col(a)
            feature_names.extend([f"ctx_raw_{safe}_delta_k{k}", f"ctx_raw_{safe}_z_k{k}"])
        feature_names.extend([f"ctx_raw_z_abs_mean_k{k}", f"ctx_raw_z_abs_max_k{k}", f"ctx_raw_mean_vec_l2_k{k}"])

    oof_df = pd.DataFrame(index=X.index, columns=feature_names, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_idx = _maybe_subsample(np.asarray(tr_idx), MAX_SUPPORT_ROWS, random_state + fold * 97)
        X_tr = X_std[tr_idx]
        X_va = X_std[va_idx]
        kmax = min(max(K_VALUES), len(X_tr))
        if kmax < 1:
            continue
        nn = NearestNeighbors(n_neighbors=kmax, metric="euclidean")
        nn.fit(X_tr)

        feats = _compute_compact_features(X_va, X_tr, nn, anchor_names)
        for name, arr in feats.items():
            if name in oof_df.columns:
                oof_df.iloc[va_idx, oof_df.columns.get_loc(name)] = arr

    oof_df = oof_df.apply(lambda s: s.fillna(s.median()), axis=0)

    support_idx = np.arange(len(X))
    support_idx = _maybe_subsample(support_idx, MAX_SUPPORT_ROWS, random_state + 777)

    support_state = {
        "preprocess": prep,
        "support_X": X_std[support_idx],
        "anchor_names": anchor_names,
        "feature_names": feature_names,
    }

    candidates: list[CandidateFeature] = []
    for name in feature_names:
        if name.endswith("_delta_k5") or name.endswith("_delta_k15"):
            formula = "delta"
        elif name.endswith("_z_k5") or name.endswith("_z_k15"):
            formula = "local_z"
        elif "z_abs_mean" in name:
            formula = "z_abs_mean"
        elif "z_abs_max" in name:
            formula = "z_abs_max"
        else:
            formula = "mean_vec_l2"

        src = list(anchor_names) if "ctx_raw_z_abs" in name or "mean_vec_l2" in name else [
            a for a in anchor_names if _safe_col(a) in name
        ]
        if not src:
            src = list(anchor_names[:1])

        candidates.append(
            CandidateFeature(
                name=name,
                source_columns=src,
                strategy="context_knn",
                formula_name=formula,
                feature_type="context_knn",
                metadata={
                    "context_feature": name,
                    "context_kind": formula,
                    "anchor_features": anchor_names,
                },
            )
        )

    return oof_df, candidates, support_state
