from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..mining.lgbm_trainer import train_fast_lgbm
from ..records import CandidateFeature
from .formulas import absdiff, div, mul, sub

TOP_RESIDUAL_FEATURES = 20
TOP_RESIDUAL_PAIRS = 30
MAX_RESIDUAL_CANDIDATES = 50
MIN_ROWS = 120


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < 10:
        return 0.0
    x = joined.iloc[:, 0].to_numpy(dtype=float)
    y = joined.iloc[:, 1].to_numpy(dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def _oof_residuals_regression(X: pd.DataFrame, y: pd.Series, random_state: int) -> pd.Series:
    n_splits = min(5, len(X))
    if n_splits < 2:
        baseline = float(y.mean())
        return y - baseline

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.Series(index=X.index, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        model, _ = train_fast_lgbm(X_tr, y_tr.to_numpy(), task="regression", random_state=random_state + fold * 11)
        pred = model.predict(X_va)
        oof.iloc[va_idx] = pred

    oof = oof.fillna(float(y.mean()))
    return y - oof


def build_residual_numeric_candidates(
    X: pd.DataFrame,
    y,
    task: str,
    random_state: int,
) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    if task != "regression":
        return pd.DataFrame(index=X.index), []
    if X.shape[0] < MIN_ROWS or X.shape[1] < 2:
        return pd.DataFrame(index=X.index), []

    y_s = pd.Series(y, index=X.index).astype(float)
    residual = _oof_residuals_regression(X, y_s, random_state=random_state)

    feature_scores = []
    for col in X.columns:
        score = abs(_safe_corr(X[col], residual))
        feature_scores.append((col, score))
    feature_scores.sort(key=lambda t: t[1], reverse=True)

    top_cols = [c for c, s in feature_scores[:TOP_RESIDUAL_FEATURES] if s > 0.0]
    if len(top_cols) < 2:
        return pd.DataFrame(index=X.index), []

    pair_scores = []
    for i in range(len(top_cols)):
        for j in range(i + 1, len(top_cols)):
            a, b = top_cols[i], top_cols[j]
            sa = next(s for c, s in feature_scores if c == a)
            sb = next(s for c, s in feature_scores if c == b)
            pair_scores.append(((a, b), sa + sb))
    pair_scores.sort(key=lambda t: t[1], reverse=True)
    top_pairs = [pair for pair, _ in pair_scores[:TOP_RESIDUAL_PAIRS]]

    generated: list[tuple[str, str, str, str, pd.Series]] = []
    for a, b in top_pairs:
        generated.extend(
            [
                (f"res_{a}_mul_{b}", a, b, "mul", mul(X[a], X[b])),
                (f"res_{a}_div_{b}", a, b, "div", div(X[a], X[b])),
                (f"res_{b}_div_{a}", b, a, "div", div(X[b], X[a])),
                (f"res_{a}_sub_{b}", a, b, "sub", sub(X[a], X[b])),
                (f"res_{a}_absdiff_{b}", a, b, "absdiff", absdiff(X[a], X[b])),
            ]
        )

    scored_generated = []
    for name, a, b, formula_name, values in generated:
        corr = abs(_safe_corr(values, residual))
        scored_generated.append((corr, name, a, b, formula_name, values))
    scored_generated.sort(key=lambda t: t[0], reverse=True)

    frames: list[pd.Series] = []
    candidates: list[CandidateFeature] = []
    for corr, name, a, b, formula_name, values in scored_generated[:MAX_RESIDUAL_CANDIDATES]:
        if corr <= 0.0:
            continue
        frames.append(values.rename(name))
        candidates.append(
            CandidateFeature(
                name=name,
                source_columns=[a, b],
                strategy="residual_numeric",
                formula_name=formula_name,
                feature_type="numeric_formula",
                metadata={
                    "residual_corr": float(corr),
                    "task": task,
                },
            )
        )

    if not frames:
        return pd.DataFrame(index=X.index), []
    return pd.concat(frames, axis=1), candidates
