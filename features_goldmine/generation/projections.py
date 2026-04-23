from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
import warnings

from ..records import CandidateFeature

MAX_PCA_ROWS_FIT = 50_000
MAX_PCA_FEATURES = 400
MAX_PCA_COMPONENTS = 32
MIN_PCA_ROWS = 100
MIN_PCA_FEATURES = 3
CLIP_LOW_Q = 0.005
CLIP_HIGH_Q = 0.995

MAX_ICA_ROWS_FIT = 30_000
MAX_ICA_FEATURES = 200
MAX_ICA_COMPONENTS = 16
MIN_ICA_ROWS = 300
MIN_ICA_FEATURES = 5


def _prepare_selected_columns(X: pd.DataFrame) -> list[str]:
    if X.shape[1] <= MAX_PCA_FEATURES:
        return X.columns.tolist()
    variances = X.var(numeric_only=True).sort_values(ascending=False)
    return variances.head(MAX_PCA_FEATURES).index.tolist()


def _safe_std(std: np.ndarray) -> np.ndarray:
    out = std.copy()
    out[~np.isfinite(out)] = 1.0
    out[out < 1e-12] = 1.0
    return out


def _fit_preprocess_params(X_sel: pd.DataFrame, fit_idx: np.ndarray) -> dict:
    X_fit = X_sel.iloc[fit_idx]

    medians = X_fit.median(axis=0).fillna(0.0)
    X_median = X_fit.fillna(medians)

    q_low = X_median.quantile(CLIP_LOW_Q, axis=0)
    q_high = X_median.quantile(CLIP_HIGH_Q, axis=0)

    X_clip = X_median.clip(lower=q_low, upper=q_high, axis=1)
    mean = X_clip.mean(axis=0).to_numpy(dtype=float)
    std = _safe_std(X_clip.std(axis=0).to_numpy(dtype=float))

    return {
        "columns": X_sel.columns.tolist(),
        "medians": medians.to_dict(),
        "clip_low": q_low.to_dict(),
        "clip_high": q_high.to_dict(),
        "mean": mean,
        "std": std,
    }


def _apply_preprocess(X: pd.DataFrame, params: dict) -> np.ndarray:
    cols = params["columns"]
    X_sel = X[cols]

    med = pd.Series(params["medians"])
    low = pd.Series(params["clip_low"])
    high = pd.Series(params["clip_high"])

    X_work = X_sel.fillna(med)
    X_work = X_work.clip(lower=low, upper=high, axis=1)

    arr = X_work.to_numpy(dtype=float)
    arr = (arr - params["mean"]) / params["std"]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def build_projection_candidates(
    X: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    n_rows, n_features = X.shape
    if n_rows < MIN_PCA_ROWS or n_features < MIN_PCA_FEATURES:
        return pd.DataFrame(index=X.index), []

    selected_cols = _prepare_selected_columns(X)
    if len(selected_cols) < MIN_PCA_FEATURES:
        return pd.DataFrame(index=X.index), []

    X_sel = X[selected_cols]

    rs = np.random.RandomState(random_state)
    if len(X_sel) > MAX_PCA_ROWS_FIT:
        fit_idx = rs.choice(len(X_sel), size=MAX_PCA_ROWS_FIT, replace=False)
        fit_idx.sort()
    else:
        fit_idx = np.arange(len(X_sel))

    params = _fit_preprocess_params(X_sel, fit_idx)

    X_fit_std = _apply_preprocess(X_sel.iloc[fit_idx], params)
    max_components = min(
        MAX_PCA_COMPONENTS,
        len(selected_cols),
        max(1, X_fit_std.shape[0] - 1),
    )
    if max_components < 1:
        return pd.DataFrame(index=X.index), []

    suggested = max(3, int(np.sqrt(len(selected_cols))))
    n_components = min(suggested, max_components)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca.fit(X_fit_std)

    X_full_std = _apply_preprocess(X_sel, params)
    comps = pca.transform(X_full_std)

    frames: list[pd.Series] = []
    candidates: list[CandidateFeature] = []

    for i in range(n_components):
        name = f"pca_comp_{i + 1:03d}"
        values = pd.Series(comps[:, i], index=X.index, name=name)
        frames.append(values)

        loadings = pca.components_[i]
        top_idx = np.argsort(np.abs(loadings))[::-1][:3]
        top_columns = [selected_cols[j] for j in top_idx]

        candidates.append(
            CandidateFeature(
                name=name,
                source_columns=top_columns,
                strategy="projection_numeric",
                formula_name="pca_component",
                feature_type="projection_pca",
                metadata={
                    "component_index": i,
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[i]),
                    "selected_columns": selected_cols,
                    "loadings": loadings.tolist(),
                    "preprocess": {
                        "medians": params["medians"],
                        "clip_low": params["clip_low"],
                        "clip_high": params["clip_high"],
                        "mean": params["mean"].tolist(),
                        "std": params["std"].tolist(),
                    },
                    "limits": {
                        "max_rows_fit": MAX_PCA_ROWS_FIT,
                        "max_features": MAX_PCA_FEATURES,
                        "max_components": MAX_PCA_COMPONENTS,
                    },
                },
            )
        )

    if not frames:
        return pd.DataFrame(index=X.index), []

    return pd.concat(frames, axis=1), candidates


def _fit_ica_with_retry(X_fit_std: np.ndarray, n_components: int, random_state: int):
    components = n_components
    for _ in range(2):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=ConvergenceWarning)
            ica = FastICA(
                n_components=components,
                random_state=random_state,
                max_iter=400,
                tol=1e-4,
                whiten="unit-variance",
            )
            ica.fit(X_fit_std)
            has_conv_warning = any(issubclass(x.category, ConvergenceWarning) for x in w)
        if not has_conv_warning:
            return ica, components
        if components <= 2:
            break
        components = max(1, components // 2)
    return None, 0


def build_ica_candidates(
    X: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    n_rows, n_features = X.shape
    if n_rows < MIN_ICA_ROWS or n_features < MIN_ICA_FEATURES:
        return pd.DataFrame(index=X.index), []

    cols_all = _prepare_selected_columns(X)
    selected_cols = cols_all[:MAX_ICA_FEATURES]
    if len(selected_cols) < MIN_ICA_FEATURES:
        return pd.DataFrame(index=X.index), []

    X_sel = X[selected_cols]

    rs = np.random.RandomState(random_state + 17)
    if len(X_sel) > MAX_ICA_ROWS_FIT:
        fit_idx = rs.choice(len(X_sel), size=MAX_ICA_ROWS_FIT, replace=False)
        fit_idx.sort()
    else:
        fit_idx = np.arange(len(X_sel))

    params = _fit_preprocess_params(X_sel, fit_idx)
    X_fit_std = _apply_preprocess(X_sel.iloc[fit_idx], params)

    max_components = min(
        MAX_ICA_COMPONENTS,
        len(selected_cols),
        max(1, X_fit_std.shape[0] - 1),
    )
    if max_components < 1:
        return pd.DataFrame(index=X.index), []

    suggested = max(3, int(np.sqrt(len(selected_cols))))
    n_components = min(suggested, max_components)
    ica, final_components = _fit_ica_with_retry(
        X_fit_std=X_fit_std,
        n_components=n_components,
        random_state=random_state,
    )
    if ica is None or final_components < 1:
        return pd.DataFrame(index=X.index), []

    X_full_std = _apply_preprocess(X_sel, params)
    comps = ica.transform(X_full_std)

    frames: list[pd.Series] = []
    candidates: list[CandidateFeature] = []

    for i in range(final_components):
        name = f"ica_comp_{i + 1:03d}"
        values = pd.Series(comps[:, i], index=X.index, name=name)
        frames.append(values)

        comp_weights = ica.components_[i]
        top_idx = np.argsort(np.abs(comp_weights))[::-1][:3]
        top_columns = [selected_cols[j] for j in top_idx]

        candidates.append(
            CandidateFeature(
                name=name,
                source_columns=top_columns,
                strategy="projection_numeric",
                formula_name="ica_component",
                feature_type="projection_ica",
                metadata={
                    "component_index": i,
                    "selected_columns": selected_cols,
                    "weights": comp_weights.tolist(),
                    "ica_mean": ica.mean_.tolist() if hasattr(ica, "mean_") else [0.0] * len(selected_cols),
                    "preprocess": {
                        "medians": params["medians"],
                        "clip_low": params["clip_low"],
                        "clip_high": params["clip_high"],
                        "mean": params["mean"].tolist(),
                        "std": params["std"].tolist(),
                    },
                    "limits": {
                        "max_rows_fit": MAX_ICA_ROWS_FIT,
                        "max_features": MAX_ICA_FEATURES,
                        "max_components": MAX_ICA_COMPONENTS,
                    },
                },
            )
        )

    if not frames:
        return pd.DataFrame(index=X.index), []
    return pd.concat(frames, axis=1), candidates
