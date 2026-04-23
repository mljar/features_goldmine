from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..records import CandidateFeature


def _series_fingerprint(series: pd.Series) -> str:
    values = np.nan_to_num(series.to_numpy(dtype=float), nan=0.0, posinf=1e12, neginf=-1e12)
    values = np.round(values, 8)
    return hashlib.md5(values.tobytes()).hexdigest()


def quick_filter_candidates(
    X_raw: pd.DataFrame,
    X_candidates: pd.DataFrame,
    candidates: list[CandidateFeature],
) -> tuple[pd.DataFrame, list[CandidateFeature], dict[str, str]]:
    if X_candidates.empty:
        return X_candidates, candidates, {}

    keep_names: list[str] = []
    rejected: dict[str, str] = {}
    seen = set()

    for cand in candidates:
        name = cand.name
        series = X_candidates[name]

        finite = np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
        finite_ratio = float(finite.mean())
        if finite_ratio < 0.95:
            rejected[name] = "too_many_nan_or_inf"
            continue

        clean = series.replace([np.inf, -np.inf], np.nan)
        non_na = clean.dropna()
        if non_na.empty:
            rejected[name] = "all_missing_after_clean"
            continue

        if float(non_na.nunique()) <= 1:
            rejected[name] = "constant"
            continue

        var = float(non_na.var())
        if var < 1e-12:
            rejected[name] = "very_low_variance"
            continue

        if cand.feature_type == "binary_rule":
            support = float(non_na.mean())
            if support < 0.01 or support > 0.99:
                rejected[name] = "rule_support_too_small_or_large"
                continue

        fp = _series_fingerprint(clean)
        if fp in seen:
            rejected[name] = "duplicate_candidate"
            continue
        seen.add(fp)

        parent_corr_too_high = False
        for parent in cand.source_columns:
            if parent not in X_raw.columns:
                continue
            parent_s = X_raw[parent].replace([np.inf, -np.inf], np.nan)
            both = pd.concat([clean, parent_s], axis=1).dropna()
            if both.empty:
                continue
            corr = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
            if np.isfinite(corr) and abs(corr) >= 0.999:
                parent_corr_too_high = True
                break
        if parent_corr_too_high:
            rejected[name] = "almost_identical_to_parent"
            continue

        keep_names.append(name)

    kept_candidates = [cand for cand in candidates if cand.name in set(keep_names)]
    return X_candidates[keep_names].copy(), kept_candidates, rejected
