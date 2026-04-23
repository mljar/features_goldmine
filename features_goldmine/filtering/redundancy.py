from __future__ import annotations

import numpy as np
import pandas as pd


def _composite_score(eval_row: dict) -> float:
    return float(eval_row["mean_gain"] * 0.6 + eval_row["top_frequency"] * 0.3 - eval_row["median_rank"] * 0.1)


def prune_redundant_survivors(
    X_candidates: pd.DataFrame,
    evaluations: list[dict],
    corr_threshold: float = 0.995,
) -> list[dict]:
    if not evaluations:
        return []

    ranked = sorted(evaluations, key=_composite_score, reverse=True)
    selected: list[dict] = []

    for row in ranked:
        name = row["name"]
        if name not in X_candidates.columns:
            continue
        keep = True
        for kept in selected:
            kname = kept["name"]
            joined = pd.concat([X_candidates[name], X_candidates[kname]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if joined.empty:
                continue
            corr = float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])
            if np.isfinite(corr) and abs(corr) >= corr_threshold:
                keep = False
                break
        if keep:
            selected.append(row)

    return selected
