from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..records import CandidateFeature

MAX_GROUPS = 12
GROUP_SIZE = 4
TOP_PAIRS_FOR_GROUPS = 80


def _build_groups(ranked_pairs: list[dict], available_cols: set[str]) -> list[tuple[str, ...]]:
    pair_rows = ranked_pairs[:TOP_PAIRS_FOR_GROUPS]
    neighbors = defaultdict(list)

    for row in pair_rows:
        a, b = row["pair"]
        if a not in available_cols or b not in available_cols:
            continue
        score = float(row.get("score", 0.0))
        neighbors[a].append((b, score))
        neighbors[b].append((a, score))

    feature_strength = []
    for f, items in neighbors.items():
        strength = sum(s for _, s in items)
        feature_strength.append((f, strength))
    feature_strength.sort(key=lambda t: t[1], reverse=True)

    groups: list[tuple[str, ...]] = []
    seen = set()
    for seed, _ in feature_strength:
        rel = sorted(neighbors[seed], key=lambda t: t[1], reverse=True)
        cols = [seed] + [f for f, _ in rel[: GROUP_SIZE - 1]]
        cols = [c for c in cols if c in available_cols]
        cols = sorted(set(cols))
        if len(cols) < 2:
            continue
        key = tuple(cols)
        if key in seen:
            continue
        seen.add(key)
        groups.append(key)
        if len(groups) >= MAX_GROUPS:
            break

    return groups


def _safe_stat(series: pd.Series, mode: str) -> pd.Series:
    if mode == "mean":
        return series
    return series


def build_grouped_row_stats_candidates(
    X: pd.DataFrame,
    ranked_pairs: list[dict],
) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    if X.shape[1] < 2:
        return pd.DataFrame(index=X.index), []

    groups = _build_groups(ranked_pairs, set(X.columns.tolist()))
    if not groups:
        return pd.DataFrame(index=X.index), []

    frames: list[pd.Series] = []
    candidates: list[CandidateFeature] = []

    stat_defs = ["mean", "std", "min", "max"]

    for i, cols in enumerate(groups, start=1):
        subset = X[list(cols)]
        subset = subset.replace([np.inf, -np.inf], np.nan)

        stat_values = {
            "mean": subset.mean(axis=1),
            "std": subset.std(axis=1).fillna(0.0),
            "min": subset.min(axis=1),
            "max": subset.max(axis=1),
        }

        for stat in stat_defs:
            name = f"grpstat_{i:03d}_{stat}"
            values = stat_values[stat].astype(float)
            frames.append(values.rename(name))
            candidates.append(
                CandidateFeature(
                    name=name,
                    source_columns=list(cols),
                    strategy="grouped_row_stats",
                    formula_name=f"group_{stat}",
                    feature_type="grouped_row_stats",
                    metadata={
                        "group_index": i,
                        "group_columns": list(cols),
                        "stat": stat,
                    },
                )
            )

    if not frames:
        return pd.DataFrame(index=X.index), []
    return pd.concat(frames, axis=1), candidates
