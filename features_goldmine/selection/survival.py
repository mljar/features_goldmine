from __future__ import annotations

import numpy as np
import pandas as pd

from ..mining.lgbm_trainer import train_fast_lgbm
from .importance import rank_from_importance


def evaluate_survival(
    X_full: pd.DataFrame,
    y,
    candidate_names: list[str],
    task: str,
    random_state: int = 42,
    selectivity: str = "balanced",
    ignore_selectivity: bool = False,
) -> list[dict]:
    if not candidate_names:
        return []

    profiles = {
        "relaxed": {
            "n_repeats": 2,
            "top_frequency_min": 0.3,
            "rank_frac_max": 0.85,
            "top_k_frac": 1.25,
            "top_k_min": 12,
        },
        "balanced": {
            "n_repeats": 3,
            "top_frequency_min": 0.4,
            "rank_frac_max": 0.7,
            "top_k_frac": 1.0,
            "top_k_min": 10,
        },
        "strict": {
            "n_repeats": 5,
            "top_frequency_min": 0.8,
            "rank_frac_max": 0.5,
            "top_k_frac": 0.8,
            "top_k_min": 8,
        },
    }
    if selectivity not in profiles:
        raise ValueError(f"Unsupported selectivity: {selectivity}. Use one of {list(profiles)}")
    cfg = profiles[selectivity]
    n_repeats = int(cfg["n_repeats"])

    stats = {
        name: {"gains": [], "ranks": [], "top_hits": 0}
        for name in candidate_names
    }

    top_k = max(int(cfg["top_k_min"]), int(np.sqrt(X_full.shape[1]) * float(cfg["top_k_frac"])))

    for i in range(n_repeats):
        seed = random_state + i * 17
        model, _ = train_fast_lgbm(X_full, y, task=task, random_state=seed)
        gains = model.booster_.feature_importance(importance_type="gain")
        names = model.booster_.feature_name()
        ranks = rank_from_importance(np.asarray(gains, dtype=float))

        for idx, feat in enumerate(names):
            if feat not in stats:
                continue
            gain_value = float(gains[idx])
            rank_value = int(ranks[idx])
            stats[feat]["gains"].append(gain_value)
            stats[feat]["ranks"].append(rank_value)
            if rank_value <= top_k:
                stats[feat]["top_hits"] += 1

    out: list[dict] = []
    for name, values in stats.items():
        if not values["gains"]:
            continue
        mean_gain = float(np.mean(values["gains"]))
        median_rank = float(np.median(values["ranks"]))
        top_frequency = float(values["top_hits"] / n_repeats)
        out.append(
            {
                "name": name,
                "mean_gain": mean_gain,
                "median_rank": median_rank,
                "top_frequency": top_frequency,
            }
        )

    if ignore_selectivity:
        ranked = [row for row in out if row["mean_gain"] > 0.0]
        ranked.sort(key=lambda r: (r["top_frequency"], r["mean_gain"], -r["median_rank"]), reverse=True)
        return ranked

    survivors = [
        row
        for row in out
        if row["top_frequency"] >= float(cfg["top_frequency_min"])
        and row["mean_gain"] > 0.0
        and row["median_rank"] <= (X_full.shape[1] * float(cfg["rank_frac_max"]))
    ]
    survivors.sort(key=lambda r: (r["top_frequency"], r["mean_gain"], -r["median_rank"]), reverse=True)
    return survivors
