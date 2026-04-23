from __future__ import annotations

import numpy as np


def rank_from_importance(gains: np.ndarray) -> np.ndarray:
    order = np.argsort(-gains)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(gains) + 1)
    return ranks
