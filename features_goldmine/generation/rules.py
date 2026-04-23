from __future__ import annotations

import pandas as pd


def apply_rule(X: pd.DataFrame, conditions: list[dict]) -> pd.Series:
    mask = pd.Series(True, index=X.index)
    for cond in conditions:
        feature = cond["feature"]
        threshold = float(cond["threshold"])
        op = cond["op"]
        if op == "<=":
            mask = mask & (X[feature] <= threshold)
        else:
            mask = mask & (X[feature] > threshold)
    return mask.astype("int8")
