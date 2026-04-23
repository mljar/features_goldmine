from __future__ import annotations

import re

import pandas as pd

from ..records import CandidateFeature
from .formulas import absdiff, div, mul, sub
from .rules import apply_rule


def _safe_col(name: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "col"


def _safe_threshold(value: float) -> str:
    # Compact readable threshold token for names, e.g. 3.14159 -> 3p142, -0.25 -> m0p25
    rounded = f"{float(value):.3f}"
    rounded = rounded.replace("-", "m").replace(".", "p")
    return rounded


def _op_token(op: str) -> str:
    if op == "<=":
        return "le"
    if op == ">":
        return "gt"
    return "op"


def build_candidates(
    X: pd.DataFrame,
    ranked_pairs: list[dict],
    paths: list[dict],
    max_pairs: int = 30,
    max_rules: int = 20,
) -> tuple[pd.DataFrame, list[CandidateFeature]]:
    frames: list[pd.Series] = []
    records: list[CandidateFeature] = []

    top_pairs = ranked_pairs[:max_pairs]

    for row in top_pairs:
        a, b = row["pair"]
        if a not in X.columns or b not in X.columns:
            continue
        if (not pd.api.types.is_numeric_dtype(X[a])) or (not pd.api.types.is_numeric_dtype(X[b])):
            continue
        pair_meta = {
            "interaction_score": row["score"],
            "pair_count": row["count"],
            "pair_total_gain": row["total_gain"],
        }
        sa, sb = _safe_col(a), _safe_col(b)

        features = [
            (f"{sa}_mul_{sb}", "mul", mul(X[a], X[b])),
            (f"{sa}_div_{sb}", "div", div(X[a], X[b])),
            (f"{sb}_div_{sa}", "div", div(X[b], X[a])),
            (f"{sa}_sub_{sb}", "sub", sub(X[a], X[b])),
            (f"{sa}_absdiff_{sb}", "absdiff", absdiff(X[a], X[b])),
        ]

        for name, formula_name, values in features:
            frames.append(values.rename(name))
            records.append(
                CandidateFeature(
                    name=name,
                    source_columns=[a, b],
                    strategy="lightgbm_paths",
                    formula_name=formula_name,
                    feature_type="numeric_formula",
                    metadata=dict(pair_meta),
                )
            )

    strong_paths = sorted(paths, key=lambda p: (p.get("path_gain", 0.0), -p.get("depth", 0.0)), reverse=True)
    rule_idx = 0
    for p in strong_paths:
        conds = p.get("conditions", [])
        if len(conds) < 2:
            continue
        c1, c2 = conds[0], conds[1]
        if c1["feature"] == c2["feature"]:
            continue
        if c1["feature"] not in X.columns or c2["feature"] not in X.columns:
            continue
        if (not pd.api.types.is_numeric_dtype(X[c1["feature"]])) or (not pd.api.types.is_numeric_dtype(X[c2["feature"]])):
            continue
        if not isinstance(c1.get("threshold"), (int, float)) or not isinstance(c2.get("threshold"), (int, float)):
            continue
        selected = [
            {"feature": c1["feature"], "op": c1["op"], "threshold": c1["threshold"]},
            {"feature": c2["feature"], "op": c2["op"], "threshold": c2["threshold"]},
        ]
        c1_name = _safe_col(c1["feature"])
        c2_name = _safe_col(c2["feature"])
        c1_op = _op_token(c1["op"])
        c2_op = _op_token(c2["op"])
        c1_thr = _safe_threshold(c1["threshold"])
        c2_thr = _safe_threshold(c2["threshold"])
        name = (
            f"rule_{c1_name}_{c1_op}_{c1_thr}"
            f"_and_{c2_name}_{c2_op}_{c2_thr}_{rule_idx:03d}"
        )
        vals = apply_rule(X, selected)
        frames.append(vals.rename(name))
        records.append(
            CandidateFeature(
                name=name,
                source_columns=[c1["feature"], c2["feature"]],
                strategy="lightgbm_paths",
                formula_name="path_rule",
                feature_type="binary_rule",
                metadata={
                    "conditions": selected,
                    "path_gain": p.get("path_gain", 0.0),
                    "path_depth": p.get("depth", 0),
                },
            )
        )
        rule_idx += 1
        if rule_idx >= max_rules:
            break

    if not frames:
        return pd.DataFrame(index=X.index), []

    X_candidates = pd.concat(frames, axis=1)
    return X_candidates, records
