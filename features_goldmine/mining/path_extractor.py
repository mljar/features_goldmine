from __future__ import annotations

from copy import deepcopy


def _coerce_threshold(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def extract_paths(
    booster_dump: dict,
    feature_names: list[str],
    max_trees: int | None = None,
    max_paths: int | None = None,
) -> list[dict]:
    paths: list[dict] = []

    def walk(node: dict, tree_index: int, conditions: list[dict], gains: list[float]):
        if max_paths is not None and len(paths) >= max_paths:
            return
        if "split_index" not in node:
            paths.append(
                {
                    "tree_index": tree_index,
                    "features": [c["feature"] for c in conditions],
                    "conditions": deepcopy(conditions),
                    "path_gain": float(sum(gains)),
                    "depth": len(conditions),
                    "leaf_value": float(node.get("leaf_value", 0.0)),
                }
            )
            return

        split_feature_idx = int(node["split_feature"])
        split_gain = float(node.get("split_gain", 0.0))
        threshold = _coerce_threshold(node.get("threshold", 0.0))
        feature_name = feature_names[split_feature_idx]
        decision_type = node.get("decision_type", "<=")
        default_left = bool(node.get("default_left", True))

        left_condition = {
            "feature": feature_name,
            "op": "<=",
            "threshold": threshold,
            "gain": split_gain,
            "decision_type": decision_type,
            "default_left": default_left,
            "direction": "left",
        }
        walk(node["left_child"], tree_index, conditions + [left_condition], gains + [split_gain])

        right_condition = {
            "feature": feature_name,
            "op": ">",
            "threshold": threshold,
            "gain": split_gain,
            "decision_type": decision_type,
            "default_left": default_left,
            "direction": "right",
        }
        walk(node["right_child"], tree_index, conditions + [right_condition], gains + [split_gain])

    tree_info = booster_dump.get("tree_info", [])
    if max_trees is not None:
        tree_info = tree_info[:max_trees]
    for tree in tree_info:
        if max_paths is not None and len(paths) >= max_paths:
            break
        walk(tree["tree_structure"], int(tree["tree_index"]), [], [])

    return paths
