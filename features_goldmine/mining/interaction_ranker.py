from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def rank_feature_pairs(paths: list[dict]) -> list[dict]:
    stats = defaultdict(lambda: {"count": 0, "total_gain": 0.0, "total_depth": 0.0, "parent_child": 0})

    for path in paths:
        features = path.get("features", [])
        if len(features) < 2:
            continue

        uniq = []
        seen = set()
        for feature in features:
            if feature not in seen:
                uniq.append(feature)
                seen.add(feature)

        path_gain = float(path.get("path_gain", 0.0))
        depth = float(path.get("depth", len(features)))

        for a, b in combinations(sorted(uniq), 2):
            key = (a, b)
            stats[key]["count"] += 1
            stats[key]["total_gain"] += path_gain
            stats[key]["total_depth"] += depth

        for i in range(len(features) - 1):
            a, b = features[i], features[i + 1]
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            stats[key]["parent_child"] += 1

    ranked = []
    for (a, b), v in stats.items():
        count = v["count"]
        mean_depth = v["total_depth"] / count if count else 0.0
        score = (v["total_gain"] * 0.7) + (count * 0.2) + (v["parent_child"] * 0.1) - (mean_depth * 0.05)
        ranked.append(
            {
                "pair": (a, b),
                "score": float(score),
                "count": int(count),
                "total_gain": float(v["total_gain"]),
                "mean_depth": float(mean_depth),
                "parent_child": int(v["parent_child"]),
            }
        )

    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked
