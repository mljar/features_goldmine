from __future__ import annotations


def build_report(
    strategy: str,
    n_raw_features: int,
    n_paths: int,
    n_ranked_pairs: int,
    n_candidates: int,
    n_after_filter: int,
    n_survivors_before_redundancy: int,
    n_final: int,
    rejected_reasons: dict[str, str],
    selected_features: list[dict],
) -> dict:
    reason_counts: dict[str, int] = {}
    for reason in rejected_reasons.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "strategy": strategy,
        "summary": {
            "n_raw_features": n_raw_features,
            "n_paths": n_paths,
            "n_ranked_pairs": n_ranked_pairs,
            "n_candidates": n_candidates,
            "n_after_filter": n_after_filter,
            "n_survivors_before_redundancy": n_survivors_before_redundancy,
            "n_final": n_final,
        },
        "rejections": {
            "count": len(rejected_reasons),
            "reason_counts": reason_counts,
            "by_feature": rejected_reasons,
        },
        "selected": selected_features,
    }
