from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateFeature:
    name: str
    source_columns: list[str]
    strategy: str
    formula_name: str
    feature_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class GoldenFeature:
    name: str
    source_columns: list[str]
    strategy: str
    formula_name: str
    mean_gain: float
    top_frequency: float
    median_rank: float
    feature_type: str
    metadata: dict = field(default_factory=dict)
