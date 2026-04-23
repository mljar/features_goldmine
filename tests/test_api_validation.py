from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from features_goldmine import GoldenFeatures


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown strategies"):
        GoldenFeatures(include_strategies=["not_a_strategy"])


def test_empty_enabled_strategies_raises() -> None:
    with pytest.raises(ValueError, match="No strategies enabled"):
        GoldenFeatures(include_strategies=["path"], exclude_strategies=["path"])


def test_categorical_only_fit_transform_works() -> None:
    X = pd.DataFrame(
        {
            "cat_a": ["x", "y", "x", "z", "y", "x", "z", "y", "x", "z"],
            "cat_b": ["a", "b", "a", "a", "c", "b", "c", "a", "b", "c"],
        }
    )
    y = pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    gf = GoldenFeatures(verbose=0, selectivity="relaxed")
    X_gold = gf.fit_transform(X, y)
    assert isinstance(X_gold, pd.DataFrame)
    assert X_gold.shape[0] == X.shape[0]


def test_max_selected_features_validation() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        GoldenFeatures(max_selected_features=0)


def test_to_json_summary_after_fit() -> None:
    ds = load_breast_cancer(as_frame=True)
    X = ds.data.iloc[:150]
    y = ds.target.iloc[:150]
    gf = GoldenFeatures(verbose=0, max_selected_features=2)
    gf.fit(X, y)
    text = gf.to_json_summary()
    assert "selected_feature_names" in text


def test_integer_target_with_many_unique_values_is_regression() -> None:
    gf = GoldenFeatures(verbose=0)
    y = pd.Series(range(50), dtype="int64").to_numpy()
    assert gf._infer_task(y) == "regression"


def test_integer_target_with_few_unique_values_is_multiclass() -> None:
    gf = GoldenFeatures(verbose=0)
    y = pd.Series([0, 1, 2, 0, 1, 2, 1, 0], dtype="int64").to_numpy()
    assert gf._infer_task(y) == "multiclass"
