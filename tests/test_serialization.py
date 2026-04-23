from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from features_goldmine import GoldenFeatures


def _load_small_breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    ds = load_breast_cancer(as_frame=True)
    X = ds.data.iloc[:220].copy()
    y = ds.target.iloc[:220].copy()
    return X, y


def test_save_load_roundtrip_transform_equality(tmp_path: Path) -> None:
    X, y = _load_small_breast_cancer()
    gf = GoldenFeatures(verbose=0, selectivity="balanced", max_selected_features=3)
    X_gold = gf.fit_transform(X, y)

    path = tmp_path / "gf_model.joblib"
    gf.save(str(path))
    gf_loaded = GoldenFeatures.load(str(path))

    X_gold_loaded = gf_loaded.transform(X)
    assert gf_loaded.selected_feature_names_ == gf.selected_feature_names_
    assert X_gold.columns.tolist() == X_gold_loaded.columns.tolist()
    assert X_gold.equals(X_gold_loaded)


def test_transform_schema_validation_missing_column(tmp_path: Path) -> None:
    X, y = _load_small_breast_cancer()
    gf = GoldenFeatures(verbose=0, selectivity="balanced", max_selected_features=2)
    gf.fit(X, y)

    X_bad = X.drop(columns=[X.columns[0]])
    with pytest.raises(ValueError, match="missing"):
        gf.transform(X_bad)


def test_save_requires_fitted(tmp_path: Path) -> None:
    gf = GoldenFeatures(verbose=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        gf.save(str(tmp_path / "gf_model.joblib"))


def test_load_version_guard(tmp_path: Path) -> None:
    X, y = _load_small_breast_cancer()
    gf = GoldenFeatures(verbose=0, selectivity="balanced", max_selected_features=1)
    gf.fit(X, y)

    path = tmp_path / "gf_model.joblib"
    gf.save(str(path))

    payload = joblib.load(path)
    assert isinstance(payload, dict)
    payload["serialization_version"] = "999"
    joblib.dump(payload, path)

    with pytest.raises(ValueError, match="serialization version"):
        GoldenFeatures.load(str(path))
