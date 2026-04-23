from __future__ import annotations

from sklearn.datasets import load_breast_cancer

from features_goldmine import GoldenFeatures


def test_max_selected_features_cap_and_transform_shape() -> None:
    ds = load_breast_cancer(as_frame=True)
    X = ds.data.iloc[:260].copy()
    y = ds.target.iloc[:260].copy()

    gf = GoldenFeatures(verbose=0, selectivity="strict", max_selected_features=1)
    X_gold = gf.fit_transform(X, y)

    assert len(gf.selected_feature_names_) <= 1
    assert X_gold.shape[1] == len(gf.selected_feature_names_)
