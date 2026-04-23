from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from features_goldmine import GoldenFeatures


def _fit_eval_logloss(
    X: pd.DataFrame,
    y: pd.Series,
    use_golden: bool,
    random_state: int = 42,
    gf_verbose: int = 1,
    gf_selectivity: str = "balanced",
) -> float:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )
    classes = sorted(y.unique())

    if use_golden:
        gf = GoldenFeatures(
            random_state=random_state,
            verbose=gf_verbose,
            selectivity=gf_selectivity,
        )
        X_train_gold = gf.fit_transform(X_train, y_train)
        X_test_gold = gf.transform(X_test)
        X_train = pd.concat([X_train, X_train_gold], axis=1)
        X_test = pd.concat([X_test, X_test_gold], axis=1)
        print(
            f"[Split] created={len(gf.selected_feature_names_)} features: "
            f"{gf.selected_feature_names_}"
        )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=int(len(classes)),
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    return float(log_loss(y_test, proba, labels=classes))


def main() -> None:
    dataset = load_iris(as_frame=True)
    X = dataset.data
    y = dataset.target

    print("Dataset: iris")
    print(f"Rows={len(X)}, Features={X.shape[1]}, Classes={y.nunique()}")
    print("Mode: single train/test split (test_size=0.25), selectivity=balanced, gf_verbose=1")

    baseline = _fit_eval_logloss(X, y, use_golden=False)
    golden = _fit_eval_logloss(
        X,
        y,
        use_golden=True,
        gf_verbose=1,
        gf_selectivity="balanced",
    )

    delta = golden - baseline
    improvement_pct = ((baseline - golden) / baseline * 100.0) if baseline != 0 else 0.0
    print(f"Baseline LogLoss (single split): {baseline:.6f}")
    print(f"Golden   LogLoss (single split): {golden:.6f}")
    print(f"Delta LogLoss (golden - baseline): {delta:+.6f}")
    print(f"LogLoss improvement vs baseline: {improvement_pct:+.2f}%")


if __name__ == "__main__":
    main()
