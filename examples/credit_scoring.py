from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features_goldmine import GoldenFeatures

DATA_URL = "https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/master/credit/data.csv"


def _prepare_X(X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()
    for col in X_out.columns:
        if not pd.api.types.is_numeric_dtype(X_out[col]):
            X_out[col] = X_out[col].astype("category")
    return X_out


def _fit_eval_auc(
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

    X_train = _prepare_X(X_train)
    X_test = _prepare_X(X_test)

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
        objective="binary",
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
    proba = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, proba))


def main() -> None:
    df = pd.read_csv(DATA_URL, sep=None, engine="python")

    id_col = df.columns[0]
    target_col = df.columns[1]

    X = df.iloc[:, 2:].copy()
    y = pd.Categorical(df.iloc[:, 1]).codes
    y = pd.Series(y, name=target_col)

    print("Dataset: credit_scoring")
    print(f"Loaded: {DATA_URL}")
    print(f"Dropped id column: '{id_col}'")
    print(
        f"Rows={len(X)}, Features={X.shape[1]}, "
        f"Numeric={X.select_dtypes(include=['number']).shape[1]}, "
        f"Categorical={X.select_dtypes(exclude=['number']).shape[1]}, "
        f"Target='{target_col}'"
    )
    print("Mode: single train/test split (test_size=0.25), selectivity=balanced, gf_verbose=1")

    baseline = _fit_eval_auc(X, y, use_golden=False)
    golden = _fit_eval_auc(
        X,
        y,
        use_golden=True,
        gf_verbose=1,
        gf_selectivity="balanced",
    )

    delta = golden - baseline
    improvement_pct = (delta / baseline * 100.0) if baseline != 0 else 0.0
    print(f"Baseline AUC (single split): {baseline:.6f}")
    print(f"Golden   AUC (single split): {golden:.6f}")
    print(f"Delta AUC (golden - baseline): {delta:+.6f}")
    print(f"AUC improvement vs baseline: {improvement_pct:+.4f}%")


if __name__ == "__main__":
    main()
