from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load

from .generation.context_knn import compute_context_features_from_state
from .records import GoldenFeature
from .reporting.report import build_report
from .strategies.lightgbm_paths import LightGBMPathsStrategy


class GoldenFeatures:
    _SERIALIZATION_VERSION = "1"
    _INTEGER_REGRESSION_UNIQUE_THRESHOLD = 20
    _ALL_STRATEGIES = {
        "path",
        "projection_pca",
        "projection_ica",
        "grouped_row_stats",
        "context_knn",
        "residual_numeric",
        "categorical_frequency",
        "categorical_oof_target",
        "categorical_group_deviation",
        "categorical_prototypes",
        "categorical_hash_cross",
    }

    def __init__(
        self,
        random_state: int = 42,
        verbose: int | bool = 0,
        selectivity: str = "balanced",
        max_selected_features: int | None = None,
        include_strategies: list[str] | None = None,
        exclude_strategies: list[str] | None = None,
    ):
        self.random_state = random_state
        self.verbose = int(verbose)
        self.selectivity = selectivity
        self.max_selected_features = max_selected_features
        if self.selectivity not in {"relaxed", "balanced", "strict"}:
            raise ValueError("selectivity must be one of: relaxed, balanced, strict")
        if self.max_selected_features is not None and self.max_selected_features < 1:
            raise ValueError("max_selected_features must be >= 1 or None")
        include_set = set(include_strategies) if include_strategies is not None else set(self._ALL_STRATEGIES)
        exclude_set = set(exclude_strategies) if exclude_strategies is not None else set()
        unknown = (include_set | exclude_set) - set(self._ALL_STRATEGIES)
        if unknown:
            raise ValueError(
                f"Unknown strategies: {sorted(unknown)}. "
                f"Allowed: {sorted(self._ALL_STRATEGIES)}"
            )
        enabled = include_set - exclude_set
        if not enabled:
            raise ValueError("No strategies enabled after include/exclude filtering.")
        self.enabled_strategies_ = sorted(enabled)
        self.strategy_name_ = "lightgbm_paths"
        self.selected_feature_names_: list[str] = []
        self.golden_features_: list[GoldenFeature] = []
        self.report_: dict[str, Any] = {}
        self._selected_metadata_by_name: dict[str, dict] = {}
        self._context_knn_state: dict | None = None
        self._fit_columns_: list[str] = []
        self._fit_numeric_columns_: list[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y) -> "GoldenFeatures":
        self._log("fit: validating inputs")
        X_all = self._validate_X(X)
        X_num = self._select_numeric(X_all)
        self._fit_columns_ = X_all.columns.astype(str).tolist()
        self._fit_numeric_columns_ = X_num.columns.astype(str).tolist()
        y_arr = self._validate_y(y)
        task = self._infer_task(y_arr)
        n_cat = int(X_all.shape[1] - X_num.shape[1])
        self._log(
            f"fit: task={task}, rows={X_all.shape[0]}, total_features={X_all.shape[1]}, "
            f"numeric_features={X_num.shape[1]}, categorical_features={n_cat}, "
            f"selectivity={self.selectivity}, max_selected_features={self.max_selected_features}, "
            f"enabled_strategies={self.enabled_strategies_}"
        )

        strategy = LightGBMPathsStrategy()
        result = strategy.run(
            X_all,
            y_arr,
            task=task,
            random_state=self.random_state,
            X_full=X_all,
            selectivity=self.selectivity,
            ignore_survival_selectivity=self.max_selected_features is not None,
            enabled_strategies=set(self.enabled_strategies_),
            verbose=self.verbose,
            logger=self._log,
        )
        self._context_knn_state = result.get("context_knn_state")

        kept_by_name = {c.name: c for c in result["kept_candidates"]}
        scores_by_name = {row["name"]: row for row in result["pruned"]}
        if self.max_selected_features is not None and len(result["pruned"]) > self.max_selected_features:
            pruned_limited = result["pruned"][: self.max_selected_features]
            self._log(
                f"fit: applying max_selected_features={self.max_selected_features} "
                f"(kept {len(pruned_limited)} of {len(result['pruned'])})"
            )
            scores_by_name = {row["name"]: row for row in pruned_limited}

        self.golden_features_ = []
        self._selected_metadata_by_name = {}

        for name, score in scores_by_name.items():
            cand = kept_by_name[name]
            meta = dict(cand.metadata)
            meta["feature_type"] = cand.feature_type
            gf = GoldenFeature(
                name=name,
                source_columns=list(cand.source_columns),
                strategy=cand.strategy,
                formula_name=cand.formula_name,
                mean_gain=float(score["mean_gain"]),
                top_frequency=float(score["top_frequency"]),
                median_rank=float(score["median_rank"]),
                feature_type=cand.feature_type,
                metadata=meta,
            )
            self.golden_features_.append(gf)
            self._selected_metadata_by_name[name] = {
                "source_columns": list(cand.source_columns),
                "formula_name": cand.formula_name,
                "feature_type": cand.feature_type,
                "metadata": dict(cand.metadata),
            }

        self.selected_feature_names_ = [g.name for g in self.golden_features_]

        selected_payload = [
            {
                "name": g.name,
                "source_columns": g.source_columns,
                "strategy": g.strategy,
                "formula_name": g.formula_name,
                "feature_type": g.feature_type,
                "mean_gain": g.mean_gain,
                "top_frequency": g.top_frequency,
                "median_rank": g.median_rank,
                "metadata": g.metadata,
            }
            for g in self.golden_features_
        ]

        self.report_ = build_report(
            strategy=self.strategy_name_,
            n_raw_features=int(X_all.shape[1]),
            n_paths=len(result["paths"]),
            n_ranked_pairs=len(result["ranked_pairs"]),
            n_candidates=int(result["X_candidates"].shape[1]),
            n_after_filter=int(result["X_filtered"].shape[1]),
            n_survivors_before_redundancy=len(result["survival"]),
            n_final=len(self.golden_features_),
            rejected_reasons=result["rejected"],
            selected_features=selected_payload,
        )

        self._log(
            "fit: completed "
            f"(candidates={result['X_candidates'].shape[1]}, "
            f"after_filter={result['X_filtered'].shape[1]}, "
            f"survivors={len(result['survival'])}, final={len(self.golden_features_)})"
        )
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X_all = self._validate_X(X)
        self._validate_transform_schema(X_all)
        X_num = self._select_numeric(X_all)
        self._log(f"transform: generating {len(self.selected_feature_names_)} golden features")

        output = pd.DataFrame(index=X_num.index)
        context_cache = None
        for name in self.selected_feature_names_:
            desc = self._selected_metadata_by_name[name]
            cols = desc["source_columns"]
            formula_name = desc["formula_name"]
            feature_type = desc["feature_type"]

            if feature_type == "numeric_formula":
                a = X_num[cols[0]]
                b = X_num[cols[1]] if len(cols) > 1 else None
                if formula_name == "mul":
                    output[name] = a * b
                elif formula_name == "div":
                    output[name] = a / (b + 1e-9)
                elif formula_name == "sub":
                    output[name] = a - b
                elif formula_name == "absdiff":
                    output[name] = np.abs(a - b)
                else:
                    raise ValueError(f"Unsupported formula: {formula_name}")
            elif feature_type == "binary_rule":
                conds = desc["metadata"].get("conditions", [])
                mask = pd.Series(True, index=X_num.index)
                for cond in conds:
                    c = X_num[cond["feature"]]
                    if cond["op"] == "<=":
                        mask = mask & (c <= float(cond["threshold"]))
                    else:
                        mask = mask & (c > float(cond["threshold"]))
                output[name] = mask.astype("int8")
            elif feature_type == "projection_pca":
                output[name] = self._transform_projection_feature(X_num, desc["metadata"], index=X_num.index)
            elif feature_type == "projection_ica":
                output[name] = self._transform_ica_feature(X_num, desc["metadata"], index=X_num.index)
            elif feature_type in {"categorical_frequency", "categorical_oof_target"}:
                col = desc["metadata"]["column"]
                mapping = desc["metadata"]["mapping"]
                fallback = float(desc["metadata"]["fallback"])
                output[name] = X_all[col].astype("object").map(mapping).fillna(fallback).astype(float)
            elif feature_type == "categorical_group_deviation":
                output[name] = self._transform_categorical_group_deviation_feature(
                    X_all,
                    metadata=desc["metadata"],
                    formula_name=formula_name,
                    index=X_num.index,
                )
            elif feature_type == "categorical_prototype":
                output[name] = self._transform_categorical_prototype_feature(
                    X_all,
                    metadata=desc["metadata"],
                    formula_name=formula_name,
                    index=X_num.index,
                )
            elif feature_type == "categorical_hash_cross":
                output[name] = self._transform_categorical_hash_cross_feature(
                    X_all,
                    metadata=desc["metadata"],
                    index=X_num.index,
                )
            elif feature_type == "grouped_row_stats":
                output[name] = self._transform_grouped_row_stat(X_num, desc["metadata"], index=X_num.index)
            elif feature_type == "context_knn":
                if context_cache is None:
                    if self._context_knn_state is None:
                        raise ValueError("Missing context_knn state; cannot transform context features.")
                    context_cache = compute_context_features_from_state(X_all, self._context_knn_state)
                output[name] = context_cache[name]
            else:
                raise ValueError(f"Unsupported feature_type: {feature_type}")

        return output

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _validate_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.empty:
            raise ValueError("X must not be empty")
        return X.copy()

    @staticmethod
    def _select_numeric(X: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        return X[numeric_cols].copy()

    def _validate_y(self, y) -> np.ndarray:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(y_arr) == 0:
            raise ValueError("y must not be empty")
        return y_arr

    def _infer_task(self, y: np.ndarray) -> str:
        if np.issubdtype(y.dtype, np.floating):
            uniq = np.unique(y)
            if len(uniq) <= 10 and np.all(np.isin(uniq, [0.0, 1.0])):
                return "binary"
            return "regression"

        uniq = np.unique(y)
        if len(uniq) <= 2:
            return "binary"
        if np.issubdtype(y.dtype, np.integer) and len(uniq) > self._INTEGER_REGRESSION_UNIQUE_THRESHOLD:
            return "regression"
        return "multiclass"

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("GoldenFeatures is not fitted. Call fit() first.")

    def _validate_transform_schema(self, X: pd.DataFrame):
        if not self._fit_columns_:
            return
        x_cols = X.columns.astype(str).tolist()
        missing = [c for c in self._fit_columns_ if c not in x_cols]
        if missing:
            show = missing[:10]
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(
                f"Input is missing {len(missing)} training columns required for transform: {show}{suffix}"
            )

    def save(self, path: str) -> None:
        self._check_fitted()
        p = Path(path)
        if p.suffix.lower() not in {".joblib", ".pkl"}:
            p = p.with_suffix(".joblib")
        payload = {
            "serialization_version": self._SERIALIZATION_VERSION,
            "package_version": "1.0.0",
            "model": self,
        }
        dump(payload, p)

    @classmethod
    def load(cls, path: str) -> "GoldenFeatures":
        payload = load(path)
        if isinstance(payload, cls):
            obj = payload
        elif isinstance(payload, dict) and "model" in payload:
            version = str(payload.get("serialization_version", "0"))
            if version != cls._SERIALIZATION_VERSION:
                raise ValueError(
                    f"Unsupported GoldenFeatures serialization version: {version}. "
                    f"Expected {cls._SERIALIZATION_VERSION}."
                )
            obj = payload["model"]
            if not isinstance(obj, cls):
                raise TypeError("Serialized payload does not contain a GoldenFeatures model.")
        else:
            raise TypeError("Unsupported file format for GoldenFeatures.load")
        obj._check_fitted()
        return obj

    def to_json_summary(self) -> str:
        data = {
            "strategy": self.strategy_name_,
            "n_selected": len(self.selected_feature_names_),
            "selected_feature_names": list(self.selected_feature_names_),
            "fit_columns": list(self._fit_columns_),
            "fit_numeric_columns": list(self._fit_numeric_columns_),
            "selectivity": self.selectivity,
            "max_selected_features": self.max_selected_features,
        }
        return json.dumps(data, ensure_ascii=True, indent=2)

    @staticmethod
    def _transform_projection_feature(X: pd.DataFrame, metadata: dict, index) -> pd.Series:
        cols = metadata["selected_columns"]
        prep = metadata["preprocess"]
        loadings = np.asarray(metadata["loadings"], dtype=float)

        X_sel = X[cols]
        med = pd.Series(prep["medians"])
        low = pd.Series(prep["clip_low"])
        high = pd.Series(prep["clip_high"])

        X_work = X_sel.fillna(med)
        X_work = X_work.clip(lower=low, upper=high, axis=1)

        arr = X_work.to_numpy(dtype=float)
        mean = np.asarray(prep["mean"], dtype=float)
        std = np.asarray(prep["std"], dtype=float)
        std = np.where(np.abs(std) < 1e-12, 1.0, std)
        arr = (arr - mean) / std
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        values = arr @ loadings
        return pd.Series(values, index=index)

    @staticmethod
    def _transform_ica_feature(X: pd.DataFrame, metadata: dict, index) -> pd.Series:
        cols = metadata["selected_columns"]
        prep = metadata["preprocess"]
        weights = np.asarray(metadata["weights"], dtype=float)
        ica_mean = np.asarray(metadata.get("ica_mean", [0.0] * len(cols)), dtype=float)

        X_sel = X[cols]
        med = pd.Series(prep["medians"])
        low = pd.Series(prep["clip_low"])
        high = pd.Series(prep["clip_high"])

        X_work = X_sel.fillna(med)
        X_work = X_work.clip(lower=low, upper=high, axis=1)

        arr = X_work.to_numpy(dtype=float)
        mean = np.asarray(prep["mean"], dtype=float)
        std = np.asarray(prep["std"], dtype=float)
        std = np.where(np.abs(std) < 1e-12, 1.0, std)
        arr = (arr - mean) / std
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        centered = arr - ica_mean
        values = centered @ weights
        return pd.Series(values, index=index)

    @staticmethod
    def _transform_grouped_row_stat(X: pd.DataFrame, metadata: dict, index) -> pd.Series:
        cols = metadata["group_columns"]
        stat = metadata["stat"]
        subset = X[cols].replace([np.inf, -np.inf], np.nan)
        if stat == "mean":
            values = subset.mean(axis=1)
        elif stat == "std":
            values = subset.std(axis=1).fillna(0.0)
        elif stat == "min":
            values = subset.min(axis=1)
        elif stat == "max":
            values = subset.max(axis=1)
        else:
            raise ValueError(f"Unsupported grouped row stat: {stat}")
        return pd.Series(values.astype(float), index=index)

    @staticmethod
    def _transform_categorical_group_deviation_feature(
        X: pd.DataFrame,
        metadata: dict,
        formula_name: str,
        index,
    ) -> pd.Series:
        cat_col = metadata["column_cat"]
        cat_s = X[cat_col].astype("object")

        if formula_name == "log_count":
            mapping = metadata["mapping_count"]
            fallback = float(metadata["fallback_count"])
            values = cat_s.map(mapping).fillna(fallback).astype(float)
            return pd.Series(values, index=index)

        num_col = metadata["column_num"]
        num_s = pd.to_numeric(X[num_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mean_map = metadata["mapping_mean"]
        std_map = metadata["mapping_std"]
        fallback_mean = float(metadata["fallback_mean"])
        fallback_std = float(metadata["fallback_std"])
        mean_s = cat_s.map(mean_map).fillna(fallback_mean).astype(float)
        std_s = cat_s.map(std_map).fillna(fallback_std).astype(float)
        num_s = num_s.fillna(fallback_mean)
        delta = num_s - mean_s

        if formula_name == "minus_group_mean":
            return pd.Series(delta.astype(float), index=index)
        if formula_name == "group_z":
            values = delta / (std_s + 1e-9)
            return pd.Series(values.astype(float), index=index)
        raise ValueError(f"Unsupported categorical_group_deviation formula: {formula_name}")

    @staticmethod
    def _transform_categorical_prototype_feature(
        X: pd.DataFrame,
        metadata: dict,
        formula_name: str,
        index,
    ) -> pd.Series:
        cat_col = metadata["column_cat"]
        anchors = metadata["anchor_columns"]
        proto_map = metadata["prototype_map"]
        global_proto = metadata["global_prototype"]

        cat_s = X[cat_col].astype("object")
        num = X[anchors].replace([np.inf, -np.inf], np.nan).astype(float)
        global_s = pd.Series(global_proto).reindex(anchors).fillna(0.0)
        num = num.fillna(global_s)

        proto_rows = pd.DataFrame(
            [proto_map.get(v, global_proto) for v in cat_s],
            index=X.index,
            columns=anchors,
        ).astype(float)
        delta = num - proto_rows

        if formula_name == "prototype_l2":
            vals = np.sqrt(np.sum(np.square(delta.to_numpy(dtype=float)), axis=1))
            return pd.Series(vals, index=index, dtype=float)
        if formula_name == "prototype_zabs_mean":
            std_s = pd.Series(metadata.get("anchor_std", {})).reindex(anchors).fillna(1.0).replace(0.0, 1.0)
            z = delta.divide(std_s + 1e-9, axis=1)
            vals = np.abs(z.to_numpy(dtype=float)).mean(axis=1)
            return pd.Series(vals, index=index, dtype=float)
        raise ValueError(f"Unsupported categorical_prototype formula: {formula_name}")

    @staticmethod
    def _transform_categorical_hash_cross_feature(
        X: pd.DataFrame,
        metadata: dict,
        index,
    ) -> pd.Series:
        import hashlib

        c1 = metadata["column_1"]
        c2 = metadata["column_2"]
        n_buckets = int(metadata["n_buckets"])
        bucket_freq = metadata["bucket_freq"]
        fallback = float(metadata["fallback"])

        s1 = X[c1].astype("object")
        s1 = s1.where(s1.notna(), "__nan__").astype(str)
        s2 = X[c2].astype("object")
        s2 = s2.where(s2.notna(), "__nan__").astype(str)
        tokens = s1 + "||" + s2

        def to_bucket(t: str) -> int:
            h = hashlib.md5(t.encode("utf-8")).hexdigest()
            return int(h, 16) % max(1, n_buckets)

        buckets = tokens.map(to_bucket)
        vals = buckets.map(bucket_freq).fillna(fallback).astype(float)
        return pd.Series(vals, index=index)

    def _log(self, message: str):
        if self.verbose:
            print(f"[GoldenFeatures] {message}")
