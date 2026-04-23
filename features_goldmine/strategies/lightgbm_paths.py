from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..filtering.quick_filters import quick_filter_candidates
from ..filtering.redundancy import prune_redundant_survivors
from ..generation.candidate_builder import build_candidates
from ..generation.categorical import build_frequency_candidates, build_oof_target_candidates
from ..generation.categorical_hash_cross import build_categorical_hash_cross_candidates
from ..generation.categorical_group_deviation import build_categorical_group_deviation_candidates
from ..generation.categorical_prototypes import build_categorical_prototype_candidates
from ..generation.context_knn import build_context_knn_candidates
from ..generation.grouped_stats import build_grouped_row_stats_candidates
from ..generation.projections import build_ica_candidates, build_projection_candidates
from ..generation.residuals import build_residual_numeric_candidates
from ..mining.interaction_ranker import rank_feature_pairs
from ..mining.lgbm_trainer import train_fast_lgbm
from ..mining.path_extractor import extract_paths
from ..selection.survival import evaluate_survival
from .base import BaseStrategy


class LightGBMPathsStrategy(BaseStrategy):
    name = "lightgbm_paths"
    _MINING_REPEATS = 3
    _MINING_MAX_ROWS = 30_000
    _MINING_MAX_TREES = 80
    _MINING_MAX_PATHS_PER_REPEAT = 20_000

    @staticmethod
    def _normalize_feature_name(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

    @staticmethod
    def _resolve_feature_name(name: str, available_columns: set[str], norm_map: dict[str, str]) -> str:
        if name in available_columns:
            return name
        space_variant = name.replace("_", " ")
        if space_variant in available_columns:
            return space_variant
        norm = LightGBMPathsStrategy._normalize_feature_name(name)
        if norm in norm_map:
            return norm_map[norm]
        return name

    def _restore_original_feature_names(self, ranked_pairs: list[dict], paths: list[dict], X: pd.DataFrame) -> None:
        columns = set(X.columns.astype(str).tolist())
        norm_map: dict[str, str] = {}
        for col in X.columns.astype(str).tolist():
            key = self._normalize_feature_name(col)
            if key not in norm_map:
                norm_map[key] = col

        for row in ranked_pairs:
            a, b = row["pair"]
            row["pair"] = (
                self._resolve_feature_name(a, columns, norm_map),
                self._resolve_feature_name(b, columns, norm_map),
            )

        for path in paths:
            if "features" in path:
                path["features"] = [self._resolve_feature_name(f, columns, norm_map) for f in path["features"]]
            if "conditions" in path:
                for cond in path["conditions"]:
                    cond["feature"] = self._resolve_feature_name(cond["feature"], columns, norm_map)

    def run(
        self,
        X: pd.DataFrame,
        y,
        task: str,
        random_state: int,
        X_full: pd.DataFrame | None = None,
        selectivity: str = "balanced",
        ignore_survival_selectivity: bool = False,
        enabled_strategies: set[str] | None = None,
        verbose: int | bool = 0,
        logger=None,
    ):
        verbosity = int(verbose)
        X_all = X
        X_num = X_all.select_dtypes(include=[np.number]).copy()

        def log(msg: str):
            if verbosity:
                if logger is not None:
                    logger(msg)
                else:
                    print(f"[GoldenFeatures] {msg}")

        log(f"stage1: training fast LightGBM on raw features ({self._MINING_REPEATS} repeats)")
        mining_models = []
        paths: list[dict] = []
        if len(X_all) > self._MINING_MAX_ROWS:
            X_mine = X_all.sample(n=self._MINING_MAX_ROWS, random_state=random_state)
            y_s = pd.Series(y, index=X_all.index)
            y_mine = y_s.loc[X_mine.index].to_numpy()
            log(
                f"stage1: sampling rows for mining "
                f"({self._MINING_MAX_ROWS}/{len(X_all)})"
            )
        else:
            X_mine = X_all
            y_mine = y
        for i in range(self._MINING_REPEATS):
            seed = random_state + i * 101
            model, booster_dump = train_fast_lgbm(X_mine, y_mine, task=task, random_state=seed)
            mining_models.append(model)
            feature_names = list(model.booster_.feature_name())
            repeat_paths = extract_paths(
                booster_dump,
                feature_names,
                max_trees=self._MINING_MAX_TREES,
                max_paths=self._MINING_MAX_PATHS_PER_REPEAT,
            )
            paths.extend(repeat_paths)
            log(
                f"stage1: repeat={i + 1}/{self._MINING_REPEATS}, "
                f"seed={seed}, paths={len(repeat_paths)}"
            )
        model = mining_models[0]
        log(f"stage1: trained with {len(model.booster_.feature_name())} raw features")
        log(f"stage2: extracted total {len(paths)} paths across repeats")

        log("stage3: ranking feature interactions")
        ranked_pairs = rank_feature_pairs(paths)
        log(f"stage3: ranked {len(ranked_pairs)} interaction pairs")
        self._restore_original_feature_names(ranked_pairs=ranked_pairs, paths=paths, X=X_all)

        enabled = enabled_strategies or {
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
        log(f"stage4: building candidate engineered features (enabled={sorted(enabled)})")
        X_source = X_full if X_full is not None else X_all
        X_candidates_paths, candidates_paths = (
            build_candidates(X_num, ranked_pairs, paths)
            if "path" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        X_candidates_proj, candidates_proj = (
            build_projection_candidates(X_num, random_state=random_state)
            if "projection_pca" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        X_candidates_ica, candidates_ica = (
            build_ica_candidates(X_num, random_state=random_state)
            if "projection_ica" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        X_candidates_freq, candidates_freq = (
            build_frequency_candidates(X_source)
            if "categorical_frequency" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        X_candidates_oof, candidates_oof = (
            build_oof_target_candidates(X_source, y=y, task=task, random_state=random_state)
            if "categorical_oof_target" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        raw_importance_order = [
            name
            for name, _gain in sorted(
                zip(model.booster_.feature_name(), model.booster_.feature_importance(importance_type="gain")),
                key=lambda t: float(t[1]),
                reverse=True,
            )
        ]
        X_candidates_gde, candidates_gde = (
            build_categorical_group_deviation_candidates(X_source, raw_importance_order=raw_importance_order)
            if "categorical_group_deviation" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        X_candidates_cproto, candidates_cproto = (
            build_categorical_prototype_candidates(X_source, raw_importance_order=raw_importance_order)
            if "categorical_prototypes" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        X_candidates_hcross, candidates_hcross = (
            build_categorical_hash_cross_candidates(X_source)
            if "categorical_hash_cross" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        X_candidates_grp, candidates_grp = (
            build_grouped_row_stats_candidates(X_num, ranked_pairs=ranked_pairs)
            if "grouped_row_stats" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        if "context_knn" in enabled:
            X_candidates_ctx, candidates_ctx, context_knn_state = build_context_knn_candidates(
                X_source,
                y=y,
                task=task,
                random_state=random_state,
                ranked_pairs=ranked_pairs,
            )
        else:
            X_candidates_ctx, candidates_ctx, context_knn_state = pd.DataFrame(index=X.index), [], None
        X_candidates_resid, candidates_resid = (
            build_residual_numeric_candidates(X_num, y=y, task=task, random_state=random_state)
            if "residual_numeric" in enabled
            else (pd.DataFrame(index=X.index), [])
        )
        X_candidates = pd.concat(
            [
                X_candidates_paths,
                X_candidates_proj,
                X_candidates_ica,
                X_candidates_grp,
                X_candidates_ctx,
                X_candidates_resid,
                X_candidates_freq,
                X_candidates_oof,
                X_candidates_gde,
                X_candidates_cproto,
                X_candidates_hcross,
            ],
            axis=1,
        )
        candidates = (
            candidates_paths
            + candidates_proj
            + candidates_ica
            + candidates_grp
            + candidates_ctx
            + candidates_resid
            + candidates_freq
            + candidates_oof
            + candidates_gde
            + candidates_cproto
            + candidates_hcross
        )
        parts = []
        if X_candidates_paths.shape[1] > 0:
            parts.append(f"{X_candidates_paths.shape[1]} path candidates")
        if X_candidates_proj.shape[1] > 0:
            parts.append(f"{X_candidates_proj.shape[1]} projection candidates")
        if X_candidates_ica.shape[1] > 0:
            parts.append(f"{X_candidates_ica.shape[1]} ica candidates")
        if X_candidates_grp.shape[1] > 0:
            parts.append(f"{X_candidates_grp.shape[1]} grouped-stats candidates")
        if X_candidates_ctx.shape[1] > 0:
            parts.append(f"{X_candidates_ctx.shape[1]} context-knn candidates")
        if X_candidates_resid.shape[1] > 0:
            parts.append(f"{X_candidates_resid.shape[1]} residual candidates")
        if X_candidates_freq.shape[1] > 0:
            parts.append(f"{X_candidates_freq.shape[1]} frequency candidates")
        if X_candidates_oof.shape[1] > 0:
            parts.append(f"{X_candidates_oof.shape[1]} oof candidates")
        if X_candidates_gde.shape[1] > 0:
            parts.append(f"{X_candidates_gde.shape[1]} grouped-deviation candidates")
        if X_candidates_cproto.shape[1] > 0:
            parts.append(f"{X_candidates_cproto.shape[1]} categorical-prototype candidates")
        if X_candidates_hcross.shape[1] > 0:
            parts.append(f"{X_candidates_hcross.shape[1]} hash-cross candidates")
        breakdown = " + ".join(parts) if parts else "0 candidates"
        log(f"stage4: generated {breakdown} = {X_candidates.shape[1]} total")

        log("stage5: quick filtering candidates")
        X_filtered, kept_candidates, rejected = quick_filter_candidates(X_num, X_candidates, candidates)
        log(
            f"stage5: kept {X_filtered.shape[1]} candidates, rejected {len(rejected)}"
        )

        log("stage6: survival competition with repeated LightGBM")
        if ignore_survival_selectivity:
            log("stage6: max_selected_features set -> selectivity thresholds ignored; ranking all candidates")
        X_full = pd.concat([X_num, X_filtered], axis=1)
        survival = evaluate_survival(
            X_full,
            y,
            candidate_names=[c.name for c in kept_candidates],
            task=task,
            random_state=random_state,
            selectivity=selectivity,
            ignore_selectivity=ignore_survival_selectivity,
        )
        log(f"stage6: {len(survival)} candidates survived")

        log("stage7: redundancy pruning")
        pruned = prune_redundant_survivors(X_filtered, survival)
        log(f"stage7: final survivors after pruning = {len(pruned)}")

        return {
            "model": model,
            "paths": paths,
            "ranked_pairs": ranked_pairs,
            "X_candidates": X_candidates,
            "X_filtered": X_filtered,
            "kept_candidates": kept_candidates,
            "rejected": rejected,
            "survival": survival,
            "pruned": pruned,
            "context_knn_state": context_knn_state,
        }
