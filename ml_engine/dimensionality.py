"""
AceML Studio – Dimensionality Reduction
=========================================
PCA, feature-importance selection, variance-threshold, correlation filter.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from config import Config

logger = logging.getLogger("aceml.dimensionality")


class DimensionalityReducer:
    """Dimensionality reduction and feature-selection utilities."""

    # ------------------------------------------------------------------ #
    #  PCA
    # ------------------------------------------------------------------ #
    @staticmethod
    def pca_reduce(df: pd.DataFrame, n_components: int | float | None = None,
                   columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
        """
        n_components: int (exact), float 0-1 (variance ratio), None (auto 95 %).
        Returns (reduced_df, info_dict).
        """
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            logger.warning("PCA: no numeric columns found")
            return df, {"error": "No numeric columns"}

        logger.info("PCA: reducing %d numeric columns (n_components=%s)", len(cols), n_components)
        X = df[cols].fillna(0).values
        if n_components is None:
            n_components = Config.PCA_DEFAULT_VARIANCE

        pca = PCA(n_components=n_components, random_state=Config.DEFAULT_RANDOM_STATE)
        transformed = pca.fit_transform(X)

        pca_cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
        pca_df = pd.DataFrame(transformed, columns=pca_cols, index=df.index)

        # Keep non-PCA columns
        other = [c for c in df.columns if c not in cols]
        result = pd.concat([df[other], pca_df], axis=1)

        info = {
            "n_components": int(transformed.shape[1]),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "original_features": len(cols),
            "removed_features": len(cols) - transformed.shape[1],
        }
        logger.info("PCA complete: %d→%d components (cumulative variance=%.4f)",
                    len(cols), transformed.shape[1],
                    np.cumsum(pca.explained_variance_ratio_)[-1])
        return result, info

    # ------------------------------------------------------------------ #
    #  Variance Threshold
    # ------------------------------------------------------------------ #
    @staticmethod
    def variance_threshold(df: pd.DataFrame, threshold: float | None = None,
                           columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
        thresh = threshold if threshold is not None else Config.LOW_VARIANCE_THRESHOLD
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            return df, {"removed": []}

        X = df[cols].fillna(0)
        selector = VarianceThreshold(threshold=thresh)
        selector.fit(X)
        keep_mask = selector.get_support()
        removed = [c for c, keep in zip(cols, keep_mask) if not keep]
        kept = [c for c, keep in zip(cols, keep_mask) if keep]

        other = [c for c in df.columns if c not in cols]
        result = df[other + kept].copy()

        logger.info("Variance threshold (%.4f): removed %d columns %s", thresh, len(removed), removed)
        return result, {"removed": removed, "kept": kept, "threshold": thresh}

    # ------------------------------------------------------------------ #
    #  Correlation-based Filter
    # ------------------------------------------------------------------ #
    @staticmethod
    def correlation_filter(df: pd.DataFrame, threshold: float | None = None,
                           columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
        thresh = threshold if threshold is not None else Config.HIGH_CORRELATION_THRESHOLD
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        if len(cols) < 2:
            return df, {"removed": []}

        corr = df[cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > thresh)]

        result = df.drop(columns=to_drop).copy()
        logger.info("Correlation filter (threshold=%.4f): removed %d columns %s", thresh, len(to_drop), to_drop)
        return result, {"removed": to_drop, "threshold": thresh}

    # ------------------------------------------------------------------ #
    #  Feature Importance Selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def feature_importance_selection(df: pd.DataFrame, target: str,
                                     task: str = "classification",
                                     top_k: int | None = None) -> tuple[pd.DataFrame, dict]:
        if target not in df.columns:
            logger.error("Feature importance: target '%s' not found in columns", target)
            return df, {"error": f"Target '{target}' not found"}

        X_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
        if not X_cols:
            logger.warning("Feature importance: no numeric features available")
            return df, {"error": "No numeric features"}

        X = df[X_cols].fillna(0)
        y = df[target]

        if task == "classification":
            model = RandomForestClassifier(n_estimators=50, random_state=Config.DEFAULT_RANDOM_STATE, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=Config.DEFAULT_RANDOM_STATE, n_jobs=-1)

        model.fit(X, y)
        importances = dict(zip(X_cols, model.feature_importances_.tolist()))
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        k = top_k or max(1, len(X_cols) // 2)
        selected = [name for name, _ in sorted_imp[:k]]

        other = [c for c in df.columns if c not in X_cols]
        result = df[other + selected].copy()

        logger.info("Feature importance selection: kept top %d of %d features (task=%s)",
                    k, len(X_cols), task)
        return result, {
            "importances": {n: round(v, 6) for n, v in sorted_imp},
            "selected": selected,
            "removed": [n for n, _ in sorted_imp[k:]],
        }
