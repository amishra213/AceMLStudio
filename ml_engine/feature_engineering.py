"""
AceML Studio – Feature Engineering
====================================
Extract date features, text features, interaction terms, polynomial
features, and custom binning.  Each method returns a new DataFrame.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger("aceml.feature_engineering")


class FeatureEngineer:
    """Collection of feature-engineering utilities."""

    # ------------------------------------------------------------------ #
    #  Date / Datetime Features
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_date_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        For each datetime column, create: year, month, day, weekday,
        quarter, is_weekend, day_of_year.
        """
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            dt = pd.to_datetime(df[col], errors="coerce")
            prefix = col
            df[f"{prefix}_year"] = dt.dt.year.astype("Int64")
            df[f"{prefix}_month"] = dt.dt.month.astype("Int64")
            df[f"{prefix}_day"] = dt.dt.day.astype("Int64")
            df[f"{prefix}_weekday"] = dt.dt.weekday.astype("Int64")
            df[f"{prefix}_quarter"] = dt.dt.quarter.astype("Int64")
            df[f"{prefix}_is_weekend"] = dt.dt.weekday.isin([5, 6]).astype(int)
            df[f"{prefix}_day_of_year"] = dt.dt.dayofyear.astype("Int64")
        return df

    # ------------------------------------------------------------------ #
    #  Interaction Features  (col_a * col_b for numeric pairs)
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_interactions(df: pd.DataFrame, column_pairs: list[list[str]]) -> pd.DataFrame:
        """column_pairs = [["col_a", "col_b"], ...]"""
        df = df.copy()
        for pair in column_pairs:
            if len(pair) != 2:
                continue
            a, b = pair
            if a in df.columns and b in df.columns:
                df[f"{a}_x_{b}"] = df[a] * df[b]
        return df

    # ------------------------------------------------------------------ #
    #  Polynomial Features
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_polynomial(df: pd.DataFrame, columns: list[str], degree: int = 2,
                          interaction_only: bool = False) -> pd.DataFrame:
        """
        Create polynomial features with safety limits to prevent memory explosion.
        
        Safety limits:
        - Max 20 input columns
        - Max degree of 3
        - Max 1000 output features
        """
        df = df.copy()
        valid = [c for c in columns if c in df.columns and np.issubdtype(df[c].dtype, np.number)]  # type: ignore
        if not valid:
            return df
        
        # Safety checks
        MAX_INPUT_COLS = 20
        MAX_DEGREE = 3
        MAX_OUTPUT_FEATURES = 1000
        
        if len(valid) > MAX_INPUT_COLS:
            logger.error(
                "BLOCKED: Polynomial features on %d columns requested. Maximum is %d to prevent memory issues. "
                "Select fewer columns or use feature importance to reduce dimensionality first.",
                len(valid), MAX_INPUT_COLS
            )
            raise ValueError(
                f"Too many columns for polynomial features ({len(valid)}). "
                f"Maximum is {MAX_INPUT_COLS}. Select fewer columns or reduce dimensions first."
            )
        
        if degree > MAX_DEGREE:
            logger.warning("Degree %d exceeds maximum %d, capping at %d", degree, MAX_DEGREE, MAX_DEGREE)
            degree = MAX_DEGREE
        
        # Estimate output features before creating them
        from math import comb
        if interaction_only:
            n_output = sum(comb(len(valid), i) for i in range(1, degree + 1))
        else:
            n_output = comb(len(valid) + degree, degree) - 1
        
        if n_output > MAX_OUTPUT_FEATURES:
            logger.error(
                "BLOCKED: Polynomial expansion would create %d features (max: %d). "
                "Reduce input columns or degree.",
                n_output, MAX_OUTPUT_FEATURES
            )
            raise ValueError(
                f"Polynomial expansion would create {n_output:,} features (max: {MAX_OUTPUT_FEATURES}). "
                f"Reduce the number of input columns ({len(valid)}) or degree ({degree})."
            )
        
        logger.info(
            "Creating polynomial features: %d columns, degree=%d, interaction_only=%s → ~%d features",
            len(valid), degree, interaction_only, n_output
        )
        
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        arr = poly.fit_transform(df[valid].fillna(0))
        names = poly.get_feature_names_out(valid)
        poly_df = pd.DataFrame(arr, columns=names, index=df.index)
        # Drop original columns that are duplicated
        new_cols = [c for c in poly_df.columns if c not in df.columns]
        logger.info("Added %d polynomial features", len(new_cols))
        return pd.concat([df, poly_df[new_cols]], axis=1)

    # ------------------------------------------------------------------ #
    #  Binning / Bucketing
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_bins(df: pd.DataFrame, column: str, bins: int = 5,
                    labels: list[str] | None = None, strategy: str = "quantile") -> pd.DataFrame:
        """strategy: 'quantile' or 'uniform'."""
        df = df.copy()
        if column not in df.columns:
            return df
        if strategy == "quantile":
            df[f"{column}_bin"] = pd.qcut(df[column], q=bins, labels=labels, duplicates="drop")
        else:
            df[f"{column}_bin"] = pd.cut(df[column], bins=bins, labels=labels)
        return df

    # ------------------------------------------------------------------ #
    #  Text Length Feature
    # ------------------------------------------------------------------ #
    @staticmethod
    def text_length_feature(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns and df[col].dtype == object:
                df[f"{col}_length"] = df[col].fillna("").str.len()
                df[f"{col}_word_count"] = df[col].fillna("").str.split().str.len()
        return df

    # ------------------------------------------------------------------ #
    #  Math Transforms
    # ------------------------------------------------------------------ #
    @staticmethod
    def log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):  # type: ignore
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
        return df

    @staticmethod
    def sqrt_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):  # type: ignore
                df[f"{col}_sqrt"] = np.sqrt(df[col].clip(lower=0))
        return df

    # ------------------------------------------------------------------ #
    #  Ratio Features
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_ratios(df: pd.DataFrame, column_pairs: list[list[str]]) -> pd.DataFrame:
        """column_pairs = [["numerator", "denominator"], ...]"""
        df = df.copy()
        for pair in column_pairs:
            if len(pair) != 2:
                continue
            num, den = pair
            if num in df.columns and den in df.columns:
                df[f"{num}_div_{den}"] = df[num] / df[den].replace(0, np.nan)
        return df

    # ------------------------------------------------------------------ #
    #  Apply multiple operations
    # ------------------------------------------------------------------ #
    @classmethod
    def apply_operations(cls, df: pd.DataFrame, operations: list[dict]) -> tuple[pd.DataFrame, list[str]]:
        log = []
        logger.info("Applying %d feature-engineering operations (initial cols=%d)", len(operations), df.shape[1])
        for op in operations:
            action = op.get("action")
            params = op.get("params", {})
            before_cols = df.shape[1]
            try:
                fn = getattr(cls, action, None)  # type: ignore
                if fn is None:
                    logger.warning("Unknown feature action: %s", action)
                    log.append(f"Unknown feature action: {action}")
                    continue
                df = fn(df, **params)
                new_cols = df.shape[1] - before_cols
                log.append(f"Applied '{action}' — added {new_cols} columns")
                logger.debug("FE: %s — added %d cols (total=%d)", action, new_cols, df.shape[1])
            except Exception as e:
                logger.error("Feature engineering action '%s' failed: %s", action, e, exc_info=True)
                log.append(f"Error in '{action}': {e}")
        logger.info("Feature engineering complete (final cols=%d)", df.shape[1])
        return df, log
