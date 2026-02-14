"""
AceML Studio – Data Cleaner
=============================
Operations: handle missing values, remove duplicates, treat outliers,
fix data types.  Each method returns a *new* DataFrame.
"""

import logging
import numpy as np
import pandas as pd
from config import Config

logger = logging.getLogger("aceml.data_cleaning")


class DataCleaner:
    """Stateless cleaning utilities – every method is a class method returning a new DataFrame."""

    # ------------------------------------------------------------------ #
    #  Missing Values
    # ------------------------------------------------------------------ #
    @staticmethod
    def drop_missing(df: pd.DataFrame, columns: list[str] | None = None, how: str = "any") -> pd.DataFrame:
        if columns:
            return df.dropna(subset=columns, how=how).reset_index(drop=True)  # type: ignore
        return df.dropna(how=how).reset_index(drop=True)  # type: ignore

    @staticmethod
    def impute(df: pd.DataFrame, strategy: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Strategy: mean, median, mode, zero, ffill, bfill."""
        df = df.copy()
        cols = columns or df.columns.tolist()
        for col in cols:
            if col not in df.columns:
                continue
            if strategy == "mean" and np.issubdtype(df[col].dtype, np.number):  # type: ignore
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median" and np.issubdtype(df[col].dtype, np.number):  # type: ignore
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                mode_val = df[col].mode()
                if len(mode_val):
                    df[col].fillna(mode_val.iloc[0], inplace=True)
            elif strategy == "zero":
                df[col].fillna(0, inplace=True)
            elif strategy == "ffill":
                df[col] = df[col].ffill()  # type: ignore
            elif strategy == "bfill":
                df[col] = df[col].bfill()  # type: ignore
        return df

    # ------------------------------------------------------------------ #
    #  Duplicates
    # ------------------------------------------------------------------ #
    @staticmethod
    def drop_duplicates(df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first") -> pd.DataFrame:
        return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)  # type: ignore

    # ------------------------------------------------------------------ #
    #  Outliers
    # ------------------------------------------------------------------ #
    @staticmethod
    def clip_outliers(df: pd.DataFrame, columns: list[str] | None = None, multiplier: float | None = None) -> pd.DataFrame:
        df = df.copy()
        mult = multiplier or Config.OUTLIER_IQR_MULTIPLIER
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            if not np.issubdtype(df[col].dtype, np.number):  # type: ignore
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - mult * iqr, q3 + mult * iqr
            df[col] = df[col].clip(lower, upper)
        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: list[str] | None = None, multiplier: float | None = None) -> pd.DataFrame:
        mult = multiplier or Config.OUTLIER_IQR_MULTIPLIER
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        mask = pd.Series(True, index=df.index)
        for col in cols:
            if not np.issubdtype(df[col].dtype, np.number):  # type: ignore
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - mult * iqr, q3 + mult * iqr
            mask &= (df[col] >= lower) & (df[col] <= upper)
        return df[mask].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Drop Columns
    # ------------------------------------------------------------------ #
    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Drop specified columns from the DataFrame."""
        if not columns:
            return df
        # Filter out columns that don't exist
        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            return df
        return df.drop(columns=valid_cols)

    # ------------------------------------------------------------------ #
    #  Data Type Fixes
    # ------------------------------------------------------------------ #
    @staticmethod
    def convert_to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def convert_to_datetime(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")  # type: ignore
        return df

    @staticmethod
    def convert_to_category(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            df[col] = df[col].astype("category")
        return df

    # ------------------------------------------------------------------ #
    #  Convenience: apply a list of operations
    # ------------------------------------------------------------------ #
    @classmethod
    def apply_operations(cls, df: pd.DataFrame, operations: list[dict]) -> tuple[pd.DataFrame, list[str]]:
        """
        Apply a sequence of cleaning operations.
        Each operation dict: {"action": "...", "params": {...}}
        Returns (cleaned_df, log_messages).
        """
        log = []
        logger.info("Applying %d cleaning operations (initial shape=%s)", len(operations), df.shape)
        for op in operations:
            action = op.get("action")
            params = op.get("params", {})
            before = df.shape[0]
            try:
                if action == "drop_missing":
                    df = cls.drop_missing(df, **params)
                elif action == "impute":
                    df = cls.impute(df, **params)
                elif action == "drop_duplicates":
                    df = cls.drop_duplicates(df, **params)
                elif action == "clip_outliers":
                    df = cls.clip_outliers(df, **params)
                elif action == "remove_outliers":
                    df = cls.remove_outliers(df, **params)
                elif action == "convert_to_numeric":
                    df = cls.convert_to_numeric(df, **params)
                elif action == "convert_to_datetime":
                    df = cls.convert_to_datetime(df, **params)
                elif action == "convert_to_category":
                    df = cls.convert_to_category(df, **params)
                elif action == "drop_columns":
                    before_cols = df.shape[1]
                    df = cls.drop_columns(df, **params)
                    after_cols = df.shape[1]
                    dropped = before_cols - after_cols
                    log.append(f"Dropped {dropped} column(s)")
                    logger.debug("Cleaning: drop_columns (cols %d→%d)", before_cols, after_cols)
                    continue
                else:
                    logger.warning("Unknown cleaning action: %s", action)
                    log.append(f"Unknown action: {action}")
                    continue
                after = df.shape[0]
                removed = before - after
                msg = f"Applied '{action}'"
                if removed:
                    msg += f" — removed {removed} rows"
                log.append(msg)
                logger.debug("Cleaning: %s (rows %d→%d)", action, before, after)
            except Exception as e:
                logger.error("Cleaning action '%s' failed: %s", action, e, exc_info=True)
                log.append(f"Error in '{action}': {e}")
        logger.info("Cleaning complete (final shape=%s)", df.shape)
        return df, log
