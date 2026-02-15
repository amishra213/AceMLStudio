"""
AceML Studio – Data Transformations
=====================================
Scaling, encoding, log/power transforms.  Each method returns a new
DataFrame and optionally the fitted transformer for inverse operations.
"""

import logging
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype  # type: ignore
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    PowerTransformer,
)

logger = logging.getLogger("aceml.transformations")


class DataTransformer:
    """Scaling and encoding utilities."""

    # ------------------------------------------------------------------ #
    #  Scaling
    # ------------------------------------------------------------------ #
    @staticmethod
    def scale(df: pd.DataFrame, columns: list[str], method: str = "standard") -> tuple[pd.DataFrame, object]:
        """
        method: 'standard', 'minmax', 'robust'
        Returns (transformed_df, fitted_scaler).
        """
        df = df.copy()
        valid = [c for c in columns if c in df.columns and np.issubdtype(df[c].dtype, np.number)]  # type: ignore
        if not valid:
            return df, None

        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        scaler_cls = scalers.get(method, StandardScaler)
        scaler = scaler_cls()
        df[valid] = scaler.fit_transform(df[valid].fillna(0))
        return df, scaler

    # ------------------------------------------------------------------ #
    #  Encoding – One-Hot
    # ------------------------------------------------------------------ #
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: list[str], drop_first: bool = False,
                       max_cardinality: int = 50) -> pd.DataFrame:
        """
        One-hot encode categorical columns.
        CRITICAL: max_cardinality is hard-capped at 100 to prevent memory explosion.
        Columns with >100 unique values should use label encoding or target encoding instead.
        """
        df = df.copy()
        # Hard limit to prevent memory explosion
        HARD_MAX_CARDINALITY = 100
        max_cardinality = min(max_cardinality, HARD_MAX_CARDINALITY)
        
        skipped = []
        for col in columns:
            if col not in df.columns:
                continue
            
            n_unique = df[col].nunique()
            if n_unique > max_cardinality:
                skipped.append(f"{col} ({n_unique:,} unique values)")
                logger.warning(
                    "Skipping one-hot encoding for '%s' - has %d unique values (max: %d). "
                    "Use label_encode or target_encode instead.",
                    col, n_unique, max_cardinality
                )
                continue
            
            # Additional safety check: prevent creating more than 500 total columns
            if n_unique > 500:
                skipped.append(f"{col} (would create {n_unique} columns)")
                logger.error(
                    "BLOCKED: Column '%s' would create %d columns. This would cause memory issues.",
                    col, n_unique
                )
                continue
            
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            logger.info("One-hot encoded '%s' → %d columns", col, dummies.shape[1])
        
        if skipped:
            logger.warning("Skipped one-hot encoding for high-cardinality columns: %s", ", ".join(skipped))
        
        return df

    # ------------------------------------------------------------------ #
    #  Encoding – Label
    # ------------------------------------------------------------------ #
    @staticmethod
    def label_encode(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
        df = df.copy()
        encoders = {}
        for col in columns:
            if col not in df.columns:
                continue
            
            # Store original info for logging
            unique_before = df[col].nunique()
            original_dtype = df[col].dtype
            
            # Convert categorical to object first to allow new values
            if is_categorical_dtype(df[col]):
                df[col] = df[col].astype('object')
            
            # Create label encoder and fit on non-null values
            le = LabelEncoder()
            null_mask = df[col].isna()
            non_null_values = df[col][~null_mask].astype(str)
            
            le.fit(non_null_values)
            encoded_values = le.transform(non_null_values)
            
            # CRITICAL FIX: Properly assign encoded values with correct dtype
            # Use float64 if there are NaNs (to preserve them), otherwise int64
            if null_mask.any():
                # Create float column to accommodate NaNs
                new_col = pd.Series(np.nan, index=df.index, dtype='float64')
                new_col.loc[~null_mask] = encoded_values  # type: ignore
                df[col] = new_col
            else:
                # No NaNs: use int64 for efficiency
                # Convert to list to satisfy type checker
                df[col] = pd.Series(list(encoded_values), dtype='int64', index=df.index)  # type: ignore
            
            encoders[col] = le
            logger.info(
                "Label encoded '%s': %d unique %s values → %d encoded integers (dtype: %s → %s)",
                col, unique_before, original_dtype, len(le.classes_), original_dtype, df[col].dtype
            )
        return df, encoders

    # ------------------------------------------------------------------ #
    #  Encoding – Ordinal
    # ------------------------------------------------------------------ #
    @staticmethod
    def ordinal_encode(df: pd.DataFrame, columns: list[str], categories_order: dict | None = None) -> tuple[pd.DataFrame, dict]:
        """
        Ordinal encode categorical columns with user-defined or auto-detected order.
        
        Args:
            df: DataFrame to encode
            columns: List of column names to encode
            categories_order: Dict mapping column names to ordered lists of categories.
                             If None or column not in dict, auto-detects sorted order.
        
        Returns:
            (encoded_df, mapping_dict) where mapping_dict contains the category mappings
        """
        df = df.copy()
        mappings = {}
        categories_order = categories_order or {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Get unique categories
            unique_cats = df[col].dropna().unique()
            original_dtype = df[col].dtype
            
            # Use provided order or auto-detect (sorted)
            if col in categories_order:
                ordered_cats = categories_order[col]
            else:
                ordered_cats = sorted(unique_cats.astype(str))
            
            # Create mapping
            cat_mapping = {cat: idx for idx, cat in enumerate(ordered_cats)}
            mappings[col] = cat_mapping
            
            # CRITICAL FIX: Apply encoding with explicit dtype conversion
            # Use .map() then explicitly convert to handle NaNs properly
            null_mask = df[col].isna()
            encoded = df[col].map(cat_mapping)
            
            if null_mask.any():
                # Preserve NaNs by using float64
                df[col] = encoded.astype('float64')
            else:
                # No NaNs: use int64 for efficiency
                df[col] = encoded.astype('int64')
            
            logger.info(
                "Ordinal encoded '%s': %d categories (dtype: %s → %s)", 
                col, len(ordered_cats), original_dtype, df[col].dtype
            )
        
        return df, mappings

    # ------------------------------------------------------------------ #
    #  Encoding – Target (mean encoding)
    # ------------------------------------------------------------------ #
    @staticmethod
    def target_encode(df: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
        df = df.copy()
        if target not in df.columns:
            return df
        for col in columns:
            if col not in df.columns:
                continue
            
            unique_before = df[col].nunique()
            means = df.groupby(col)[target].mean()
            
            # CRITICAL FIX: Create new column with explicit float64 dtype
            # This ensures the encoded column is properly numeric
            df[f"{col}_target_enc"] = df[col].map(means).astype('float64')
            
            logger.info(
                "Target encoded '%s': %d unique values → mean encoding (new column: %s_target_enc, dtype: float64)",
                col, unique_before, col
            )
        return df

    # ------------------------------------------------------------------ #
    #  Data Type Conversion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_numeric_string(value) -> str:
        """
        Clean a value for numeric conversion by removing currency symbols,
        commas, percent signs, and other common formatting.
        
        Examples:
            '$1,234.56' -> '1234.56'
            '€99.99' -> '99.99'
            '1,234' -> '1234'
            '45%' -> '45'
        """
        if pd.isna(value):
            return value
        
        s = str(value).strip()
        
        # Remove common currency symbols: $, €, £, ¥, ₹, ₽, etc.
        s = re.sub(r'[$€£¥₹₽¢₩₪₦₨฿₴₸₵₡₢₣₤₧₮₰₲₳₴₵₶₷₸₹₺₻₼₽₾₿]', '', s)
        
        # Remove commas used as thousand separators
        s = re.sub(r',', '', s)
        
        # Remove percent signs
        s = re.sub(r'%', '', s)
        
        # Remove spaces
        s = s.replace(' ', '')
        
        # Handle parentheses for negative numbers (accounting format)
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        
        return s
    
    @staticmethod
    def convert_dtype(df: pd.DataFrame, columns: list[str], target_dtype: str) -> pd.DataFrame:
        """
        Convert columns to specified data type.
        Automatically cleans currency symbols and formatting for numeric conversions.
        
        Args:
            df: DataFrame to convert
            columns: List of column names to convert
            target_dtype: Target data type ('int', 'float', 'str', 'category', 'datetime')
        
        Returns:
            DataFrame with converted columns
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Convert categorical to object first to avoid assignment issues
            if is_categorical_dtype(df[col]):
                df[col] = df[col].astype('object')
            
            try:
                original_non_null = df[col].notna().sum()
                
                if target_dtype == 'int':
                    # Clean and convert to numeric
                    cleaned = df[col].apply(DataTransformer._clean_numeric_string)
                    df[col] = pd.to_numeric(cleaned, errors='coerce').astype('Int64')  # Nullable integer
                    
                elif target_dtype == 'float':
                    # Clean and convert to numeric
                    cleaned = df[col].apply(DataTransformer._clean_numeric_string)
                    df[col] = pd.to_numeric(cleaned, errors='coerce').astype(float)
                    
                elif target_dtype == 'str' or target_dtype == 'string':
                    df[col] = df[col].astype(str)
                    
                elif target_dtype == 'category':
                    df[col] = df[col].astype('category')
                    
                elif target_dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                elif target_dtype == 'bool' or target_dtype == 'boolean':
                    df[col] = df[col].astype(bool)
                    
                else:
                    logger.warning("Unknown dtype '%s' for column '%s'", target_dtype, col)
                    continue
                
                # Check for data loss
                final_non_null = df[col].notna().sum()
                if final_non_null < original_non_null:
                    lost = original_non_null - final_non_null
                    loss_pct = (lost / original_non_null * 100) if original_non_null > 0 else 0
                    logger.warning(
                        "Column '%s' conversion to %s caused data loss: %d values (%.1f%%) became null",
                        col, target_dtype, lost, loss_pct
                    )
                
                logger.info("Converted column '%s' to %s (preserved %d/%d values)", 
                          col, target_dtype, final_non_null, original_non_null)
                          
            except Exception as e:
                logger.error("Failed to convert column '%s' to %s: %s", col, target_dtype, e)
                raise ValueError(f"Cannot convert column '{col}' to {target_dtype}: {str(e)}")
        
        return df

    # ------------------------------------------------------------------ #
    #  Log Transform
    # ------------------------------------------------------------------ #
    @staticmethod
    def log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):  # type: ignore
                df[col] = np.log1p(df[col].clip(lower=0))
        return df

    # ------------------------------------------------------------------ #
    #  Power Transform (Yeo-Johnson)
    # ------------------------------------------------------------------ #
    @staticmethod
    def power_transform(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, object]:
        df = df.copy()
        valid = [c for c in columns if c in df.columns and np.issubdtype(df[c].dtype, np.number)]  # type: ignore
        if not valid:
            return df, None
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        df[valid] = pt.fit_transform(df[valid].fillna(0))
        return df, pt

    # ------------------------------------------------------------------ #
    #  Apply a sequence of transformations
    # ------------------------------------------------------------------ #
    @classmethod
    def apply_operations(cls, df: pd.DataFrame, operations: list[dict]) -> tuple[pd.DataFrame, list[str]]:
        log = []
        logger.info("Applying %d transformation operations (shape=%s)", len(operations), df.shape)
        
        for op in operations:
            action = op.get("action")
            params = op.get("params", {})
            initial_shape = df.shape
            
            try:
                if action == "scale":
                    df, _ = cls.scale(df, **params)
                    log.append(f"✓ Scaled {len(params.get('columns', []))} columns")
                    
                elif action == "one_hot_encode":
                    df = cls.one_hot_encode(df, **params)
                    new_cols = df.shape[1] - initial_shape[1]
                    if new_cols > 0:
                        log.append(f"✓ One-hot encoded → added {new_cols} columns")
                    else:
                        log.append(f"⚠ One-hot encoding skipped (high cardinality columns)")
                    
                elif action == "label_encode":
                    df, _ = cls.label_encode(df, **params)
                    log.append(f"✓ Label encoded {len(params.get('columns', []))} columns")
                    
                elif action == "ordinal_encode":
                    df, mappings = cls.ordinal_encode(df, **params)
                    log.append(f"✓ Ordinal encoded {len(params.get('columns', []))} columns")
                    
                elif action == "target_encode":
                    df = cls.target_encode(df, **params)
                    log.append(f"✓ Target encoded {len(params.get('columns', []))} columns")
                    
                elif action == "convert_dtype":
                    df = cls.convert_dtype(df, **params)
                    dtype = params.get('target_dtype', 'unknown')
                    log.append(f"✓ Converted {len(params.get('columns', []))} columns to {dtype}")
                    
                elif action == "log_transform":
                    df = cls.log_transform(df, **params)
                    log.append(f"✓ Log transformed {len(params.get('columns', []))} columns")
                    
                elif action == "power_transform":
                    df, _ = cls.power_transform(df, **params)
                    log.append(f"✓ Power transformed {len(params.get('columns', []))} columns")
                    
                else:
                    logger.warning("Unknown transform action: %s", action)
                    log.append(f"❌ Unknown transform action: {action}")
                    continue
                    
                logger.debug("Transform: %s applied (shape=%s)", action, df.shape)
                
            except ValueError as e:
                # User-facing errors (e.g., too many columns)
                error_msg = str(e)
                logger.error("Transform action '%s' blocked: %s", action, error_msg)
                log.append(f"❌ {action} blocked: {error_msg}")
                raise  # Re-raise to stop pipeline
                
            except Exception as e:
                logger.error("Transform action '%s' failed: %s", action, e, exc_info=True)
                log.append(f"❌ Error in '{action}': {e}")
                raise  # Re-raise to stop pipeline
                
        logger.info("Transformation pipeline complete (final shape=%s)", df.shape)
        return df, log
