"""
AceML Studio – Feature Engineering
====================================
Extract date features, text features, interaction terms, polynomial
features, and custom binning.  Each method returns a new DataFrame.
"""

import logging
from typing import Any

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
        valid = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
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
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
        return df

    @staticmethod
    def sqrt_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
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
    #  Column Combinations - Arithmetic
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_arithmetic_column(df: pd.DataFrame, col_a: str, col_b: str, 
                                  operation: str, new_col_name: str | None = None) -> pd.DataFrame:
        """
        Create new column from arithmetic operations on two existing columns.
        operation: 'add', 'subtract', 'multiply', 'divide', 'power', 'modulo'
        """
        df = df.copy()
        if col_a not in df.columns or col_b not in df.columns:
            logger.warning("Columns '%s' or '%s' not found", col_a, col_b)
            return df
        
        if new_col_name is None:
            new_col_name = f"{col_a}_{operation}_{col_b}"
        
        try:
            if operation == "add":
                df[new_col_name] = df[col_a] + df[col_b]
            elif operation == "subtract":
                df[new_col_name] = df[col_a] - df[col_b]
            elif operation == "multiply":
                df[new_col_name] = df[col_a] * df[col_b]
            elif operation == "divide":
                df[new_col_name] = df[col_a] / df[col_b].replace(0, np.nan)
            elif operation == "power":
                df[new_col_name] = df[col_a] ** df[col_b]
            elif operation == "modulo":
                df[new_col_name] = df[col_a] % df[col_b].replace(0, np.nan)
            else:
                logger.warning("Unknown arithmetic operation: %s", operation)
                return df
            
            logger.info("Created column '%s' = %s %s %s", new_col_name, col_a, operation, col_b)
        except Exception as e:
            logger.error("Failed to create arithmetic column: %s", e)
            raise ValueError(f"Cannot perform {operation} on columns '{col_a}' and '{col_b}': {str(e)}")
        
        return df

    # ------------------------------------------------------------------ #
    #  Column Combinations - Aggregations Across Columns
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_row_aggregate(df: pd.DataFrame, columns: list[str], 
                             aggregation: str, new_col_name: str | None = None) -> pd.DataFrame:
        """
        Create new column by aggregating across multiple columns for each row.
        aggregation: 'sum', 'mean', 'median', 'min', 'max', 'std'
        """
        df = df.copy()
        valid = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        
        if not valid:
            logger.warning("No valid numeric columns found in: %s", columns)
            return df
        
        if new_col_name is None:
            new_col_name = f"row_{aggregation}_{'_'.join(valid[:3])}"
            if len(valid) > 3:
                new_col_name += "_etc"
        
        try:
            if aggregation == "sum":
                df[new_col_name] = df[valid].sum(axis=1)
            elif aggregation == "mean":
                df[new_col_name] = df[valid].mean(axis=1)
            elif aggregation == "median":
                df[new_col_name] = df[valid].median(axis=1)
            elif aggregation == "min":
                df[new_col_name] = df[valid].min(axis=1)
            elif aggregation == "max":
                df[new_col_name] = df[valid].max(axis=1)
            elif aggregation == "std":
                df[new_col_name] = df[valid].std(axis=1)
            else:
                logger.warning("Unknown aggregation: %s", aggregation)
                return df
            
            logger.info("Created column '%s' = %s across %d columns", new_col_name, aggregation, len(valid))
        except Exception as e:
            logger.error("Failed to create row aggregate: %s", e)
            raise ValueError(f"Cannot compute {aggregation} across columns: {str(e)}")
        
        return df

    # ------------------------------------------------------------------ #
    #  String Extraction - Substring
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_substring(df: pd.DataFrame, column: str, start: int, end: int | None = None,
                          new_col_name: str | None = None) -> pd.DataFrame:
        """Extract substring from text column."""
        df = df.copy()
        if column not in df.columns:
            logger.warning("Column '%s' not found", column)
            return df
        
        if new_col_name is None:
            new_col_name = f"{column}_substr_{start}_{end or 'end'}"
        
        try:
            df[new_col_name] = df[column].astype(str).str[start:end]
            logger.info("Extracted substring from '%s' → '%s'", column, new_col_name)
        except Exception as e:
            logger.error("Failed to extract substring: %s", e)
            raise ValueError(f"Cannot extract substring from '{column}': {str(e)}")
        
        return df

    # ------------------------------------------------------------------ #
    #  String Extraction - Pattern (Regex)
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_pattern(df: pd.DataFrame, column: str, pattern: str,
                        new_col_name: str | None = None) -> pd.DataFrame:
        """Extract text matching regex pattern."""
        df = df.copy()
        if column not in df.columns:
            logger.warning("Column '%s' not found", column)
            return df
        
        if new_col_name is None:
            new_col_name = f"{column}_extracted"
        
        try:
            # Extract first match of pattern
            # Use expand=False to get a Series (only first capturing group)
            extracted = df[column].astype(str).str.extract(f'({pattern})', expand=False)
            # If multiple columns returned, take the first one
            if isinstance(extracted, pd.DataFrame):
                first_col: pd.Series = extracted.iloc[:, 0]  # type: ignore[call-overload]
                df[new_col_name] = first_col
            else:
                df[new_col_name] = extracted
            logger.info("Extracted pattern '%s' from '%s' → '%s'", pattern, column, new_col_name)
        except Exception as e:
            logger.error("Failed to extract pattern: %s", e)
            raise ValueError(f"Invalid regex pattern '{pattern}': {str(e)}")
        
        return df

    # ------------------------------------------------------------------ #
    #  String Split
    # ------------------------------------------------------------------ #
    @staticmethod
    def split_column(df: pd.DataFrame, column: str, delimiter: str = " ",
                     max_splits: int = 2, prefix: str | None = None) -> pd.DataFrame:
        """Split column into multiple columns based on delimiter."""
        df = df.copy()
        if column not in df.columns:
            logger.warning("Column '%s' not found", column)
            return df
        
        if prefix is None:
            prefix = column
        
        try:
            split_data = df[column].astype(str).str.split(delimiter, n=max_splits, expand=True)
            for i in range(split_data.shape[1]):
                df[f"{prefix}_part{i+1}"] = split_data[i]
            logger.info("Split '%s' into %d columns", column, split_data.shape[1])
        except Exception as e:
            logger.error("Failed to split column: %s", e)
            raise ValueError(f"Cannot split column '{column}': {str(e)}")
        
        return df

    # ------------------------------------------------------------------ #
    #  String Concatenation
    # ------------------------------------------------------------------ #
    @staticmethod
    def concatenate_columns(df: pd.DataFrame, columns: list[str], 
                           separator: str = " ", new_col_name: str | None = None) -> pd.DataFrame:
        """Concatenate multiple columns into one."""
        df = df.copy()
        valid = [c for c in columns if c in df.columns]
        
        if not valid:
            logger.warning("No valid columns found in: %s", columns)
            return df
        
        if new_col_name is None:
            new_col_name = "_".join(valid[:3]) + "_combined"
        
        try:
            # Convert to string and concatenate
            df[new_col_name] = df[valid].astype(str).agg(separator.join, axis=1)
            logger.info("Concatenated %d columns → '%s'", len(valid), new_col_name)
        except Exception as e:
            logger.error("Failed to concatenate columns: %s", e)
            raise ValueError(f"Cannot concatenate columns: {str(e)}")
        
        return df

    # ------------------------------------------------------------------ #
    #  Conditional Column (If-Then-Else)
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_conditional_column(df: pd.DataFrame, column: str, condition: str,
                                   value_if_true: str | float | int,
                                   value_if_false: str | float | int,
                                   new_col_name: str | None = None) -> pd.DataFrame:
        """
        Create column based on condition.
        condition examples: '>50', '==100', '==Yes', 'contains:text', 'startswith:A'
        """
        df = df.copy()
        if column not in df.columns:
            logger.warning("Column '%s' not found", column)
            return df
        
        if new_col_name is None:
            new_col_name = f"{column}_conditional"
        
        try:
            # Parse condition
            mask = None  # Initialize mask
            if condition.startswith('contains:'):
                search_str = condition.split(':', 1)[1]
                mask = df[column].astype(str).str.contains(search_str, na=False, case=False)
            elif condition.startswith('startswith:'):
                prefix = condition.split(':', 1)[1]
                mask = df[column].astype(str).str.startswith(prefix, na=False)
            elif condition.startswith('endswith:'):
                suffix = condition.split(':', 1)[1]
                mask = df[column].astype(str).str.endswith(suffix, na=False)
            elif any(op in condition for op in ['>=', '<=', '==', '!=', '>', '<']):
                # Detect which operator is used
                import operator
                ops = {'>=': operator.ge, '<=': operator.le, '==': operator.eq,
                       '!=': operator.ne, '>': operator.gt, '<': operator.lt}
                for op_str, op_func in ops.items():
                    if op_str in condition:
                        value_str = condition.replace(op_str, '').strip()
                        
                        # Try to convert to numeric, otherwise use string comparison
                        try:
                            threshold = float(value_str)
                            mask = op_func(df[column], threshold)
                        except ValueError:
                            # String comparison
                            mask = op_func(df[column].astype(str), value_str)
                        break
            else:
                raise ValueError(f"Unsupported condition format: {condition}")
            
            if mask is None:
                raise ValueError("Could not create mask from condition")
            df[new_col_name] = mask.map({True: value_if_true, False: value_if_false})
            logger.info("Created conditional column '%s' based on '%s %s'", 
                       new_col_name, column, condition)
        except Exception as e:
            logger.error("Failed to create conditional column: %s", e)
            raise ValueError(f"Cannot create conditional column: {str(e)}")
        
        return df

    # ------------------------------------------------------------------ #
    #  Custom Formula (Safe Evaluation)
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_custom_column(df: pd.DataFrame, formula: str, new_col_name: str) -> pd.DataFrame:
        """
        Create column from custom formula using existing columns.
        Example: "col_a * 2 + col_b" or "np.sqrt(col_a)"
        
        Security: Only column names and safe math operations allowed.
        """
        df = df.copy()
        
        try:
            # Build safe namespace with only columns and safe functions
            safe_dict: dict[str, Any] = {col: df[col] for col in df.columns}
            safe_dict.update({
                'np': np,
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
            })
            
            # Evaluate formula
            result = eval(formula, {"__builtins__": {}}, safe_dict)
            df[new_col_name] = result
            
            logger.info("Created custom column '%s' from formula: %s", new_col_name, formula)
        except Exception as e:
            logger.error("Failed to evaluate formula '%s': %s", formula, e)
            raise ValueError(f"Invalid formula '{formula}': {str(e)}")
        
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
