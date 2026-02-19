"""
AceML Studio – Data Quality Analyzer
======================================
Comprehensive data quality analysis: missing values, duplicates, outliers,
class imbalance, correlations, and an overall quality score.
"""

import logging
import numpy as np
import pandas as pd
from config import Config

logger = logging.getLogger("aceml.data_quality")


class DataQualityAnalyzer:
    """Analyze a DataFrame for data-quality issues and return a structured report."""

    def __init__(self, df: pd.DataFrame, target_column: str | None = None):
        self.df = df
        self.target = target_column

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def full_report(self) -> dict:
        """Run all checks and return a unified quality report."""
        logger.info("Running full quality report on %d rows x %d cols (target=%s)",
                    self.df.shape[0], self.df.shape[1], self.target)
        missing = self._missing_values()
        duplicates = self._duplicates()
        outliers = self._outliers()
        class_imbalance = self._class_imbalance()
        correlations = self._high_correlations()
        dtype_issues = self._dtype_issues()
        constant_cols = self._constant_columns()
        high_cardinality = self._high_cardinality_columns()
        issues = self._compile_issues(missing, duplicates, outliers, class_imbalance, constant_cols, high_cardinality)
        score = self._quality_score(issues)

        logger.info("Quality report complete: score=%d, issues=%d (critical=%d, warning=%d, info=%d)",
                    score,
                    len(issues),
                    sum(1 for i in issues if i['severity'] == 'critical'),
                    sum(1 for i in issues if i['severity'] == 'warning'),
                    sum(1 for i in issues if i['severity'] == 'info'))
        if missing:
            logger.debug("Missing values found in %d columns", len(missing))
        if duplicates['count'] > 0:
            logger.debug("Duplicate rows: %d (%.2f%%)", duplicates['count'], duplicates['percentage'])
        if outliers:
            logger.debug("Outliers detected in %d columns", len(outliers))
        if class_imbalance and class_imbalance.get('is_imbalanced'):
            logger.warning("Class imbalance detected for target '%s' (ratio=%.4f)",
                          self.target, class_imbalance['majority_minority_ratio'])
        if high_cardinality:
            logger.warning("High cardinality columns detected: %d", len(high_cardinality))

        return {
            "shape": {"rows": int(self.df.shape[0]), "columns": int(self.df.shape[1])},
            "missing_values": missing,
            "duplicates": duplicates,
            "outliers": outliers,
            "class_imbalance": class_imbalance,
            "high_correlations": correlations,
            "dtype_issues": dtype_issues,
            "constant_columns": constant_cols,
            "high_cardinality_columns": high_cardinality,
            "issues": issues,
            "quality_score": score,
        }

    # ------------------------------------------------------------------ #
    #  Missing Values
    # ------------------------------------------------------------------ #
    def _missing_values(self) -> dict:
        total = len(self.df)
        result = {}
        for col in self.df.columns:
            cnt = int(self.df[col].isna().sum())
            pct = round(cnt / total * 100, 2) if total else 0
            if cnt > 0:
                result[col] = {"count": cnt, "percentage": pct}
        return result

    # ------------------------------------------------------------------ #
    #  Duplicates
    # ------------------------------------------------------------------ #
    def _duplicates(self) -> dict:
        cnt = int(self.df.duplicated().sum())
        pct = round(cnt / len(self.df) * 100, 2) if len(self.df) else 0
        return {"count": cnt, "percentage": pct}

    # ------------------------------------------------------------------ #
    #  Outliers (IQR method on numeric columns)
    # ------------------------------------------------------------------ #
    def _outliers(self) -> dict:
        result = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            q1 = float(self.df[col].quantile(0.25))
            q3 = float(self.df[col].quantile(0.75))
            iqr = q3 - q1
            mult = Config.OUTLIER_IQR_MULTIPLIER
            lower = q1 - mult * iqr
            upper = q3 + mult * iqr
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            cnt = int(mask.sum())
            if cnt > 0:
                pct = round(cnt / len(self.df) * 100, 2)
                result[col] = {
                    "count": cnt,
                    "percentage": pct,
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                }
        return result

    # ------------------------------------------------------------------ #
    #  Class Imbalance (classification target)
    # ------------------------------------------------------------------ #
    def _class_imbalance(self) -> dict | None:
        if self.target is None or self.target not in self.df.columns:
            return None
        vc = self.df[self.target].value_counts()
        distribution = {str(k): int(v) for k, v in vc.items()}
        majority = int(vc.iloc[0])
        minority = int(vc.iloc[-1])
        ratio = round(minority / majority, 4) if majority else 0
        imbalanced = ratio < Config.CLASS_IMBALANCE_RATIO
        return {
            "target": self.target,
            "distribution": distribution,
            "majority_minority_ratio": ratio,
            "is_imbalanced": imbalanced,
        }

    # ------------------------------------------------------------------ #
    #  High Correlations
    # ------------------------------------------------------------------ #
    def _high_correlations(self) -> list:
        numeric = self.df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return []
        corr = numeric.corr().abs()
        pairs = []
        seen = set()
        threshold = Config.HIGH_CORRELATION_THRESHOLD
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if val >= threshold:  # type: ignore
                    c1, c2 = corr.columns[i], corr.columns[j]
                    key = tuple(sorted([c1, c2]))
                    if key not in seen:
                        seen.add(key)
                        pairs.append({"col_a": c1, "col_b": c2, "correlation": round(float(val), 4)})  # type: ignore
        return pairs

    # ------------------------------------------------------------------ #
    #  Data-type Issues
    # ------------------------------------------------------------------ #
    def _dtype_issues(self) -> list:
        issues = []
        for col in self.df.columns:
            if self.df[col].dtype == object:
                non_null = self.df[col].dropna()
                if len(non_null) == 0:
                    continue
                # Check if numeric hidden as string
                numeric_parseable = pd.to_numeric(non_null, errors="coerce").notna().mean()
                if numeric_parseable > 0.8:
                    issues.append({
                        "column": col,
                        "issue": "Likely numeric stored as string",
                        "parseable_pct": round(numeric_parseable * 100, 1),
                    })
                # Check if datetime hidden as string
                try:
                    pd.to_datetime(non_null.head(50), errors="raise", format="mixed")  # type: ignore
                    issues.append({"column": col, "issue": "Likely datetime stored as string"})
                except Exception:
                    pass
        return issues

    # ------------------------------------------------------------------ #
    #  Constant / near-constant columns
    # ------------------------------------------------------------------ #
    def _constant_columns(self) -> list:
        result = []
        for col in self.df.columns:
            nuniq = self.df[col].nunique(dropna=True)
            if nuniq <= 1:
                result.append({"column": col, "unique_values": int(nuniq), "type": "constant"})
            elif nuniq / len(self.df) < Config.LOW_VARIANCE_THRESHOLD and self.df[col].dtype == object:
                result.append({"column": col, "unique_values": int(nuniq), "type": "near_constant"})
        return result

    # ------------------------------------------------------------------ #
    #  High Cardinality Columns (risky for one-hot encoding)
    # ------------------------------------------------------------------ #
    def _high_cardinality_columns(self) -> list:
        """
        Detect categorical columns with very high cardinality that would cause
        memory issues if one-hot encoded.
        """
        result = []
        # Check object and category dtypes
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            nuniq = self.df[col].nunique(dropna=True)
            
            # Severity levels based on unique values
            if nuniq > 1000:
                severity = "critical"
                message = f"Column '{col}' has {nuniq:,} unique values - DO NOT one-hot encode! Use label encoding or target encoding instead."
            elif nuniq > 100:
                severity = "warning"
                message = f"Column '{col}' has {nuniq:,} unique values - one-hot encoding blocked. Use label encoding or target encoding."
            elif nuniq > 50:
                severity = "info"
                message = f"Column '{col}' has {nuniq} unique values - consider label encoding instead of one-hot."
            else:
                continue  # Low cardinality, safe for one-hot
            
            result.append({
                "column": col,
                "unique_values": int(nuniq),
                "severity": severity,
                "message": message
            })
            
            logger.info("High cardinality detected: %s", message)
        
        return result

    # ------------------------------------------------------------------ #
    #  Compile flat issue list
    # ------------------------------------------------------------------ #
    def _compile_issues(self, missing, duplicates, outliers, imbalance, constant_cols, high_cardinality) -> list:
        issues = []

        # Missing values
        for col, info in missing.items():
            severity = "critical" if info["percentage"] > Config.MISSING_VALUE_CRITICAL_PCT else \
                       "warning" if info["percentage"] > Config.MISSING_VALUE_WARN_PCT else "info"
            issues.append({
                "type": "missing_values",
                "column": col,
                "severity": severity,
                "message": f"Column '{col}' has {info['percentage']}% missing values ({info['count']} rows).",
            })

        # Duplicates
        if duplicates["percentage"] > Config.DUPLICATE_WARN_PCT:
            issues.append({
                "type": "duplicates",
                "severity": "warning",
                "message": f"{duplicates['count']} duplicate rows ({duplicates['percentage']}%).",
            })
        
        # High Cardinality (important for preventing memory issues)
        for hc_info in high_cardinality:
            issues.append({
                "type": "high_cardinality",
                "column": hc_info["column"],
                "severity": hc_info["severity"],
                "message": hc_info["message"],
            })

        # Outliers
        for col, info in outliers.items():
            if info["percentage"] > 5:
                issues.append({
                    "type": "outliers",
                    "column": col,
                    "severity": "warning",
                    "message": f"Column '{col}' has {info['count']} outliers ({info['percentage']}%).",
                })

        # Class imbalance
        if imbalance and imbalance["is_imbalanced"]:
            issues.append({
                "type": "class_imbalance",
                "severity": "warning",
                "message": f"Target '{imbalance['target']}' is imbalanced (minority/majority ratio: {imbalance['majority_minority_ratio']}).",
            })

        # Constants
        for c in constant_cols:
            issues.append({
                "type": "constant_column",
                "column": c["column"],
                "severity": "info",
                "message": f"Column '{c['column']}' is {c['type']} ({c['unique_values']} unique values).",
            })

        return issues

    # ------------------------------------------------------------------ #
    #  Overall quality score (0-100)
    # ------------------------------------------------------------------ #
    def _quality_score(self, issues: list) -> int:
        score = 100
        for issue in issues:
            if issue["severity"] == "critical":
                score -= 15
            elif issue["severity"] == "warning":
                score -= 5
            else:
                score -= 1
        return max(0, score)
