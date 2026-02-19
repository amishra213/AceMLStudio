"""
AceML Studio – Anomaly Detection Engine
==========================================
Multi-method anomaly / outlier detection supporting:
  • Statistical methods: Z-Score, Modified Z-Score, IQR, Grubbs
  • ML methods: Isolation Forest, One-Class SVM, Local Outlier Factor (LOF)
  • Clustering-based: DBSCAN
  • Deep Learning: Autoencoder (sklearn MLPRegressor-based)
  • Multivariate detection with feature contribution analysis
  • Streaming / incremental detection interface
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger("aceml.anomaly_detection")

warnings.filterwarnings("ignore", category=FutureWarning)


# ════════════════════════════════════════════════════════════════════
#  Anomaly Detection Engine
# ════════════════════════════════════════════════════════════════════

class AnomalyDetectionEngine:
    """Comprehensive anomaly / outlier detection toolkit."""

    # ================================================================
    #  STATISTICAL METHODS
    # ================================================================

    # ----------------------------------------------------------------
    #  Z-Score
    # ----------------------------------------------------------------
    @staticmethod
    def zscore_detection(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """Detect outliers using Z-Score on specified (or all numeric) columns."""
        start_time = time.time()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].copy()
        means = data.mean()
        stds = data.std()
        stds = stds.replace(0, np.nan)  # avoid division by zero

        z_scores = (data - means) / stds
        is_anomaly = (z_scores.abs() > threshold).any(axis=1)

        anomaly_indices = df.index[is_anomaly].tolist()
        anomaly_count = int(is_anomaly.sum())

        # Per-column stats
        col_stats: Dict[str, Any] = {}
        for col in columns:
            col_z = z_scores[col].dropna()
            col_outliers = col_z.abs() > threshold
            col_stats[col] = {
                "outlier_count": int(col_outliers.sum()),
                "max_zscore": round(float(col_z.abs().max()), 4),
                "mean_zscore": round(float(col_z.abs().mean()), 4),
            }

        return {
            "method": "zscore",
            "threshold": threshold,
            "columns_analyzed": columns,
            "total_rows": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
            "anomaly_indices": anomaly_indices[:500],  # cap for response size
            "column_stats": col_stats,
            "labels": is_anomaly.astype(int).tolist(),
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ----------------------------------------------------------------
    #  Modified Z-Score (MAD-based, robust)
    # ----------------------------------------------------------------
    @staticmethod
    def modified_zscore_detection(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 3.5,
    ) -> Dict[str, Any]:
        """Detect outliers using Modified Z-Score (Median Absolute Deviation)."""
        start_time = time.time()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].copy()
        medians = data.median()
        mad = (data - medians).abs().median()
        mad = mad.replace(0, np.nan)
        modified_z = 0.6745 * (data - medians) / mad

        is_anomaly = (modified_z.abs() > threshold).any(axis=1)
        anomaly_count = int(is_anomaly.sum())

        return {
            "method": "modified_zscore",
            "threshold": threshold,
            "columns_analyzed": columns,
            "total_rows": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
            "anomaly_indices": df.index[is_anomaly].tolist()[:500],
            "labels": is_anomaly.astype(int).tolist(),
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ----------------------------------------------------------------
    #  IQR (Interquartile Range)
    # ----------------------------------------------------------------
    @staticmethod
    def iqr_detection(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5,
    ) -> Dict[str, Any]:
        """Detect outliers via IQR method."""
        start_time = time.time()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].copy()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        is_anomaly = ((data < lower) | (data > upper)).any(axis=1)
        anomaly_count = int(is_anomaly.sum())

        col_bounds: Dict[str, Any] = {}
        for col in columns:
            col_data = data[col]
            col_outliers = (col_data < lower[col]) | (col_data > upper[col])
            col_bounds[col] = {
                "lower_bound": round(float(lower[col]), 4),
                "upper_bound": round(float(upper[col]), 4),
                "q1": round(float(Q1[col]), 4),
                "q3": round(float(Q3[col]), 4),
                "iqr": round(float(IQR[col]), 4),
                "outlier_count": int(col_outliers.sum()),
            }

        return {
            "method": "iqr",
            "multiplier": multiplier,
            "columns_analyzed": columns,
            "total_rows": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
            "anomaly_indices": df.index[is_anomaly].tolist()[:500],
            "column_bounds": col_bounds,
            "labels": is_anomaly.astype(int).tolist(),
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  ML-BASED METHODS
    # ================================================================

    # ----------------------------------------------------------------
    #  Isolation Forest
    # ----------------------------------------------------------------
    @staticmethod
    def isolation_forest_detection(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Detect anomalies with Isolation Forest."""
        start_time = time.time()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        predictions = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
        scores = model.decision_function(X_scaled)

        is_anomaly = predictions == -1
        anomaly_count = int(is_anomaly.sum())

        # Feature contribution (approximate via single-feature scores)
        feature_contributions: Dict[str, float] = {}
        for i, col in enumerate(columns):
            single_model = IsolationForest(
                contamination=contamination,
                n_estimators=50,
                random_state=random_state,
            )
            single_scores = single_model.fit(X_scaled[:, i:i+1]).decision_function(X_scaled[:, i:i+1])
            # Correlation between single feature scores and overall scores
            corr = float(np.corrcoef(scores, single_scores)[0, 1])
            feature_contributions[col] = round(abs(corr), 4)
        feature_contributions = dict(sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True))

        return {
            "method": "isolation_forest",
            "contamination": contamination,
            "n_estimators": n_estimators,
            "columns_analyzed": columns,
            "total_rows": len(data),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(data), 4) if len(data) > 0 else 0,
            "anomaly_indices": data.index[is_anomaly].tolist()[:500],
            "anomaly_scores": [round(float(s), 4) for s in scores],
            "score_threshold": round(float(np.percentile(scores, contamination * 100)), 4),
            "feature_contributions": feature_contributions,
            "labels": [1 if a else 0 for a in is_anomaly],
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ----------------------------------------------------------------
    #  One-Class SVM
    # ----------------------------------------------------------------
    @staticmethod
    def one_class_svm_detection(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str = "scale",
    ) -> Dict[str, Any]:
        """Detect anomalies with One-Class SVM."""
        start_time = time.time()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].dropna()
        # Subsample for large datasets (SVM is O(n²))
        max_samples = 10000
        if len(data) > max_samples:
            data_sample = data.sample(max_samples, random_state=42)
            logger.info("Subsampled to %d rows for One-Class SVM", max_samples)
        else:
            data_sample = data

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_sample)

        kernel_lit: Any = kernel
        gamma_lit: Any = gamma
        model = OneClassSVM(kernel=kernel_lit, nu=nu, gamma=gamma_lit)
        predictions = model.fit_predict(X_scaled)
        scores = model.decision_function(X_scaled)

        is_anomaly = predictions == -1
        anomaly_count = int(is_anomaly.sum())

        return {
            "method": "one_class_svm",
            "kernel": kernel,
            "nu": nu,
            "columns_analyzed": columns,
            "total_rows": len(data_sample),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(data_sample), 4) if len(data_sample) > 0 else 0,
            "anomaly_indices": data_sample.index[is_anomaly].tolist()[:500],
            "anomaly_scores": [round(float(s), 4) for s in scores],
            "labels": [1 if a else 0 for a in is_anomaly],
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ----------------------------------------------------------------
    #  Local Outlier Factor (LOF)
    # ----------------------------------------------------------------
    @staticmethod
    def lof_detection(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        n_neighbors: int = 20,
        contamination: float = 0.05,
    ) -> Dict[str, Any]:
        """Detect anomalies with Local Outlier Factor."""
        start_time = time.time()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        model = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(data) - 1),
            contamination=contamination,
            n_jobs=-1,
        )
        predictions = model.fit_predict(X_scaled)
        scores = model.negative_outlier_factor_

        is_anomaly = predictions == -1
        anomaly_count = int(is_anomaly.sum())

        return {
            "method": "lof",
            "n_neighbors": n_neighbors,
            "contamination": contamination,
            "columns_analyzed": columns,
            "total_rows": len(data),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(data), 4) if len(data) > 0 else 0,
            "anomaly_indices": data.index[is_anomaly].tolist()[:500],
            "lof_scores": [round(float(s), 4) for s in scores],
            "labels": [1 if a else 0 for a in is_anomaly],
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ----------------------------------------------------------------
    #  DBSCAN-based
    # ----------------------------------------------------------------
    @staticmethod
    def dbscan_detection(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> Dict[str, Any]:
        """Detect anomalies as DBSCAN noise points (label = -1)."""
        start_time = time.time()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X_scaled)

        is_anomaly = labels == -1
        anomaly_count = int(is_anomaly.sum())
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return {
            "method": "dbscan",
            "eps": eps,
            "min_samples": min_samples,
            "columns_analyzed": columns,
            "total_rows": len(data),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(data), 4) if len(data) > 0 else 0,
            "n_clusters": n_clusters,
            "anomaly_indices": data.index[is_anomaly].tolist()[:500],
            "cluster_labels": labels.tolist(),
            "labels": [1 if a else 0 for a in is_anomaly],
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ----------------------------------------------------------------
    #  Autoencoder-based (MLP Reconstruction Error)
    # ----------------------------------------------------------------
    @staticmethod
    def autoencoder_detection(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        encoding_dim: Optional[int] = None,
        contamination: float = 0.05,
        max_iter: int = 500,
    ) -> Dict[str, Any]:
        """Detect anomalies via autoencoder reconstruction error."""
        start_time = time.time()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        n_features = X_scaled.shape[1]
        if encoding_dim is None:
            encoding_dim = max(2, n_features // 3)

        # Build autoencoder as an MLPRegressor (input → bottleneck → output)
        hidden_layers = (n_features, encoding_dim, n_features)
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        model.fit(X_scaled, X_scaled)  # reconstruct self

        # Reconstruction error per sample
        X_reconstructed = model.predict(X_scaled)
        recon_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

        # Threshold = top contamination percentile
        threshold = float(np.percentile(recon_error, (1 - contamination) * 100))
        is_anomaly = recon_error > threshold
        anomaly_count = int(is_anomaly.sum())

        # Per-feature reconstruction error for anomalies
        feature_errors: Dict[str, float] = {}
        for i, col in enumerate(columns):
            feature_errors[col] = round(float(np.mean((X_scaled[:, i] - X_reconstructed[:, i]) ** 2)), 4)
        feature_errors = dict(sorted(feature_errors.items(), key=lambda x: x[1], reverse=True))

        return {
            "method": "autoencoder",
            "encoding_dim": encoding_dim,
            "contamination": contamination,
            "columns_analyzed": columns,
            "total_rows": len(data),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(data), 4) if len(data) > 0 else 0,
            "anomaly_indices": data.index[is_anomaly].tolist()[:500],
            "reconstruction_errors": [round(float(e), 6) for e in recon_error],
            "threshold": round(threshold, 6),
            "feature_errors": feature_errors,
            "labels": [1 if a else 0 for a in is_anomaly],
            "duration_sec": round(time.time() - start_time, 3),
        }

    # ================================================================
    #  ENSEMBLE / COMPARISON
    # ================================================================
    @classmethod
    def detect_all(
        cls,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        contamination: float = 0.05,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple anomaly-detection methods and return comparison + consensus.
        ``methods`` may include: zscore, modified_zscore, iqr, isolation_forest,
        one_class_svm, lof, dbscan, autoencoder.
        """
        if methods is None:
            methods = ["zscore", "iqr", "isolation_forest", "lof"]

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        all_results: Dict[str, Any] = {
            "columns_analyzed": columns,
            "total_rows": len(df),
            "contamination": contamination,
            "methods": {},
        }

        method_map = {
            "zscore": lambda: cls.zscore_detection(df, columns),
            "modified_zscore": lambda: cls.modified_zscore_detection(df, columns),
            "iqr": lambda: cls.iqr_detection(df, columns),
            "isolation_forest": lambda: cls.isolation_forest_detection(df, columns, contamination=contamination),
            "one_class_svm": lambda: cls.one_class_svm_detection(df, columns, nu=contamination),
            "lof": lambda: cls.lof_detection(df, columns, contamination=contamination),
            "dbscan": lambda: cls.dbscan_detection(df, columns),
            "autoencoder": lambda: cls.autoencoder_detection(df, columns, contamination=contamination),
        }

        label_sets: Dict[str, List[int]] = {}
        for method_name in methods:
            func = method_map.get(method_name)
            if func is None:
                all_results["methods"][method_name] = {"error": f"Unknown method: {method_name}"}
                continue
            try:
                logger.info("Running anomaly detection: %s", method_name)
                res = func()
                all_results["methods"][method_name] = res
                if "labels" in res:
                    label_sets[method_name] = res["labels"]
            except Exception as e:
                logger.error("Anomaly detection %s failed: %s", method_name, e, exc_info=True)
                all_results["methods"][method_name] = {"error": str(e)}

        # Consensus: a row is an anomaly if flagged by majority
        if label_sets:
            label_matrix = np.array(list(label_sets.values()))  # shape (n_methods, n_rows)
            n_methods_used = len(label_sets)
            votes = label_matrix.sum(axis=0)
            majority_threshold = n_methods_used / 2
            consensus = (votes >= majority_threshold).astype(int)

            consensus_count = int(consensus.sum())
            all_results["consensus"] = {
                "anomaly_count": consensus_count,
                "anomaly_rate": round(consensus_count / len(df), 4) if len(df) > 0 else 0,
                "methods_used": list(label_sets.keys()),
                "majority_threshold": majority_threshold,
                "anomaly_indices": [int(i) for i in np.where(consensus == 1)[0]][:500],
                "votes_per_row": votes.tolist(),
            }

        return all_results

    @staticmethod
    def get_available_methods() -> Dict[str, Any]:
        """List available detection methods."""
        return {
            "statistical": {
                "zscore": "Z-Score — simple, fast, assumes normality",
                "modified_zscore": "Modified Z-Score (MAD) — robust to non-normal data",
                "iqr": "Interquartile Range — non-parametric, robust",
            },
            "ml_based": {
                "isolation_forest": "Isolation Forest — tree-based, handles high dimensions",
                "one_class_svm": "One-Class SVM — kernel-based novelty detection",
                "lof": "Local Outlier Factor — density-based, local context",
                "dbscan": "DBSCAN — clustering-based, labels noise as anomalies",
                "autoencoder": "Autoencoder — neural network reconstruction error",
            },
        }

    # ================================================================
    #  Evaluation (when ground truth is available)
    # ================================================================
    @staticmethod
    def evaluate_detection(
        y_true: List[int],
        y_pred: List[int],
    ) -> Dict[str, Any]:
        """Evaluate anomaly detection against known labels."""
        y_t = np.array(y_true)
        y_p = np.array(y_pred)
        try:
            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            return {
                "precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
                "recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_t, y_p, zero_division=0)), 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "confusion_matrix": cm.tolist(),
            }
        except Exception as e:
            return {"error": str(e)}
