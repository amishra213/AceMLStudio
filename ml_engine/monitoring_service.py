"""
AceML Studio – Real-Time Monitoring Service
==============================================
Detect model drift, track performance degradation, and generate alerts.

Drift Detection Methods:
  • Statistical Drift (Kolmogorov-Smirnov test for numeric features)
  • Categorical Drift (Chi-square test for categorical features)
  • Prediction Drift (Distribution shift in predictions)
  • Target Drift (Distribution shift in actual targets, if available)

Performance Monitoring:
  • Track key metrics over time (accuracy, precision, recall, RMSE, etc.)
  • Detect performance degradation thresholds
  • Alert on metric violations
  • Baseline comparison (training vs. production)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("aceml.monitoring")


# ════════════════════════════════════════════════════════════════════
#  Enums & Data Structures
# ════════════════════════════════════════════════════════════════════

class DriftType(str, Enum):
    """Types of drift that can be detected."""
    FEATURE_DRIFT = "feature_drift"          # Input data distribution changed
    PREDICTION_DRIFT = "prediction_drift"    # Model output distribution changed
    TARGET_DRIFT = "target_drift"            # Target label distribution changed
    PERFORMANCE_DRIFT = "performance_drift"  # Model metrics degraded


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftMetric:
    """Single drift detection metric."""
    feature_name: str
    drift_type: str                    # feature_drift, prediction_drift, etc.
    method: str                        # ks_test, chi_square, etc.
    test_statistic: float = 0.0        # e.g., KS statistic
    p_value: float = 0.0               # p-value from statistical test
    threshold: float = 0.05            # Significance level (default α=0.05)
    is_drifted: bool = False           # True if p_value < threshold
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    current_mean: Optional[float] = None
    current_std: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d


@dataclass
class PerformanceMetric:
    """Track a single performance metric over time."""
    metric_name: str                   # accuracy, precision, auc, rmse, etc.
    metric_value: float = 0.0
    baseline_value: float = 0.0        # Value during training
    degradation_pct: float = 0.0       # % change from baseline
    timestamp: float = field(default_factory=time.time)
    is_degraded: bool = False          # True if degradation > threshold

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d


@dataclass
class MonitoringSession:
    """Monitoring state for a trained model."""
    session_id: str
    model_name: str
    baseline_df: Optional[pd.DataFrame] = None  # Training data stats
    baseline_metrics: Dict[str, float] = field(default_factory=dict)  # Training metrics
    drift_history: List[DriftMetric] = field(default_factory=list)
    performance_history: List[PerformanceMetric] = field(default_factory=list)
    prediction_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_monitored: float = field(default_factory=time.time)

    def elapsed_hours(self) -> float:
        return round((time.time() - self.created_at) / 3600, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "prediction_count": self.prediction_count,
            "elapsed_hours": self.elapsed_hours(),
            "drift_count": sum(1 for d in self.drift_history if d.is_drifted),
            "degraded_metrics": sum(1 for p in self.performance_history if p.is_degraded),
            "drift_history": [d.to_dict() for d in self.drift_history[-10:]],  # Last 10
            "performance_history": [p.to_dict() for p in self.performance_history[-10:]],
        }


# ════════════════════════════════════════════════════════════════════
#  Monitoring Service
# ════════════════════════════════════════════════════════════════════

class MonitoringService:
    """
    Real-time monitoring for production models.

    Usage:
        monitor = MonitoringService()
        session = monitor.create_session("model_v1", baseline_df, baseline_metrics)
        monitor.log_predictions(session.session_id, predictions_df, actuals_df)
        drift_report = monitor.detect_drift(session.session_id, new_df)
        perf_report = monitor.check_performance(session.session_id, metrics)
    """

    def __init__(self):
        self._sessions: Dict[str, MonitoringSession] = {}

    # ────────────────────────────────────────────────────────────────
    #  Session Management
    # ────────────────────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        model_name: str,
        baseline_df: Optional[pd.DataFrame] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> MonitoringSession:
        """Create a new monitoring session for a model."""
        session = MonitoringSession(
            session_id=session_id,
            model_name=model_name,
            baseline_df=baseline_df.copy() if baseline_df is not None else None,
            baseline_metrics=baseline_metrics or {},
        )
        self._sessions[session_id] = session
        logger.info("Monitoring session created: %s (model=%s)", session_id, model_name)
        return session

    def get_session(self, session_id: str) -> Optional[MonitoringSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    # ────────────────────────────────────────────────────────────────
    #  Drift Detection
    # ────────────────────────────────────────────────────────────────

    def detect_drift(
        self,
        session_id: str,
        new_df: pd.DataFrame,
        drift_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect data drift between baseline and new data.

        Returns:
            {
                "has_drift": bool,
                "drift_count": int,
                "drifted_features": [feature names],
                "drift_details": [DriftMetric dicts],
                "summary": {...}
            }
        """
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        if session.baseline_df is None or len(session.baseline_df) == 0:
            return {"error": "No baseline data available for drift detection"}

        baseline = session.baseline_df
        new_data = new_df.dropna()

        if len(new_data) == 0:
            return {"error": "New data is empty"}

        drift_metrics: List[DriftMetric] = []

        # Numeric columns: KS test
        numeric_cols = baseline.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in new_data.columns:
                continue
            baseline_vals = np.asarray(baseline[col].dropna(), dtype=float)
            new_vals = np.asarray(new_data[col].dropna(), dtype=float)

            if len(baseline_vals) < 2 or len(new_vals) < 2:
                continue

            ks_result: Tuple[Any, Any] = stats.ks_2samp(baseline_vals, new_vals)
            ks_stat = float(ks_result[0])  # type: ignore
            p_val = float(ks_result[1])    # type: ignore
            is_drifted = p_val < drift_threshold

            metric = DriftMetric(
                feature_name=col,
                drift_type=DriftType.FEATURE_DRIFT,
                method="ks_test",
                test_statistic=ks_stat,
                p_value=p_val,
                threshold=drift_threshold,
                is_drifted=is_drifted,
                baseline_mean=float(np.mean(baseline_vals)),
                baseline_std=float(np.std(baseline_vals)),
                current_mean=float(np.mean(new_vals)),
                current_std=float(np.std(new_vals)),
            )
            drift_metrics.append(metric)

        # Categorical columns: Chi-square test
        cat_cols = baseline.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if col not in new_data.columns:
                continue
            baseline_vals = baseline[col].dropna()
            new_vals = new_data[col].dropna()

            if len(baseline_vals) < 2 or len(new_vals) < 2:
                continue

            # Get common categories
            all_cats = set(baseline_vals.unique()) | set(new_vals.unique())
            baseline_counts = baseline_vals.value_counts()
            new_counts = new_vals.value_counts()

            # Chi-square test
            try:
                chi2_result: Tuple[Any, Any] = stats.chisquare(
                    [new_counts.get(cat, 0) for cat in all_cats],
                    [baseline_counts.get(cat, 1) for cat in all_cats],
                )
                chi2_stat = float(chi2_result[0])  # type: ignore
                p_val = float(chi2_result[1])      # type: ignore
                is_drifted = p_val < drift_threshold

                metric = DriftMetric(
                    feature_name=col,
                    drift_type=DriftType.FEATURE_DRIFT,
                    method="chi_square",
                    test_statistic=float(chi2_stat),
                    p_value=float(p_val),
                    threshold=drift_threshold,
                    is_drifted=is_drifted,
                )
                drift_metrics.append(metric)
            except Exception as e:
                logger.debug("Chi-square test failed for %s: %s", col, e)

        # Store in session
        session.drift_history.extend(drift_metrics)
        session.last_monitored = time.time()

        drifted = [m for m in drift_metrics if m.is_drifted]
        return {
            "has_drift": len(drifted) > 0,
            "drift_count": len(drifted),
            "drifted_features": [m.feature_name for m in drifted],
            "drift_details": [m.to_dict() for m in drift_metrics],
            "summary": {
                "total_features_monitored": len(drift_metrics),
                "drifted_percentage": round(len(drifted) / max(len(drift_metrics), 1) * 100, 1),
            },
        }

    # ────────────────────────────────────────────────────────────────
    #  Performance Monitoring
    # ────────────────────────────────────────────────────────────────

    def check_performance(
        self,
        session_id: str,
        current_metrics: Dict[str, float],
        degradation_threshold_pct: float = 5.0,  # Alert if metric drops > 5%
    ) -> Dict[str, Any]:
        """
        Check if current metrics have degraded from baseline.

        Args:
            session_id: Monitoring session ID
            current_metrics: Dict like {"accuracy": 0.92, "precision": 0.88, ...}
            degradation_threshold_pct: Threshold % change to trigger alert

        Returns:
            {
                "has_degradation": bool,
                "degraded_metrics": [metric names],
                "performance_details": [PerformanceMetric dicts],
                "summary": {...}
            }
        """
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        perf_metrics: List[PerformanceMetric] = []

        for metric_name, current_value in current_metrics.items():
            baseline_value = session.baseline_metrics.get(metric_name, 0.0)

            if baseline_value == 0:
                degradation_pct = 0.0
                is_degraded = False
            else:
                degradation_pct = round((baseline_value - current_value) / baseline_value * 100, 2)
                # For metrics where lower is worse (accuracy, AUC, F1, etc.)
                is_degraded = degradation_pct > degradation_threshold_pct

            metric = PerformanceMetric(
                metric_name=metric_name,
                metric_value=round(current_value, 4),
                baseline_value=round(baseline_value, 4),
                degradation_pct=degradation_pct,
                is_degraded=is_degraded,
            )
            perf_metrics.append(metric)

        session.performance_history.extend(perf_metrics)
        session.last_monitored = time.time()

        degraded = [m for m in perf_metrics if m.is_degraded]
        return {
            "has_degradation": len(degraded) > 0,
            "degraded_metrics": [m.metric_name for m in degraded],
            "performance_details": [m.to_dict() for m in perf_metrics],
            "summary": {
                "total_metrics_monitored": len(perf_metrics),
                "degraded_count": len(degraded),
                "avg_degradation_pct": round(
                    np.mean([abs(m.degradation_pct) for m in perf_metrics]), 2
                ),
            },
        }

    # ────────────────────────────────────────────────────────────────
    #  Prediction Tracking
    # ────────────────────────────────────────────────────────────────

    def log_predictions(
        self,
        session_id: str,
        predictions: pd.Series,
        actuals: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Log predictions made by the model for tracking.

        Args:
            session_id: Monitoring session ID
            predictions: Model predictions (Series)
            actuals: Actual target values (optional, Series)

        Returns:
            {
                "logged_count": int,
                "predictions_summary": {...},
                "actuals_summary": {...} if provided
            }
        """
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        session.prediction_count += len(predictions)
        session.last_monitored = time.time()

        predictions_clean = predictions.dropna()
        result = {
            "logged_count": len(predictions),
            "predictions_summary": {
                "mean": round(float(predictions_clean.mean()), 4),
                "std": round(float(predictions_clean.std()), 4),
                "min": round(float(predictions_clean.min()), 4),
                "max": round(float(predictions_clean.max()), 4),
                "unique_count": int(predictions_clean.nunique()),
            },
        }

        if actuals is not None:
            actuals_clean = actuals.dropna()
            result["actuals_summary"] = {
                "mean": round(float(actuals_clean.mean()), 4),
                "std": round(float(actuals_clean.std()), 4),
                "min": round(float(actuals_clean.min()), 4),
                "max": round(float(actuals_clean.max()), 4),
                "unique_count": int(actuals_clean.nunique()),
            }

        return result

    # ────────────────────────────────────────────────────────────────
    #  Summary & Reporting
    # ────────────────────────────────────────────────────────────────

    def get_monitoring_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a comprehensive monitoring summary for a session."""
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        recent_drifts = session.drift_history[-20:]
        recent_perf = session.performance_history[-20:]

        drifted = sum(1 for d in recent_drifts if d.is_drifted)
        degraded = sum(1 for p in recent_perf if p.is_degraded)

        return {
            "session_id": session.session_id,
            "model_name": session.model_name,
            "created_at": datetime.fromtimestamp(session.created_at).isoformat(),
            "last_monitored": datetime.fromtimestamp(session.last_monitored).isoformat(),
            "elapsed_hours": session.elapsed_hours(),
            "prediction_count": session.prediction_count,
            "drift_status": {
                "detected_drifts": drifted,
                "monitoring_features": len(session.baseline_df.columns) if session.baseline_df is not None else 0,
            },
            "performance_status": {
                "degraded_metrics": degraded,
                "monitored_metrics": len(session.baseline_metrics),
            },
            "recent_drift_details": [d.to_dict() for d in recent_drifts[-5:]],
            "recent_performance_details": [p.to_dict() for p in recent_perf[-5:]],
        }
