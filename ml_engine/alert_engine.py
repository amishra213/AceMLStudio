"""
AceML Studio – Alert Engine
=============================
Manage alert rules, evaluate metrics, and generate notifications.

Features:
  • Define custom alert rules (threshold-based, statistical)
  • Evaluate rules against monitored metrics
  • Alert history and acknowledgment tracking
  • Severity levels (INFO, WARNING, CRITICAL)
  • Flexible metric operators (>, <, >=, <=, ==, !=)
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("aceml.alerts")


# ════════════════════════════════════════════════════════════════════
#  Enums & Data Structures
# ════════════════════════════════════════════════════════════════════

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertOperator(str, Enum):
    """Comparison operators for alert thresholds."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


class AlertStatus(str, Enum):
    """Alert status tracking."""
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    MUTED = "muted"


@dataclass
class AlertRule:
    """Defines an alert condition."""
    rule_id: str                       # Unique identifier
    name: str                          # Human-readable name
    description: str = ""
    metric_name: str = ""              # e.g., "accuracy", "ks_statistic"
    operator: str = ">"                # Comparison operator
    threshold: float = 0.0             # Alert if metric [operator] threshold
    severity: str = "warning"          # info, warning, critical
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = datetime.fromtimestamp(self.created_at).isoformat()
        d["updated_at"] = datetime.fromtimestamp(self.updated_at).isoformat()
        return d


@dataclass
class Alert:
    """A triggered alert instance."""
    alert_id: str
    rule_id: str
    rule_name: str
    metric_name: str
    metric_value: float
    threshold: float
    operator: str
    severity: str
    status: str = "triggered"
    message: str = ""
    triggered_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["triggered_at"] = datetime.fromtimestamp(self.triggered_at).isoformat()
        if self.acknowledged_at is not None:
            d["acknowledged_at"] = datetime.fromtimestamp(self.acknowledged_at).isoformat()
        return d


@dataclass
class AlertSession:
    """Tracks alerts for a monitored model."""
    session_id: str
    model_name: str
    rules: Dict[str, AlertRule] = field(default_factory=dict)
    alert_history: List[Alert] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    acknowledged_ids: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "total_alerts": len(self.alert_history),
            "triggered_alerts": sum(1 for a in self.alert_history if a.status == "triggered"),
            "acknowledged_alerts": len(self.acknowledged_ids),
        }


# ════════════════════════════════════════════════════════════════════
#  Alert Engine
# ════════════════════════════════════════════════════════════════════

class AlertEngine:
    """
    Manages alert rules and evaluates metrics against them.

    Usage:
        engine = AlertEngine()
        session = engine.create_session("model_v1", "MyModel")
        engine.create_rule(session.session_id, "high_drift", "ks_statistic", ">", 0.5, "critical")
        alerts = engine.evaluate(session.session_id, {"ks_statistic": 0.6})
        engine.acknowledge_alert("alert_123", "admin")
    """

    def __init__(self):
        self._sessions: Dict[str, AlertSession] = {}

    # ────────────────────────────────────────────────────────────────
    #  Session Management
    # ────────────────────────────────────────────────────────────────

    def create_session(self, session_id: str, model_name: str) -> AlertSession:
        """Create a new alert session."""
        session = AlertSession(session_id=session_id, model_name=model_name)
        self._sessions[session_id] = session
        logger.info("Alert session created: %s (model=%s)", session_id, model_name)
        return session

    def get_session(self, session_id: str) -> Optional[AlertSession]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]

    # ────────────────────────────────────────────────────────────────
    #  Rule Management
    # ────────────────────────────────────────────────────────────────

    def create_rule(
        self,
        session_id: str,
        rule_name: str,
        metric_name: str,
        operator: str,
        threshold: float,
        severity: str = "warning",
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Create a new alert rule.

        Args:
            session_id: Alert session ID
            rule_name: Name for the rule
            metric_name: Metric to monitor (e.g., "accuracy", "ks_statistic")
            operator: Comparison operator (>, <, >=, <=, ==, !=)
            threshold: Threshold value
            severity: Alert severity (info, warning, critical)
            description: Optional description

        Returns:
            {"success": True, "rule_id": "...", "rule": {...}} or error dict
        """
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        # Validate operator
        valid_ops = [op.value for op in AlertOperator]
        if operator not in valid_ops:
            return {"error": f"Invalid operator {operator!r}. Must be one of {valid_ops}"}

        # Validate severity
        valid_severities = [s.value for s in AlertSeverity]
        if severity not in valid_severities:
            return {"error": f"Invalid severity {severity!r}. Must be one of {valid_severities}"}

        rule_id = str(uuid.uuid4())[:8]
        rule = AlertRule(
            rule_id=rule_id,
            name=rule_name,
            description=description,
            metric_name=metric_name,
            operator=operator,
            threshold=threshold,
            severity=severity,
        )

        session.rules[rule_id] = rule
        logger.info(
            "Alert rule created: %s (session=%s, metric=%s %s %s)",
            rule_id, session_id, metric_name, operator, threshold
        )

        return {
            "success": True,
            "rule_id": rule_id,
            "rule": rule.to_dict(),
        }

    def update_rule(
        self,
        session_id: str,
        rule_id: str,
        **updates,
    ) -> Dict[str, Any]:
        """
        Update an existing alert rule.

        Args:
            session_id: Alert session ID
            rule_id: Rule to update
            **updates: Fields to update (name, threshold, severity, enabled, etc.)

        Returns:
            {"success": True, "rule": {...}} or error dict
        """
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        if rule_id not in session.rules:
            return {"error": f"Rule {rule_id!r} not found in session"}

        rule = session.rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key) and key not in ["rule_id", "created_at"]:
                setattr(rule, key, value)

        rule.updated_at = time.time()
        logger.info("Alert rule updated: %s", rule_id)

        return {
            "success": True,
            "rule": rule.to_dict(),
        }

    def delete_rule(self, session_id: str, rule_id: str) -> Dict[str, Any]:
        """Delete an alert rule."""
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        if rule_id not in session.rules:
            return {"error": f"Rule {rule_id!r} not found"}

        del session.rules[rule_id]
        logger.info("Alert rule deleted: %s", rule_id)
        return {"success": True}

    def get_rules(self, session_id: str) -> Dict[str, Any]:
        """Get all rules for a session."""
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        return {
            "session_id": session_id,
            "total_rules": len(session.rules),
            "enabled_rules": sum(1 for r in session.rules.values() if r.enabled),
            "rules": [r.to_dict() for r in session.rules.values()],
        }

    # ────────────────────────────────────────────────────────────────
    #  Rule Evaluation
    # ────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        session_id: str,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Evaluate all rules against provided metrics.

        Args:
            session_id: Alert session ID
            metrics: Dict like {"accuracy": 0.92, "ks_statistic": 0.4, ...}

        Returns:
            {
                "triggered_alerts": [Alert dicts],
                "total_triggered": int,
                "evaluation_time": float
            }
        """
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        start_time = time.time()
        triggered: List[Alert] = []

        for rule in session.rules.values():
            if not rule.enabled:
                continue

            metric_value = metrics.get(rule.metric_name)
            if metric_value is None:
                continue

            # Evaluate condition
            condition_met = self._evaluate_condition(
                metric_value,
                rule.operator,
                rule.threshold,
            )

            if condition_met:
                alert = Alert(
                    alert_id=str(uuid.uuid4())[:12],
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    operator=rule.operator,
                    severity=rule.severity,
                    message=f"{rule.metric_name} {rule.operator} {rule.threshold} (got {metric_value:.4f})",
                )
                triggered.append(alert)
                session.alert_history.append(alert)

        elapsed = round(time.time() - start_time, 4)
        logger.info(
            "Evaluated %d rules, triggered %d alerts (%.4fs)",
            len(session.rules), len(triggered), elapsed
        )

        return {
            "triggered_alerts": [a.to_dict() for a in triggered],
            "total_triggered": len(triggered),
            "evaluation_time_seconds": elapsed,
        }

    @staticmethod
    def _evaluate_condition(value: float, operator: str, threshold: float) -> bool:
        """Evaluate metric against threshold using operator."""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 1e-6
        elif operator == "!=":
            return abs(value - threshold) >= 1e-6
        return False

    # ────────────────────────────────────────────────────────────────
    #  Alert Management
    # ────────────────────────────────────────────────────────────────

    def acknowledge_alert(
        self,
        session_id: str,
        alert_id: str,
        acknowledged_by: str = "system",
    ) -> Dict[str, Any]:
        """Mark an alert as acknowledged."""
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        # Find alert in history
        alert = None
        for a in session.alert_history:
            if a.alert_id == alert_id:
                alert = a
                break

        if alert is None:
            return {"error": f"Alert {alert_id!r} not found"}

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = time.time()
        alert.acknowledged_by = acknowledged_by
        session.acknowledged_ids.add(alert_id)

        logger.info("Alert acknowledged: %s (by %s)", alert_id, acknowledged_by)
        return {"success": True, "alert": alert.to_dict()}

    def get_alert_history(
        self,
        session_id: str,
        limit: int = 100,
        status_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get alert history for a session.

        Args:
            session_id: Alert session ID
            limit: Max alerts to return (most recent first)
            status_filter: Filter by status (triggered, acknowledged, resolved, muted)

        Returns:
            {
                "session_id": str,
                "total_alerts": int,
                "alerts": [Alert dicts]
            }
        """
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        alerts = session.alert_history
        if status_filter:
            alerts = [a for a in alerts if a.status == status_filter]

        # Most recent first
        alerts = sorted(alerts, key=lambda a: a.triggered_at, reverse=True)[:limit]

        return {
            "session_id": session_id,
            "total_alerts": len(session.alert_history),
            "returned_alerts": len(alerts),
            "alerts": [a.to_dict() for a in alerts],
        }

    def get_active_alerts(self, session_id: str) -> Dict[str, Any]:
        """Get only triggered (unacknowledged) alerts."""
        return self.get_alert_history(session_id, status_filter="triggered")

    def clear_alert_history(self, session_id: str) -> Dict[str, Any]:
        """Clear all alerts for a session."""
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        count = len(session.alert_history)
        session.alert_history.clear()
        session.acknowledged_ids.clear()

        logger.info("Alert history cleared: %s (removed %d alerts)", session_id, count)
        return {"success": True, "cleared_alerts": count}

    # ────────────────────────────────────────────────────────────────
    #  Summary & Reporting
    # ────────────────────────────────────────────────────────────────

    def get_alert_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of alert status for a session."""
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"Session {session_id!r} not found"}

        alerts = session.alert_history
        triggered = [a for a in alerts if a.status == "triggered"]
        acknowledged = [a for a in alerts if a.status == "acknowledged"]
        critical = [a for a in triggered if a.severity == "critical"]

        return {
            "session_id": session_id,
            "model_name": session.model_name,
            "created_at": datetime.fromtimestamp(session.created_at).isoformat(),
            "rule_stats": {
                "total_rules": len(session.rules),
                "enabled_rules": sum(1 for r in session.rules.values() if r.enabled),
            },
            "alert_stats": {
                "total_alerts": len(alerts),
                "triggered_alerts": len(triggered),
                "critical_alerts": len(critical),
                "acknowledged_alerts": len(acknowledged),
            },
            "recent_triggered": [a.to_dict() for a in triggered[-5:]],
        }
