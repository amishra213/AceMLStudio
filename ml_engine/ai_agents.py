"""
AceML Studio – AI Agents Engine
==================================
Autonomous multi-step ML agents that orchestrate the full pipeline via
plan → execute → reflect → iterate loops, powered by LLMAnalyzer.

Agents available:
  • DataAnalystAgent      – Explores dataset, identifies issues, produces summary insights
  • FeatureEngineerAgent  – Autonomously suggests and applies feature-engineering steps
  • ModelSelectionAgent   – Recommends models, trains, evaluates, and ranks them
  • AutoMLAgent           – End-to-end agent: data → features → model → report

Each agent stores its session in a dict with steps, status, and outputs so
the frontend can poll progress or retrieve the final report.

Design:
  - All agents are stateless classes; running state is kept in AgentSession objects.
  - AgentOrchestrator manages active sessions (keyed by session_id).
  - LLM calls are routed through LLMAnalyzer._call_llm.
"""

import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("aceml.agents")


# ════════════════════════════════════════════════════════════════════
#  Enums & data structures
# ════════════════════════════════════════════════════════════════════

class AgentType(str, Enum):
    DATA_ANALYST     = "data_analyst"
    FEATURE_ENGINEER = "feature_engineer"
    MODEL_SELECTION  = "model_selection"
    AUTOML           = "automl"


class AgentStatus(str, Enum):
    IDLE      = "idle"
    PLANNING  = "planning"
    RUNNING   = "running"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED    = "failed"
    STOPPED   = "stopped"


class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    SKIPPED   = "skipped"
    FAILED    = "failed"


@dataclass
class AgentStep:
    """A single agent action step."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    status: str = StepStatus.PENDING
    started_at: float = 0.0
    finished_at: float = 0.0
    result: Dict[str, Any] = field(default_factory=dict)
    llm_reasoning: str = ""
    error: str = ""

    def duration(self) -> float:
        if self.started_at and self.finished_at:
            return round(self.finished_at - self.started_at, 3)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_sec"] = self.duration()
        return d


@dataclass
class AgentSession:
    """Full lifecycle of one agent run."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    agent_type: str = AgentType.DATA_ANALYST
    status: str = AgentStatus.IDLE
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    steps: List[AgentStep] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)       # LLM-generated plan (step names)
    final_report: str = ""
    context: Dict[str, Any] = field(default_factory=dict)  # dataset summary etc.
    error: str = ""
    iterations: int = 0
    max_iterations: int = 5

    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.created_at, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "elapsed_sec": self.elapsed(),
            "steps": [s.to_dict() for s in self.steps],
            "plan": self.plan,
            "final_report": self.final_report,
            "context": self.context,
            "error": self.error,
            "iterations": self.iterations,
        }


# ════════════════════════════════════════════════════════════════════
#  Base Agent
# ════════════════════════════════════════════════════════════════════

class BaseAgent:
    """Base class shared by all agents."""

    AGENT_TYPE: str = AgentType.DATA_ANALYST
    MAX_ITERATIONS: int = 5

    def __init__(self, llm_analyzer_class: Any) -> None:
        self._llm = llm_analyzer_class

    # ----------------------------------------------------------------
    #  LLM helpers
    # ----------------------------------------------------------------
    def _llm_call(self, prompt: str) -> str:
        try:
            return self._llm._call_llm(prompt)
        except Exception as exc:  # pragma: no cover
            logger.error("LLM call failed in agent: %s", exc)
            return f"[LLM Error] {exc}"

    def _llm_plan(self, session: AgentSession) -> List[str]:
        """Ask LLM to build a step-by-step plan given the context."""
        ctx_str = json.dumps(session.context, indent=2, default=str)
        prompt = (
            f"You are an autonomous {self.AGENT_TYPE.replace('_', ' ')} agent in AceML Studio.\n"
            f"Given the following dataset context, produce a numbered action plan with up to "
            f"{self.MAX_ITERATIONS} concrete steps.\n"
            "Each step must be one short imperative sentence (e.g. 'Check missing values').\n"
            "Return ONLY a JSON array of strings — no prose, no markdown fences.\n\n"
            f"Context:\n{ctx_str}"
        )
        raw = self._llm_call(prompt)
        # Parse JSON plan
        try:
            # Strip possible markdown fences
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1].lstrip("json").strip()
            plan = json.loads(clean)
            if isinstance(plan, list):
                return [str(p) for p in plan[:self.MAX_ITERATIONS]]
        except Exception:
            pass
        # Fallback: split by newlines
        lines = [l.lstrip("0123456789.-) ").strip() for l in raw.splitlines() if l.strip()]
        return lines[:self.MAX_ITERATIONS] or ["Analyse dataset"]

    def _llm_reflect(self, session: AgentSession, step_results: List[Dict]) -> str:
        """Ask LLM to reflect on completed steps and produce insights."""
        results_str = json.dumps(step_results, indent=2, default=str)
        prompt = (
            f"You are an autonomous {self.AGENT_TYPE.replace('_', ' ')} agent.\n"
            "The following steps have been completed. Reflect on the outcomes and provide:\n"
            "1. A brief summary of findings (3–5 bullet points)\n"
            "2. Key recommendations or next actions\n"
            "3. Any warnings or concerns\n\n"
            f"Steps completed:\n{results_str}"
        )
        return self._llm_call(prompt)

    # ----------------------------------------------------------------
    #  Step execution (to be overridden)
    # ----------------------------------------------------------------
    def execute_step(
        self,
        step: AgentStep,
        session: AgentSession,
        df: Optional[pd.DataFrame],
    ) -> None:
        """Perform one step.  Subclasses override _run_step_logic."""
        step.status = StepStatus.RUNNING
        step.started_at = time.time()
        try:
            result = self._run_step_logic(step.name, session, df)
            step.result = result
            step.status = StepStatus.COMPLETED
        except Exception as exc:  # pragma: no cover
            step.error = str(exc)
            step.status = StepStatus.FAILED
            logger.error("Agent step '%s' failed: %s", step.name, exc, exc_info=True)
        finally:
            step.finished_at = time.time()

    def _run_step_logic(
        self,
        step_name: str,
        session: AgentSession,
        df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Override in subclasses. Returns a JSON-serialisable result dict."""
        return {"step": step_name, "note": "No logic defined for base agent."}

    # ----------------------------------------------------------------
    #  Top-level run
    # ----------------------------------------------------------------
    def run(self, session: AgentSession, df: Optional[pd.DataFrame]) -> None:
        """Execute the full agent loop synchronously."""
        try:
            session.status = AgentStatus.PLANNING

            # Build context
            if df is not None:
                session.context.update(_summarise_df(df))

            # LLM plans the steps
            session.plan = self._llm_plan(session)
            logger.info("Agent %s planned %d steps", self.AGENT_TYPE, len(session.plan))

            # Build step objects
            session.steps = [
                AgentStep(name=name, description=f"Step: {name}")
                for name in session.plan
            ]

            session.status = AgentStatus.RUNNING

            # Execute steps
            for step in session.steps:
                if session.status == AgentStatus.STOPPED:
                    break
                session.iterations += 1
                self.execute_step(step, session, df)

            # Reflect
            session.status = AgentStatus.REFLECTING
            completed_results = [
                {"step": s.name, "result": s.result, "error": s.error}
                for s in session.steps
            ]
            session.final_report = self._llm_reflect(session, completed_results)

            session.status = AgentStatus.COMPLETED

        except Exception as exc:  # pragma: no cover
            session.error = str(exc)
            session.status = AgentStatus.FAILED
            logger.error("Agent %s run failed: %s", self.AGENT_TYPE, exc, exc_info=True)
        finally:
            session.finished_at = time.time()


# ════════════════════════════════════════════════════════════════════
#  Data Analyst Agent
# ════════════════════════════════════════════════════════════════════

class DataAnalystAgent(BaseAgent):
    """
    Autonomously explores a dataset:
      - Profile columns (dtypes, missing, cardinality, distributions)
      - Detect data-quality issues (outliers, skew, leakage risk)
      - Compute correlations
      - Generate a narrative analysis report
    """

    AGENT_TYPE = AgentType.DATA_ANALYST
    MAX_ITERATIONS = 6

    def _run_step_logic(
        self, step_name: str, session: AgentSession, df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        if df is None:
            return {"error": "No dataset loaded"}

        name_lower = step_name.lower()

        # ── Column profiling ────────────────────────────────────────
        if any(k in name_lower for k in ("profile", "column", "dtype", "type", "schema")):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
            profile = []
            for col in df.columns:
                series = df[col]
                entry: Dict[str, Any] = {
                    "column": col,
                    "dtype": str(series.dtype),
                    "null_count": int(series.isna().sum()),
                    "null_pct": round(float(series.isna().mean() * 100), 2),
                    "unique": int(series.nunique()),
                }
                if pd.api.types.is_numeric_dtype(series):
                    entry["mean"] = round(float(series.mean()), 4)
                    entry["std"] = round(float(series.std()), 4)
                    entry["min"] = round(float(series.min()), 4)
                    entry["max"] = round(float(series.max()), 4)
                    entry["skew"] = round(float(series.skew()), 4)  # type: ignore[arg-type]
                else:
                    top_vals = series.value_counts().head(5)
                    entry["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
                profile.append(entry)

            # Store in session for later steps
            session.context["profile"] = profile
            return {
                "total_columns": len(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": cat_cols,
                "datetime_columns": dt_cols,
                "profile": profile[:30],  # cap
            }

        # ── Missing values ──────────────────────────────────────────
        if any(k in name_lower for k in ("miss", "null", "nan", "empty")):
            missing = df.isna().sum()
            return {
                "total_missing_cells": int(missing.sum()),
                "pct_missing_overall": round(float(missing.sum() / df.size * 100), 2),
                "columns_with_missing": [
                    {"column": col, "missing": int(missing[col]),
                     "pct": round(float(missing[col] / len(df) * 100), 2)}
                    for col in missing[missing > 0].index.tolist()
                ],
            }

        # ── Outlier detection ───────────────────────────────────────
        if any(k in name_lower for k in ("outlier", "anomal", "extreme")):
            numeric_df = df.select_dtypes(include=[np.number])
            outlier_info = []
            for col in numeric_df.columns:
                s = numeric_df[col].dropna()
                if len(s) == 0:
                    continue
                q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_out = int(((s < lower) | (s > upper)).sum())
                if n_out > 0:
                    outlier_info.append({
                        "column": col,
                        "n_outliers": n_out,
                        "pct_outliers": round(n_out / len(s) * 100, 2),
                        "lower_fence": round(lower, 4),
                        "upper_fence": round(upper, 4),
                    })
            session.context["outliers"] = outlier_info
            return {"outlier_columns": outlier_info}

        # ── Correlations ────────────────────────────────────────────
        if any(k in name_lower for k in ("corr", "relation", "associat")):
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                return {"message": "Fewer than 2 numeric columns, correlation skipped"}
            corr = numeric_df.corr()
            # Top pairs
            pairs: List[Dict[str, Any]] = []
            cols = list(corr.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    v = float(corr.iloc[i, j])  # type: ignore[arg-type]
                    if abs(v) > 0.5:
                        pairs.append({"col_a": cols[i], "col_b": cols[j], "correlation": round(v, 4)})
            pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            return {"strong_correlations": pairs[:30], "total_features": len(cols)}

        # ── Class imbalance / target analysis ───────────────────────
        if any(k in name_lower for k in ("target", "label", "class", "imbalanc", "distribut")):
            target = session.context.get("target")
            if target and target in df.columns:
                vc = df[target].value_counts()
                return {
                    "target": target,
                    "unique_classes": int(vc.nunique()),
                    "class_distribution": {str(k): int(v) for k, v in vc.items()},
                    "imbalance_ratio": round(float(vc.max() / vc.min()), 2) if vc.min() > 0 else None,
                }
            # No target: general summary
            return {
                "rows": len(df),
                "cols": len(df.columns),
                "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
            }

        # ── Duplicate detection ─────────────────────────────────────
        if any(k in name_lower for k in ("duplic", "unique row")):
            n_dupl = int(df.duplicated().sum())
            return {
                "n_duplicate_rows": n_dupl,
                "pct_duplicates": round(n_dupl / len(df) * 100, 2),
            }

        # ── Summary statistics (fallback) ───────────────────────────
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            return {"rows": len(df), "cols": len(df.columns)}
        desc = numeric_df.describe().to_dict()
        return {
            "rows": len(df),
            "cols": len(df.columns),
            "summary": {
                col: {k: round(float(v), 4) for k, v in stats.items()}
                for col, stats in desc.items()
            },
        }


# ════════════════════════════════════════════════════════════════════
#  Feature Engineer Agent
# ════════════════════════════════════════════════════════════════════

class FeatureEngineerAgent(BaseAgent):
    """
    Autonomously explores and recommends feature-engineering steps:
      - Datetime feature extraction
      - Interaction features
      - Encoding recommendations
      - Transformation recommendations (log, sqrt, box-cox)
      - Feature importance proxy (variance, correlation with target)
    """

    AGENT_TYPE = AgentType.FEATURE_ENGINEER
    MAX_ITERATIONS = 5

    def _run_step_logic(
        self, step_name: str, session: AgentSession, df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        if df is None:
            return {"error": "No dataset loaded"}

        name_lower = step_name.lower()
        target = session.context.get("target")

        # ── Datetime features ────────────────────────────────────────
        if any(k in name_lower for k in ("date", "time", "temporal")):
            dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
            # Also check object columns parseable as dates
            for col in df.select_dtypes(include=["object"]).columns:
                try:
                    parsed = pd.to_datetime(df[col].head(20), errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        dt_cols.append(col)
                except Exception:
                    pass
            suggestions = []
            for col in dt_cols:
                suggestions.append({
                    "column": col,
                    "recommended_features": [
                        f"{col}_year", f"{col}_month", f"{col}_day",
                        f"{col}_dayofweek", f"{col}_hour", f"{col}_quarter",
                        f"{col}_is_weekend",
                    ],
                    "reason": "Datetime columns should be decomposed into cyclical/ordinal features",
                })
            return {"datetime_suggestions": suggestions, "datetime_columns_found": len(dt_cols)}

        # ── Encoding recommendations ─────────────────────────────────
        if any(k in name_lower for k in ("encod", "categor", "nominal", "ordinal")):
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            recommendations = []
            for col in cat_cols:
                n_unique = df[col].nunique()
                rec: Dict[str, Any] = {"column": col, "unique_values": n_unique}
                if n_unique <= 5:
                    rec["encoding"] = "one_hot"
                    rec["reason"] = "Low cardinality — one-hot is safe"
                elif n_unique <= 20:
                    rec["encoding"] = "ordinal_or_onehot"
                    rec["reason"] = "Medium cardinality — consider ordinal or one-hot"
                elif target and target in df.columns:
                    rec["encoding"] = "target_encoding"
                    rec["reason"] = "High cardinality with target available — use target encoding"
                else:
                    rec["encoding"] = "frequency_encoding"
                    rec["reason"] = "High cardinality without target — use frequency encoding"
                recommendations.append(rec)
            return {"encoding_recommendations": recommendations}

        # ── Transformation recommendations ──────────────────────────
        if any(k in name_lower for k in ("transform", "skew", "log", "scale", "normaliz")):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            recs = []
            for col in numeric_cols:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                skew_val = float(s.skew())  # type: ignore[arg-type]
                rec: Dict[str, Any] = {
                    "column": col,
                    "skewness": round(skew_val, 4),
                }
                if skew_val > 1.0 and float(s.min()) >= 0:
                    rec["recommendation"] = "log1p"
                    rec["reason"] = "Right-skewed with non-negative values"
                elif skew_val < -1.0:
                    rec["recommendation"] = "reflect_then_log"
                    rec["reason"] = "Left-skewed — reflect and apply log"
                elif abs(skew_val) > 0.5:
                    rec["recommendation"] = "sqrt_or_boxcox"
                    rec["reason"] = "Moderate skew — square root or Box-Cox"
                else:
                    rec["recommendation"] = "standardize"
                    rec["reason"] = "Near-normal — StandardScaler sufficient"
                recs.append(rec)
            return {"transformation_recommendations": recs}

        # ── Interaction features ─────────────────────────────────────
        if any(k in name_lower for k in ("interact", "cross", "product", "ratio")):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = target if target and target in numeric_cols else None
            if target_col:
                numeric_cols = [c for c in numeric_cols if c != target_col]

            suggestions = []
            # Top correlated pairs as interaction candidates
            if len(numeric_cols) >= 2:
                sample_df = df[numeric_cols].dropna()
                corr = sample_df.corr().abs()
                already_seen: set = set()
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        c1, c2 = numeric_cols[i], numeric_cols[j]
                        key = f"{c1}__{c2}"
                        if key in already_seen:
                            continue
                        already_seen.add(key)
                        v = float(corr.iloc[i, j])  # type: ignore[arg-type]
                        suggestions.append({
                            "feature_a": c1,
                            "feature_b": c2,
                            "base_correlation": round(v, 4),
                            "suggested_interactions": [
                                f"{c1}_x_{c2} (product)",
                                f"{c1}_div_{c2} (ratio)" if float(df[c2].replace(0, np.nan).dropna().min()) > 0 else None,
                            ],
                        })
                        if len(suggestions) >= 15:
                            break
                    if len(suggestions) >= 15:
                        break
            return {"interaction_suggestions": suggestions}

        # ── Feature importance proxy (via variance / correlation) ────
        if any(k in name_lower for k in ("import", "select", "rank", "relevant")):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            ranking = []
            if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
                target_series = df[target].dropna()
                for col in numeric_cols:
                    if col == target:
                        continue
                    s = df[col].dropna()
                    common_idx = s.index.intersection(target_series.index)
                    if len(common_idx) < 10:
                        continue
                    corr_val = float(np.corrcoef(
                        np.array(s.loc[common_idx].values, dtype=float),
                        np.array(target_series.loc[common_idx].values, dtype=float),
                    )[0, 1])
                    ranking.append({
                        "column": col,
                        "abs_correlation_with_target": round(abs(corr_val), 4),
                        "correlation": round(corr_val, 4),
                    })
                ranking.sort(key=lambda x: x["abs_correlation_with_target"], reverse=True)
            else:
                # Variance-based ranking
                for col in numeric_cols:
                    variance = float(df[col].var())  # type: ignore[arg-type]
                    ranking.append({"column": col, "variance": round(variance, 6)})
                ranking.sort(key=lambda x: x.get("variance", 0), reverse=True)
            return {"feature_ranking": ranking[:30]}

        # ── Fallback ─────────────────────────────────────────────────
        return {
            "step": step_name,
            "columns": len(df.columns),
            "rows": len(df),
            "numeric": len(df.select_dtypes(include=[np.number]).columns),
            "categorical": len(df.select_dtypes(include=["object", "category"]).columns),
        }


# ════════════════════════════════════════════════════════════════════
#  Model Selection Agent
# ════════════════════════════════════════════════════════════════════

class ModelSelectionAgent(BaseAgent):
    """
    Recommends, trains baseline versions, evaluates, and ranks ML models.
    Uses LLM reasoning to map dataset characteristics to model families.
    """

    AGENT_TYPE = AgentType.MODEL_SELECTION
    MAX_ITERATIONS = 5

    # Model catalogue
    CLASSIFICATION_MODELS = [
        "logistic_regression", "random_forest_clf", "gradient_boosting_clf",
        "xgboost_clf", "svm_clf", "knn_clf",
    ]
    REGRESSION_MODELS = [
        "linear_regression", "random_forest_reg", "gradient_boosting_reg",
        "xgboost_reg", "lasso", "ridge",
    ]

    def _run_step_logic(
        self, step_name: str, session: AgentSession, df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        if df is None:
            return {"error": "No dataset loaded"}

        name_lower = step_name.lower()
        target = session.context.get("target")
        task = session.context.get("task", "classification")

        # ── Task / target analysis ───────────────────────────────────
        if any(k in name_lower for k in ("task", "problem", "target", "type")):
            if not target:
                return {"message": "No target variable specified in context"}
            if target not in df.columns:
                return {"message": f"Target column '{target}' not found in dataset"}
            target_series = df[target].dropna()
            n_unique = target_series.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(target_series)
            inferred_task = "regression" if (is_numeric and n_unique > 20) else "classification"
            session.context["task"] = inferred_task
            session.context["n_classes"] = n_unique if inferred_task == "classification" else None
            return {
                "target": target,
                "unique_values": int(n_unique),
                "dtype": str(target_series.dtype),
                "inferred_task": inferred_task,
                "n_samples": len(df),
                "class_balance": {
                    str(k): int(v)
                    for k, v in target_series.value_counts().head(10).items()
                } if inferred_task == "classification" else None,
            }

        # ── Model recommendation ─────────────────────────────────────
        if any(k in name_lower for k in ("recommend", "suggest", "cand", "select model")):
            task_type = session.context.get("task", "classification")
            n_samples = len(df) if df is not None else 0
            n_features = len(df.columns) if df is not None else 0

            if task_type == "classification":
                candidates = self.CLASSIFICATION_MODELS
            else:
                candidates = self.REGRESSION_MODELS

            # Heuristic scoring
            scored: List[Dict[str, Any]] = []
            for model_name in candidates:
                score = 50
                reason = []
                if n_samples < 1000 and "knn" in model_name:
                    score += 10
                    reason.append("small dataset suits KNN")
                if n_samples > 10000 and "gradient_boosting" in model_name:
                    score += 15
                    reason.append("large dataset — GBM handles well")
                if n_features > 50 and "lasso" in model_name:
                    score += 10
                    reason.append("high-dimensional data — regularization helps")
                if n_features < 20 and "linear" in model_name:
                    score += 5
                    reason.append("low-dimensional — linear baseline is useful")
                if not reason:
                    reason.append("general-purpose baseline")
                scored.append({
                    "model": model_name,
                    "score": score,
                    "reason": "; ".join(reason),
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            session.context["recommended_models"] = [m["model"] for m in scored[:3]]
            return {"recommendations": scored, "top_3": [m["model"] for m in scored[:3]]}

        # ── Baseline evaluation ──────────────────────────────────────
        if any(k in name_lower for k in ("train", "fit", "baseline", "evaluat")):
            if not target or target not in df.columns:
                return {"error": "No target variable for training"}
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
            from sklearn.linear_model import LogisticRegression, LinearRegression

            task_type = session.context.get("task", "classification")
            numeric_df = df.select_dtypes(include=[np.number]).dropna()

            feature_cols = [c for c in numeric_df.columns if c != target]
            if not feature_cols:
                return {"error": "No numeric feature columns found for training"}

            X = numeric_df[feature_cols].values
            if task_type == "classification":
                le = LabelEncoder()
                y = le.fit_transform(df[target].dropna().values[:len(X)])
                models_to_eval = {
                    "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
                    "random_forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                    "gradient_boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
                }
                scoring = "f1_weighted"
            else:
                y = numeric_df[target].values
                X = numeric_df[feature_cols].values
                models_to_eval = {
                    "linear_regression": LinearRegression(),
                    "random_forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                    "gradient_boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
                }
                scoring = "r2"

            results = []
            for model_name, model_obj in models_to_eval.items():
                try:
                    cv_n = min(5, len(X))
                    scores = cross_val_score(model_obj, X, y, cv=cv_n, scoring=scoring, n_jobs=-1)
                    results.append({
                        "model": model_name,
                        "cv_mean": round(float(scores.mean()), 4),
                        "cv_std": round(float(scores.std()), 4),
                        "scoring": scoring,
                    })
                except Exception as exc:
                    results.append({"model": model_name, "error": str(exc)})

            results.sort(key=lambda x: x.get("cv_mean", -999), reverse=True)
            session.context["baseline_results"] = results
            return {"baseline_results": results, "best_model": results[0] if results else None}

        # ── Risk / leakage analysis ──────────────────────────────────
        if any(k in name_lower for k in ("leak", "risk", "sanity")):
            issues = []
            if target and target in df.columns:
                numeric_df = df.select_dtypes(include=[np.number])
                if target in numeric_df.columns:
                    corr = numeric_df.corr()[target].drop(target).abs()
                    perfect = corr[corr > 0.99].index.tolist()
                    if perfect:
                        issues.append({
                            "type": "potential_leakage",
                            "columns": perfect,
                            "detail": "Correlation > 0.99 with target — likely data leakage",
                        })
            n_dupl = int(df.duplicated().sum())
            if n_dupl > 0:
                issues.append({
                    "type": "duplicate_rows",
                    "count": n_dupl,
                    "detail": "Duplicate rows may cause train/test bleed",
                })
            return {"data_quality_risks": issues, "n_risks": len(issues)}

        # ── Fallback ─────────────────────────────────────────────────
        return {"step": step_name, "rows": len(df), "cols": len(df.columns)}


# ════════════════════════════════════════════════════════════════════
#  AutoML Agent  (orchestrates all agents end-to-end)
# ════════════════════════════════════════════════════════════════════

class AutoMLAgent(BaseAgent):
    """
    End-to-end AutoML agent that chains Data Analyst → Feature Engineer →
    Model Selection and produces a consolidated report.
    """

    AGENT_TYPE = AgentType.AUTOML
    MAX_ITERATIONS = 12

    def run(self, session: AgentSession, df: Optional[pd.DataFrame]) -> None:
        """Override to run three sub-agents in sequence."""
        try:
            session.status = AgentStatus.RUNNING
            if df is not None:
                session.context.update(_summarise_df(df))

            sub_agents = [
                DataAnalystAgent(self._llm),
                FeatureEngineerAgent(self._llm),
                ModelSelectionAgent(self._llm),
            ]
            sub_reports = []
            for sub_agent in sub_agents:
                sub_session = AgentSession(
                    agent_type=sub_agent.AGENT_TYPE,
                    context=dict(session.context),
                    max_iterations=sub_agent.MAX_ITERATIONS,
                )
                sub_agent.run(sub_session, df)
                session.steps.extend(sub_session.steps)
                session.iterations += sub_session.iterations
                sub_reports.append({
                    "agent": sub_agent.AGENT_TYPE,
                    "report": sub_session.final_report,
                    "status": sub_session.status,
                })
                # Propagate context updates back
                session.context.update(sub_session.context)
                if session.status == AgentStatus.STOPPED:
                    break

            # Final consolidated report
            session.status = AgentStatus.REFLECTING
            reports_str = json.dumps(sub_reports, indent=2, default=str)
            session.final_report = self._llm_call(
                f"You are the AceML AutoML Agent. The following sub-agents have completed their analysis:\n\n"
                f"{reports_str}\n\n"
                "Produce a final consolidated AutoML report with:\n"
                "1. Dataset overview and key findings\n"
                "2. Recommended feature engineering steps (prioritised)\n"
                "3. Best model recommendation with justification\n"
                "4. Suggested next steps\n"
                "5. Any warnings or caveats\n"
                "Be concise and actionable."
            )
            session.status = AgentStatus.COMPLETED

        except Exception as exc:
            session.error = str(exc)
            session.status = AgentStatus.FAILED
            logger.error("AutoML agent failed: %s", exc, exc_info=True)
        finally:
            session.finished_at = time.time()


# ════════════════════════════════════════════════════════════════════
#  Agent Orchestrator
# ════════════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """
    Manages active agent sessions.

    Usage:
        orchestrator = AgentOrchestrator(LLMAnalyzer)
        session = orchestrator.create_session("data_analyst", context={"target": "price"})
        orchestrator.run_sync(session.session_id, df)
        result = orchestrator.get_session(session.session_id)
    """

    AGENT_MAP = {
        AgentType.DATA_ANALYST:     DataAnalystAgent,
        AgentType.FEATURE_ENGINEER: FeatureEngineerAgent,
        AgentType.MODEL_SELECTION:  ModelSelectionAgent,
        AgentType.AUTOML:           AutoMLAgent,
    }

    def __init__(self, llm_analyzer_class: Any) -> None:
        self._llm = llm_analyzer_class
        self._sessions: Dict[str, AgentSession] = {}

    # ----------------------------------------------------------------
    def create_session(
        self,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentSession:
        session = AgentSession(
            agent_type=agent_type,
            context=context or {},
        )
        self._sessions[session.session_id] = session
        logger.info("Agent session created: %s (type=%s)", session.session_id, agent_type)
        return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        return self._sessions.get(session_id)

    def stop_session(self, session_id: str) -> bool:
        s = self._sessions.get(session_id)
        if s and s.status in (AgentStatus.RUNNING, AgentStatus.PLANNING, AgentStatus.REFLECTING):
            s.status = AgentStatus.STOPPED
            s.finished_at = time.time()
            return True
        return False

    def run_sync(self, session_id: str, df: Optional[pd.DataFrame]) -> AgentSession:
        """Run the agent synchronously (blocks until complete)."""
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id!r} not found")
        agent_cls = self.AGENT_MAP.get(
            AgentType(session.agent_type), DataAnalystAgent  # type: ignore[call-overload]
        )
        agent = agent_cls(self._llm)
        agent.run(session, df)
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "agent_type": s.agent_type,
                "status": s.status,
                "elapsed_sec": s.elapsed(),
                "n_steps": len(s.steps),
            }
            for s in self._sessions.values()
        ]

    def clear_completed(self) -> int:
        before = len(self._sessions)
        self._sessions = {
            k: v for k, v in self._sessions.items()
            if v.status not in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.STOPPED)
        }
        return before - len(self._sessions)

    @staticmethod
    def available_agent_types() -> List[Dict[str, Any]]:
        return [
            {
                "type": AgentType.DATA_ANALYST,
                "name": "Data Analyst Agent",
                "description": (
                    "Autonomously profiles the dataset, identifies data-quality issues, "
                    "detects outliers, and generates a narrative analysis report."
                ),
                "max_iterations": DataAnalystAgent.MAX_ITERATIONS,
            },
            {
                "type": AgentType.FEATURE_ENGINEER,
                "name": "Feature Engineer Agent",
                "description": (
                    "Recommends datetime decomposition, encoding strategies, "
                    "transformation techniques, and interaction features."
                ),
                "max_iterations": FeatureEngineerAgent.MAX_ITERATIONS,
            },
            {
                "type": AgentType.MODEL_SELECTION,
                "name": "Model Selection Agent",
                "description": (
                    "Infers the ML task, recommends candidate models, trains baselines "
                    "with cross-validation, and ranks models by performance."
                ),
                "max_iterations": ModelSelectionAgent.MAX_ITERATIONS,
            },
            {
                "type": AgentType.AUTOML,
                "name": "AutoML Agent",
                "description": (
                    "Full end-to-end agent that chains Data Analyst → Feature Engineer → "
                    "Model Selection and produces a consolidated AutoML report."
                ),
                "max_iterations": AutoMLAgent.MAX_ITERATIONS,
            },
        ]


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def _summarise_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce a lightweight JSON-serialisable summary of a DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "datetime_columns": dt_cols,
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
    }
