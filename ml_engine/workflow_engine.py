"""
AceML Studio – Iterative Workflow Engine
==========================================
LLM-driven, iterative data preparation workflow that orchestrates:
  data analysis → data cleanup → transformations → feature engineering →
  dimensionality reduction.

The engine uses an LLM to plan, analyze after each step, decide whether
to iterate further, and select the best parameters at every stage.
Workflow state is fully serialisable so the frontend can poll / control it.

**Bi-directional navigation** — steps can be:
  • skipped and deferred to a later queue
  • run in any order by ID
  • re-ordered (moved up / down) within the iteration
  • re-run after completion or failure
  • deferred steps can be recalled back into the active queue at any position
Each mutation saves a DataFrame snapshot so you can safely go back.
"""

import logging
import time
import uuid
import json
import copy
import traceback
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from ml_engine.data_quality import DataQualityAnalyzer
from ml_engine.data_cleaning import DataCleaner
from ml_engine.data_loader import DataLoader
from ml_engine.feature_engineering import FeatureEngineer
from ml_engine.transformations import DataTransformer
from ml_engine.dimensionality import DimensionalityReducer

logger = logging.getLogger("aceml.workflow_engine")


# ════════════════════════════════════════════════════════════════════
#  Enums & Data Structures
# ════════════════════════════════════════════════════════════════════

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class StepType(str, Enum):
    DATA_ANALYSIS = "data_analysis"
    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    TRANSFORMATIONS = "transformations"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    EVALUATION = "evaluation"  # LLM evaluates current state


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    DEFERRED = "deferred"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """A single step in the workflow."""
    id: str = ""
    step_type: str = ""
    title: str = ""
    description: str = ""
    status: str = StepStatus.PENDING
    operations: list = field(default_factory=list)  # actual ops to execute
    llm_rationale: str = ""  # why the LLM chose this
    result_summary: str = ""
    error: str = ""
    started_at: float = 0
    finished_at: float = 0
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    # Bi-directional tracking
    run_count: int = 0        # how many times this step has been executed
    deferred_at: float = 0    # timestamp when deferred (0 = never)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkflowIteration:
    """One full pass through the pipeline steps."""
    iteration_number: int = 0
    steps: list = field(default_factory=list)       # list of WorkflowStep dicts (active queue)
    deferred_steps: list = field(default_factory=list)  # list of WorkflowStep dicts parked for later
    planned_at: float = 0
    completed_at: float = 0
    llm_plan: str = ""
    llm_evaluation: str = ""
    should_continue: bool = False
    improvement_summary: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class WorkflowState:
    """Full state of a workflow run — serialisable to JSON for the frontend."""
    workflow_id: str = ""
    status: str = WorkflowStatus.PENDING
    target_column: str = ""
    task_type: str = "classification"
    objectives: str = ""
    max_iterations: int = 5
    current_iteration: int = 0
    iterations: list = field(default_factory=list)  # list of WorkflowIteration dicts
    created_at: float = 0
    updated_at: float = 0
    error: str = ""
    # Snapshot metrics
    initial_quality_score: int = 0
    current_quality_score: int = 0
    initial_shape: dict = field(default_factory=dict)
    current_shape: dict = field(default_factory=dict)
    # Config
    auto_approve: bool = True  # run without waiting for user approval
    enabled_steps: list = field(default_factory=lambda: [
        StepType.DATA_ANALYSIS,
        StepType.DATA_CLEANING,
        StepType.FEATURE_ENGINEERING,
        StepType.TRANSFORMATIONS,
        StepType.DIMENSIONALITY_REDUCTION,
    ])
    # Bi-directional navigation info (not serialised — just for frontend hints)
    supports_navigation: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ════════════════════════════════════════════════════════════════════
#  Helper: DataFrame snapshot for LLM context
# ════════════════════════════════════════════════════════════════════

def _df_snapshot(df: pd.DataFrame, target: str | None = None) -> dict:
    """Create a concise snapshot of a DataFrame for LLM context."""
    info = DataLoader.get_info(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    snapshot = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist()[:50],  # cap for token limit
        "numeric_columns": num_cols[:30],
        "categorical_columns": cat_cols[:30],
        "dtypes": {c: str(df[c].dtype) for c in df.columns[:50]},
        "missing_pct": {
            c: round(df[c].isna().mean() * 100, 2)
            for c in df.columns if df[c].isna().any()
        },
        "sample_values": {
            c: df[c].dropna().head(3).tolist()
            for c in df.columns[:20]
        },
    }
    if target:
        snapshot["target_column"] = target
        if target in df.columns:
            snapshot["target_nunique"] = int(df[target].nunique())
    return snapshot


def _quality_metrics(df: pd.DataFrame, target: str | None = None) -> dict:
    """Run a quick quality check and return key metrics."""
    try:
        analyzer = DataQualityAnalyzer(df, target_column=target)
        report = analyzer.full_report()
        return {
            "quality_score": report.get("quality_score", 0),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_pct": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2) if len(df) > 0 else 0,
            "duplicate_rows": int(df.duplicated().sum()),
            "n_issues": len(report.get("issues", [])),
            "critical_issues": sum(1 for i in report.get("issues", []) if i.get("severity") == "critical"),
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "high_cardinality_cols": len(report.get("high_cardinality_columns", [])),
            "outlier_cols": len(report.get("outliers", {})),
        }
    except Exception as e:
        logger.error("Quality metrics failed: %s", e)
        return {"quality_score": 0, "error": str(e)}


# ════════════════════════════════════════════════════════════════════
#  Workflow Engine
# ════════════════════════════════════════════════════════════════════

class WorkflowEngine:
    """
    Orchestrates the iterative data-preparation workflow.

    **Bi-directional navigation** — the engine supports:
      • run_next_step()        — execute the next pending step in order
      • run_step_by_id(id)     — jump to and execute any step (pending, deferred, or re-run)
      • defer_step(id)         — skip a step now, park it in the deferred queue
      • recall_step(id, pos)   — move a deferred step back into the active queue
      • reorder_steps(id, dir) — move a step up or down in the pending queue
      • rerun_step(id)         — re-run a completed or failed step
      • skip_step(id)          — permanently skip a step (won't come back)

    Each step execution saves a DataFrame snapshot keyed by step-id, so
    re-running or going back is safe.

    Usage:
        engine = WorkflowEngine(df, target, task, objectives)
        state = engine.start()          # kicks off planning + iteration 1
        state = engine.run_next_step()  # execute next pending step
        state = engine.run_step_by_id(step_id)  # jump to any step
        state = engine.defer_step(step_id)      # skip now, come back later
        state = engine.reorder_steps(step_id, "up")  # move step earlier
        state = engine.run_full_iteration()     # run all remaining
    """

    # Maximum snapshots to keep (to avoid unbounded memory)
    MAX_SNAPSHOTS = 30

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        objectives: str = "",
        max_iterations: int = 5,
        auto_approve: bool = True,
        enabled_steps: list | None = None,
    ):
        self.df = df.copy()
        self.original_df = df.copy()
        # Snapshot store: step_id → DataFrame copy (before that step ran)
        self._snapshots: dict[str, pd.DataFrame] = {}
        self.state = WorkflowState(
            workflow_id=str(uuid.uuid4())[:12],
            target_column=target_column,
            task_type=task_type,
            objectives=objectives,
            max_iterations=max_iterations,
            auto_approve=auto_approve,
            created_at=time.time(),
            updated_at=time.time(),
            initial_shape={"rows": len(df), "columns": len(df.columns)},
            current_shape={"rows": len(df), "columns": len(df.columns)},
        )
        if enabled_steps:
            self.state.enabled_steps = enabled_steps

        # Initial quality snapshot
        metrics = _quality_metrics(df, target_column)
        self.state.initial_quality_score = metrics.get("quality_score", 0)
        self.state.current_quality_score = metrics.get("quality_score", 0)

        logger.info(
            "WorkflowEngine created: id=%s, target=%s, task=%s, shape=%s, quality=%d",
            self.state.workflow_id, target_column, task_type,
            df.shape, self.state.initial_quality_score,
        )

    # ──────────────────────────────────────────────────────────────
    #  Snapshot helpers (for safe bi-directional navigation)
    # ──────────────────────────────────────────────────────────────

    def _save_snapshot(self, key: str):
        """Save a copy of the current DataFrame under *key*."""
        if len(self._snapshots) >= self.MAX_SNAPSHOTS:
            # Evict oldest snapshot (by insertion order — Python 3.7+ dicts)
            oldest = next(iter(self._snapshots))
            del self._snapshots[oldest]
        self._snapshots[key] = self.df.copy()

    def _restore_snapshot(self, key: str) -> bool:
        """Restore the DataFrame to the snapshot saved under *key*.
        Returns True if restored, False if snapshot not found."""
        snap = self._snapshots.get(key)
        if snap is not None:
            self.df = snap.copy()
            self.state.current_shape = {"rows": len(self.df), "columns": len(self.df.columns)}
            metrics = _quality_metrics(self.df, self.state.target_column)
            self.state.current_quality_score = metrics.get("quality_score", 0)
            return True
        return False

    # ──────────────────────────────────────────────────────────────
    #  Internal helpers for locating steps
    # ──────────────────────────────────────────────────────────────

    def _current_iteration(self) -> dict | None:
        """Return the last (current) iteration dict, or None."""
        if self.state.iterations:
            return self.state.iterations[-1]
        return None

    def _find_step_in_iteration(self, iteration_dict: dict, step_id: str) -> tuple[str, int, dict | None]:
        """Locate a step by id in active or deferred lists.
        Returns (location, index, step_dict) where location is 'steps' or 'deferred_steps'."""
        for loc_key in ("steps", "deferred_steps"):
            for idx, s in enumerate(iteration_dict.get(loc_key, [])):
                if s["id"] == step_id:
                    return loc_key, idx, s
        return "", -1, None

    def _all_active_done(self, iteration_dict: dict) -> bool:
        """Check if every step in the active queue is finished (completed / skipped / failed)."""
        terminal = {StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED}
        return all(s["status"] in terminal for s in iteration_dict.get("steps", []))

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return serialisable workflow state."""
        self.state.updated_at = time.time()
        return self.state.to_dict()

    def get_dataframe(self) -> pd.DataFrame:
        """Return the current working DataFrame."""
        return self.df

    def start(self) -> dict:
        """Plan the first iteration and start execution."""
        self.state.status = WorkflowStatus.PLANNING
        self.state.updated_at = time.time()

        try:
            # Plan iteration 1
            iteration = self._plan_iteration()
            self.state.iterations.append(iteration.to_dict())
            self.state.current_iteration = 1
            self.state.status = WorkflowStatus.RUNNING
            logger.info("Workflow %s: iteration 1 planned with %d steps",
                        self.state.workflow_id, len(iteration.steps))
            return self.get_state()
        except Exception as e:
            logger.error("Workflow start failed: %s", e, exc_info=True)
            self.state.status = WorkflowStatus.FAILED
            self.state.error = str(e)
            return self.get_state()

    def run_next_step(self) -> dict:
        """Execute the next pending step in the current iteration."""
        if self.state.status not in (WorkflowStatus.RUNNING,):
            return self.get_state()

        iteration_dict = self._current_iteration()
        if not iteration_dict:
            return self.get_state()

        steps = iteration_dict["steps"]

        # Find next pending step
        for i, step_dict in enumerate(steps):
            if step_dict["status"] == StepStatus.PENDING:
                steps[i] = self._execute_step(step_dict)
                self.state.updated_at = time.time()

                # Check if all active steps done
                if self._all_active_done(iteration_dict):
                    self._maybe_finish_iteration(iteration_dict)

                return self.get_state()

        # All done — evaluate if not already
        if not iteration_dict.get("completed_at"):
            self._maybe_finish_iteration(iteration_dict)

        return self.get_state()

    def run_step_by_id(self, step_id: str) -> dict:
        """Execute a specific step by its ID, regardless of queue order.

        Works for:
          • pending steps (runs immediately, out of order)
          • deferred steps (recalls and runs)
          • completed/failed steps (re-runs with snapshot restore)
        """
        if self.state.status not in (WorkflowStatus.RUNNING,):
            return self.get_state()

        iteration_dict = self._current_iteration()
        if not iteration_dict:
            return self.get_state()

        loc, idx, step = self._find_step_in_iteration(iteration_dict, step_id)
        if step is None:
            logger.warning("run_step_by_id: step %s not found", step_id)
            return self.get_state()

        # If deferred, move it back into active steps at current execution point
        if loc == "deferred_steps":
            iteration_dict["deferred_steps"].pop(idx)
            # Insert it at the end of the active queue (before evaluation)
            insert_pos = self._find_insert_position(iteration_dict["steps"])
            iteration_dict["steps"].insert(insert_pos, step)
            loc = "steps"
            idx = insert_pos

        # If already completed or failed — re-run: restore snapshot first
        if step["status"] in (StepStatus.COMPLETED, StepStatus.FAILED):
            self._restore_snapshot(step["id"])
            step["status"] = StepStatus.PENDING
            step["error"] = ""
            step["result_summary"] = ""
            step["metrics_before"] = {}
            step["metrics_after"] = {}
            step["started_at"] = 0
            step["finished_at"] = 0
            # Invalidate all steps that came *after* this one in the queue
            self._invalidate_subsequent_steps(iteration_dict["steps"], idx)

        # Reset deferred status if it was deferred
        if step["status"] == StepStatus.DEFERRED:
            step["status"] = StepStatus.PENDING

        # Execute
        result = self._execute_step(step)
        iteration_dict["steps"][idx] = result
        self.state.updated_at = time.time()

        if self._all_active_done(iteration_dict):
            self._maybe_finish_iteration(iteration_dict)

        return self.get_state()

    def defer_step(self, step_id: str) -> dict:
        """Defer a pending step — remove from active queue, park in deferred list.
        The step can be recalled later."""
        iteration_dict = self._current_iteration()
        if not iteration_dict:
            return self.get_state()

        loc, idx, step = self._find_step_in_iteration(iteration_dict, step_id)
        if step is None or loc != "steps":
            return self.get_state()

        if step["status"] not in (StepStatus.PENDING,):
            logger.warning("Cannot defer step %s in status %s", step_id, step["status"])
            return self.get_state()

        # Move from active → deferred
        iteration_dict["steps"].pop(idx)
        step["status"] = StepStatus.DEFERRED
        step["deferred_at"] = time.time()
        iteration_dict.setdefault("deferred_steps", []).append(step)
        logger.info("Step %s deferred", step_id)

        # If all remaining active steps are done, maybe finish
        if self._all_active_done(iteration_dict):
            self._maybe_finish_iteration(iteration_dict)

        self.state.updated_at = time.time()
        return self.get_state()

    def recall_step(self, step_id: str, position: int | None = None) -> dict:
        """Move a deferred step back into the active queue.
        position: index to insert at (None = append before evaluation step)."""
        iteration_dict = self._current_iteration()
        if not iteration_dict:
            return self.get_state()

        loc, idx, step = self._find_step_in_iteration(iteration_dict, step_id)
        if step is None or loc != "deferred_steps":
            return self.get_state()

        # Remove from deferred
        iteration_dict["deferred_steps"].pop(idx)
        step["status"] = StepStatus.PENDING
        step["deferred_at"] = 0

        # Insert into active queue
        insert_pos = self._find_insert_position(iteration_dict["steps"])
        if position is not None and 0 <= position <= len(iteration_dict["steps"]):
            insert_pos = position
        iteration_dict["steps"].insert(insert_pos, step)

        # If iteration was completed/awaiting and we just added a pending step, resume
        if self.state.status in (WorkflowStatus.COMPLETED, WorkflowStatus.AWAITING_APPROVAL):
            self.state.status = WorkflowStatus.RUNNING
            iteration_dict["completed_at"] = 0  # un-complete the iteration

        logger.info("Step %s recalled to position %d", step_id, insert_pos)
        self.state.updated_at = time.time()
        return self.get_state()

    def reorder_steps(self, step_id: str, direction: str) -> dict:
        """Move a pending step up or down in the active queue.
        direction: 'up' (earlier) or 'down' (later)."""
        iteration_dict = self._current_iteration()
        if not iteration_dict:
            return self.get_state()

        steps = iteration_dict["steps"]
        loc, idx, step = self._find_step_in_iteration(iteration_dict, step_id)
        if step is None or loc != "steps":
            return self.get_state()

        # Only allow reordering pending steps
        if step["status"] != StepStatus.PENDING:
            return self.get_state()

        if direction == "up" and idx > 0:
            # Don't move above a completed/running step
            target_idx = idx - 1
            if steps[target_idx]["status"] in (StepStatus.COMPLETED, StepStatus.RUNNING):
                return self.get_state()
            steps[idx], steps[target_idx] = steps[target_idx], steps[idx]
            logger.info("Step %s moved up to position %d", step_id, target_idx)

        elif direction == "down" and idx < len(steps) - 1:
            target_idx = idx + 1
            # Don't swap with evaluation step (always last)
            if steps[target_idx].get("step_type") == StepType.EVALUATION:
                return self.get_state()
            steps[idx], steps[target_idx] = steps[target_idx], steps[idx]
            logger.info("Step %s moved down to position %d", step_id, target_idx)

        self.state.updated_at = time.time()
        return self.get_state()

    def rerun_step(self, step_id: str) -> dict:
        """Re-run a previously completed or failed step.
        Restores the DataFrame snapshot from before that step ran."""
        return self.run_step_by_id(step_id)  # run_step_by_id already handles re-run logic

    def run_full_iteration(self) -> dict:
        """Execute all pending steps in the current iteration (in order).
        Skips deferred steps — they stay in the deferred queue."""
        if self.state.status not in (WorkflowStatus.RUNNING,):
            return self.get_state()

        iteration_dict = self._current_iteration()
        if not iteration_dict:
            return self.get_state()

        steps = iteration_dict["steps"]

        for i, step_dict in enumerate(steps):
            if step_dict["status"] == StepStatus.PENDING:
                steps[i] = self._execute_step(step_dict)
                if steps[i]["status"] == StepStatus.FAILED and not self.state.auto_approve:
                    break  # stop on failure in manual mode

        self._maybe_finish_iteration(iteration_dict)
        self.state.updated_at = time.time()
        return self.get_state()

    def continue_workflow(self) -> dict:
        """
        After an iteration completes, decide whether to do another.
        If auto_approve is True, automatically starts the next iteration
        if the LLM recommends continuing.
        """
        if self.state.status == WorkflowStatus.COMPLETED:
            return self.get_state()

        last_iter = self.state.iterations[-1]
        if last_iter.get("should_continue") and self.state.current_iteration < self.state.max_iterations:
            # Plan and start next iteration
            self.state.status = WorkflowStatus.PLANNING
            try:
                iteration = self._plan_iteration()
                self.state.iterations.append(iteration.to_dict())
                self.state.current_iteration += 1
                self.state.status = WorkflowStatus.RUNNING
                logger.info("Workflow %s: iteration %d planned with %d steps",
                            self.state.workflow_id, self.state.current_iteration,
                            len(iteration.steps))
            except Exception as e:
                logger.error("Workflow continuation failed: %s", e, exc_info=True)
                self.state.status = WorkflowStatus.FAILED
                self.state.error = str(e)
        else:
            self.state.status = WorkflowStatus.COMPLETED
            logger.info("Workflow %s completed after %d iterations",
                        self.state.workflow_id, self.state.current_iteration)

        return self.get_state()

    def run_all(self) -> dict:
        """Run the entire workflow end-to-end (all iterations)."""
        self.start()
        while self.state.status == WorkflowStatus.RUNNING:
            self.run_full_iteration()
            if self.state.status in (WorkflowStatus.FAILED, WorkflowStatus.ABORTED):
                break
            self.continue_workflow()
        return self.get_state()

    def approve_iteration(self) -> dict:
        """User approves the current plan — resume execution."""
        if self.state.status == WorkflowStatus.AWAITING_APPROVAL:
            self.state.status = WorkflowStatus.RUNNING
        return self.get_state()

    def abort(self) -> dict:
        """Abort the workflow."""
        self.state.status = WorkflowStatus.ABORTED
        self.state.error = "Aborted by user"
        logger.info("Workflow %s aborted by user", self.state.workflow_id)
        return self.get_state()

    def skip_step(self, step_id: str) -> dict:
        """Permanently skip a specific pending step (won't come back)."""
        iteration_dict = self._current_iteration()
        if not iteration_dict:
            return self.get_state()

        loc, idx, step = self._find_step_in_iteration(iteration_dict, step_id)
        if step is None:
            return self.get_state()

        if step["status"] in (StepStatus.PENDING, StepStatus.DEFERRED):
            # If in deferred, remove from there
            if loc == "deferred_steps":
                iteration_dict["deferred_steps"].pop(idx)
                iteration_dict["steps"].append(step)  # add back to show as skipped
            step["status"] = StepStatus.SKIPPED
            logger.info("Step %s skipped permanently", step_id)

        # If all active steps are now done, maybe finish
        if self._all_active_done(iteration_dict):
            self._maybe_finish_iteration(iteration_dict)

        self.state.updated_at = time.time()
        return self.get_state()

    # ──────────────────────────────────────────────────────────────
    #  Internal: Iteration finish logic
    # ──────────────────────────────────────────────────────────────

    def _maybe_finish_iteration(self, iteration_dict: dict):
        """Check if iteration can be marked as finished.
        Won't finish if there are still deferred steps (user might recall them)."""
        deferred = iteration_dict.get("deferred_steps", [])
        if deferred and self.state.status == WorkflowStatus.RUNNING:
            # There are deferred steps — don't auto-finish; let the user decide
            logger.info("Iteration has %d deferred step(s) — waiting for user action", len(deferred))
            return

        if not iteration_dict.get("completed_at"):
            iteration_dict["completed_at"] = time.time()
        self._evaluate_iteration(iteration_dict)

    def _find_insert_position(self, steps: list) -> int:
        """Find the best position to insert a step: just before the evaluation step (last),
        or at the end if no evaluation step."""
        for i in reversed(range(len(steps))):
            if steps[i].get("step_type") == StepType.EVALUATION:
                return i
        return len(steps)

    def _invalidate_subsequent_steps(self, steps: list, from_idx: int):
        """When re-running a step, mark all subsequent completed steps as pending
        since their inputs have changed."""
        for i in range(from_idx + 1, len(steps)):
            if steps[i]["status"] in (StepStatus.COMPLETED, StepStatus.FAILED):
                steps[i]["status"] = StepStatus.PENDING
                steps[i]["result_summary"] = ""
                steps[i]["error"] = ""
                steps[i]["metrics_before"] = {}
                steps[i]["metrics_after"] = {}
                steps[i]["started_at"] = 0
                steps[i]["finished_at"] = 0
                logger.info("Step %s invalidated (input changed by re-run)", steps[i]["id"])

    # ──────────────────────────────────────────────────────────────
    #  Planning (LLM-driven)
    # ──────────────────────────────────────────────────────────────

    def _plan_iteration(self) -> WorkflowIteration:
        """Ask the LLM to plan the next iteration of steps."""
        from llm_engine.analyzer import LLMAnalyzer

        iteration = WorkflowIteration(
            iteration_number=self.state.current_iteration + 1,
            planned_at=time.time(),
        )

        snapshot = _df_snapshot(self.df, self.state.target_column)
        quality = _quality_metrics(self.df, self.state.target_column)

        # Build context for the LLM
        history_summary = ""
        if self.state.iterations:
            for prev in self.state.iterations:
                history_summary += f"\n### Iteration {prev.get('iteration_number', '?')}\n"
                history_summary += f"Plan: {prev.get('llm_plan', 'N/A')[:300]}\n"
                history_summary += f"Evaluation: {prev.get('llm_evaluation', 'N/A')[:300]}\n"
                for s in prev.get("steps", []):
                    history_summary += f"  - {s.get('title', '?')}: {s.get('status', '?')} — {s.get('result_summary', '')[:100]}\n"

        plan_response = LLMAnalyzer.plan_workflow_iteration(
            data_snapshot=snapshot,
            quality_metrics=quality,
            target_column=self.state.target_column,
            task_type=self.state.task_type,
            objectives=self.state.objectives,
            iteration_number=iteration.iteration_number,
            max_iterations=self.state.max_iterations,
            enabled_steps=[s.value if isinstance(s, StepType) else s for s in self.state.enabled_steps],
            history_summary=history_summary,
        )

        iteration.llm_plan = plan_response.get("plan_summary", "")

        # Parse LLM response into concrete steps
        steps = self._parse_plan_to_steps(plan_response)
        iteration.steps = [s.to_dict() for s in steps]

        logger.info("Planned iteration %d: %d steps — %s",
                     iteration.iteration_number, len(steps),
                     [s.title for s in steps])
        return iteration

    def _parse_plan_to_steps(self, plan: dict) -> list[WorkflowStep]:
        """Convert LLM plan JSON into executable WorkflowStep objects."""
        steps = []
        raw_steps = plan.get("steps", [])

        for raw in raw_steps:
            step_type = raw.get("step_type", "")
            # Validate step type is enabled
            if step_type not in [s.value if isinstance(s, StepType) else s for s in self.state.enabled_steps]:
                if step_type != StepType.EVALUATION:
                    continue

            step = WorkflowStep(
                id=str(uuid.uuid4())[:8],
                step_type=step_type,
                title=raw.get("title", step_type),
                description=raw.get("description", ""),
                operations=raw.get("operations", []),
                llm_rationale=raw.get("rationale", ""),
            )
            steps.append(step)

        # Always add an evaluation step at the end
        steps.append(WorkflowStep(
            id=str(uuid.uuid4())[:8],
            step_type=StepType.EVALUATION,
            title="Evaluate Results",
            description="LLM evaluates the data state after all operations",
        ))

        return steps

    # ──────────────────────────────────────────────────────────────
    #  Step Execution
    # ──────────────────────────────────────────────────────────────

    def _execute_step(self, step_dict: dict) -> dict:
        """Execute a single workflow step and return updated step dict."""
        # Save a snapshot *before* this step mutates the DataFrame
        self._save_snapshot(step_dict["id"])

        step_dict["status"] = StepStatus.RUNNING
        step_dict["started_at"] = time.time()
        step_dict["metrics_before"] = _quality_metrics(self.df, self.state.target_column)
        step_dict["run_count"] = step_dict.get("run_count", 0) + 1

        step_type = step_dict["step_type"]
        logger.info("Executing step: %s (%s)", step_dict["title"], step_type)

        try:
            if step_type == StepType.DATA_ANALYSIS:
                result = self._exec_data_analysis(step_dict)
            elif step_type == StepType.DATA_CLEANING:
                result = self._exec_data_cleaning(step_dict)
            elif step_type == StepType.FEATURE_ENGINEERING:
                result = self._exec_feature_engineering(step_dict)
            elif step_type == StepType.TRANSFORMATIONS:
                result = self._exec_transformations(step_dict)
            elif step_type == StepType.DIMENSIONALITY_REDUCTION:
                result = self._exec_dimensionality(step_dict)
            elif step_type == StepType.EVALUATION:
                result = self._exec_evaluation(step_dict)
            else:
                result = f"Unknown step type: {step_type}"

            step_dict["status"] = StepStatus.COMPLETED
            step_dict["result_summary"] = str(result)[:500]
            step_dict["metrics_after"] = _quality_metrics(self.df, self.state.target_column)

            # Update global metrics
            self.state.current_quality_score = step_dict["metrics_after"].get("quality_score", 0)
            self.state.current_shape = step_dict["metrics_after"].get("shape", {})

            logger.info("Step completed: %s — quality %d→%d",
                        step_dict["title"],
                        step_dict["metrics_before"].get("quality_score", 0),
                        step_dict["metrics_after"].get("quality_score", 0))

        except Exception as e:
            step_dict["status"] = StepStatus.FAILED
            step_dict["error"] = str(e)
            step_dict["result_summary"] = f"Failed: {e}"
            logger.error("Step failed: %s — %s", step_dict["title"], e, exc_info=True)

        step_dict["finished_at"] = time.time()
        return step_dict

    def _exec_data_analysis(self, step: dict) -> str:
        """Run data quality analysis and return summary."""
        analyzer = DataQualityAnalyzer(self.df, target_column=self.state.target_column)
        report = analyzer.full_report()
        score = report.get("quality_score", 0)
        n_issues = len(report.get("issues", []))
        return f"Quality score: {score}/100, {n_issues} issues found, shape: {self.df.shape}"

    def _exec_data_cleaning(self, step: dict) -> str:
        """Execute cleaning operations from the step plan."""
        operations = step.get("operations", [])
        if not operations:
            return "No cleaning operations specified"

        before_shape = self.df.shape
        self.df, log = DataCleaner.apply_operations(self.df, operations)
        after_shape = self.df.shape

        summary = f"Shape {before_shape}→{after_shape}. " + " | ".join(log)
        return summary

    def _exec_feature_engineering(self, step: dict) -> str:
        """Execute feature engineering operations."""
        operations = step.get("operations", [])
        if not operations:
            return "No feature engineering operations specified"

        before_cols = self.df.shape[1]
        self.df, log = FeatureEngineer.apply_operations(self.df, operations)
        after_cols = self.df.shape[1]

        summary = f"Columns {before_cols}→{after_cols}. " + " | ".join(log)
        return summary

    def _exec_transformations(self, step: dict) -> str:
        """Execute transformation operations."""
        operations = step.get("operations", [])
        if not operations:
            return "No transformation operations specified"

        before_shape = self.df.shape
        self.df, log = DataTransformer.apply_operations(self.df, operations)
        after_shape = self.df.shape

        summary = f"Shape {before_shape}→{after_shape}. " + " | ".join(log)
        return summary

    def _exec_dimensionality(self, step: dict) -> str:
        """Execute dimensionality reduction."""
        operations = step.get("operations", [])
        if not operations:
            return "No dimensionality operations specified"

        results = []
        for op in operations:
            method = op.get("method", "pca")
            params = op.get("params", {})
            before_cols = self.df.shape[1]

            if method == "pca":
                self.df, info = DimensionalityReducer.pca_reduce(self.df, **params)
            elif method == "variance_threshold":
                self.df, info = DimensionalityReducer.variance_threshold(self.df, **params)
            elif method == "correlation_filter":
                self.df, info = DimensionalityReducer.correlation_filter(self.df, **params)
            elif method == "feature_importance":
                target = params.pop("target", self.state.target_column)
                task = params.pop("task", self.state.task_type)
                self.df, info = DimensionalityReducer.feature_importance_selection(
                    self.df, target, task, **params
                )
            else:
                results.append(f"Unknown method: {method}")
                continue

            results.append(f"{method}: cols {before_cols}→{self.df.shape[1]}")

        return " | ".join(results)

    def _exec_evaluation(self, step: dict) -> str:
        """LLM evaluates the current data state."""
        from llm_engine.analyzer import LLMAnalyzer

        snapshot = _df_snapshot(self.df, self.state.target_column)
        quality = _quality_metrics(self.df, self.state.target_column)

        evaluation = LLMAnalyzer.evaluate_workflow_step(
            data_snapshot=snapshot,
            quality_metrics=quality,
            target_column=self.state.target_column,
            task_type=self.state.task_type,
            objectives=self.state.objectives,
        )
        return evaluation

    # ──────────────────────────────────────────────────────────────
    #  Iteration Evaluation
    # ──────────────────────────────────────────────────────────────

    def _evaluate_iteration(self, iteration_dict: dict):
        """Ask the LLM whether another iteration is needed."""
        from llm_engine.analyzer import LLMAnalyzer

        snapshot = _df_snapshot(self.df, self.state.target_column)
        quality = _quality_metrics(self.df, self.state.target_column)

        # Gather step results (active + deferred)
        step_summaries = []
        for s in iteration_dict.get("steps", []):
            step_summaries.append({
                "type": s.get("step_type"),
                "title": s.get("title"),
                "status": s.get("status"),
                "result": s.get("result_summary", "")[:200],
                "quality_before": s.get("metrics_before", {}).get("quality_score", "?"),
                "quality_after": s.get("metrics_after", {}).get("quality_score", "?"),
            })
        for s in iteration_dict.get("deferred_steps", []):
            step_summaries.append({
                "type": s.get("step_type"),
                "title": s.get("title"),
                "status": "deferred",
                "result": "Step was deferred by user",
                "quality_before": "?",
                "quality_after": "?",
            })

        eval_result = LLMAnalyzer.evaluate_workflow_iteration(
            data_snapshot=snapshot,
            quality_metrics=quality,
            target_column=self.state.target_column,
            task_type=self.state.task_type,
            objectives=self.state.objectives,
            iteration_number=iteration_dict.get("iteration_number", 1),
            max_iterations=self.state.max_iterations,
            step_summaries=step_summaries,
            initial_quality=self.state.initial_quality_score,
        )

        iteration_dict["llm_evaluation"] = eval_result.get("evaluation", "")
        iteration_dict["should_continue"] = eval_result.get("should_continue", False)
        iteration_dict["improvement_summary"] = eval_result.get("improvement_summary", "")

        logger.info(
            "Iteration %d evaluation: should_continue=%s, quality=%d→%d",
            iteration_dict.get("iteration_number", 0),
            eval_result.get("should_continue"),
            self.state.initial_quality_score,
            quality.get("quality_score", 0),
        )

        # If not continuing or reached max, mark completed
        if not eval_result.get("should_continue") or self.state.current_iteration >= self.state.max_iterations:
            if self.state.auto_approve:
                self.state.status = WorkflowStatus.COMPLETED
            else:
                self.state.status = WorkflowStatus.AWAITING_APPROVAL
        elif not self.state.auto_approve:
            self.state.status = WorkflowStatus.AWAITING_APPROVAL
