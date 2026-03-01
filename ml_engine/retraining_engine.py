"""
AceML Studio – Retraining Engine
==================================
Manage and execute automated ML model retraining workflows triggered by
scheduled DB extractions or on-demand runs.

Full workflow
-------------
  1. Load freshly extracted CSV data
  2. (Optional) Data cleaning via DataCleaner
  3. (Optional) Feature engineering via FeatureEngineer
  4. Preprocessing – null imputation, categorical encoding, column selection
  5. Train / validation / test split  (supervised tasks)
  6. Train model via ModelTrainer
  7. Evaluate with ModelEvaluator
  8. Compare with current production model (optional)
  9. Register new version in ModelRegistry
 10. Auto-promote if configured and performance improves
 11. Track run as an experiment via ExperimentTracker
"""

import json
import logging
import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config
from .data_loader import DataLoader
from .data_cleaning import DataCleaner
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator
from .model_registry import ModelRegistry
from .experiment_tracker import ExperimentTracker

logger = logging.getLogger("aceml.retraining_engine")

# ── Storage paths ──────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(__file__))
RETRAIN_JOBS_DIR = os.path.join(_BASE, "retrain_jobs")
RETRAIN_HISTORY_DIR = os.path.join(_BASE, "retrain_history")


# ═══════════════════════════════════════════════════════════════════════════════
#  Retraining Job Manager  – CRUD + persistence
# ═══════════════════════════════════════════════════════════════════════════════

class RetrainingJobManager:
    """
    Create, update, delete and persist retraining job definitions.

    Each job stores:
    • Which DB extraction query to use for fresh data
    • Which registered model to retrain (name + algorithm key)
    • Target column, optional feature-column list
    • Cleaning / feature-engineering flags
    • Auto-promotion config (metric, threshold)
    • Schedule configuration
    """

    def __init__(self):
        os.makedirs(RETRAIN_JOBS_DIR, exist_ok=True)
        os.makedirs(RETRAIN_HISTORY_DIR, exist_ok=True)
        self._jobs: Dict[str, Dict] = {}
        self._load_all()

    # ── Persistence helpers ────────────────────────────────────────────────────

    def _load_all(self):
        """Load all persisted job JSON files from disk."""
        for fname in os.listdir(RETRAIN_JOBS_DIR):
            if fname.endswith(".json"):
                fpath = os.path.join(RETRAIN_JOBS_DIR, fname)
                job_id = fname[:-5]
                try:
                    with open(fpath) as f:
                        self._jobs[job_id] = json.load(f)
                except Exception as exc:
                    logger.warning("Could not load retrain job %s: %s", fname, exc)

    def _persist(self, job_id: str):
        """Write a single job to disk."""
        fpath = os.path.join(RETRAIN_JOBS_DIR, f"{job_id}.json")
        with open(fpath, "w") as f:
            json.dump(self._jobs[job_id], f, indent=2, default=str)

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def create_job(
        self,
        name: str,
        query_id: str,
        model_name: str,
        model_key: str,
        task: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        apply_cleaning: bool = True,
        apply_feature_engineering: bool = False,
        compare_with_production: bool = True,
        auto_promote: bool = False,
        promotion_metric: str = "val_score",
        promotion_threshold: float = 0.0,
        hyperparams: Optional[Dict[str, Any]] = None,
        schedule_enabled: bool = False,
        schedule_interval_minutes: int = 60,
        schedule_frequency_unit: str = "minutes",
        schedule_frequency_value: int = 60,
        schedule_start_date: Optional[str] = None,
        schedule_end_date: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Dict:
        """
        Create and persist a new retraining job.

        Args:
            name:                        Human-readable job name
            query_id:                    Extraction query whose CSV triggers this job
            model_name:                  Registry model name (e.g. 'fraud_detector')
            model_key:                   Algorithm key (e.g. 'random_forest_clf')
            task:                        'classification' | 'regression' | 'unsupervised'
            target_column:               Name of the target/label column
            feature_columns:             Explicit feature list (empty = use all non-target)
            apply_cleaning:              Run DataCleaner before training
            apply_feature_engineering:   Run FeatureEngineer before training
            compare_with_production:     Compare new model with current production
            auto_promote:                Automatically promote if improvement > threshold
            promotion_metric:            Metric key used for comparison
            promotion_threshold:         Minimum delta to trigger auto-promotion
            hyperparams:                 Override default hyperparameters
            schedule_enabled:            Whether to trigger this job on the query schedule
            schedule_interval_minutes:   Legacy alias – kept for backward compat (prefer frequency fields)
            schedule_frequency_unit:     Unit for the interval: 'minutes'|'hours'|'days'|'weeks'
            schedule_frequency_value:    Numeric interval value (e.g. 2 hours, 1 day)
            schedule_start_date:         ISO datetime string – schedule active from this point
            schedule_end_date:           ISO datetime string – schedule expires after this point
            job_id:                      Optional explicit ID (generated if None)

        Returns:
            Full job dict
        """
        jid = job_id or str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        job: Dict[str, Any] = {
            "job_id": jid,
            "name": name,
            "query_id": query_id,
            "model_name": model_name,
            "model_key": model_key,
            "task": task,
            "target_column": target_column,
            "feature_columns": feature_columns or [],
            "apply_cleaning": apply_cleaning,
            "apply_feature_engineering": apply_feature_engineering,
            "compare_with_production": compare_with_production,
            "auto_promote": auto_promote,
            "promotion_metric": promotion_metric,
            "promotion_threshold": promotion_threshold,
            "hyperparams": hyperparams or {},
            "schedule_enabled": schedule_enabled,
            "schedule_interval_minutes": schedule_interval_minutes,
            "schedule_frequency_unit": schedule_frequency_unit,
            "schedule_frequency_value": schedule_frequency_value,
            "schedule_start_date": schedule_start_date,
            "schedule_end_date": schedule_end_date,
            "created_at": now,
            "updated_at": now,
            "last_run_at": None,
            "last_run_status": None,
            "last_run_id": None,
            "run_count": 0,
        }
        self._jobs[jid] = job
        self._persist(jid)
        logger.info("Created retraining job: '%s' (%s)", name, jid)
        return job

    def update_job(self, job_id: str, updates: Dict) -> Optional[Dict]:
        """Update mutable fields of an existing job."""
        if job_id not in self._jobs:
            return None
        immutable = {"job_id", "created_at"}
        for k, v in updates.items():
            if k not in immutable:
                self._jobs[job_id][k] = v
        self._jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._persist(job_id)
        return self._jobs[job_id]

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[Dict]:
        return sorted(self._jobs.values(), key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_job(self, job_id: str) -> bool:
        if job_id not in self._jobs:
            return False
        del self._jobs[job_id]
        fpath = os.path.join(RETRAIN_JOBS_DIR, f"{job_id}.json")
        if os.path.exists(fpath):
            os.remove(fpath)
        logger.info("Deleted retraining job: %s", job_id)
        return True

    def list_jobs_for_query(self, query_id: str) -> List[Dict]:
        """Return all retraining jobs that are linked to a specific extraction query."""
        return [j for j in self._jobs.values() if j.get("query_id") == query_id]

    def _update_run_stats(self, job_id: str, run_id: str, status: str):
        """Called by pipeline after each run to update last-run metadata."""
        if job_id in self._jobs:
            self._jobs[job_id]["last_run_at"] = datetime.now(timezone.utc).isoformat()
            self._jobs[job_id]["last_run_status"] = status
            self._jobs[job_id]["last_run_id"] = run_id
            self._jobs[job_id]["run_count"] = self._jobs[job_id].get("run_count", 0) + 1
            self._persist(job_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  Retraining Pipeline  – end-to-end execution
# ═══════════════════════════════════════════════════════════════════════════════

class RetrainingPipeline:
    """
    Execute a complete retraining workflow given a job definition and a CSV file.

    Steps
    -----
    1. load_data           – Load the CSV produced by the extraction job
    2. data_cleaning       – (optional) Auto-clean via DataCleaner
    3. feature_engineering – (optional) Polynomial / interaction features
    4. preprocessing       – Null imputation, categorical encoding, column selection
    5. split_data          – Train / val / test split (supervised) or full X (unsupervised)
    6. train_model         – Train via ModelTrainer
    7. evaluate            – Score with ModelEvaluator
    8. register_model      – Persist new version in ModelRegistry
    9. compare_promote     – Compare vs production, auto-promote if configured
   10. track_experiment    – Save run to ExperimentTracker
    """

    def __init__(self):
        self.job_manager = RetrainingJobManager()
        self.registry = ModelRegistry()
        self.tracker = ExperimentTracker()

    # ── Public entry-point ─────────────────────────────────────────────────────

    def run(self, job_id: str, data_file_path: str) -> Dict:
        """
        Execute the full retraining pipeline synchronously.

        Args:
            job_id:         Retraining job ID
            data_file_path: Absolute path to the CSV with fresh data

        Returns:
            Run result dict containing status, metrics, new version, etc.
        """
        run_id = str(uuid.uuid4())[:8]
        start_time = datetime.now(timezone.utc)

        job = self.job_manager.get_job(job_id)
        if not job:
            return {"status": "error", "message": f"Retraining job not found: {job_id}"}

        logger.info(
            "Retraining run %s started – job '%s' (%s), data: %s",
            run_id, job["name"], job_id, data_file_path,
        )

        run_result: Dict[str, Any] = {
            "run_id": run_id,
            "job_id": job_id,
            "job_name": job["name"],
            "started_at": start_time.isoformat(),
            "data_file": data_file_path,
            "status": "running",
            "steps": [],
            "metrics": {},
            "new_model_version": None,
            "new_model_id": None,
            "promoted": False,
            "comparison": {},
            "experiment_id": None,
            "error": None,
        }

        try:
            # ── Step 1 ─ Load Data ─────────────────────────────────────────────
            self._step_start(run_result, "load_data")
            loader = DataLoader()
            df = loader.load(data_file_path)

            if df is None or df.empty:
                raise ValueError("Extracted data is empty – nothing to train on")

            target_col = job["target_column"]
            if target_col not in df.columns:
                raise ValueError(
                    f"Target column '{target_col}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

            self._step_done(run_result, rows=len(df), columns=len(df.columns))
            logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

            # ── Step 2 ─ Data Cleaning (optional) ─────────────────────────────
            if job.get("apply_cleaning", True):
                self._step_start(run_result, "data_cleaning")
                try:
                    df = DataCleaner.drop_duplicates(df)
                    df = DataCleaner.drop_missing(df, how="all")  # drop all-null rows
                    df = DataCleaner.impute(df, strategy="median")  # median-fill numerics
                    self._step_done(run_result, rows_after=len(df))
                    logger.info("Cleaning done – %d rows remaining", len(df))
                except Exception as exc:
                    logger.warning("DataCleaner failed (skipping): %s", exc)
                    self._step_skip(run_result, warning=str(exc))

            # ── Step 3 ─ Feature Engineering (optional) ───────────────────────
            if job.get("apply_feature_engineering", False):
                self._step_start(run_result, "feature_engineering")
                try:
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    num_cols = [c for c in num_cols if c != target_col]
                    if num_cols:
                        df = FeatureEngineer.create_polynomial(df, num_cols, degree=2)
                    self._step_done(run_result, columns_after=len(df.columns))
                except Exception as exc:
                    logger.warning("FeatureEngineer failed (skipping): %s", exc)
                    self._step_skip(run_result, warning=str(exc))

            # ── Step 4 ─ Preprocessing ─────────────────────────────────────────
            self._step_start(run_result, "preprocessing")
            df = self._preprocess(df, target_col, job.get("feature_columns") or [])
            if len(df) < 10:
                raise ValueError(f"Only {len(df)} rows remain after preprocessing – insufficient for training")
            self._step_done(run_result, shape=list(df.shape))

            # ── Step 5 ─ Split Data ────────────────────────────────────────────
            self._step_start(run_result, "split_data")
            task = job["task"]
            feature_cols = job.get("feature_columns") or [c for c in df.columns if c != target_col]
            feature_cols = [c for c in feature_cols if c in df.columns and c != target_col]

            if task in ("classification", "regression"):
                splits = ModelTrainer.split_data(df[feature_cols + [target_col]], target_col)
                X_train = splits["X_train"]
                X_val   = splits["X_val"]
                X_test  = splits["X_test"]
                y_train = splits["y_train"]
                y_val   = splits["y_val"]
                y_test  = splits["y_test"]
                self._step_done(run_result, split_info=splits.get("split_info", {}))
            else:
                # Unsupervised – no labels required
                X_train = df[feature_cols]
                X_val = X_test = y_train = y_val = y_test = None
                self._step_done(run_result, rows=len(df))

            feature_names = list(X_train.columns)

            # ── Step 6 ─ Train Model ───────────────────────────────────────────
            self._step_start(run_result, "train_model")
            model_key = job["model_key"]
            hyperparams = job.get("hyperparams") or {}

            model, train_info = ModelTrainer.train(
                model_key=model_key,
                task=task,
                X_train=X_train,
                y_train=y_train,
                hyperparams=hyperparams if hyperparams else None,
            )
            self._step_done(
                run_result,
                training_time_sec=train_info.get("training_time_sec"),
                execution_mode=train_info.get("execution_mode"),
            )
            logger.info("Trained '%s' in %.2fs", model_key, train_info.get("training_time_sec", 0))

            # ── Step 7 ─ Evaluate ──────────────────────────────────────────────
            self._step_start(run_result, "evaluate")
            metrics: Dict[str, Any] = {}

            if task != "unsupervised":
                eval_result = ModelEvaluator.evaluate(model, X_test, y_test, task, feature_names)
                metrics = dict(eval_result.get("metrics", {}))
                metrics["val_score"]   = round(float(model.score(X_val, y_val)), 4)   # type: ignore[union-attr]
                metrics["train_score"] = round(float(model.score(X_train, y_train)), 4)  # type: ignore[union-attr]
            else:
                try:
                    from sklearn.metrics import silhouette_score, davies_bouldin_score  # noqa: PLC0415
                    if hasattr(model, "labels_"):
                        labels = model.labels_  # type: ignore[union-attr]
                    elif hasattr(model, "predict"):
                        labels = model.predict(X_train)  # type: ignore[union-attr]
                    else:
                        labels = None
                    if labels is not None and len(set(labels)) > 1:
                        metrics["silhouette_score"]   = round(float(silhouette_score(X_train, labels)), 4)
                        metrics["davies_bouldin_score"] = round(float(davies_bouldin_score(X_train, labels)), 4)
                        metrics["n_clusters"] = len(set(labels)) - (1 if -1 in labels else 0)
                except Exception as exc:
                    logger.warning("Could not compute unsupervised metrics: %s", exc)

            run_result["metrics"] = metrics
            self._step_done(run_result, metrics_preview={
                k: v for k, v in list(metrics.items())[:6] if isinstance(v, (int, float))
            })

            # ── Step 8 ─ Register New Model Version ───────────────────────────
            self._step_start(run_result, "register_model")
            model_name = job["model_name"]

            reg_result = self.registry.register_model(
                name=model_name,
                model=model,
                model_type=model_key,
                task=task,
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                hyperparameters=train_info.get("hyperparams", {}),
                metadata={
                    "rows": len(X_train),
                    "features": len(feature_names),
                    "feature_names": feature_names,
                    "target_column": target_col,
                    "source_query": job.get("query_id"),
                    "retrain_job_id": job_id,
                    "retrain_run_id": run_id,
                },
                description=f"Auto-retrained by job '{job['name']}' (run {run_id})",
            )

            new_model_id = reg_result.get("model_id")
            new_version  = reg_result.get("version", "?")
            run_result["new_model_version"] = new_version
            run_result["new_model_id"]      = new_model_id
            self._step_done(run_result, model_id=new_model_id, version=new_version)
            logger.info("Registered %s v%s (id=%s)", model_name, new_version, new_model_id)

            # ── Step 9 ─ Compare & Auto-Promote ───────────────────────────────
            if job.get("compare_with_production", True):
                self._step_start(run_result, "compare_promote")
                should_promote = False
                comparison: Dict[str, Any] = {}

                prod_models = self.registry.list_models(name=model_name, status="production")

                if prod_models:
                    prod = prod_models[0]
                    prod_metrics = prod.get("metrics", {})
                    promo_metric = job.get("promotion_metric", "val_score")

                    new_val  = metrics.get(promo_metric)
                    prod_val = prod_metrics.get(promo_metric)

                    if new_val is not None and prod_val is not None:
                        improvement = float(new_val) - float(prod_val)
                        threshold   = float(job.get("promotion_threshold", 0.0))
                        should_promote = improvement > threshold
                        comparison = {
                            "metric":       promo_metric,
                            "new_value":    new_val,
                            "prod_value":   prod_val,
                            "improvement":  round(improvement, 6),
                            "threshold":    threshold,
                            "should_promote": should_promote,
                        }
                        logger.info(
                            "Compare %s: new=%.4f  prod=%.4f  delta=%.4f  threshold=%.4f  promote=%s",
                            promo_metric, new_val, prod_val, improvement, threshold, should_promote,
                        )
                    else:
                        comparison = {
                            "note": f"Metric '{promo_metric}' not available for comparison",
                            "new_metrics": metrics,
                            "prod_metrics": prod_metrics,
                        }
                else:
                    # No production model exists → promote by default
                    should_promote = True
                    comparison = {"note": "No existing production model – promoting automatically"}

                if should_promote and job.get("auto_promote", False) and new_model_id:
                    self.registry.promote_model(
                        new_model_id,
                        "production",
                        notes=f"Auto-promoted by retraining job '{job['name']}' run {run_id}",
                    )
                    run_result["promoted"] = True
                    logger.info("Auto-promoted %s v%s to production", model_name, new_version)

                run_result["comparison"] = comparison
                self._step_done(
                    run_result,
                    comparison=comparison,
                    promoted=run_result["promoted"],
                )

            # ── Step 10 ─ Track Experiment ─────────────────────────────────────
            non_dict_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            exp = self.tracker.save_experiment(
                name=f"[retrain] {job['name']} v{new_version}",
                task=task,
                model_key=model_key,
                hyperparams=train_info.get("hyperparams", {}),
                metrics=non_dict_metrics,
                data_info={
                    "rows":       len(df),
                    "target":     target_col,
                    "source":     f"db_retrain_job:{job_id}",
                    "run_id":     run_id,
                    "query_id":   job.get("query_id"),
                },
                notes=f"Automated retraining. Job: {job['name']} (id={job_id})",
            )
            run_result["experiment_id"] = exp.get("id")

            # ── Finalise ───────────────────────────────────────────────────────
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            run_result["status"]          = "success"
            run_result["elapsed_seconds"] = round(elapsed, 2)
            run_result["completed_at"]    = datetime.now(timezone.utc).isoformat()

            logger.info(
                "Retraining run %s completed in %.2fs – %s v%s%s",
                run_id, elapsed, model_name, new_version,
                " (PROMOTED)" if run_result["promoted"] else "",
            )

        except Exception as exc:
            run_result["status"]       = "error"
            run_result["error"]        = str(exc)
            run_result["traceback"]    = traceback.format_exc()
            run_result["completed_at"] = datetime.now(timezone.utc).isoformat()
            # Mark any still-running step as failed
            for step in run_result["steps"]:
                if step.get("status") == "running":
                    step["status"] = "error"
                    step["error"]  = str(exc)
            logger.error("Retraining run %s FAILED: %s", run_id, exc, exc_info=True)

        # ── Persist & update job stats ─────────────────────────────────────────
        self._save_run_history(run_result)
        self.job_manager._update_run_stats(job_id, run_id, run_result["status"])

        return run_result

    # ── Preprocessing helper ───────────────────────────────────────────────────

    def _preprocess(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """
        Inline preprocessing:
        - Filter to requested feature columns (or keep all non-target)
        - Drop rows where target is null
        - Median-fill numeric nulls
        - Label-encode categorical columns
        - Zero-fill remaining nulls
        """
        df = df.copy()

        # Drop rows with missing target
        df = df.dropna(subset=[target_col])

        # Select columns
        if feature_cols:
            keep = list(dict.fromkeys(
                [c for c in feature_cols if c in df.columns] + [target_col]
            ))
            df = df[keep]
        else:
            df = df  # keep all columns

        # Impute numeric nulls with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != target_col:
                df[col] = df[col].fillna(df[col].median())

        # Label-encode object / category columns (skip target)
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col != target_col:
                df[col] = df[col].astype("category").cat.codes

        # Fill any residual nulls
        df = df.fillna(0)

        return df

    # ── Run history helpers ────────────────────────────────────────────────────

    @staticmethod
    def _step_start(run_result: Dict, step_name: str):
        run_result["steps"].append({"step": step_name, "status": "running"})

    @staticmethod
    def _step_done(run_result: Dict, **extra):
        run_result["steps"][-1]["status"] = "done"
        run_result["steps"][-1].update(extra)

    @staticmethod
    def _step_skip(run_result: Dict, **extra):
        run_result["steps"][-1]["status"] = "skipped"
        run_result["steps"][-1].update(extra)

    def _save_run_history(self, run_result: Dict):
        run_id = run_result.get("run_id", str(uuid.uuid4())[:8])
        fpath = os.path.join(RETRAIN_HISTORY_DIR, f"{run_id}.json")
        try:
            with open(fpath, "w") as f:
                json.dump(run_result, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not persist run history file: %s", exc)

    # ── History queries ────────────────────────────────────────────────────────

    def get_run_history(
        self,
        job_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Return recent run records, optionally filtered by job_id."""
        runs: List[Dict] = []
        for fname in os.listdir(RETRAIN_HISTORY_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(RETRAIN_HISTORY_DIR, fname)
            try:
                with open(fpath) as f:
                    run = json.load(f)
                if job_id is None or run.get("job_id") == job_id:
                    runs.append(run)
            except Exception:
                pass
        runs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return runs[:limit]

    def get_run(self, run_id: str) -> Optional[Dict]:
        fpath = os.path.join(RETRAIN_HISTORY_DIR, f"{run_id}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                return json.load(f)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Singleton accessors
# ═══════════════════════════════════════════════════════════════════════════════

_pipeline_instance: Optional[RetrainingPipeline] = None
_job_manager_instance: Optional[RetrainingJobManager] = None


def get_retraining_pipeline() -> RetrainingPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RetrainingPipeline()
    return _pipeline_instance


def get_retraining_job_manager() -> RetrainingJobManager:
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = RetrainingJobManager()
    return _job_manager_instance
