"""
AceML Studio – Model Training
================================
Train classification & regression models.  Supports train/val/test split,
cross-validation, and multiple algorithm families.
"""

import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from config import Config
from .cloud_gpu import get_cloud_gpu_manager

logger = logging.getLogger("aceml.model_training")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
    logger.debug("XGBoost available")
except ImportError:
    HAS_XGB = False
    logger.info("XGBoost not installed — XGB models disabled")


# ------------------------------------------------------------------ #
#  Model Registry
# ------------------------------------------------------------------ #
CLF_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "decision_tree_clf": DecisionTreeClassifier,
    "random_forest_clf": RandomForestClassifier,
    "gradient_boosting_clf": GradientBoostingClassifier,
    "svm_clf": SVC,
    "knn_clf": KNeighborsClassifier,
    "mlp_clf": MLPClassifier,
}

REG_REGISTRY: dict[str, type] = {
    "linear_regression": LinearRegression,
    "decision_tree_reg": DecisionTreeRegressor,
    "random_forest_reg": RandomForestRegressor,
    "gradient_boosting_reg": GradientBoostingRegressor,
    "svr_reg": SVR,
    "knn_reg": KNeighborsRegressor,
    "mlp_reg": MLPRegressor,
}

if HAS_XGB:
    CLF_REGISTRY["xgboost_clf"] = XGBClassifier
    REG_REGISTRY["xgboost_reg"] = XGBRegressor


def _default_params(key: str) -> dict:
    """Sensible defaults for quick training."""
    defaults = {
        "logistic_regression": {"max_iter": 1000, "random_state": Config.DEFAULT_RANDOM_STATE},
        "decision_tree_clf": {"random_state": Config.DEFAULT_RANDOM_STATE, "max_depth": 10},
        "random_forest_clf": {"n_estimators": 100, "random_state": Config.DEFAULT_RANDOM_STATE, "n_jobs": -1},
        "gradient_boosting_clf": {"n_estimators": 100, "random_state": Config.DEFAULT_RANDOM_STATE},
        "xgboost_clf": {"n_estimators": 100, "random_state": Config.DEFAULT_RANDOM_STATE, "use_label_encoder": False, "eval_metric": "logloss"},
        "svm_clf": {"probability": True, "random_state": Config.DEFAULT_RANDOM_STATE},
        "knn_clf": {"n_neighbors": 5},
        "mlp_clf": {"max_iter": 500, "random_state": Config.DEFAULT_RANDOM_STATE},
        "linear_regression": {},
        "decision_tree_reg": {"random_state": Config.DEFAULT_RANDOM_STATE, "max_depth": 10},
        "random_forest_reg": {"n_estimators": 100, "random_state": Config.DEFAULT_RANDOM_STATE, "n_jobs": -1},
        "gradient_boosting_reg": {"n_estimators": 100, "random_state": Config.DEFAULT_RANDOM_STATE},
        "xgboost_reg": {"n_estimators": 100, "random_state": Config.DEFAULT_RANDOM_STATE},
        "svr_reg": {},
        "knn_reg": {"n_neighbors": 5},
        "mlp_reg": {"max_iter": 500, "random_state": Config.DEFAULT_RANDOM_STATE},
    }
    return defaults.get(key, {})


class ModelTrainer:
    """Train ML models with split and cross-validation support."""

    @staticmethod
    def get_available_models(task: str) -> dict:
        if task == "classification":
            return Config.CLASSIFICATION_MODELS
        elif task == "regression":
            return Config.REGRESSION_MODELS
        elif task == "unsupervised":
            return Config.UNSUPERVISED_MODELS
        return Config.REGRESSION_MODELS

    # ------------------------------------------------------------------ #
    #  Split helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def split_data(df: pd.DataFrame, target: str,
                   test_size: float | None = None,
                   val_size: float | None = None,
                   random_state: int | None = None):
        ts = test_size or Config.DEFAULT_TEST_SIZE
        vs = val_size or Config.DEFAULT_VALIDATION_SIZE
        rs = random_state or Config.DEFAULT_RANDOM_STATE

        X = df.drop(columns=[target])
        y = df[target]

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
        relative_val = vs / (1 - ts)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=relative_val, random_state=rs)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "split_info": {
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
            }
        }

    # ------------------------------------------------------------------ #
    #  Train single model
    # ------------------------------------------------------------------ #
    @staticmethod
    def train(model_key: str, task: str, X_train, y_train,
              hyperparams: dict | None = None,
              use_cloud_gpu: bool | None = None) -> tuple[object, dict]:
        registry = CLF_REGISTRY if task == "classification" else REG_REGISTRY
        model_cls = registry.get(model_key)
        if model_cls is None:
            logger.error("Unknown model key: %s (task=%s)", model_key, task)
            raise ValueError(f"Unknown model: {model_key}")

        params = _default_params(model_key)
        if hyperparams:
            params.update(hyperparams)

        # Check if cloud GPU should be used
        use_gpu = use_cloud_gpu if use_cloud_gpu is not None else Config.CLOUD_GPU_ENABLED
        
        if use_gpu:
            try:
                logger.info("Attempting cloud GPU training for '%s' (%s)", model_key, task)
                gpu_manager = get_cloud_gpu_manager()
                model, gpu_info = gpu_manager.execute_training(model_cls, X_train, y_train, params)
                
                info = {
                    "model_key": model_key,
                    "task": task,
                    "hyperparams": params,
                    "training_time_sec": gpu_info.get("training_time_sec", 0),
                    "n_train_samples": len(X_train),
                    "n_features": X_train.shape[1],
                    "execution_mode": "cloud_gpu",
                    "cloud_provider": gpu_info.get("provider"),
                    "cloud_details": gpu_info
                }
                return model, info
                
            except Exception as e:
                logger.error("Cloud GPU training failed: %s", e, exc_info=True)
                if Config.GPU_FALLBACK_TO_LOCAL:
                    logger.warning("Falling back to local training")
                else:
                    raise

        # Local training (default or fallback)
        logger.info("Training '%s' (%s) on %d samples x %d features (LOCAL)",
                    model_key, task, len(X_train), X_train.shape[1])
        logger.debug("Hyperparams: %s", params)
        model = model_cls(**params)
        start = time.time()
        model.fit(X_train, y_train)
        duration = round(time.time() - start, 2)
        logger.info("Model '%s' trained in %.2fs", model_key, duration)

        info = {
            "model_key": model_key,
            "task": task,
            "hyperparams": params,
            "training_time_sec": duration,
            "n_train_samples": len(X_train),
            "n_features": X_train.shape[1],
            "execution_mode": "local"
        }
        return model, info

    # ------------------------------------------------------------------ #
    #  Cross Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def cross_validate(model_key: str, task: str, X, y,
                       cv: int | None = None,
                       hyperparams: dict | None = None) -> dict:
        registry = CLF_REGISTRY if task == "classification" else REG_REGISTRY
        model_cls = registry.get(model_key)
        if model_cls is None:
            raise ValueError(f"Unknown model: {model_key}")

        params = _default_params(model_key)
        if hyperparams:
            params.update(hyperparams)
        model = model_cls(**params)

        folds = cv or Config.DEFAULT_CV_FOLDS
        scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"

        results = cross_validate(model, X, y, cv=folds, scoring=scoring,
                                 return_train_score=True, n_jobs=-1)
        return {
            "cv_folds": folds,
            "scoring": scoring,
            "test_scores": results["test_score"].tolist(),
            "train_scores": results["train_score"].tolist(),
            "mean_test_score": round(float(results["test_score"].mean()), 4),
            "std_test_score": round(float(results["test_score"].std()), 4),
            "mean_train_score": round(float(results["train_score"].mean()), 4),
            "fit_times": results["fit_time"].tolist(),
        }

    # ------------------------------------------------------------------ #
    #  Train multiple models for comparison
    # ------------------------------------------------------------------ #
    @classmethod
    def train_multiple(cls, model_keys: list[str], task: str,
                       X_train, y_train, X_val, y_val,
                       use_cloud_gpu: bool | None = None) -> list[dict]:
        logger.info("Training %d models for comparison (task=%s)", len(model_keys), task)
        results = []
        for key in model_keys:
            try:
                model, info = cls.train(key, task, X_train, y_train, use_cloud_gpu=use_cloud_gpu)
                train_score = round(float(model.score(X_train, y_train)), 4)  # type: ignore
                val_score = round(float(model.score(X_val, y_val)), 4)  # type: ignore
                logger.info("  %s: train=%.4f, val=%.4f (%.2fs) [%s]",
                            key, train_score, val_score, info['training_time_sec'],
                            info.get('execution_mode', 'local'))
                results.append({
                    **info,
                    "train_score": train_score,
                    "val_score": val_score,
                    "model": model,
                    "status": "success",
                })
            except Exception as e:
                logger.error("Failed to train '%s': %s", key, e, exc_info=True)
                results.append({
                    "model_key": key,
                    "status": "error",
                    "error": str(e),
                })
        return results
