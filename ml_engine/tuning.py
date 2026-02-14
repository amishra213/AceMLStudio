"""
AceML Studio – Hyperparameter Tuning
======================================
Grid search, random search, and Optuna-based tuning.
"""

import time
import logging
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from config import Config
from .model_training import CLF_REGISTRY, REG_REGISTRY, _default_params
from .cloud_gpu import get_cloud_gpu_manager

logger = logging.getLogger("aceml.tuning")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
    logger.debug("Optuna available")
except ImportError:
    HAS_OPTUNA = False
    logger.info("Optuna not installed — Optuna tuning disabled")


# ------------------------------------------------------------------ #
#  Default param grids per model
# ------------------------------------------------------------------ #
PARAM_GRIDS: dict[str, dict] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "max_iter": [1000],
    },
    "decision_tree_clf": {
        "max_depth": [3, 5, 10, 20, None], "min_samples_split": [2, 5, 10],
    },
    "random_forest_clf": {
        "n_estimators": [50, 100, 200], "max_depth": [5, 10, 20, None], "min_samples_split": [2, 5],
    },
    "gradient_boosting_clf": {
        "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7],
    },
    "xgboost_clf": {
        "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7], "subsample": [0.8, 1.0],
    },
    "svm_clf": {
        "C": [0.1, 1, 10], "kernel": ["rbf", "linear"],
    },
    "knn_clf": {
        "n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"],
    },
    "mlp_clf": {
        "hidden_layer_sizes": [(64,), (128,), (64, 32)], "learning_rate_init": [0.001, 0.01],
    },
    "linear_regression": {},
    "decision_tree_reg": {
        "max_depth": [3, 5, 10, 20, None], "min_samples_split": [2, 5, 10],
    },
    "random_forest_reg": {
        "n_estimators": [50, 100, 200], "max_depth": [5, 10, 20, None], "min_samples_split": [2, 5],
    },
    "gradient_boosting_reg": {
        "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7],
    },
    "xgboost_reg": {
        "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7], "subsample": [0.8, 1.0],
    },
    "svr_reg": {
        "C": [0.1, 1, 10], "kernel": ["rbf", "linear"],
    },
    "knn_reg": {
        "n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"],
    },
    "mlp_reg": {
        "hidden_layer_sizes": [(64,), (128,), (64, 32)], "learning_rate_init": [0.001, 0.01],
    },
}


class HyperparameterTuner:
    """Grid, Random, and Optuna hyperparameter search."""

    # ------------------------------------------------------------------ #
    #  Grid Search
    # ------------------------------------------------------------------ #
    @staticmethod
    def grid_search(model_key: str, task: str, X, y,
                    param_grid: dict | None = None,
                    cv: int | None = None,
                    use_cloud_gpu: bool | None = None) -> dict:
        registry = CLF_REGISTRY if task == "classification" else REG_REGISTRY
        model_cls = registry.get(model_key)
        if model_cls is None:
            logger.error("Grid search: unknown model '%s'", model_key)
            raise ValueError(f"Unknown model: {model_key}")

        base_params = _default_params(model_key)
        model = model_cls(**base_params)
        grid = param_grid or PARAM_GRIDS.get(model_key, {})
        if not grid:
            logger.warning("Grid search: no param grid for '%s'", model_key)
            return {"error": "No param grid available for this model"}

        scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"
        folds = cv or Config.DEFAULT_CV_FOLDS

        # Check if cloud GPU should be used
        use_gpu = use_cloud_gpu if use_cloud_gpu is not None else Config.CLOUD_GPU_ENABLED
        
        if use_gpu:
            try:
                logger.info("Attempting cloud GPU grid search for '%s'", model_key)
                gpu_manager = get_cloud_gpu_manager()
                gpu_result = gpu_manager.execute_tuning(
                    model_cls, X, y, task, method="grid",
                    param_grid=grid, cv=folds
                )
                
                # Check if cloud execution was successful
                if gpu_result.get("status") != "fallback_to_local":
                    logger.info("Cloud GPU grid search completed successfully")
                    return gpu_result
                else:
                    logger.warning("Cloud GPU tuning fell back to local, continuing with local execution")
                    
            except Exception as e:
                logger.error("Cloud GPU grid search failed: %s", e, exc_info=True)
                if not Config.GPU_FALLBACK_TO_LOCAL:
                    raise
                logger.warning("Falling back to local grid search")

        # Local grid search (default or fallback)
        logger.info("Grid search (LOCAL): model=%s, %d param combos, cv=%d", model_key, 
                    np.prod([len(v) for v in grid.values()]) if grid else 0, folds)
        start = time.time()
        search = GridSearchCV(model, grid, cv=folds, scoring=scoring, n_jobs=-1, return_train_score=True)
        search.fit(X, y)
        duration = round(time.time() - start, 2)
        logger.info("Grid search complete in %.2fs — best_score=%.4f, best_params=%s",
                    duration, search.best_score_, search.best_params_)

        return {
            "method": "grid_search",
            "best_params": search.best_params_,
            "best_score": round(float(search.best_score_), 4),
            "total_fits": len(search.cv_results_["mean_test_score"]),
            "duration_sec": duration,
            "all_results": _extract_cv_results(search),
            "best_model": search.best_estimator_,
            "execution_mode": "local"
        }

    # ------------------------------------------------------------------ #
    #  Random Search
    # ------------------------------------------------------------------ #
    @staticmethod
    def random_search(model_key: str, task: str, X, y,
                      param_distributions: dict | None = None,
                      n_iter: int | None = None,
                      cv: int | None = None,
                      use_cloud_gpu: bool | None = None) -> dict:
        registry = CLF_REGISTRY if task == "classification" else REG_REGISTRY
        model_cls = registry.get(model_key)
        if model_cls is None:
            raise ValueError(f"Unknown model: {model_key}")

        base_params = _default_params(model_key)
        model = model_cls(**base_params)
        dist = param_distributions or PARAM_GRIDS.get(model_key, {})
        if not dist:
            return {"error": "No param distributions available"}

        scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"
        folds = cv or Config.DEFAULT_CV_FOLDS
        iterations = n_iter or min(Config.MAX_TUNING_ITERATIONS, 20)

        # Check if cloud GPU should be used
        use_gpu = use_cloud_gpu if use_cloud_gpu is not None else Config.CLOUD_GPU_ENABLED
        
        if use_gpu:
            try:
                logger.info("Attempting cloud GPU random search for '%s'", model_key)
                gpu_manager = get_cloud_gpu_manager()
                gpu_result = gpu_manager.execute_tuning(
                    model_cls, X, y, task, method="random",
                    param_grid=dist, n_iter=iterations, cv=folds
                )
                
                # Check if cloud execution was successful
                if gpu_result.get("status") != "fallback_to_local":
                    logger.info("Cloud GPU random search completed successfully")
                    return gpu_result
                else:
                    logger.warning("Cloud GPU tuning fell back to local, continuing with local execution")
                    
            except Exception as e:
                logger.error("Cloud GPU random search failed: %s", e, exc_info=True)
                if not Config.GPU_FALLBACK_TO_LOCAL:
                    raise
                logger.warning("Falling back to local random search")

        # Local random search (default or fallback)
        logger.info("Random search (LOCAL): model=%s, n_iter=%d, cv=%d", model_key, iterations, folds)
        start = time.time()
        search = RandomizedSearchCV(
            model, dist, n_iter=iterations, cv=folds,
            scoring=scoring, n_jobs=-1, random_state=Config.DEFAULT_RANDOM_STATE,
            return_train_score=True,
        )
        search.fit(X, y)
        duration = round(time.time() - start, 2)
        logger.info("Random search complete in %.2fs — best_score=%.4f, best_params=%s",
                    duration, search.best_score_, search.best_params_)

        return {
            "method": "random_search",
            "best_params": search.best_params_,
            "best_score": round(float(search.best_score_), 4),
            "n_iterations": iterations,
            "duration_sec": duration,
            "all_results": _extract_cv_results(search),
            "best_model": search.best_estimator_,
            "execution_mode": "local"
        }

    # ------------------------------------------------------------------ #
    #  Optuna Search
    # ------------------------------------------------------------------ #
    @staticmethod
    def optuna_search(model_key: str, task: str, X, y,
                      n_trials: int | None = None,
                      cv: int | None = None,
                      use_cloud_gpu: bool | None = None) -> dict:
        if not HAS_OPTUNA:
            logger.error("Optuna not installed")
            return {"error": "Optuna not installed. Run: pip install optuna"}

        registry = CLF_REGISTRY if task == "classification" else REG_REGISTRY
        model_cls = registry.get(model_key)
        if model_cls is None:
            logger.error("Optuna search: unknown model '%s'", model_key)
            raise ValueError(f"Unknown model: {model_key}")

        folds = cv or Config.DEFAULT_CV_FOLDS
        trials = n_trials or 50
        scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"
        trial_results = []

        # Check if cloud GPU should be used
        use_gpu = use_cloud_gpu if use_cloud_gpu is not None else Config.CLOUD_GPU_ENABLED
        
        if use_gpu:
            try:
                logger.info("Attempting cloud GPU optuna search for '%s'", model_key)
                gpu_manager = get_cloud_gpu_manager()
                gpu_result = gpu_manager.execute_tuning(
                    model_cls, X, y, task, method="optuna",
                    n_iter=trials, cv=folds
                )
                
                # Check if cloud execution was successful
                if gpu_result.get("status") != "fallback_to_local":
                    logger.info("Cloud GPU optuna search completed successfully")
                    return gpu_result
                else:
                    logger.warning("Cloud GPU tuning fell back to local, continuing with local execution")
                    
            except Exception as e:
                logger.error("Cloud GPU optuna search failed: %s", e, exc_info=True)
                if not Config.GPU_FALLBACK_TO_LOCAL:
                    raise
                logger.warning("Falling back to local optuna search")

        # Local optuna search (default or fallback)
        logger.info("Optuna search (LOCAL): model=%s, n_trials=%d, cv=%d", model_key, trials, folds)

        def objective(trial):
            params = _optuna_suggest(trial, model_key)
            base = _default_params(model_key)
            base.update(params)
            model = model_cls(**base)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=folds, scoring=scoring, n_jobs=-1)
            mean_score = scores.mean()
            trial_results.append({"params": params, "score": round(float(mean_score), 4)})
            return mean_score

        start = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, timeout=Config.TUNING_TIMEOUT_SECONDS)
        duration = round(time.time() - start, 2)
        logger.info("Optuna search complete in %.2fs — best_score=%.4f, best_params=%s",
                    duration, study.best_value, study.best_params)

        # Re-train best model
        best_params = study.best_params
        base = _default_params(model_key)
        base.update(best_params)
        best_model = model_cls(**base)
        best_model.fit(X, y)

        return {
            "method": "optuna",
            "best_params": best_params,
            "best_score": round(float(study.best_value), 4),
            "n_trials": trials,
            "duration_sec": duration,
            "trial_results": trial_results,
            "best_model": best_model,
            "execution_mode": "local"
        }


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def _extract_cv_results(search) -> list[dict]:
    results = []
    for i in range(len(search.cv_results_["mean_test_score"])):
        results.append({
            "params": search.cv_results_["params"][i],
            "mean_test_score": round(float(search.cv_results_["mean_test_score"][i]), 4),
            "std_test_score": round(float(search.cv_results_["std_test_score"][i]), 4),
            "mean_train_score": round(float(search.cv_results_["mean_train_score"][i]), 4),
            "rank": int(search.cv_results_["rank_test_score"][i]),
        })
    return sorted(results, key=lambda x: x["rank"])


def _optuna_suggest(trial, model_key: str) -> dict:
    """Suggest hyperparameters based on model type."""
    suggestions = {
        "logistic_regression": lambda t: {"C": t.suggest_float("C", 0.001, 100, log=True)},
        "decision_tree_clf": lambda t: {
            "max_depth": t.suggest_int("max_depth", 2, 30),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        },
        "random_forest_clf": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 300),
            "max_depth": t.suggest_int("max_depth", 3, 25),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 15),
        },
        "gradient_boosting_clf": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 300),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 10),
        },
        "xgboost_clf": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 300),
            "learning_rate": t.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 10),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
        },
        "svm_clf": lambda t: {
            "C": t.suggest_float("C", 0.01, 100, log=True),
            "kernel": t.suggest_categorical("kernel", ["rbf", "linear"]),
        },
        "knn_clf": lambda t: {
            "n_neighbors": t.suggest_int("n_neighbors", 1, 25),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
        },
        "mlp_clf": lambda t: {
            "hidden_layer_sizes": t.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (64, 32), (128, 64)]),
            "learning_rate_init": t.suggest_float("learning_rate_init", 0.0001, 0.1, log=True),
        },
    }
    # Regression mirrors classification with _reg suffix
    reg_map = {
        "decision_tree_reg": "decision_tree_clf",
        "random_forest_reg": "random_forest_clf",
        "gradient_boosting_reg": "gradient_boosting_clf",
        "xgboost_reg": "xgboost_clf",
        "svr_reg": "svm_clf",
        "knn_reg": "knn_clf",
        "mlp_reg": "mlp_clf",
    }
    key = reg_map.get(model_key, model_key)
    fn = suggestions.get(key, lambda t: {})
    return fn(trial)
