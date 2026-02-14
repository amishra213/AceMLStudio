"""
AceML Studio – Model Evaluation
=================================
Classification & regression metrics, confusion matrix, ROC data,
feature importances, and learning-curve helpers.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import learning_curve
from config import Config

logger = logging.getLogger("aceml.evaluation")


class ModelEvaluator:
    """Compute evaluation metrics for trained models."""

    # ------------------------------------------------------------------ #
    #  Classification Metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
        labels = sorted(list(set(y_true) | set(y_pred)))
        is_binary = len(labels) <= 2
        avg = "binary" if is_binary else "weighted"

        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        }

        # ROC-AUC (needs probabilities)
        if y_prob is not None:
            try:
                if is_binary:
                    prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, prob)), 4)
                else:
                    metrics["roc_auc"] = round(float(
                        roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                    ), 4)
            except Exception:
                metrics["roc_auc"] = None  # type: ignore

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics["confusion_matrix"] = {  # type: ignore
            "labels": [str(l) for l in labels],
            "matrix": cm.tolist(),
        }

        # Per-class report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["per_class"] = {  # type: ignore
            str(k): {m: round(float(v), 4) for m, v in vals.items()}  # type: ignore
            for k, vals in report.items() if isinstance(vals, dict)  # type: ignore
        }

        return metrics

    # ------------------------------------------------------------------ #
    #  Regression Metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def regression_metrics(y_true, y_pred) -> dict:
        return {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
            "r2_score": round(float(r2_score(y_true, y_pred)), 4),
            "explained_variance": round(float(explained_variance_score(y_true, y_pred)), 4),
            "residuals": {
                "mean": round(float(np.mean(y_true - y_pred)), 4),
                "std": round(float(np.std(y_true - y_pred)), 4),
            },
        }

    # ------------------------------------------------------------------ #
    #  ROC Curve Data
    # ------------------------------------------------------------------ #
    @staticmethod
    def roc_curve_data(y_true, y_prob) -> dict | None:
        try:
            prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            fpr, tpr, thresholds = roc_curve(y_true, prob)
            return {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Precision-Recall Curve Data
    # ------------------------------------------------------------------ #
    @staticmethod
    def pr_curve_data(y_true, y_prob) -> dict | None:
        try:
            prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            precision, recall, thresholds = precision_recall_curve(y_true, prob)
            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Feature Importance
    # ------------------------------------------------------------------ #
    @staticmethod
    def feature_importance(model, feature_names: list[str]) -> dict | None:
        imp = None
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                imp = np.abs(coef).mean(axis=0)
            else:
                imp = np.abs(coef)

        if imp is None:
            return None

        importance_dict = {name: round(float(val), 6) for name, val in zip(feature_names, imp)}
        sorted_imp = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return sorted_imp

    # ------------------------------------------------------------------ #
    #  Learning Curves
    # ------------------------------------------------------------------ #
    @staticmethod
    def learning_curves(model, X, y, cv: int = 5) -> dict:
        train_sizes, train_scores, val_scores, fit_times, score_times = learning_curve(  # type: ignore
            model, X, y, cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
            n_jobs=-1,
            random_state=Config.DEFAULT_RANDOM_STATE,
            return_times=True,
        )
        return {
            "train_sizes": train_sizes.tolist(),
            "train_mean": train_scores.mean(axis=1).tolist(),
            "train_std": train_scores.std(axis=1).tolist(),
            "val_mean": val_scores.mean(axis=1).tolist(),
            "val_std": val_scores.std(axis=1).tolist(),
        }

    # ------------------------------------------------------------------ #
    #  Full evaluation convenience
    # ------------------------------------------------------------------ #
    @classmethod
    def evaluate(cls, model, X, y, task: str, feature_names: list[str] | None = None) -> dict:
        logger.info("Evaluating model (task=%s, samples=%d, features=%d)", task, len(y), X.shape[1])
        y_pred = model.predict(X)
        result = {"task": task}

        if task == "classification":
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)
            result["metrics"] = cls.classification_metrics(y, y_pred, y_prob)  # type: ignore
            logger.info("Classification metrics: accuracy=%.4f, f1=%.4f",
                        result['metrics'].get('accuracy', 0),
                        result['metrics'].get('f1_score', 0))
            if y_prob is not None:
                result["roc_curve"] = cls.roc_curve_data(y, y_prob)  # type: ignore
                result["pr_curve"] = cls.pr_curve_data(y, y_prob)  # type: ignore
        else:
            result["metrics"] = cls.regression_metrics(y, y_pred)  # type: ignore
            logger.info("Regression metrics: r2=%.4f, rmse=%.4f",
                        result['metrics'].get('r2_score', 0),
                        result['metrics'].get('rmse', 0))

        if feature_names:
            result["feature_importance"] = cls.feature_importance(model, feature_names)  # type: ignore

        return result
