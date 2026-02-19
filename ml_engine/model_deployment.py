"""
Model Deployment Service
========================
Handles model serving, prediction APIs, and monitoring.
Provides REST endpoints for real-time inference.
"""

import os
import json
import time
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import pandas as pd
import numpy as np

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Logs predictions for monitoring and analysis.
    Tracks input features, predictions, confidence scores, and timing.
    """
    
    def __init__(self, log_db_path: str = "models/predictions.db"):
        """
        Initialize prediction logger.
        
        Args:
            log_db_path: Path to SQLite database for prediction logs
        """
        self.log_db_path = Path(log_db_path)
        self.log_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for prediction logging."""
        conn = sqlite3.connect(str(self.log_db_path))
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                input_features TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                latency_ms REAL,
                client_id TEXT,
                request_id TEXT
            )
        """)
        
        # Performance metrics table (aggregated stats)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                metric_date DATE DEFAULT (DATE('now')),
                total_predictions INTEGER DEFAULT 0,
                avg_latency_ms REAL,
                p95_latency_ms REAL,
                p99_latency_ms REAL,
                error_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Prediction logger database initialized at: %s", self.log_db_path)
    
    def log_prediction(
        self,
        model_id: int,
        model_name: str,
        model_version: str,
        input_features: Dict[str, Any],
        prediction: Any,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
        client_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        Log a single prediction.
        
        Args:
            model_id: Model ID from registry
            model_name: Model name
            model_version: Model version
            input_features: Dictionary of input features
            prediction: Model prediction
            confidence: Confidence score (for classification)
            latency_ms: Prediction latency in milliseconds
            client_id: Optional client identifier
            request_id: Optional request identifier
        """
        try:
            conn = sqlite3.connect(str(self.log_db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions 
                (model_id, model_name, model_version, input_features, prediction, 
                 confidence, latency_ms, client_id, request_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                model_name,
                model_version,
                json.dumps(input_features),
                json.dumps(prediction) if not isinstance(prediction, str) else prediction,
                confidence,
                latency_ms,
                client_id,
                request_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error("Failed to log prediction: %s", e)
    
    def get_recent_predictions(self, model_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent predictions for a model."""
        conn = sqlite3.connect(str(self.log_db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, prediction_time, input_features, prediction, confidence, latency_ms, request_id
            FROM predictions WHERE model_id = ? ORDER BY prediction_time DESC LIMIT ?
        """, (model_id, limit))
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                "id": row[0],
                "timestamp": row[1],
                "input_features": json.loads(row[2]),
                "prediction": json.loads(row[3]) if row[3].startswith('[') or row[3].startswith('{') else row[3],
                "confidence": row[4],
                "latency_ms": row[5],
                "request_id": row[6]
            })
        
        conn.close()
        return predictions
    
    def get_performance_stats(self, model_id: int, days: int = 7) -> Dict[str, Any]:
        """
        Get performance statistics for a model.
        
        Args:
            model_id: Model ID
            days: Number of days to look back
        
        Returns:
            Dict with performance metrics
        """
        conn = sqlite3.connect(str(self.log_db_path))
        cursor = conn.cursor()
        
        # Get predictions in time window
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                AVG(latency_ms) as avg_latency,
                AVG(confidence) as avg_confidence,
                MIN(prediction_time) as first_prediction,
                MAX(prediction_time) as last_prediction
            FROM predictions 
            WHERE model_id = ? 
            AND prediction_time >= datetime('now', '-' || ? || ' days')
        """, (model_id, days))
        
        row = cursor.fetchone()
        
        # Get latency percentiles
        cursor.execute("""
            SELECT latency_ms FROM predictions 
            WHERE model_id = ? 
            AND prediction_time >= datetime('now', '-' || ? || ' days')
            AND latency_ms IS NOT NULL
            ORDER BY latency_ms
        """, (model_id, days))
        
        latencies = [r[0] for r in cursor.fetchall()]
        
        conn.close()
        
        p50 = np.percentile(latencies, 50) if latencies else None
        p95 = np.percentile(latencies, 95) if latencies else None
        p99 = np.percentile(latencies, 99) if latencies else None
        
        return {
            "total_predictions": row[0] or 0,
            "avg_latency_ms": round(row[1], 2) if row[1] else None,
            "avg_confidence": round(row[2], 4) if row[2] else None,
            "first_prediction": row[3],
            "last_prediction": row[4],
            "latency_p50_ms": round(p50, 2) if p50 else None,
            "latency_p95_ms": round(p95, 2) if p95 else None,
            "latency_p99_ms": round(p99, 2) if p99 else None,
            "days": days
        }


class ModelDeploymentService:
    """
    Manages model deployment and serves predictions.
    Integrates with ModelRegistry for version management.
    """
    
    def __init__(self, registry: ModelRegistry, prediction_logger: PredictionLogger):
        """
        Initialize deployment service.
        
        Args:
            registry: ModelRegistry instance
            prediction_logger: PredictionLogger instance
        """
        self.registry = registry
        self.logger = prediction_logger
        self._deployed_models: Dict[str, Any] = {}
        
        logger.info("ModelDeploymentService initialized")
    
    def deploy_model(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Deploy a model for serving predictions.
        
        Args:
            model_name: Name of model to deploy
            version: Specific version (defaults to production version)
        
        Returns:
            Deployment info
        """
        try:
            # Load model from registry
            if version:
                model_data = self.registry.load_model(name=model_name, version=version)
            else:
                # Load production model
                model_data = self.registry.load_model(name=model_name, status='production')
            
            # Cache model in memory
            deployment_key = f"{model_name}:{model_data['version']}"
            self._deployed_models[deployment_key] = {
                "model": model_data["model"],
                "model_id": model_data["model_id"],
                "version": model_data["version"],
                "model_type": model_data["model_type"],
                "task": model_data["task"],
                "metadata": model_data["metadata"],
                "deployed_at": datetime.now().isoformat()
            }
            
            logger.info("Model deployed: %s v%s", model_name, model_data["version"])
            
            return {
                "model_name": model_name,
                "version": model_data["version"],
                "model_id": model_data["model_id"],
                "deployment_key": deployment_key,
                "status": "deployed",
                "deployed_at": self._deployed_models[deployment_key]["deployed_at"]
            }
            
        except Exception as e:
            logger.error("Failed to deploy model %s: %s", model_name, e, exc_info=True)
            raise
    
    def predict(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        version: Optional[str] = None,
        log_prediction: bool = True,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction using deployed model.
        
        Args:
            model_name: Name of model
            input_data: Input features as dictionary
            version: Specific version (defaults to latest deployed)
            log_prediction: Whether to log the prediction
            request_id: Optional request identifier
        
        Returns:
            Dict with prediction, confidence, and metadata
        """
        start_time = time.time()
        
        try:
            # Find deployed model
            deployment_key = self._find_deployment(model_name, version)
            if not deployment_key:
                # Try to deploy the model
                self.deploy_model(model_name, version)
                deployment_key = self._find_deployment(model_name, version)
                
            if not deployment_key:
                raise ValueError(f"Model {model_name} (version: {version}) not deployed")
            
            deployment = self._deployed_models[deployment_key]
            model = deployment["model"]
            task = deployment["task"]
            
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            if task == "classification":
                prediction = model.predict(input_df)[0]
                
                # Get confidence if available
                confidence = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df)[0]
                    confidence = float(max(proba))
                    prediction_proba = proba.tolist()
                else:
                    prediction_proba = None
                
                # Convert prediction to standard Python type
                try:
                    pred_value = int(prediction) if np.issubdtype(type(prediction), np.integer) else str(prediction)
                except (TypeError, ValueError):
                    pred_value = str(prediction)
                
                result = {
                    "prediction": pred_value,
                    "confidence": confidence,
                    "probabilities": prediction_proba
                }
                
            elif task == "regression":
                prediction = model.predict(input_df)[0]
                result = {
                    "prediction": float(prediction)
                }
                confidence = None
                
            elif task == "unsupervised":
                # Clustering
                prediction = model.predict(input_df)[0]
                result = {
                    "cluster": int(prediction)
                }
                confidence = None
                
            else:
                raise ValueError(f"Unsupported task type: {task}")
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log prediction
            if log_prediction:
                self.logger.log_prediction(
                    model_id=deployment["model_id"],
                    model_name=model_name,
                    model_version=deployment["version"],
                    input_features=input_data,
                    prediction=result["prediction"] if "prediction" in result else result.get("cluster"),
                    confidence=confidence,
                    latency_ms=latency_ms,
                    request_id=request_id
                )
            
            # Add metadata to result and create final response
            final_result: Dict[str, Any] = {
                **result,
                "model_name": model_name,
                "model_version": deployment["version"],
                "model_type": deployment["model_type"],
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            return final_result
            
        except Exception as e:
            logger.error("Prediction failed for %s: %s", model_name, e, exc_info=True)
            raise
    
    def batch_predict(
        self,
        model_name: str,
        input_data: List[Dict[str, Any]],
        version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            model_name: Name of model
            input_data: List of input feature dictionaries
            version: Specific version
        
        Returns:
            List of prediction results
        """
        results = []
        for i, single_input in enumerate(input_data):
            try:
                result = self.predict(
                    model_name=model_name,
                    input_data=single_input,
                    version=version,
                    log_prediction=True,
                    request_id=f"batch_{i}"
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "input_index": i
                })
        
        return results
    
    def get_deployed_models(self) -> List[Dict[str, Any]]:
        """Get list of currently deployed models."""
        return [
            {
                "deployment_key": key,
                "model_name": key.split(':')[0],
                "version": info["version"],
                "model_type": info["model_type"],
                "task": info["task"],
                "deployed_at": info["deployed_at"]
            }
            for key, info in self._deployed_models.items()
        ]
    
    def undeploy_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """
        Remove model from deployed cache.
        
        Args:
            model_name: Model name
            version: Specific version (removes all if None)
        
        Returns:
            True if successful
        """
        if version:
            deployment_key = f"{model_name}:{version}"
            if deployment_key in self._deployed_models:
                del self._deployed_models[deployment_key]
                logger.info("Model undeployed: %s", deployment_key)
                return True
        else:
            # Remove all versions
            keys_to_remove = [k for k in self._deployed_models.keys() if k.startswith(f"{model_name}:")]
            for key in keys_to_remove:
                del self._deployed_models[key]
                logger.info("Model undeployed: %s", key)
            return len(keys_to_remove) > 0
        
        return False
    
    def _find_deployment(self, model_name: str, version: Optional[str] = None) -> Optional[str]:
        """Find deployment key for a model."""
        if version:
            key = f"{model_name}:{version}"
            return key if key in self._deployed_models else None
        else:
            # Find latest deployed version
            matching_keys = [k for k in self._deployed_models.keys() if k.startswith(f"{model_name}:")]
            return matching_keys[0] if matching_keys else None
