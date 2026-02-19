"""
Model Registry & Versioning System
===================================
Manages model versions, metadata, and deployment lifecycle.
Provides version control, rollback, and model lineage tracking.
"""

import os
import json
import pickle
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for managing ML models with versioning and metadata.
    
    Features:
    - Model versioning with semantic versioning support
    - Metadata tracking (metrics, hyperparameters, dataset info)
    - Model lifecycle management (staging, production, archived)
    - Model lineage and provenance tracking
    - Rollback capability
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Base directory for storing models and metadata
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.db_path = self.registry_path / "registry.db"
        self._init_database()
        
        logger.info("ModelRegistry initialized at: %s", self.registry_path)
    
    def _init_database(self):
        """Initialize SQLite database for model metadata."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                task TEXT NOT NULL,
                status TEXT DEFAULT 'staging',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_path TEXT NOT NULL,
                model_hash TEXT NOT NULL,
                description TEXT,
                UNIQUE(name, version)
            )
        """)
        
        # Model metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                dataset TEXT DEFAULT 'test',
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            )
        """)
        
        # Model metadata table (for hyperparameters, config, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            )
        """)
        
        # Model lineage table (parent-child relationships)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                child_model_id INTEGER NOT NULL,
                parent_model_id INTEGER NOT NULL,
                relationship_type TEXT DEFAULT 'tuned_from',
                FOREIGN KEY (child_model_id) REFERENCES models(id) ON DELETE CASCADE,
                FOREIGN KEY (parent_model_id) REFERENCES models(id) ON DELETE CASCADE
            )
        """)
        
        # Deployment history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deployment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deployed_by TEXT,
                notes TEXT,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Model registry database initialized")
    
    def register_model(
        self,
        name: str,
        model: Any,
        model_type: str,
        task: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        parent_model_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Register a new model with the registry.
        
        Args:
            name: Model name (e.g., 'fraud_detector', 'churn_predictor')
            model: Trained model object
            model_type: Model algorithm type (e.g., 'random_forest', 'xgboost')
            task: ML task type ('classification', 'regression', 'clustering')
            version: Semantic version (auto-incremented if None)
            metrics: Model performance metrics
            hyperparameters: Model hyperparameters
            metadata: Additional metadata (dataset info, feature names, etc.)
            description: Human-readable description
            parent_model_id: ID of parent model (for lineage tracking)
        
        Returns:
            Dict with registration details including model_id and version
        """
        try:
            # Auto-generate version if not provided
            if version is None:
                version = self._get_next_version(name)
            
            # Serialize and save model
            model_filename = f"{name}_v{version.replace('.', '_')}.pkl"
            model_path = self.models_dir / model_filename
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Calculate model hash for integrity checking
            model_hash = self._calculate_file_hash(model_path)
            
            # Insert into database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO models (name, version, model_type, task, model_path, model_hash, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, version, model_type, task, str(model_path), model_hash, description))
            
            model_id = cursor.lastrowid
            
            # Store metrics
            if metrics:
                for metric_name, metric_value in metrics.items():
                    cursor.execute("""
                        INSERT INTO model_metrics (model_id, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    """, (model_id, metric_name, float(metric_value)))
            
            # Store hyperparameters
            if hyperparameters:
                for key, value in hyperparameters.items():
                    cursor.execute("""
                        INSERT INTO model_metadata (model_id, key, value)
                        VALUES (?, ?, ?)
                    """, (model_id, f"hyperparam_{key}", json.dumps(value)))
            
            # Store additional metadata
            if metadata:
                for key, value in metadata.items():
                    cursor.execute("""
                        INSERT INTO model_metadata (model_id, key, value)
                        VALUES (?, ?, ?)
                    """, (model_id, f"metadata_{key}", json.dumps(value)))
            
            # Store lineage
            if parent_model_id:
                cursor.execute("""
                    INSERT INTO model_lineage (child_model_id, parent_model_id, relationship_type)
                    VALUES (?, ?, ?)
                """, (model_id, parent_model_id, 'tuned_from'))
            
            # Log deployment history
            cursor.execute("""
                INSERT INTO deployment_history (model_id, action, to_status)
                VALUES (?, ?, ?)
            """, (model_id, 'registered', 'staging'))
            
            conn.commit()
            conn.close()
            
            logger.info("Model registered: %s v%s (ID: %d)", name, version, model_id)
            
            return {
                "model_id": model_id,
                "name": name,
                "version": version,
                "model_type": model_type,
                "task": task,
                "status": "staging",
                "model_path": str(model_path),
                "model_hash": model_hash
            }
        
        except Exception as e:
            logger.error("Failed to register model: %s", e, exc_info=True)
            raise
    
    def load_model(self, model_id: Optional[int] = None, name: Optional[str] = None, 
                   version: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model from the registry.
        
        Args:
            model_id: Specific model ID to load
            name: Model name (requires version or status)
            version: Specific version to load
            status: Load model with specific status ('production', 'staging')
        
        Returns:
            Dict with model object and metadata
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Build query based on provided parameters
        if model_id:
            cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        elif name and version:
            cursor.execute("SELECT * FROM models WHERE name = ? AND version = ?", (name, version))
        elif name and status:
            cursor.execute("""
                SELECT * FROM models WHERE name = ? AND status = ? 
                ORDER BY created_at DESC LIMIT 1
            """, (name, status))
        elif name:
            # Get latest version
            cursor.execute("""
                SELECT * FROM models WHERE name = ? 
                ORDER BY created_at DESC LIMIT 1
            """, (name,))
        else:
            conn.close()
            raise ValueError("Must provide model_id or name (with optional version/status)")
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Model not found with provided criteria")
        
        # Parse model record
        model_record = {
            "id": row[0],
            "name": row[1],
            "version": row[2],
            "model_type": row[3],
            "task": row[4],
            "status": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "model_path": row[8],
            "model_hash": row[9],
            "description": row[10]
        }
        
        # Load model object
        model_path = Path(model_record["model_path"])
        if not model_path.exists():
            conn.close()
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Verify model hash
        current_hash = self._calculate_file_hash(model_path)
        if current_hash != model_record["model_hash"]:
            logger.warning("Model hash mismatch - file may be corrupted: %s", model_path)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metrics
        cursor.execute("SELECT metric_name, metric_value, dataset FROM model_metrics WHERE model_id = ?", 
                      (model_record["id"],))
        metrics = {}
        for metric_row in cursor.fetchall():
            key = f"{metric_row[0]}_{metric_row[2]}" if metric_row[2] != 'test' else metric_row[0]
            metrics[key] = metric_row[1]
        
        # Load metadata
        cursor.execute("SELECT key, value FROM model_metadata WHERE model_id = ?", 
                      (model_record["id"],))
        metadata = {}
        for meta_row in cursor.fetchall():
            metadata[meta_row[0]] = json.loads(meta_row[1])
        
        conn.close()
        
        logger.info("Model loaded: %s v%s (ID: %d)", model_record["name"], 
                   model_record["version"], model_record["id"])
        
        return {
            "model": model,
            "model_id": model_record["id"],
            "name": model_record["name"],
            "version": model_record["version"],
            "model_type": model_record["model_type"],
            "task": model_record["task"],
            "status": model_record["status"],
            "metrics": metrics,
            "metadata": metadata,
            "description": model_record["description"]
        }
    
    def promote_model(self, model_id: int, to_status: str, notes: Optional[str] = None) -> bool:
        """
        Promote model to a new status (staging -> production, etc.).
        
        Args:
            model_id: Model ID to promote
            to_status: Target status ('production', 'archived', 'staging')
            notes: Optional notes about the promotion
        
        Returns:
            True if successful
        """
        valid_statuses = ['staging', 'production', 'archived']
        if to_status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get current status
        cursor.execute("SELECT status, name, version FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Model ID {model_id} not found")
        
        from_status, name, version = row
        
        # If promoting to production, demote current production model
        if to_status == 'production':
            cursor.execute("""
                UPDATE models SET status = 'archived', updated_at = CURRENT_TIMESTAMP 
                WHERE name = ? AND status = 'production'
            """, (name,))
        
        # Update model status
        cursor.execute("""
            UPDATE models SET status = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, (to_status, model_id))
        
        # Log deployment history
        cursor.execute("""
            INSERT INTO deployment_history (model_id, action, from_status, to_status, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (model_id, 'promoted', from_status, to_status, notes))
        
        conn.commit()
        conn.close()
        
        logger.info("Model promoted: %s v%s (ID: %d) %s -> %s", 
                   name, version, model_id, from_status, to_status)
        
        return True
    
    def list_models(self, name: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all models with optional filtering.
        
        Args:
            name: Filter by model name
            status: Filter by status
        
        Returns:
            List of model records with metrics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM models"
        params = []
        conditions = []
        
        if name:
            conditions.append("name = ?")
            params.append(name)
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        models = []
        for row in rows:
            model_id = row[0]
            
            # Get metrics
            cursor.execute("""
                SELECT metric_name, metric_value, dataset 
                FROM model_metrics WHERE model_id = ?
            """, (model_id,))
            metrics = {}
            for metric_row in cursor.fetchall():
                key = f"{metric_row[0]}_{metric_row[2]}" if metric_row[2] != 'test' else metric_row[0]
                metrics[key] = metric_row[1]
            
            models.append({
                "id": row[0],
                "name": row[1],
                "version": row[2],
                "model_type": row[3],
                "task": row[4],
                "status": row[5],
                "created_at": row[6],
                "updated_at": row[7],
                "description": row[10],
                "metrics": metrics
            })
        
        conn.close()
        return models
    
    def delete_model(self, model_id: int) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model ID to delete
        
        Returns:
            True if successful
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get model path
        cursor.execute("SELECT model_path, name, version FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Model ID {model_id} not found")
        
        model_path, name, version = row
        
        # Delete model file
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Delete from database (cascades to related tables)
        cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
        
        conn.commit()
        conn.close()
        
        logger.info("Model deleted: %s v%s (ID: %d)", name, version, model_id)
        return True
    
    def get_deployment_history(self, model_id: int) -> List[Dict[str, Any]]:
        """Get deployment history for a model."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT action, from_status, to_status, deployed_at, deployed_by, notes
            FROM deployment_history WHERE model_id = ? ORDER BY deployed_at DESC
        """, (model_id,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "action": row[0],
                "from_status": row[1],
                "to_status": row[2],
                "deployed_at": row[3],
                "deployed_by": row[4],
                "notes": row[5]
            })
        
        conn.close()
        return history
    
    def _get_next_version(self, name: str) -> str:
        """Auto-generate next semantic version for a model."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT version FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1
        """, (name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return "1.0.0"
        
        # Parse current version and increment patch version
        try:
            parts = row[0].split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{major}.{minor}.{patch + 1}"
        except:
            return "1.0.0"
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
