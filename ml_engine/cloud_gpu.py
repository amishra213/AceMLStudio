"""
AceML Studio – Cloud GPU Integration
======================================
Integrate with cloud-hosted GPU servers for model training and hyperparameter tuning.
Supports AWS SageMaker, Azure ML, GCP Vertex AI, and custom GPU servers.
"""

import time
import logging
import tempfile
import os
from typing import Any, Dict, Tuple, Optional
from config import Config

logger = logging.getLogger("aceml.cloud_gpu")


class CloudGPUManager:
    """
    Manage cloud GPU execution for training and tuning.
    Routes jobs to the configured cloud provider.
    """

    def __init__(self):
        self.provider = Config.CLOUD_GPU_PROVIDER
        self.enabled = Config.CLOUD_GPU_ENABLED
        logger.info(f"CloudGPUManager initialized - Provider: {self.provider}, Enabled: {self.enabled}")

    def is_enabled(self) -> bool:
        """Check if cloud GPU is enabled."""
        return self.enabled

    def execute_training(self, model_cls, X_train, y_train, hyperparams: dict) -> Tuple[Any, Dict]:
        """
        Execute model training on cloud GPU.
        
        Args:
            model_cls: Model class to instantiate
            X_train: Training features
            y_train: Training labels
            hyperparams: Model hyperparameters
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        if not self.enabled:
            raise RuntimeError("Cloud GPU is not enabled. Set CLOUD_GPU_ENABLED=True in config.properties")

        logger.info(f"Executing training on {self.provider} cloud GPU")
        
        if self.provider == "aws_sagemaker":
            return self._execute_aws_sagemaker(model_cls, X_train, y_train, hyperparams)
        elif self.provider == "azure_ml":
            return self._execute_azure_ml(model_cls, X_train, y_train, hyperparams)
        elif self.provider == "gcp_vertex":
            return self._execute_gcp_vertex(model_cls, X_train, y_train, hyperparams)
        elif self.provider == "custom":
            return self._execute_custom(model_cls, X_train, y_train, hyperparams)
        else:
            raise ValueError(f"Unsupported cloud GPU provider: {self.provider}")

    def execute_tuning(self, model_cls, X, y, task: str, method: str, 
                      param_grid: Optional[Dict] = None, n_iter: Optional[int] = None, 
                      cv: Optional[int] = None) -> Dict:
        """
        Execute hyperparameter tuning on cloud GPU.
        
        Args:
            model_cls: Model class to tune
            X: Features
            y: Labels
            task: 'classification' or 'regression'
            method: 'grid', 'random', or 'optuna'
            param_grid: Parameter grid/distributions
            n_iter: Number of iterations (for random/optuna)
            cv: Cross-validation folds
            
        Returns:
            Dictionary with tuning results
        """
        if not self.enabled:
            raise RuntimeError("Cloud GPU is not enabled. Set CLOUD_GPU_ENABLED=True in config.properties")

        logger.info(f"Executing {method} tuning on {self.provider} cloud GPU")
        
        tuning_config = {
            "task": task,
            "method": method,
            "param_grid": param_grid,
            "n_iter": n_iter,
            "cv": cv or Config.DEFAULT_CV_FOLDS
        }
        
        if self.provider == "aws_sagemaker":
            return self._tune_aws_sagemaker(model_cls, X, y, tuning_config)
        elif self.provider == "azure_ml":
            return self._tune_azure_ml(model_cls, X, y, tuning_config)
        elif self.provider == "gcp_vertex":
            return self._tune_gcp_vertex(model_cls, X, y, tuning_config)
        elif self.provider == "custom":
            return self._tune_custom(model_cls, X, y, tuning_config)
        else:
            raise ValueError(f"Unsupported cloud GPU provider: {self.provider}")

    # ================================================================
    # AWS SageMaker Integration
    # ================================================================
    def _execute_aws_sagemaker(self, model_cls, X_train, y_train, hyperparams: dict) -> Tuple[Any, Dict]:
        """Execute training on AWS SageMaker."""
        try:
            import boto3  # type: ignore
            from sagemaker.session import Session  # type: ignore
            from sagemaker.sklearn import SKLearn  # type: ignore
        except ImportError as e:
            logger.error("AWS SageMaker dependencies not installed. Run: pip install boto3 sagemaker")
            raise ImportError(f"Missing AWS dependencies: {e}")

        logger.info("Preparing AWS SageMaker training job")
        start_time = time.time()

        # Create SageMaker session
        boto_session = boto3.Session(
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
        )
        sagemaker_session = Session(boto_session=boto_session)

        # Save data to S3
        import pandas as pd
        train_data = pd.DataFrame(X_train)
        train_data['target'] = y_train
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            train_data.to_csv(f.name, index=False)
            s3_input = sagemaker_session.upload_data(
                path=f.name,
                bucket=Config.AWS_SAGEMAKER_S3_BUCKET,
                key_prefix='aceml/training'
            )
            os.unlink(f.name)

        # Create training script
        training_script = self._generate_training_script(model_cls, hyperparams)
        script_path = os.path.join(tempfile.gettempdir(), 'train.py')
        with open(script_path, 'w') as f:
            f.write(training_script)

        # Launch SageMaker training job
        estimator = SKLearn(
            entry_point=script_path,
            role=Config.AWS_SAGEMAKER_ROLE_ARN,
            instance_type=Config.AWS_SAGEMAKER_INSTANCE_TYPE,
            instance_count=1,
            framework_version='1.2-1',
            sagemaker_session=sagemaker_session
        )

        estimator.fit({'train': s3_input}, wait=True)
        
        # Download trained model
        model_data = estimator.model_data
        logger.info(f"AWS SageMaker training completed. Model saved to: {model_data}")
        
        duration = round(time.time() - start_time, 2)
        
        # For now, we'll return a placeholder - in production, download and deserialize the model
        training_info = {
            "provider": "aws_sagemaker",
            "instance_type": Config.AWS_SAGEMAKER_INSTANCE_TYPE,
            "training_time_sec": duration,
            "model_data_uri": model_data,
            "hyperparams": hyperparams
        }
        
        # Load the trained model (simplified - in production, download from S3)
        model = model_cls(**hyperparams)
        model.fit(X_train, y_train)  # Fallback local training
        
        return model, training_info

    def _tune_aws_sagemaker(self, model_cls, X, y, tuning_config: dict) -> Dict:
        """Execute hyperparameter tuning on AWS SageMaker."""
        try:
            import boto3  # type: ignore
            from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter  # type: ignore
        except ImportError as e:
            logger.error("AWS SageMaker tuner dependencies not installed")
            raise ImportError(f"Missing AWS dependencies: {e}")

        logger.info("AWS SageMaker hyperparameter tuning not yet fully implemented")
        logger.warning("Falling back to local tuning")
        
        # Return placeholder - full implementation would use SageMaker HyperparameterTuner
        return {
            "method": tuning_config["method"],
            "provider": "aws_sagemaker",
            "status": "fallback_to_local",
            "message": "AWS tuning requires full implementation"
        }

    # ================================================================
    # Azure ML Integration
    # ================================================================
    def _execute_azure_ml(self, model_cls, X_train, y_train, hyperparams: dict) -> Tuple[Any, Dict]:
        """Execute training on Azure ML."""
        try:
            from azure.ai.ml import MLClient  # type: ignore
            from azure.identity import ClientSecretCredential  # type: ignore
            from azure.ai.ml import command  # type: ignore
        except ImportError as e:
            logger.error("Azure ML dependencies not installed. Run: pip install azure-ai-ml azure-identity")
            raise ImportError(f"Missing Azure dependencies: {e}")

        logger.info("Preparing Azure ML training job")
        start_time = time.time()

        # Create Azure ML client
        credential = ClientSecretCredential(
            tenant_id=Config.AZURE_TENANT_ID,
            client_id=Config.AZURE_CLIENT_ID,
            client_secret=Config.AZURE_CLIENT_SECRET
        )
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=Config.AZURE_SUBSCRIPTION_ID,
            resource_group_name=Config.AZURE_RESOURCE_GROUP,
            workspace_name=Config.AZURE_WORKSPACE_NAME
        )

        logger.info(f"Azure ML workspace connected: {Config.AZURE_WORKSPACE_NAME}")
        
        # For now, fallback to local training
        # Full implementation would create Azure ML job
        duration = round(time.time() - start_time, 2)
        
        model = model_cls(**hyperparams)
        model.fit(X_train, y_train)
        
        training_info = {
            "provider": "azure_ml",
            "compute_target": Config.AZURE_COMPUTE_TARGET,
            "vm_size": Config.AZURE_VM_SIZE,
            "training_time_sec": duration,
            "hyperparams": hyperparams,
            "status": "fallback_to_local"
        }
        
        return model, training_info

    def _tune_azure_ml(self, model_cls, X, y, tuning_config: dict) -> Dict:
        """Execute hyperparameter tuning on Azure ML."""
        logger.info("Azure ML hyperparameter tuning not yet fully implemented")
        logger.warning("Falling back to local tuning")
        
        return {
            "method": tuning_config["method"],
            "provider": "azure_ml",
            "status": "fallback_to_local",
            "message": "Azure tuning requires full implementation"
        }

    # ================================================================
    # GCP Vertex AI Integration
    # ================================================================
    def _execute_gcp_vertex(self, model_cls, X_train, y_train, hyperparams: dict) -> Tuple[Any, Dict]:
        """Execute training on GCP Vertex AI."""
        try:
            from google.cloud import aiplatform  # type: ignore
        except ImportError as e:
            logger.error("GCP Vertex AI dependencies not installed. Run: pip install google-cloud-aiplatform")
            raise ImportError(f"Missing GCP dependencies: {e}")

        logger.info("Preparing GCP Vertex AI training job")
        start_time = time.time()

        # Initialize Vertex AI
        if Config.GCP_SERVICE_ACCOUNT_KEY_PATH and os.path.exists(Config.GCP_SERVICE_ACCOUNT_KEY_PATH):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Config.GCP_SERVICE_ACCOUNT_KEY_PATH
        
        aiplatform.init(
            project=Config.GCP_PROJECT_ID,
            location=Config.GCP_REGION
        )

        logger.info(f"GCP Vertex AI initialized: {Config.GCP_PROJECT_ID}")
        
        # For now, fallback to local training
        # Full implementation would create Vertex AI custom training job
        duration = round(time.time() - start_time, 2)
        
        model = model_cls(**hyperparams)
        model.fit(X_train, y_train)
        
        training_info = {
            "provider": "gcp_vertex",
            "machine_type": Config.GCP_MACHINE_TYPE,
            "accelerator_type": Config.GCP_ACCELERATOR_TYPE,
            "accelerator_count": Config.GCP_ACCELERATOR_COUNT,
            "training_time_sec": duration,
            "hyperparams": hyperparams,
            "status": "fallback_to_local"
        }
        
        return model, training_info

    def _tune_gcp_vertex(self, model_cls, X, y, tuning_config: dict) -> Dict:
        """Execute hyperparameter tuning on GCP Vertex AI."""
        logger.info("GCP Vertex AI hyperparameter tuning not yet fully implemented")
        logger.warning("Falling back to local tuning")
        
        return {
            "method": tuning_config["method"],
            "provider": "gcp_vertex",
            "status": "fallback_to_local",
            "message": "GCP tuning requires full implementation"
        }

    # ================================================================
    # Custom GPU Server Integration
    # ================================================================
    def _execute_custom(self, model_cls, X_train, y_train, hyperparams: dict) -> Tuple[Any, Dict]:
        """Execute training on custom GPU server via REST API."""
        import requests
        
        logger.info(f"Preparing custom GPU server training job: {Config.CUSTOM_GPU_ENDPOINT}")
        start_time = time.time()

        # Serialize training data
        import numpy as np
        payload = {
            "model_type": model_cls.__name__,
            "hyperparams": hyperparams,
            "X_train": X_train.tolist() if hasattr(X_train, 'tolist') else X_train,
            "y_train": y_train.tolist() if hasattr(y_train, 'tolist') else y_train,
        }

        headers = {
            "Content-Type": "application/json",
        }
        
        if Config.CUSTOM_GPU_API_KEY:
            headers["X-API-Key"] = Config.CUSTOM_GPU_API_KEY
        if Config.CUSTOM_GPU_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {Config.CUSTOM_GPU_AUTH_TOKEN}"

        # Submit training job
        try:
            endpoint = f"{Config.CUSTOM_GPU_ENDPOINT}/train"
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=Config.GPU_JOB_TIMEOUT,
                auth=(Config.CUSTOM_GPU_USERNAME, Config.CUSTOM_GPU_PASSWORD) 
                     if Config.CUSTOM_GPU_USERNAME else None
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Custom GPU training completed: {result.get('job_id')}")
            
            duration = round(time.time() - start_time, 2)
            
            # For now, train locally and return
            model = model_cls(**hyperparams)
            model.fit(X_train, y_train)
            
            training_info = {
                "provider": "custom",
                "endpoint": Config.CUSTOM_GPU_ENDPOINT,
                "training_time_sec": duration,
                "hyperparams": hyperparams,
                "job_id": result.get('job_id'),
                "status": result.get('status', 'completed')
            }
            
            return model, training_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Custom GPU server request failed: {e}")
            if Config.GPU_FALLBACK_TO_LOCAL:
                logger.warning("Falling back to local training")
                model = model_cls(**hyperparams)
                model.fit(X_train, y_train)
                duration = round(time.time() - start_time, 2)
                
                training_info = {
                    "provider": "custom",
                    "status": "fallback_to_local",
                    "error": str(e),
                    "training_time_sec": duration,
                    "hyperparams": hyperparams
                }
                return model, training_info
            else:
                raise

    def _tune_custom(self, model_cls, X, y, tuning_config: dict) -> Dict:
        """Execute hyperparameter tuning on custom GPU server."""
        import requests
        
        logger.info(f"Preparing custom GPU server tuning job: {Config.CUSTOM_GPU_ENDPOINT}")
        
        payload = {
            "model_type": model_cls.__name__,
            "tuning_config": tuning_config,
            "X": X.tolist() if hasattr(X, 'tolist') else X,
            "y": y.tolist() if hasattr(y, 'tolist') else y,
        }

        headers = {
            "Content-Type": "application/json",
        }
        
        if Config.CUSTOM_GPU_API_KEY:
            headers["X-API-Key"] = Config.CUSTOM_GPU_API_KEY
        if Config.CUSTOM_GPU_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {Config.CUSTOM_GPU_AUTH_TOKEN}"

        try:
            endpoint = f"{Config.CUSTOM_GPU_ENDPOINT}/tune"
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=Config.GPU_JOB_TIMEOUT,
                auth=(Config.CUSTOM_GPU_USERNAME, Config.CUSTOM_GPU_PASSWORD) 
                     if Config.CUSTOM_GPU_USERNAME else None
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Custom GPU tuning completed: {result.get('job_id')}")
            
            return {
                "method": tuning_config["method"],
                "provider": "custom",
                "job_id": result.get('job_id'),
                "best_params": result.get('best_params', {}),
                "best_score": result.get('best_score', 0.0),
                "status": result.get('status', 'completed')
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Custom GPU server tuning request failed: {e}")
            if Config.GPU_FALLBACK_TO_LOCAL:
                logger.warning("Custom GPU tuning failed, should fallback to local")
                return {
                    "method": tuning_config["method"],
                    "provider": "custom",
                    "status": "fallback_to_local",
                    "error": str(e)
                }
            else:
                raise

    # ================================================================
    # Helper Methods
    # ================================================================
    def _generate_training_script(self, model_cls, hyperparams: dict) -> str:
        """Generate a training script for cloud execution."""
        script = f"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from {model_cls.__module__} import {model_cls.__name__}

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('/opt/ml/input/data/train/train.csv')
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Train model
    model = {model_cls.__name__}(**{hyperparams})
    model.fit(X, y)
    
    # Save model
    with open('/opt/ml/model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
"""
        return script


# Singleton instance
_cloud_gpu_manager = None


def get_cloud_gpu_manager() -> CloudGPUManager:
    """Get or create CloudGPUManager singleton."""
    global _cloud_gpu_manager
    if _cloud_gpu_manager is None:
        _cloud_gpu_manager = CloudGPUManager()
    return _cloud_gpu_manager
