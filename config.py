"""
AceML Studio - Configuration & Constants
=========================================
Application constants, ML defaults, and thresholds.
Sensitive configuration (API keys) loaded from config.properties file.
"""

import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger("aceml.config")


def load_properties(filepath: str) -> dict:
    """Load configuration from .properties file."""
    props = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    props[key.strip()] = value.strip()
        logger.info("Loaded %d properties from %s", len(props), filepath)
    except FileNotFoundError:
        logger.warning("Properties file not found: %s — using default values", filepath)
    except Exception as e:
        logger.error("Failed to read properties file %s: %s", filepath, e)
    return props


def save_properties(filepath: str, props: dict) -> bool:
    """Save configuration to .properties file."""
    try:
        # Read existing file to preserve comments
        comments = []
        existing_keys = set()
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('#') or not stripped:
                        comments.append(line)
                    elif '=' in stripped:
                        key = stripped.split('=', 1)[0].strip()
                        existing_keys.add(key)
        
        # Write properties file
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header comments if file is new
            if not existing_keys:
                f.write("# AceML Studio Configuration Properties\n")
                f.write("# =======================================\n")
                f.write("# Auto-generated configuration file\n")
                f.write("# Last updated: " + str(os.path.getmtime(filepath) if os.path.exists(filepath) else "now") + "\n\n")
            
            # Write all properties
            for key, value in sorted(props.items()):
                f.write(f"{key}={value}\n")
        
        logger.info("Saved %d properties to %s", len(props), filepath)
        return True
    except Exception as e:
        logger.error("Failed to write properties file %s: %s", filepath, e)
        return False


# Load properties from config.properties file
_props = load_properties(os.path.join(BASE_DIR, 'config.properties'))


def _get_config(key: str, default: str = "") -> str:
    """Get configuration value from environment variable or properties file.
    
    Priority: Environment Variable > config.properties > default
    """
    return os.environ.get(key, _props.get(key, default))


class Config:
    # ──────────────────────────── Application ────────────────────────────
    APP_NAME = "AceML Studio"
    APP_VERSION = "1.0.0"
    SECRET_KEY = _get_config("SECRET_KEY", "aceml-studio-change-in-production")
    DEBUG = _get_config("DEBUG", "True").lower() in ("true", "1", "yes")

    # ──────────────────────────── File Upload ────────────────────────────
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    UPLOAD_CHUNKS_FOLDER = os.path.join(BASE_DIR, "uploads", "_chunks")
    
    # File upload limits (configurable from properties)
    MAX_FILE_UPLOAD_SIZE_MB = int(_get_config("MAX_FILE_UPLOAD_SIZE_MB", "256"))
    MAX_CONTENT_LENGTH = MAX_FILE_UPLOAD_SIZE_MB * 1024 * 1024
    ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "parquet"}

    # ──────────────────────────── Large Dataset Handling ─────────────────
    # Chunking configuration (configurable from properties)
    CHUNK_SIZE_MB = int(_get_config("CHUNK_SIZE_MB", "5"))
    CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024
    LARGE_FILE_THRESHOLD_MB = int(_get_config("LARGE_FILE_THRESHOLD_MB", "50"))
    
    # Database fallback for large files
    USE_DB_FOR_LARGE_FILES = _get_config("USE_DB_FOR_LARGE_FILES", "True").lower() in ("true", "1", "yes")
    DB_FALLBACK_THRESHOLD_MB = int(_get_config("DB_FALLBACK_THRESHOLD_MB", "500"))
    DB_PATH = os.path.join(BASE_DIR, "uploads", "large_files.db")
    
    # Memory optimization settings
    MEMORY_OPTIMIZE_THRESHOLD_MB = 100             # auto-optimize dtypes above this
    MAX_IN_MEMORY_ROWS = 5_000_000                 # sample beyond this row count
    SAMPLE_SIZE_FOR_VERY_LARGE = 1_000_000         # rows to sample from very large sets
    LOW_MEMORY_CSV_CHUNKSIZE = 50_000              # rows per chunk for low-memory CSV reads

    # ──────────────────────────── LLM Configuration ─────────────────────
    # Supported providers: "openai", "azure_openai", "anthropic", "deepseek"
    LLM_PROVIDER = _get_config("LLM_PROVIDER", "deepseek")

    # OpenAI
    OPENAI_API_KEY = _get_config("OPENAI_API_KEY", "")
    OPENAI_MODEL = _get_config("OPENAI_MODEL", "gpt-4")
    OPENAI_MAX_TOKENS = int(_get_config("OPENAI_MAX_TOKENS", "2048"))
    OPENAI_TEMPERATURE = float(_get_config("OPENAI_TEMPERATURE", "0.3"))

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = _get_config("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_KEY = _get_config("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_DEPLOYMENT = _get_config("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_OPENAI_API_VERSION = _get_config("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    # Anthropic
    ANTHROPIC_API_KEY = _get_config("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = _get_config("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

    # DeepSeek
    DEEPSEEK_API_KEY = _get_config("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL = _get_config("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_BASE_URL = _get_config("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MAX_TOKENS = int(_get_config("DEEPSEEK_MAX_TOKENS", "2048"))
    DEEPSEEK_TEMPERATURE = float(_get_config("DEEPSEEK_TEMPERATURE", "0.3"))

    # ──────────────────────────── ML Defaults ───────────────────────────
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_VALIDATION_SIZE = 0.1
    DEFAULT_CV_FOLDS = 5
    DEFAULT_RANDOM_STATE = 42

    # Supported classification models
    CLASSIFICATION_MODELS = {
        "logistic_regression": "Logistic Regression",
        "decision_tree_clf": "Decision Tree",
        "random_forest_clf": "Random Forest",
        "gradient_boosting_clf": "Gradient Boosting",
        "xgboost_clf": "XGBoost",
        "svm_clf": "SVM",
        "knn_clf": "K-Nearest Neighbors",
        "mlp_clf": "Neural Network (MLP)",
    }

    # Supported regression models
    REGRESSION_MODELS = {
        "linear_regression": "Linear Regression",
        "decision_tree_reg": "Decision Tree",
        "random_forest_reg": "Random Forest",
        "gradient_boosting_reg": "Gradient Boosting",
        "xgboost_reg": "XGBoost",
        "svr_reg": "SVR",
        "knn_reg": "K-Nearest Neighbors",
        "mlp_reg": "Neural Network (MLP)",
    }

    # ──────────────────────────── Hyperparameter Tuning ─────────────────
    MAX_TUNING_ITERATIONS = 100
    TUNING_TIMEOUT_SECONDS = 600

    # ──────────────────────────── Experiment Tracking ───────────────────
    EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

    # ──────────────────────────── Data Quality Thresholds ───────────────
    MISSING_VALUE_WARN_PCT = 5        # warn if > 5 %
    MISSING_VALUE_CRITICAL_PCT = 30   # critical if > 30 %
    DUPLICATE_WARN_PCT = 1            # warn if > 1 %
    CLASS_IMBALANCE_RATIO = 0.3       # minority < 30 % of majority
    OUTLIER_IQR_MULTIPLIER = 1.5
    HIGH_CORRELATION_THRESHOLD = 0.95
    LOW_VARIANCE_THRESHOLD = 0.01

    # ──────────────────────────── Scaling / Encoding ────────────────────
    SCALERS = ["standard", "minmax", "robust"]
    ENCODERS = ["onehot", "label", "target"]

    # ──────────────────────────── Dimensionality Reduction ──────────────
    PCA_DEFAULT_VARIANCE = 0.95       # keep 95 % variance
    MAX_PCA_COMPONENTS = 50

    # ──────────────────────────── Cloud GPU Configuration ─────────────
    # Enable cloud GPU for model training and hyperparameter tuning
    CLOUD_GPU_ENABLED = _get_config("CLOUD_GPU_ENABLED", "False").lower() in ("true", "1", "yes")
    CLOUD_GPU_PROVIDER = _get_config("CLOUD_GPU_PROVIDER", "aws_sagemaker")

    # AWS SageMaker
    AWS_REGION = _get_config("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID = _get_config("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY = _get_config("AWS_SECRET_ACCESS_KEY", "")
    AWS_SAGEMAKER_ROLE_ARN = _get_config("AWS_SAGEMAKER_ROLE_ARN", "")
    AWS_SAGEMAKER_INSTANCE_TYPE = _get_config("AWS_SAGEMAKER_INSTANCE_TYPE", "ml.p3.2xlarge")
    AWS_SAGEMAKER_S3_BUCKET = _get_config("AWS_SAGEMAKER_S3_BUCKET", "")

    # Azure ML
    AZURE_SUBSCRIPTION_ID = _get_config("AZURE_SUBSCRIPTION_ID", "")
    AZURE_RESOURCE_GROUP = _get_config("AZURE_RESOURCE_GROUP", "")
    AZURE_WORKSPACE_NAME = _get_config("AZURE_WORKSPACE_NAME", "")
    AZURE_TENANT_ID = _get_config("AZURE_TENANT_ID", "")
    AZURE_CLIENT_ID = _get_config("AZURE_CLIENT_ID", "")
    AZURE_CLIENT_SECRET = _get_config("AZURE_CLIENT_SECRET", "")
    AZURE_COMPUTE_TARGET = _get_config("AZURE_COMPUTE_TARGET", "gpu-cluster")
    AZURE_VM_SIZE = _get_config("AZURE_VM_SIZE", "Standard_NC6")

    # GCP Vertex AI
    GCP_PROJECT_ID = _get_config("GCP_PROJECT_ID", "")
    GCP_REGION = _get_config("GCP_REGION", "us-central1")
    GCP_SERVICE_ACCOUNT_KEY_PATH = _get_config("GCP_SERVICE_ACCOUNT_KEY_PATH", "")
    GCP_MACHINE_TYPE = _get_config("GCP_MACHINE_TYPE", "n1-standard-4")
    GCP_ACCELERATOR_TYPE = _get_config("GCP_ACCELERATOR_TYPE", "NVIDIA_TESLA_T4")
    GCP_ACCELERATOR_COUNT = int(_get_config("GCP_ACCELERATOR_COUNT", "1"))

    # Custom GPU Server
    CUSTOM_GPU_ENDPOINT = _get_config("CUSTOM_GPU_ENDPOINT", "")
    CUSTOM_GPU_API_KEY = _get_config("CUSTOM_GPU_API_KEY", "")
    CUSTOM_GPU_AUTH_TOKEN = _get_config("CUSTOM_GPU_AUTH_TOKEN", "")
    CUSTOM_GPU_USERNAME = _get_config("CUSTOM_GPU_USERNAME", "")
    CUSTOM_GPU_PASSWORD = _get_config("CUSTOM_GPU_PASSWORD", "")

    # GPU Training Options
    GPU_JOB_TIMEOUT = int(_get_config("GPU_JOB_TIMEOUT", "3600"))
    GPU_FALLBACK_TO_LOCAL = _get_config("GPU_FALLBACK_TO_LOCAL", "True").lower() in ("true", "1", "yes")
    GPU_MAX_RETRIES = int(_get_config("GPU_MAX_RETRIES", "3"))
