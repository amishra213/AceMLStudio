"""
AceML Studio – Flask Application
==================================
Main entry point.  Provides REST API endpoints for the full ML pipeline:
upload → quality → clean → feature-engineer → transform → reduce → train →
evaluate → tune → track experiments.  LLM-powered insights at every stage.
"""

import os
import uuid
import time
import shutil
import gc
import traceback
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

from logging_config import setup_logging, get_logger, get_recent_logs

# Initialise logging BEFORE anything else
setup_logging()

from config import Config
from ml_engine.data_loader import DataLoader
from ml_engine.data_quality import DataQualityAnalyzer
from ml_engine.data_cleaning import DataCleaner
from ml_engine.feature_engineering import FeatureEngineer
from ml_engine.transformations import DataTransformer
from ml_engine.dimensionality import DimensionalityReducer
from ml_engine.model_training import ModelTrainer
from ml_engine.evaluation import ModelEvaluator
from ml_engine.tuning import HyperparameterTuner
from ml_engine.experiment_tracker import ExperimentTracker
from ml_engine.visualizer import DataVisualizer
from ml_engine.db_storage import DataFrameDBStorage
from ml_engine.workflow_engine import WorkflowEngine
from llm_engine.analyzer import LLMAnalyzer

logger = get_logger("app")


# ────────────────────────────────────────────────────────────────────
#  Custom JSON Provider for Pandas/Numpy Types
# ────────────────────────────────────────────────────────────────────
class PandasJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that handles pandas and numpy types."""
    
    def default(self, obj):
        """Convert pandas/numpy types to JSON-serializable types."""
        if pd.isna(obj):  # Handles pd.NA, np.nan, pd.NaT, None
            return None
        if hasattr(obj, 'item'):
            # numpy scalar — convert via .item() which gives a native Python type
            val = obj.item()
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return None
            return val
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ────────────────────────────────────────────────────────────────────
#  App Init
# ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY
app.json = PandasJSONProvider(app)  # Use custom JSON provider
CORS(app)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.UPLOAD_CHUNKS_FOLDER, exist_ok=True)
os.makedirs(Config.EXPERIMENTS_DIR, exist_ok=True)

# Initialize database storage for large files
db_storage = DataFrameDBStorage(Config.DB_PATH) if Config.USE_DB_FOR_LARGE_FILES else None

logger.info("Flask app initialised: %s v%s (debug=%s)",
            Config.APP_NAME, Config.APP_VERSION, Config.DEBUG)
logger.info("Upload config: max_size=%d MB, chunk_size=%d MB, threshold=%d MB, use_db=%s",
            Config.MAX_FILE_UPLOAD_SIZE_MB, Config.CHUNK_SIZE_MB,
            Config.LARGE_FILE_THRESHOLD_MB, Config.USE_DB_FOR_LARGE_FILES)

# In-memory session store  (keyed by session_id)
# Each entry: {"df": DataFrame, "original_df": DataFrame, "models": {}, "target": str, "task": str}
DATA_STORE: dict[str, dict] = {}
tracker = ExperimentTracker()

# Active workflow engines per session
WORKFLOW_STORE: dict[str, WorkflowEngine] = {}


def _sid() -> str:
    """Get or create session id."""
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())[:12]
    return session["sid"]


def _store(sid: str | None = None) -> dict:
    sid = sid or _sid()
    if sid not in DATA_STORE:
        DATA_STORE[sid] = {}
    return DATA_STORE[sid]


def _df(sid: str | None = None) -> pd.DataFrame | None:
    """Get DataFrame for session, loading from database or disk if necessary."""
    store = _store(sid)
    
    # If DataFrame in memory, return it
    if "df" in store and store["df"] is not None:
        return store["df"]
    
    # Check if stored in database
    if store.get("stored_in_db") and db_storage is not None:
        try:
            sid_key = sid or _sid()
            logger.info("Loading DataFrame from database for session %s", sid_key)
            df_result = db_storage.retrieve_dataframe(sid_key)
            # Ensure we have a DataFrame, not an Iterator
            if df_result is not None and isinstance(df_result, pd.DataFrame):
                store["df"] = df_result
                return df_result
            else:
                logger.warning("Failed to retrieve DataFrame from database for session %s", sid_key)
        except Exception as e:
            logger.error("Error loading DataFrame from database: %s", e, exc_info=True)
    
    # Last resort: try to reload from original filepath (handles Flask reload scenario)
    # Prioritize original_filepath, fall back to regular filepath
    file_to_reload = store.get("original_filepath") or store.get("filepath")
    if file_to_reload and os.path.exists(file_to_reload):
        try:
            logger.info("Reloading DataFrame from disk: %s (session=%s)", 
                       file_to_reload, sid or _sid())
            df_result = DataLoader.load(file_to_reload)
            store["df"] = df_result
            return df_result
        except Exception as e:
            logger.error("Failed to reload DataFrame from disk (%s): %s", 
                        file_to_reload, e, exc_info=True)
    
    return store.get("df")


def _ok(data=None, message="success"):
    return jsonify({"status": "ok", "message": message, "data": data})


def _err(message, code=400):
    logger.warning("API error response [%d]: %s", code, message)
    return jsonify({"status": "error", "message": message}), code


# ────────────────────────────────────────────────────────────────────
#  Page
# ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", config=Config)


@app.route("/favicon.ico")
def favicon():
    """Handle favicon requests to prevent 404 errors."""
    from flask import send_from_directory
    import os
    favicon_path = os.path.join(app.root_path, 'static')
    # Return 204 No Content if favicon doesn't exist
    if not os.path.exists(os.path.join(favicon_path, 'favicon.ico')):
        return '', 204
    return send_from_directory(favicon_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')


# ────────────────────────────────────────────────────────────────────
#  Upload  (supports both regular and chunked uploads)
# ────────────────────────────────────────────────────────────────────

# In-flight chunked uploads: { upload_id: { "filename", "total_chunks", "received", "dir" } }
_CHUNK_UPLOADS: dict[str, dict] = {}


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Standard upload — suitable for files under the large-file threshold."""
    try:
        logger.info("Upload request received from session=%s", _sid())
        
        # Validate request
        if "file" not in request.files:
            return _err("No file part in request")
        
        f = request.files["file"]
        if not f.filename:
            return _err("No file selected")
        
        if not DataLoader.allowed_file(f.filename):
            return _err(f"Unsupported file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}")

        # Save file
        filename = f"{uuid.uuid4().hex[:8]}_{f.filename}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        try:
            f.save(filepath)
            logger.info("File saved: %s → %s", f.filename, filepath)
        except Exception as e:
            logger.error("Failed to save uploaded file: %s", e, exc_info=True)
            return _err(f"Failed to save file: {str(e)}")

        # Load and store
        try:
            result = _load_and_store(filepath, f.filename)
            return _ok(result)
        except Exception as e:
            logger.error("Failed to load uploaded file %s: %s", f.filename, e, exc_info=True)
            # Clean up the saved file on error
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            return _err(f"Failed to load file: {str(e)}")
    
    except Exception as e:
        logger.error("Unexpected error in upload_file: %s", e, exc_info=True)
        return _err(f"Upload failed: {str(e)}", 500)


# ─── chunked upload: init ─────────────────────────────────────────
@app.route("/api/upload/chunked/init", methods=["POST"])
def chunked_upload_init():
    """Client calls this before sending chunks. Returns an upload_id."""
    try:
        body = request.get_json(silent=True) or {}
        filename = body.get("filename", "")
        total_chunks = int(body.get("totalChunks", 0))
        file_size = int(body.get("fileSize", 0))

        # Validation
        if not filename:
            return _err("filename is required")
        if total_chunks <= 0:
            return _err("totalChunks must be greater than 0")
        if file_size <= 0:
            return _err("Invalid file size")
        
        # Check file size limit
        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_UPLOAD_SIZE_MB:
            return _err(f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed ({Config.MAX_FILE_UPLOAD_SIZE_MB} MB)")
        
        if not DataLoader.allowed_file(filename):
            return _err(f"Unsupported file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}")

        # Create upload session
        upload_id = uuid.uuid4().hex[:12]
        chunk_dir = os.path.join(Config.UPLOAD_CHUNKS_FOLDER, upload_id)
        
        try:
            os.makedirs(chunk_dir, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create chunk directory: %s", e, exc_info=True)
            return _err(f"Failed to initialize upload: {str(e)}", 500)

        _CHUNK_UPLOADS[upload_id] = {
            "filename": filename,
            "total_chunks": total_chunks,
            "file_size": file_size,
            "received": set(),
            "dir": chunk_dir,
            "session": _sid(),
            "created_at": time.time(),
        }
        
        logger.info("Chunked upload init: id=%s, file=%s, chunks=%d, size=%.1f MB",
                    upload_id, filename, total_chunks, file_size_mb)
        
        return _ok({
            "uploadId": upload_id,
            "chunkSize": Config.CHUNK_SIZE_BYTES,
            "message": f"Preparing to upload {filename} ({file_size_mb:.1f} MB) in {total_chunks} chunks"
        })
    
    except ValueError as e:
        logger.warning("Invalid input in chunked upload init: %s", e)
        return _err(f"Invalid input: {str(e)}", 400)
    except Exception as e:
        logger.error("Unexpected error in chunked_upload_init: %s", e, exc_info=True)
        return _err(f"Failed to initialize upload: {str(e)}", 500)


# ─── chunked upload: receive a chunk ──────────────────────────────
@app.route("/api/upload/chunked/chunk", methods=["POST"])
def chunked_upload_chunk():
    """Receive a single chunk. Expects multipart form with 'chunk' file,
    'uploadId' and 'chunkIndex' fields."""
    try:
        upload_id = request.form.get("uploadId", "")
        
        if not upload_id:
            return _err("uploadId is required")
        
        if upload_id not in _CHUNK_UPLOADS:
            return _err("Invalid or expired uploadId. Please restart the upload.", 404)

        try:
            chunk_index = int(request.form.get("chunkIndex", -1))
        except ValueError:
            return _err("Invalid chunkIndex", 400)

        meta = _CHUNK_UPLOADS[upload_id]
        
        # Validate chunk index
        if chunk_index < 0 or chunk_index >= meta["total_chunks"]:
            return _err(f"Invalid chunkIndex: {chunk_index}. Expected 0-{meta['total_chunks']-1}", 400)

        # Get chunk data
        chunk_file = request.files.get("chunk")
        if not chunk_file:
            return _err("No chunk data provided")

        # Save chunk
        chunk_path = os.path.join(meta["dir"], f"chunk_{chunk_index:06d}")
        try:
            chunk_file.save(chunk_path)
            meta["received"].add(chunk_index)
            
            logger.debug("Chunk %d/%d received for upload %s (%.1f MB total received)",
                        chunk_index + 1, meta["total_chunks"], upload_id,
                        sum(os.path.getsize(os.path.join(meta["dir"], f)) 
                            for f in os.listdir(meta["dir"])) / (1024*1024))
            
            return _ok({
                "chunkIndex": chunk_index,
                "received": len(meta["received"]),
                "totalChunks": meta["total_chunks"],
                "progress": round((len(meta["received"]) / meta["total_chunks"]) * 100, 1)
            })
        
        except IOError as e:
            logger.error("Failed to save chunk %d for upload %s: %s", chunk_index, upload_id, e, exc_info=True)
            return _err(f"Failed to save chunk: {str(e)}", 500)
    
    except Exception as e:
        logger.error("Unexpected error in chunked_upload_chunk: %s", e, exc_info=True)
        return _err(f"Failed to process chunk: {str(e)}", 500)


# ─── chunked upload: finalise ────────────────────────────────────
@app.route("/api/upload/chunked/complete", methods=["POST"])
def chunked_upload_complete():
    """After all chunks are uploaded, reassemble and load the file."""
    try:
        body = request.get_json(silent=True) or {}
        upload_id = body.get("uploadId", "")
        
        if not upload_id:
            return _err("uploadId is required")
        
        if upload_id not in _CHUNK_UPLOADS:
            return _err("Invalid or expired uploadId", 404)

        meta = _CHUNK_UPLOADS[upload_id]
        
        # Verify all chunks received
        if len(meta["received"]) < meta["total_chunks"]:
            missing = meta["total_chunks"] - len(meta["received"])
            missing_indices = [i for i in range(meta["total_chunks"]) if i not in meta["received"]]
            logger.warning("Upload incomplete for %s: %d chunks missing: %s",
                          upload_id, missing, missing_indices[:10])
            return _err(f"Upload incomplete — {missing} chunk(s) still missing. Please retry the upload.")

        # Reassemble chunks into a single file
        safe_name = f"{uuid.uuid4().hex[:8]}_{meta['filename']}"
        final_path = os.path.join(Config.UPLOAD_FOLDER, safe_name)

        logger.info("Reassembling %d chunks → %s (%.1f MB)",
                    meta["total_chunks"], final_path, meta["file_size"] / (1024*1024))
        
        try:
            with open(final_path, "wb") as out:
                for i in range(meta["total_chunks"]):
                    chunk_path = os.path.join(meta["dir"], f"chunk_{i:06d}")
                    
                    if not os.path.exists(chunk_path):
                        raise FileNotFoundError(f"Chunk {i} not found at {chunk_path}")
                    
                    try:
                        with open(chunk_path, "rb") as cp:
                            shutil.copyfileobj(cp, out)
                    except IOError as e:
                        raise IOError(f"Failed to read chunk {i}: {str(e)}")
            
            logger.info("Successfully reassembled file: %s", final_path)
            
        except Exception as e:
            logger.error("Chunk reassembly failed for %s: %s", upload_id, e, exc_info=True)
            _cleanup_chunks(upload_id)
            # Clean up partial file
            try:
                if os.path.exists(final_path):
                    os.remove(final_path)
            except:
                pass
            return _err(f"Failed to reassemble file: {str(e)}", 500)

        # Clean up chunk directory
        _cleanup_chunks(upload_id)

        # Load the reassembled file
        try:
            result = _load_and_store(final_path, meta["filename"])
            logger.info("Large file upload complete: %s", meta["filename"])
            return _ok(result)
            
        except Exception as e:
            logger.error("Failed to load reassembled file %s: %s",
                        meta["filename"], e, exc_info=True)
            # Clean up the reassembled file on error
            try:
                if os.path.exists(final_path):
                    os.remove(final_path)
            except:
                pass
            return _err(f"Failed to load file: {str(e)}", 500)
    
    except Exception as e:
        logger.error("Unexpected error in chunked_upload_complete: %s", e, exc_info=True)
        return _err(f"Failed to complete upload: {str(e)}", 500)


# ─── chunked upload: cancel ──────────────────────────────────────
@app.route("/api/upload/chunked/cancel", methods=["POST"])
def chunked_upload_cancel():
    """Cancel an in-progress chunked upload."""
    try:
        body = request.get_json(silent=True) or {}
        upload_id = body.get("uploadId", "")
        
        if upload_id:
            _cleanup_chunks(upload_id)
            logger.info("Chunked upload cancelled: %s", upload_id)
            return _ok({"cancelled": True, "uploadId": upload_id})
        else:
            return _err("uploadId is required")
    
    except Exception as e:
        logger.error("Error cancelling upload: %s", e, exc_info=True)
        return _err(f"Failed to cancel upload: {str(e)}", 500)


def _cleanup_chunks(upload_id: str):
    """Remove chunk directory and metadata. Safe to call even if upload doesn't exist."""
    try:
        meta = _CHUNK_UPLOADS.pop(upload_id, None)
        if meta and os.path.isdir(meta["dir"]):
            shutil.rmtree(meta["dir"], ignore_errors=True)
            logger.debug("Cleaned up chunks for upload %s", upload_id)
        gc.collect()
    except Exception as e:
        logger.warning("Error during chunk cleanup for %s: %s", upload_id, e)


def _load_and_store(filepath: str, original_filename: str) -> dict:
    """Common helper: load a file into a DataFrame, store it, return API data."""
    try:
        t0 = time.time()
        file_mb = DataLoader._file_size_mb(filepath)
        logger.info("Loading file: %s (%.1f MB)", original_filename, file_mb)

        # Load the DataFrame
        try:
            df = DataLoader.load(filepath)
        except Exception as e:
            logger.error("DataLoader.load failed for %s: %s", original_filename, e, exc_info=True)
            raise ValueError(f"Failed to parse file: {str(e)}")
        
        elapsed = time.time() - t0
        mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
        logger.info("File loaded in %.2fs — shape=%s, mem=%.1f MB",
                    elapsed, df.shape, mem_mb)

        store = _store()
        sid = _sid()
        
        # Determine storage strategy
        use_db = (Config.USE_DB_FOR_LARGE_FILES and 
                 db_storage and 
                 mem_mb >= Config.DB_FALLBACK_THRESHOLD_MB)
        
        if use_db:
            # Store in database to save RAM
            logger.info("Large dataset (%.1f MB) exceeds DB threshold (%.1f MB) — storing in database",
                       mem_mb, Config.DB_FALLBACK_THRESHOLD_MB)
            
            try:
                # db_storage is guaranteed to be not None here due to use_db check
                if db_storage is not None and db_storage.store_dataframe(sid, df, original_filename):
                    # Keep only a sample in memory for quick preview
                    sample_size = min(1000, len(df))
                    store["df"] = df.head(sample_size).copy()
                    store["stored_in_db"] = True
                    store["db_full_rows"] = len(df)
                    logger.info("Stored full dataset in DB, keeping %d-row sample in memory", sample_size)
                else:
                    raise Exception("Database storage failed")
                    
            except Exception as e:
                logger.warning("Failed to store in database, falling back to in-memory: %s", e)
                # Fall back to in-memory storage
                store["df"] = df
                store["stored_in_db"] = False
        else:
            # Store in memory
            store["df"] = df
            store["stored_in_db"] = False
            
            # For moderately large datasets, keep only one copy to save memory
            if mem_mb < Config.MEMORY_OPTIMIZE_THRESHOLD_MB:
                store["original_df"] = df.copy()
            else:
                # Store only the path so the original can be reloaded on demand
                logger.info("Large dataset (%.1f MB) — skipping in-memory backup, "
                           "storing path for on-demand reload", mem_mb)
                store["original_df"] = None
                store["original_filepath"] = filepath
        
        # Store metadata
        store["filepath"] = filepath
        store["filename"] = original_filename
        store["models"] = {}

        # Get info and preview
        try:
            info = DataLoader.get_info(df)
            preview = DataLoader.get_preview(df)
        except Exception as e:
            logger.error("Failed to generate info/preview: %s", e, exc_info=True)
            raise ValueError(f"Failed to process file: {str(e)}")

        # Add large dataset warnings to info
        info["large_dataset"] = bool(file_mb >= Config.LARGE_FILE_THRESHOLD_MB)
        info["file_size_mb"] = round(file_mb, 1)
        info["memory_usage_mb"] = round(mem_mb, 1)
        info["stored_in_db"] = bool(use_db)
        info["sampled"] = bool(len(df) < info.get("_original_rows", len(df) + 1))
        
        if use_db and "db_full_rows" in store:
            info["db_full_rows"] = store["db_full_rows"]
            info["preview_note"] = f"Showing preview only. Full dataset ({store['db_full_rows']:,} rows) stored in database."

        logger.info("Upload complete: %s — %d rows × %d cols, %.1f MB, db_storage=%s",
                   original_filename, info["rows"], info["columns"], file_mb, use_db)
        
        return {"info": info, "preview": preview, "filename": original_filename}
    
    except Exception as e:
        logger.error("Error in _load_and_store for %s: %s", original_filename, e, exc_info=True)
        raise


# ────────────────────────────────────────────────────────────────────
#  Data Preview & Info
# ────────────────────────────────────────────────────────────────────
@app.route("/api/session/check")
def session_check():
    """Check if there's an active session with loaded data."""
    df = _df()
    store = _store()
    
    if df is None:
        return _ok({
            "has_data": False,
            "session_id": _sid()
        })
    
    # Session has data - return info to restore UI state
    try:
        info = DataLoader.get_info(df)
        preview = DataLoader.get_preview(df, 20)
        filename = store.get("filename", "Dataset")
        
        return _ok({
            "has_data": True,
            "session_id": _sid(),
            "info": info,
            "preview": preview,
            "filename": filename
        })
    except Exception as e:
        logger.error("Failed to get session info: %s", e, exc_info=True)
        return _ok({
            "has_data": True,
            "session_id": _sid(),
            "error": str(e)
        })


# ────────────────────────────────────────────────────────────────────
#  Saved Datasets Management
# ────────────────────────────────────────────────────────────────────
@app.route("/api/datasets/save", methods=["POST"])
def save_dataset():
    """Save the current dataset permanently with a name."""
    df = _df()
    if df is None:
        return _err("No dataset loaded to save", 404)
    
    if not db_storage:
        return _err("Dataset persistence is not enabled", 503)
    
    body = request.get_json(silent=True) or {}
    dataset_name = body.get("name", "").strip()
    description = body.get("description", "").strip()
    tags = body.get("tags", [])
    
    if not dataset_name:
        return _err("Dataset name is required")
    
    # Get original filename
    store = _store()
    original_filename = store.get("filename", store.get("original_filename", ""))
    
    try:
        success = db_storage.save_dataset(
            dataset_name=dataset_name,
            df=df,
            description=description,
            original_filename=original_filename,
            tags=tags
        )
        
        if success:
            logger.info("Saved dataset '%s' with %d rows, %d cols (session=%s)",
                       dataset_name, len(df), len(df.columns), _sid())
            return _ok({
                "message": f"Dataset '{dataset_name}' saved successfully",
                "dataset_name": dataset_name,
                "rows": len(df),
                "columns": len(df.columns)
            })
        else:
            return _err("Failed to save dataset")
    
    except Exception as e:
        logger.error("Error saving dataset: %s", e, exc_info=True)
        return _err(f"Failed to save dataset: {str(e)}")


@app.route("/api/datasets/list", methods=["GET"])
def list_datasets():
    """List all saved datasets."""
    if not db_storage:
        return _err("Dataset persistence is not enabled", 503)
    
    try:
        search = request.args.get("search", None)
        tags = request.args.getlist("tags")
        
        datasets = db_storage.list_datasets(search=search, tags=tags if tags else None)
        
        logger.info("Listed %d saved datasets (session=%s)", len(datasets), _sid())
        return _ok({
            "datasets": datasets,
            "count": len(datasets)
        })
    
    except Exception as e:
        logger.error("Error listing datasets: %s", e, exc_info=True)
        return _err(f"Failed to list datasets: {str(e)}")


@app.route("/api/datasets/load/<dataset_name>", methods=["POST"])
def load_dataset(dataset_name: str):
    """Load a saved dataset into the current session."""
    if not db_storage:
        return _err("Dataset persistence is not enabled", 503)
    
    try:
        df = db_storage.load_dataset(dataset_name)
        
        if df is None:
            return _err(f"Dataset '{dataset_name}' not found", 404)
        
        # Store in session
        store = _store()
        store["df"] = df
        store["original_df"] = df.copy()
        store["filename"] = dataset_name
        store["loaded_from_saved"] = True
        
        # Get dataset info
        info = DataLoader.get_info(df)
        preview = DataLoader.get_preview(df)
        dataset_info = db_storage.get_dataset_info(dataset_name)
        
        logger.info("Loaded saved dataset '%s': %d rows x %d cols (session=%s)",
                   dataset_name, len(df), len(df.columns), _sid())
        
        return _ok({
            "message": f"Dataset '{dataset_name}' loaded successfully",
            "filename": dataset_name,
            "info": info,
            "preview": preview,
            "dataset_info": dataset_info
        })
    
    except Exception as e:
        logger.error("Error loading dataset '%s': %s", dataset_name, e, exc_info=True)
        return _err(f"Failed to load dataset: {str(e)}")


@app.route("/api/datasets/delete/<dataset_name>", methods=["DELETE"])
def delete_dataset(dataset_name: str):
    """Delete a saved dataset."""
    if not db_storage:
        return _err("Dataset persistence is not enabled", 503)
    
    try:
        success = db_storage.delete_dataset(dataset_name)
        
        if success:
            logger.info("Deleted dataset '%s' (session=%s)", dataset_name, _sid())
            return _ok({
                "message": f"Dataset '{dataset_name}' deleted successfully"
            })
        else:
            return _err(f"Dataset '{dataset_name}' not found", 404)
    
    except Exception as e:
        logger.error("Error deleting dataset '%s': %s", dataset_name, e, exc_info=True)
        return _err(f"Failed to delete dataset: {str(e)}")


@app.route("/api/datasets/info/<dataset_name>", methods=["GET"])
def get_dataset_info(dataset_name: str):
    """Get detailed information about a saved dataset."""
    if not db_storage:
        return _err("Dataset persistence is not enabled", 503)
    
    try:
        info = db_storage.get_dataset_info(dataset_name)
        
        if info is None:
            return _err(f"Dataset '{dataset_name}' not found", 404)
        
        return _ok(info)
    
    except Exception as e:
        logger.error("Error getting dataset info for '%s': %s", dataset_name, e, exc_info=True)
        return _err(f"Failed to get dataset info: {str(e)}")


# ────────────────────────────────────────────────────────────────────
#  Data Preview & Info
# ────────────────────────────────────────────────────────────────────
@app.route("/api/data/preview")
def data_preview():
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    n = request.args.get("n", 50, type=int)
    return _ok(DataLoader.get_preview(df, n))


@app.route("/api/data/info")
def data_info():
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    return _ok(DataLoader.get_info(df))


@app.route("/api/data/statistics")
def data_statistics():
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    return _ok(DataLoader.get_statistics(df))


@app.route("/api/data/columns")
def data_columns():
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    info = DataLoader.get_info(df)
    return _ok({
        "columns": info["column_names"],
        "numeric": info["numeric_columns"],
        "categorical": info["categorical_columns"],
        "datetime": info["datetime_columns"],
    })


@app.route("/api/data/current")
def data_current():
    """Get current data state with full info and preview after all transformations."""
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    
    try:
        info = DataLoader.get_info(df)
        preview = DataLoader.get_preview(df, 20)  # Show more rows for inspection
        
        # Include basic quality metrics
        quality_summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_percentage": float(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        }
        
        return _ok({
            "info": info,
            "preview": preview,
            "quality_summary": quality_summary,
            "column_details": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].notna().sum()),
                    "null": int(df[col].isna().sum()),
                    "null_pct": float(df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0
                }
                for col in df.columns
            ]
        })
    except Exception as e:
        logger.error("Failed to get current data state: %s", e, exc_info=True)
        return _err(f"Failed to get current data: {e}")


# ────────────────────────────────────────────────────────────────────
#  Data Quality
# ────────────────────────────────────────────────────────────────────
@app.route("/api/data/quality", methods=["POST"])
def data_quality():
    """Analyze data quality and return comprehensive report."""
    try:
        logger.info("Data quality analysis requested (session=%s)", _sid())
        df = _df()
        if df is None:
            return _err("No dataset loaded", 404)
        
        body = request.get_json(silent=True) or {}
        target = body.get("target") or _store().get("target")
        logger.debug("Quality analysis target=%s, shape=%s", target, df.shape)
        
        t0 = time.time()
        try:
            analyzer = DataQualityAnalyzer(df, target_column=target)
            report = analyzer.full_report()
        except Exception as e:
            logger.error("Quality analysis failed: %s", e, exc_info=True)
            return _err(f"Quality analysis failed: {str(e)}", 500)
        
        logger.info("Quality analysis complete in %.2fs — score=%s, issues=%d",
                    time.time() - t0, report.get('quality_score'), len(report.get('issues', [])))
        _store()["quality_report"] = report
        return _ok(report)
    
    except Exception as e:
        logger.error("Unexpected error in data_quality: %s", e, exc_info=True)
        return _err(f"Quality analysis failed: {str(e)}", 500)


# ────────────────────────────────────────────────────────────────────
#  Data Cleaning
# ────────────────────────────────────────────────────────────────────
@app.route("/api/data/clean", methods=["POST"])
def data_clean():
    """Apply data cleaning operations."""
    try:
        logger.info("Data cleaning requested (session=%s)", _sid())
        df = _df()
        if df is None:
            return _err("No dataset loaded", 404)
        
        body = request.get_json(silent=True) or {}
        operations = body.get("operations", [])
        if not operations:
            return _err("No operations specified")

        logger.info("Applying %d cleaning operations, initial shape=%s", len(operations), df.shape)
        t0 = time.time()
        
        try:
            cleaned, log = DataCleaner.apply_operations(df, operations)
        except Exception as e:
            logger.error("Data cleaning failed: %s", e, exc_info=True)
            return _err(f"Cleaning failed: {str(e)}", 500)
        
        logger.info("Cleaning complete in %.2fs — final shape=%s", time.time() - t0, cleaned.shape)
        for entry in log:
            logger.debug("  Cleaning log: %s", entry)
        
        _store()["df"] = cleaned
        
        try:
            info = DataLoader.get_info(cleaned)
            preview = DataLoader.get_preview(cleaned)
        except Exception as e:
            logger.error("Failed to generate info/preview after cleaning: %s", e, exc_info=True)
            return _err(f"Cleaning succeeded but failed to generate preview: {str(e)}", 500)
        
        return _ok({"log": log, "info": info, "preview": preview})
    
    except Exception as e:
        logger.error("Unexpected error in data_clean: %s", e, exc_info=True)
        return _err(f"Cleaning failed: {str(e)}", 500)


# ────────────────────────────────────────────────────────────────────
#  Feature Engineering
# ────────────────────────────────────────────────────────────────────
@app.route("/api/data/feature-engineer", methods=["POST"])
def feature_engineer():
    """Apply feature engineering operations."""
    try:
        logger.info("Feature engineering requested (session=%s)", _sid())
        df = _df()
        if df is None:
            return _err("No dataset loaded", 404)
        
        body = request.get_json(silent=True) or {}
        operations = body.get("operations", [])
        if not operations:
            return _err("No operations specified")

        logger.info("Applying %d feature-engineering operations, initial cols=%d", len(operations), df.shape[1])
        t0 = time.time()
        
        try:
            result, log = FeatureEngineer.apply_operations(df, operations)
        except Exception as e:
            logger.error("Feature engineering failed: %s", e, exc_info=True)
            return _err(f"Feature engineering failed: {str(e)}", 500)
        
        logger.info("Feature engineering complete in %.2fs — cols %d→%d",
                    time.time() - t0, df.shape[1], result.shape[1])
        for entry in log:
            logger.debug("  FE log: %s", entry)
        
        _store()["df"] = result
        
        try:
            info = DataLoader.get_info(result)
            preview = DataLoader.get_preview(result)
        except Exception as e:
            logger.error("Failed to generate info/preview after FE: %s", e, exc_info=True)
            return _err(f"Feature engineering succeeded but failed to generate preview: {str(e)}", 500)
        
        return _ok({"log": log, "info": info, "preview": preview})
    
    except Exception as e:
        logger.error("Unexpected error in feature_engineer: %s", e, exc_info=True)
        return _err(f"Feature engineering failed: {str(e)}", 500)


# ────────────────────────────────────────────────────────────────────
#  Transformations
# ────────────────────────────────────────────────────────────────────
@app.route("/api/data/transform", methods=["POST"])
def transform():
    logger.info("Data transformation requested (session=%s)", _sid())
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    body = request.get_json(silent=True) or {}
    operations = body.get("operations", [])
    if not operations:
        return _err("No operations specified")

    logger.info("Applying %d transform operations, shape=%s", len(operations), df.shape)
    t0 = time.time()
    result, log = DataTransformer.apply_operations(df, operations)
    logger.info("Transformation complete in %.2fs — shape=%s", time.time() - t0, result.shape)
    for entry in log:
        logger.debug("  Transform log: %s", entry)
    _store()["df"] = result
    info = DataLoader.get_info(result)
    return _ok({"log": log, "info": info, "preview": DataLoader.get_preview(result)})


# ────────────────────────────────────────────────────────────────────
#  Dimensionality Reduction
# ────────────────────────────────────────────────────────────────────
@app.route("/api/data/reduce", methods=["POST"])
def reduce_dimensions():
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    body = request.get_json(silent=True) or {}
    method = body.get("method", "pca")
    params = body.get("params", {})

    try:
        logger.info("Dimensionality reduction: method=%s, params=%s, shape=%s", method, params, df.shape)
        t0 = time.time()
        if method == "pca":
            result, info = DimensionalityReducer.pca_reduce(df, **params)
        elif method == "variance_threshold":
            result, info = DimensionalityReducer.variance_threshold(df, **params)
        elif method == "correlation_filter":
            result, info = DimensionalityReducer.correlation_filter(df, **params)
        elif method == "feature_importance":
            target = params.pop("target", _store().get("target"))
            task = params.pop("task", _store().get("task", "classification"))
            if not target:
                return _err("Target column required for feature importance")
            result, info = DimensionalityReducer.feature_importance_selection(df, target, task, **params)
        else:
            return _err(f"Unknown method: {method}")

        logger.info("Dimensionality reduction complete in %.2fs — cols %d→%d",
                    time.time() - t0, df.shape[1], result.shape[1])
        _store()["df"] = result
        data_info = DataLoader.get_info(result)
        return _ok({"reduction_info": info, "data_info": data_info, "preview": DataLoader.get_preview(result)})
    except Exception as e:
        logger.error("Dimensionality reduction failed: %s", e, exc_info=True)
        return _err(f"Dimensionality reduction failed: {e}")


# ────────────────────────────────────────────────────────────────────
#  Set Target & Task
# ────────────────────────────────────────────────────────────────────
@app.route("/api/config/target", methods=["POST", "GET"])
def set_target():
    if request.method == "GET":
        # Return current target/task configuration
        store = _store()
        return _ok({
            "target": store.get("target"),
            "task": store.get("task", "classification")
        })
    
    # POST: Set target and task
    body = request.get_json(silent=True) or {}
    target = body.get("target")
    task = body.get("task", "classification")
    if not target:
        return _err("target is required")
    _store()["target"] = target
    _store()["task"] = task
    logger.info("Target set: target=%s, task=%s (session=%s)", target, task, _sid())
    return _ok({"target": target, "task": task})


@app.route("/api/config/upload")
def get_upload_config():
    """Return upload configuration for frontend."""
    return _ok({
        "maxFileSizeBytes": Config.MAX_CONTENT_LENGTH,
        "maxFileSizeMB": Config.MAX_FILE_UPLOAD_SIZE_MB,
        "chunkSizeBytes": Config.CHUNK_SIZE_BYTES,
        "chunkSizeMB": Config.CHUNK_SIZE_MB,
        "largeFileThresholdBytes": Config.LARGE_FILE_THRESHOLD_MB * 1024 * 1024,
        "largeFileThresholdMB": Config.LARGE_FILE_THRESHOLD_MB,
        "useDbForLargeFiles": Config.USE_DB_FOR_LARGE_FILES,
        "dbFallbackThresholdMB": Config.DB_FALLBACK_THRESHOLD_MB,
    })


# ────────────────────────────────────────────────────────────────────
#  Data Finalize — mark data as ready for training
# ────────────────────────────────────────────────────────────────────
@app.route("/api/data/finalize", methods=["POST"])
def finalize_data():
    """Save the current processed DataFrame state and mark it as ready for training."""
    sid = _sid()
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)

    try:
        # Save the processed dataset if db_storage is available
        saved_name = None
        if db_storage:
            import time as _time
            dataset_name = f"finalized_{int(_time.time())}"
            original = DATA_STORE.get(sid, {}).get("filename", "unknown")
            db_storage.save_dataset(
                dataset_name=dataset_name,
                df=df,
                description="Auto-saved after AI Workflow completion",
                original_filename=original,
                tags=["finalized", "workflow-output"],
            )
            saved_name = dataset_name
            logger.info("Finalized data saved as '%s' (session=%s)", dataset_name, sid)

        # Compute final quality report
        store = DATA_STORE.get(sid, {})
        target = store.get("target")
        analyzer = DataQualityAnalyzer(df, target_column=target)
        quality = analyzer.full_report()

        logger.info("Data finalized: %d rows × %d cols, quality=%s (session=%s)",
                     df.shape[0], df.shape[1], quality.get("quality_score", "?"), sid)

        return _ok({
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "quality_score": quality.get("quality_score", 0),
            "saved_name": saved_name,
            "message": "Data finalized and ready for model training",
        })

    except Exception as e:
        logger.error("Error finalizing data: %s", e, exc_info=True)
        return _err(f"Failed to finalize data: {str(e)}")


# ────────────────────────────────────────────────────────────────────
#  Model Training
# ────────────────────────────────────────────────────────────────────

@app.route("/api/training/auto-fix", methods=["POST"])
def training_auto_fix():
    """AI-assisted auto-fix for training errors (high cardinality, etc.)"""
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    
    body = request.get_json(silent=True) or {}
    error_type = body.get("error_type")
    problem_columns = body.get("problem_columns", [])
    target = body.get("target")
    
    if not error_type or not problem_columns:
        return _err("Missing error_type or problem_columns", 400)
    
    try:
        from ml_engine.transformations import DataTransformer
        
        if error_type == "high_cardinality":
            logger.info("Auto-fixing high cardinality: applying label encoding to %d columns", len(problem_columns))
            
            # Filter to columns that actually exist
            existing_cols = [col for col in problem_columns if col in df.columns]
            if not existing_cols:
                return _err("Problem columns not found in dataset", 400)
            
            # Apply label encoding
            df_fixed, mappings = DataTransformer.label_encode(df, existing_cols)
            
            # Update session dataframe
            store = _store()
            store["df"] = df_fixed
            
            logger.info("Auto-fix successful: label encoded %d columns", len(existing_cols))
            return jsonify({
                "status": "ok",
                "message": f"Successfully applied Label Encoding to {len(existing_cols)} column(s): {', '.join(existing_cols[:5])}",
                "columns_fixed": existing_cols,
                "new_shape": {"rows": df_fixed.shape[0], "columns": df_fixed.shape[1]},
                "encodings": {col: len(mapping.classes_) if hasattr(mapping, 'classes_') else str(mapping) for col, mapping in mappings.items()}
            })

        elif error_type == "datetime_columns":
            logger.info("Auto-fixing datetime columns: converting %d column(s) to numeric features", len(problem_columns))
            
            existing_cols = [col for col in problem_columns if col in df.columns]
            if not existing_cols:
                return _err("Problem columns not found in dataset", 400)
            
            df_fixed = df.copy()
            converted = []
            for col in existing_cols:
                try:
                    dt_series = pd.to_datetime(df_fixed[col], errors="coerce")
                    epoch = pd.Timestamp("1970-01-01")
                    df_fixed[f"{col}_year"] = dt_series.dt.year.astype("float64")
                    df_fixed[f"{col}_month"] = dt_series.dt.month.astype("float64")
                    df_fixed[f"{col}_day"] = dt_series.dt.day.astype("float64")
                    df_fixed[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float64")
                    df_fixed[f"{col}_ordinal"] = (dt_series - epoch).dt.total_seconds().astype("float64")
                    df_fixed = df_fixed.drop(columns=[col])
                    converted.append(col)
                    logger.info("Converted datetime '%s' → 5 numeric features", col)
                except Exception as dt_err:
                    logger.warning("Failed to convert '%s', dropping: %s", col, dt_err)
                    df_fixed = df_fixed.drop(columns=[col])
                    converted.append(col)
            
            store = _store()
            store["df"] = df_fixed
            
            return jsonify({
                "status": "ok",
                "message": f"Converted {len(converted)} datetime column(s) to numeric features: {', '.join(converted[:5])}",
                "columns_fixed": converted,
                "new_shape": {"rows": df_fixed.shape[0], "columns": df_fixed.shape[1]}
            })

        elif error_type == "non_numeric_columns":
            logger.info("Auto-fixing non-numeric columns: converting/dropping %d column(s)", len(problem_columns))
            
            existing_cols = [col for col in problem_columns if col in df.columns]
            if not existing_cols:
                return _err("Problem columns not found in dataset", 400)
            
            df_fixed = df.copy()
            fixed_cols = []
            for col in existing_cols:
                dtype = df_fixed[col].dtype
                # Try datetime conversion first
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    try:
                        dt_series = pd.to_datetime(df_fixed[col], errors="coerce")
                        epoch = pd.Timestamp("1970-01-01")
                        df_fixed[f"{col}_ordinal"] = (dt_series - epoch).dt.total_seconds().astype("float64")
                        df_fixed = df_fixed.drop(columns=[col])
                        fixed_cols.append(col)
                        continue
                    except Exception:
                        pass
                # Try label encoding for object columns
                if dtype == "object" or isinstance(dtype, pd.CategoricalDtype):
                    try:
                        df_fixed, _ = DataTransformer.label_encode(df_fixed, [col])
                        fixed_cols.append(col)
                        continue
                    except Exception:
                        pass
                # Last resort: drop the column
                df_fixed = df_fixed.drop(columns=[col])
                fixed_cols.append(col)
            
            store = _store()
            store["df"] = df_fixed
            
            return jsonify({
                "status": "ok",
                "message": f"Fixed {len(fixed_cols)} non-numeric column(s): {', '.join(fixed_cols[:5])}",
                "columns_fixed": fixed_cols,
                "new_shape": {"rows": df_fixed.shape[0], "columns": df_fixed.shape[1]}
            })

        else:
            return _err(f"Unknown error_type: {error_type}", 400)
            
    except Exception as e:
        logger.error("Auto-fix failed: %s", e, exc_info=True)
        return _err(f"Auto-fix failed: {str(e)}")

@app.route("/api/model/train", methods=["POST"])
def train_model():
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    
    # CRITICAL: Early validation of dataset dimensions
    MAX_FEATURES_BEFORE_TRAINING = 10000
    MAX_ROWS_FOR_SAFETY = 10_000_000  # 10M rows
    
    if df.shape[1] > MAX_FEATURES_BEFORE_TRAINING:
        error_msg = (
            f"Your dataset has {df.shape[1]:,} columns, which exceeds the safe limit of "
            f"{MAX_FEATURES_BEFORE_TRAINING:,} for training. This often happens when high-cardinality "
            f"columns were incorrectly one-hot encoded. "
            f"\n\nSOLUTION: Click 'Reset' button, re-upload your data, and use 'Label Encoding' "
            f"or 'Target Encoding' instead of one-hot encoding for high-cardinality columns. "
            f"Or use dimensionality reduction techniques."
        )
        logger.error("Training blocked - too many features: %s", error_msg)
        return _err(error_msg, 400)
    
    if df.shape[0] > MAX_ROWS_FOR_SAFETY:
        logger.warning("Large dataset detected: %d rows - training may be slow", df.shape[0])
    
    store = _store()
    body = request.get_json(silent=True) or {}

    target = body.get("target") or store.get("target")
    task = body.get("task") or store.get("task", "classification")
    model_keys = body.get("models", [])
    hyperparams = body.get("hyperparams", {})
    test_size = body.get("test_size", Config.DEFAULT_TEST_SIZE)
    val_size = body.get("val_size", Config.DEFAULT_VALIDATION_SIZE)

    if not target or target not in df.columns:
        return _err("Valid target column required")
    if not model_keys:
        return _err("At least one model must be selected")

    # Validate that model_keys are compatible with the task
    available_models = ModelTrainer.get_available_models(task)
    invalid_models = [m for m in model_keys if m not in available_models]
    valid_model_keys = [m for m in model_keys if m in available_models]
    
    if invalid_models:
        logger.warning("Filtering out %d invalid models for task '%s': %s", 
                      len(invalid_models), task, invalid_models)
    
    if not valid_model_keys:
        return _err(f"None of the selected models are valid for {task}. Available models: {list(available_models.keys())}")
    
    model_keys = valid_model_keys  # Use only valid models

    logger.info("Training request: models=%s, task=%s, target=%s, shape=%s (session=%s)",
                model_keys, task, target, df.shape, _sid())

    store["target"] = target
    store["task"] = task

    # Prepare numeric-only features
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # CRITICAL: Validate feature count before processing
    MAX_FEATURES_ALLOWED = 10000  # Reasonable limit for most ML tasks
    if X.shape[1] > MAX_FEATURES_ALLOWED:
        error_msg = (
            f"Dataset has {X.shape[1]:,} features (columns) which exceeds the maximum "
            f"allowed limit of {MAX_FEATURES_ALLOWED:,}. This would cause memory issues during training. "
            f"Please use dimensionality reduction (PCA, feature selection) to reduce the number of features first, "
            f"or reset and re-upload your data with proper preprocessing."
        )
        logger.error("Training blocked: %s", error_msg)
        return _err(error_msg, 400)

    # AUTO-CONVERT datetime columns to numeric features
    datetime_cols = X.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if datetime_cols:
        logger.info("Auto-converting %d datetime column(s) to numeric features: %s", len(datetime_cols), datetime_cols[:5])
        for col in datetime_cols:
            try:
                dt_series = pd.to_datetime(X[col], errors="coerce")
                # Extract useful numeric features from datetime
                X[f"{col}_year"] = dt_series.dt.year.astype("float64")
                X[f"{col}_month"] = dt_series.dt.month.astype("float64")
                X[f"{col}_day"] = dt_series.dt.day.astype("float64")
                X[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float64")
                # Also create ordinal (days since epoch) for continuous representation
                epoch = pd.Timestamp("1970-01-01")
                X[f"{col}_ordinal"] = (dt_series - epoch).dt.total_seconds().astype("float64")
                # Drop original datetime column
                X = X.drop(columns=[col])
                logger.info("Converted datetime '%s' → 5 numeric features (year, month, day, dayofweek, ordinal)", col)
            except Exception as dt_err:
                logger.warning("Failed to convert datetime column '%s', dropping it: %s", col, dt_err)
                X = X.drop(columns=[col])

    # Also check for object columns that look like dates (string dates)
    for col in X.select_dtypes(include=["object"]).columns:
        try:
            sample = X[col].dropna().head(20)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() > len(sample) * 0.8:  # >80% parse as dates
                    logger.info("Detected string-date column '%s', converting to numeric features", col)
                    dt_series = pd.to_datetime(X[col], errors="coerce")
                    epoch = pd.Timestamp("1970-01-01")
                    X[f"{col}_year"] = dt_series.dt.year.astype("float64")
                    X[f"{col}_month"] = dt_series.dt.month.astype("float64")
                    X[f"{col}_day"] = dt_series.dt.day.astype("float64")
                    X[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float64")
                    X[f"{col}_ordinal"] = (dt_series - epoch).dt.total_seconds().astype("float64")
                    X = X.drop(columns=[col])
        except Exception:
            pass  # Not a date column, skip

    # Auto-encode remaining object columns WITH CARDINALITY CHECKS
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        logger.debug("Checking %d categorical columns for encoding safety", len(obj_cols))
        
        # Validate cardinality before encoding
        high_cardinality_cols = []
        safe_to_encode = []
        total_new_features = 0
        
        for col in obj_cols:
            n_unique = X[col].nunique()
            if n_unique > 100:
                high_cardinality_cols.append(f"{col} ({n_unique:,} unique values)")
                logger.warning("Column '%s' has %d unique values - too high for one-hot encoding", col, n_unique)
            elif n_unique > 1:  # Skip constant columns
                safe_to_encode.append(col)
                total_new_features += n_unique - 1  # -1 for drop_first
        
        if high_cardinality_cols:
            # Extract column names without unique counts
            problem_cols = [col.split(' (')[0] for col in high_cardinality_cols]
            error_msg = (
                f"Cannot train: {len(high_cardinality_cols)} categorical column(s) have too many unique values "
                f"for safe one-hot encoding: {', '.join(high_cardinality_cols[:3])}{'...' if len(high_cardinality_cols) > 3 else ''}. "
                f"\n\nThis would create {total_new_features + sum(int(c.split('(')[1].split()[0].replace(',', '')) for c in high_cardinality_cols):,}+ features and cause memory overflow. "
                f"\n\nSOLUTION: Use AI Auto-Fix to apply Label Encoding, or manually fix in Transformations page."
            )
            logger.error("Training blocked due to high cardinality: %s", error_msg)
            return jsonify({
                "status": "error",
                "message": error_msg,
                "error_type": "high_cardinality",
                "fixable": True,
                "problem_columns": problem_cols,
                "suggestion": "Apply Label Encoding to high-cardinality columns"
            }), 400
        
        # Check total features after encoding
        final_feature_count = X.shape[1] - len(safe_to_encode) + total_new_features
        if final_feature_count > MAX_FEATURES_ALLOWED:
            error_msg = (
                f"One-hot encoding would create {final_feature_count:,} features (current: {X.shape[1]}, "
                f"adding {total_new_features:,} from encoding), exceeding limit of {MAX_FEATURES_ALLOWED:,}. "
                f"Use Label Encoding instead or reduce dimensions first."
            )
            logger.error("Training blocked: %s", error_msg)
            return _err(error_msg, 400)
        
        if safe_to_encode:
            logger.info("Auto-encoding %d safe categorical columns: %s → %d new features",
                       len(safe_to_encode), safe_to_encode[:5], total_new_features)
            X = pd.get_dummies(X, columns=safe_to_encode, drop_first=True)
        else:
            logger.info("No categorical columns to encode (all are numeric or already encoded)")

    # FINAL SAFETY CHECK: ensure no non-numeric columns remain
    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        logger.warning("Dropping %d remaining non-numeric columns before training: %s", len(non_numeric), non_numeric[:10])
        X = X.drop(columns=non_numeric)

    X = X.fillna(0)
    logger.info("Final training feature matrix: %d rows × %d features", X.shape[0], X.shape[1])

    # Split
    t0 = time.time()
    split = ModelTrainer.split_data(
        pd.concat([X, y], axis=1), target,
        test_size=test_size, val_size=val_size
    )
    logger.debug("Data split: train=%d, val=%d, test=%d",
                 split['split_info']['train_size'],
                 split['split_info']['val_size'],
                 split['split_info']['test_size'])

    results = ModelTrainer.train_multiple(
        model_keys, task,
        split["X_train"], split["y_train"],
        split["X_val"], split["y_val"]
    )
    duration = time.time() - t0

    # Store models and split for evaluation
    for r in results:
        if r.get("status") == "success" and "model" in r:
            store.setdefault("models", {})[r["model_key"]] = r.pop("model")
            logger.info("Model trained: %s — train=%.4f, val=%.4f (%.2fs)",
                        r['model_key'], r.get('train_score', 0), r.get('val_score', 0),
                        r.get('training_time_sec', 0))
        elif r.get("status") == "error":
            logger.error("Model training failed: %s — %s", r.get('model_key'), r.get('error'))

    store["split"] = split
    store["feature_names"] = X.columns.tolist()
    logger.info("Training pipeline complete in %.2fs for %d model(s)", duration, len(model_keys))

    return _ok({"results": results, "split_info": split["split_info"]})


# ────────────────────────────────────────────────────────────────────
#  Evaluation
# ────────────────────────────────────────────────────────────────────
@app.route("/api/model/evaluate", methods=["POST"])
def evaluate_model():
    store = _store()
    models = store.get("models", {})
    split = store.get("split")
    task = store.get("task", "classification")
    feature_names = store.get("feature_names", [])

    if not models or not split:
        return _err("No trained models or split data")

    body = request.get_json(silent=True) or {}
    model_key = body.get("model_key")
    dataset = body.get("dataset", "test")  # train, val, test

    keys_to_eval = [model_key] if model_key else list(models.keys())
    logger.info("Evaluation request: models=%s, dataset=%s, task=%s (session=%s)",
                keys_to_eval, dataset, task, _sid())

    X = split[f"X_{dataset}"]
    y = split[f"y_{dataset}"]

    results = {}
    for key in keys_to_eval:
        model = models.get(key)
        if model is None:
            results[key] = {"error": "Model not found"}
            logger.warning("Evaluation: model '%s' not found in store", key)
            continue
        t0 = time.time()
        results[key] = ModelEvaluator.evaluate(model, X, y, task, feature_names)
        logger.info("Evaluated '%s' on %s set in %.2fs", key, dataset, time.time() - t0)

    # Store evaluation results for ranking
    store["eval_results"] = results
    store["eval_dataset"] = dataset

    return _ok(results)


# ────────────────────────────────────────────────────────────────────
#  Ranked Model Results
# ────────────────────────────────────────────────────────────────────
@app.route("/api/model/ranked-results", methods=["GET"])
def get_ranked_results():
    """Get trained models ranked by test performance."""
    store = _store()
    task = store.get("task", "classification")
    eval_results = store.get("eval_results", {})
    
    if not eval_results:
        return _err("No evaluation results available. Please evaluate models first.")
    
    # Extract primary metric based on task
    primary_metric = "accuracy" if task == "classification" else "r2_score"
    
    # Build ranked list
    ranked = []
    for model_key, result in eval_results.items():
        if "error" in result:
            continue
        
        metrics = result.get("metrics", {})
        score = metrics.get(primary_metric)
        
        if score is not None:
            ranked.append({
                "model_key": model_key,
                "test_score": score,
                "task": task,
                "metrics": metrics,
                "primary_metric": primary_metric
            })
    
    # Sort by test score (descending - higher is better for both accuracy and r2)
    ranked.sort(key=lambda x: x["test_score"], reverse=True)
    
    # Add rank numbers
    for idx, model in enumerate(ranked, 1):
        model["rank"] = idx
    
    logger.info("Ranked results: %d models, best=%s (%.4f)",
                len(ranked), ranked[0]["model_key"] if ranked else "none",
                ranked[0]["test_score"] if ranked else 0)
    
    return _ok({
        "ranked_models": ranked,
        "task": task,
        "primary_metric": primary_metric
    })


# ────────────────────────────────────────────────────────────────────
#  Cross Validation
# ────────────────────────────────────────────────────────────────────
@app.route("/api/model/cross-validate", methods=["POST"])
def cross_validate_model():
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    store = _store()
    body = request.get_json(silent=True) or {}

    target = body.get("target") or store.get("target")
    task = body.get("task") or store.get("task", "classification")
    model_key = body.get("model_key")
    cv = body.get("cv", Config.DEFAULT_CV_FOLDS)

    if not target or not model_key:
        return _err("target and model_key required")

    logger.info("Cross-validation: model=%s, cv=%d, task=%s (session=%s)",
                model_key, cv, task, _sid())

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # AUTO-CONVERT datetime columns to numeric features
    datetime_cols = X.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if datetime_cols:
        logger.info("Auto-converting %d datetime column(s) to numeric features: %s", len(datetime_cols), datetime_cols[:5])
        for col in datetime_cols:
            try:
                dt_series = pd.to_datetime(X[col], errors="coerce")
                X[f"{col}_year"] = dt_series.dt.year.astype("float64")
                X[f"{col}_month"] = dt_series.dt.month.astype("float64")
                X[f"{col}_day"] = dt_series.dt.day.astype("float64")
                X[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float64")
                epoch = pd.Timestamp("1970-01-01")
                X[f"{col}_ordinal"] = (dt_series - epoch).dt.total_seconds().astype("float64")
                X = X.drop(columns=[col])
                logger.info("Converted datetime '%s' → 5 numeric features", col)
            except Exception as dt_err:
                logger.warning("Failed to convert datetime column '%s', dropping it: %s", col, dt_err)
                X = X.drop(columns=[col])

    # Check for object columns that look like dates
    for col in X.select_dtypes(include=["object"]).columns:
        try:
            sample = X[col].dropna().head(20)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() > len(sample) * 0.8:
                    logger.info("Detected string-date column '%s', converting to numeric features", col)
                    dt_series = pd.to_datetime(X[col], errors="coerce")
                    epoch = pd.Timestamp("1970-01-01")
                    X[f"{col}_year"] = dt_series.dt.year.astype("float64")
                    X[f"{col}_month"] = dt_series.dt.month.astype("float64")
                    X[f"{col}_day"] = dt_series.dt.day.astype("float64")
                    X[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float64")
                    X[f"{col}_ordinal"] = (dt_series - epoch).dt.total_seconds().astype("float64")
                    X = X.drop(columns=[col])
        except Exception:
            pass

    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)
    X = X.fillna(0)

    t0 = time.time()
    result = ModelTrainer.cross_validate(model_key, task, X, y, cv=cv)
    logger.info("Cross-validation complete in %.2fs — mean_test=%.4f±%.4f",
                time.time() - t0, result.get('mean_test_score', 0), result.get('std_test_score', 0))
    return _ok(result)


# ────────────────────────────────────────────────────────────────────
#  Hyperparameter Tuning
# ────────────────────────────────────────────────────────────────────
@app.route("/api/model/tune", methods=["POST"])
def tune_model():
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    store = _store()
    body = request.get_json(silent=True) or {}

    target = body.get("target") or store.get("target")
    task = body.get("task") or store.get("task", "classification")
    model_key = body.get("model_key")
    method = body.get("method", "random_search")
    cv = body.get("cv", Config.DEFAULT_CV_FOLDS)
    n_iter = body.get("n_iter", 20)

    if not target or not model_key:
        return _err("target and model_key required")

    logger.info("Hyperparameter tuning: model=%s, method=%s, n_iter=%d, cv=%d (session=%s)",
                model_key, method, n_iter, cv, _sid())

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # AUTO-CONVERT datetime columns to numeric features
    datetime_cols = X.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if datetime_cols:
        logger.info("Auto-converting %d datetime column(s) to numeric features: %s", len(datetime_cols), datetime_cols[:5])
        for col in datetime_cols:
            try:
                dt_series = pd.to_datetime(X[col], errors="coerce")
                # Extract useful numeric features from datetime
                X[f"{col}_year"] = dt_series.dt.year.astype("float64")
                X[f"{col}_month"] = dt_series.dt.month.astype("float64")
                X[f"{col}_day"] = dt_series.dt.day.astype("float64")
                X[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float64")
                # Also create ordinal (days since epoch) for continuous representation
                epoch = pd.Timestamp("1970-01-01")
                X[f"{col}_ordinal"] = (dt_series - epoch).dt.total_seconds().astype("float64")
                # Drop original datetime column
                X = X.drop(columns=[col])
                logger.info("Converted datetime '%s' → 5 numeric features (year, month, day, dayofweek, ordinal)", col)
            except Exception as dt_err:
                logger.warning("Failed to convert datetime column '%s', dropping it: %s", col, dt_err)
                X = X.drop(columns=[col])

    # Also check for object columns that look like dates (string dates)
    for col in X.select_dtypes(include=["object"]).columns:
        try:
            sample = X[col].dropna().head(20)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() > len(sample) * 0.8:  # >80% parse as dates
                    logger.info("Detected string-date column '%s', converting to numeric features", col)
                    dt_series = pd.to_datetime(X[col], errors="coerce")
                    epoch = pd.Timestamp("1970-01-01")
                    X[f"{col}_year"] = dt_series.dt.year.astype("float64")
                    X[f"{col}_month"] = dt_series.dt.month.astype("float64")
                    X[f"{col}_day"] = dt_series.dt.day.astype("float64")
                    X[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float64")
                    X[f"{col}_ordinal"] = (dt_series - epoch).dt.total_seconds().astype("float64")
                    X = X.drop(columns=[col])
        except Exception:
            pass  # Not a date column, skip

    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)
    X = X.fillna(0)

    try:
        t0 = time.time()
        if method == "grid_search":
            result = HyperparameterTuner.grid_search(model_key, task, X, y, cv=cv)
        elif method == "random_search":
            result = HyperparameterTuner.random_search(model_key, task, X, y, n_iter=n_iter, cv=cv)
        elif method == "optuna":
            result = HyperparameterTuner.optuna_search(model_key, task, X, y, n_trials=n_iter, cv=cv)
        else:
            return _err(f"Unknown tuning method: {method}")

        logger.info("Tuning complete in %.2fs — best_score=%.4f, best_params=%s",
                    time.time() - t0, result.get('best_score', 0), result.get('best_params'))

        # Store best model
        best_model = result.pop("best_model", None)
        if best_model:
            store.setdefault("models", {})[f"{model_key}_tuned"] = best_model
            logger.info("Tuned model stored as '%s_tuned'", model_key)

        return _ok(result)
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        logger.error("Tuning failed for model=%s, method=%s: %s", model_key, method, e, exc_info=True)
        
        # Provide detailed error info for AI analysis
        return jsonify({
            "status": "error",
            "message": f"Tuning failed: {error_msg}",
            "error_details": {
                "model_key": model_key,
                "method": method,
                "task": task,
                "error_type": error_type,
                "error_message": error_msg,
                "n_iter": n_iter,
                "cv_folds": cv
            },
            "ai_analysis_available": True
        }), 400


# ────────────────────────────────────────────────────────────────────
#  Experiments
# ────────────────────────────────────────────────────────────────────
@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    logger.debug("Listing experiments")
    return _ok(tracker.list_experiments())


@app.route("/api/experiments", methods=["POST"])
def save_experiment():
    store = _store()
    body = request.get_json(silent=True) or {}
    name = body.get("name", "Unnamed Experiment")
    model_key = body.get("model_key", "")
    metrics = body.get("metrics", {})
    hyperparams = body.get("hyperparams", {})
    notes = body.get("notes", "")

    task = store.get("task", "classification")
    df = _df()
    data_info = DataLoader.get_info(df) if df is not None else {}

    record = tracker.save_experiment(
        name=name, task=task, model_key=model_key,
        hyperparams=hyperparams, metrics=metrics,
        data_info=data_info, notes=notes,
    )
    logger.info("Experiment saved: id=%s, name='%s', model=%s", record['id'], name, model_key)
    return _ok(record)


@app.route("/api/experiments/<exp_id>")
def get_experiment(exp_id):
    exp = tracker.get_experiment(exp_id)
    if not exp:
        return _err("Experiment not found", 404)
    return _ok(exp)


@app.route("/api/experiments/<exp_id>", methods=["DELETE"])
def delete_experiment(exp_id):
    tracker.delete_experiment(exp_id)
    logger.info("Experiment deleted: id=%s", exp_id)
    return _ok(message="Deleted")


@app.route("/api/experiments/compare", methods=["POST"])
def compare_experiments():
    body = request.get_json(silent=True) or {}
    ids = body.get("ids", [])
    if len(ids) < 2:
        return _err("At least 2 experiment IDs required")
    return _ok(tracker.compare_experiments(ids))


# ────────────────────────────────────────────────────────────────────
#  LLM AI Insights
# ────────────────────────────────────────────────────────────────────
@app.route("/api/llm/analyze-quality", methods=["POST"])
def llm_analyze_quality():
    logger.info("LLM quality analysis requested (session=%s)", _sid())
    store = _store()
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    report = store.get("quality_report")
    if not report:
        analyzer = DataQualityAnalyzer(df, store.get("target"))
        report = analyzer.full_report()
    summary = DataLoader.get_info(df)
    response = LLMAnalyzer.analyze_data_quality(report, summary)
    logger.info("LLM quality analysis complete (response_len=%d)", len(response))
    return _ok({"analysis": response})


@app.route("/api/llm/suggest-cleaning", methods=["POST"])
def llm_suggest_cleaning():
    logger.info("LLM cleaning suggestions requested (session=%s)", _sid())
    store = _store()
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    report = store.get("quality_report", {})
    issues = report.get("issues", [])
    col_info = DataLoader.get_info(df)
    response = LLMAnalyzer.suggest_cleaning(issues, col_info)
    logger.info("LLM cleaning suggestions complete (response_len=%d)", len(response))
    return _ok({"suggestions": response})


@app.route("/api/llm/suggest-features", methods=["POST"])
def llm_suggest_features():
    logger.info("LLM feature suggestions requested (session=%s)", _sid())
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    col_info = DataLoader.get_info(df)
    target = _store().get("target")
    response = LLMAnalyzer.suggest_features(col_info, target)
    logger.info("LLM feature suggestions complete (response_len=%d)", len(response))
    return _ok({"suggestions": response})


@app.route("/api/llm/explain-evaluation", methods=["POST"])
def llm_explain_evaluation():
    logger.info("LLM evaluation explanation requested (session=%s)", _sid())
    body = request.get_json(silent=True) or {}
    eval_results = body.get("results", {})
    task = body.get("task") or _store().get("task", "classification")
    response = LLMAnalyzer.explain_evaluation(eval_results, task)
    logger.info("LLM evaluation explanation complete (response_len=%d)", len(response))
    return _ok({"explanation": response})


@app.route("/api/llm/suggest-tuning", methods=["POST"])
def llm_suggest_tuning():
    logger.info("LLM tuning suggestions requested (session=%s)", _sid())
    body = request.get_json(silent=True) or {}
    model_key = body.get("model_key", "")
    params = body.get("params", {})
    metrics = body.get("metrics", {})
    response = LLMAnalyzer.suggest_tuning(model_key, params, metrics)
    logger.info("LLM tuning suggestions complete for model=%s (response_len=%d)", model_key, len(response))
    return _ok({"suggestions": response})


@app.route("/api/llm/analyze-tuning-error", methods=["POST"])
def llm_analyze_tuning_error():
    """Analyze a tuning failure and provide detailed guidance."""
    logger.info("LLM tuning error analysis requested (session=%s)", _sid())
    body = request.get_json(silent=True) or {}
    
    error_details = body.get("error_details", {})
    model_key = error_details.get("model_key", "unknown")
    method = error_details.get("method", "unknown")
    error_message = error_details.get("error_message", "")
    task = error_details.get("task", "classification")
    
    df = _df()
    data_info = DataLoader.get_info(df) if df is not None else {}
    
    response = LLMAnalyzer.analyze_tuning_error(
        model_key, method, error_message, data_info, task
    )
    
    logger.info("LLM tuning error analysis complete for model=%s (response_len=%d)", 
                model_key, len(response))
    return _ok({"analysis": response})


@app.route("/api/llm/ask", methods=["POST"])
def llm_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    if not question:
        return _err("Question required")
    logger.info("LLM free-form question (session=%s, len=%d)", _sid(), len(question))
    context = {}
    df = _df()
    if df is not None:
        context["data_info"] = DataLoader.get_info(df)
        context["target"] = _store().get("target")
        context["task"] = _store().get("task")
    response = LLMAnalyzer.general_question(question, context)
    logger.info("LLM answer delivered (response_len=%d)", len(response))
    return _ok({"answer": response})


# ────────────────────────────────────────────────────────────────────
#  Iterative Workflow  –  LLM-driven data preparation pipeline
# ────────────────────────────────────────────────────────────────────

@app.route("/api/workflow/start", methods=["POST"])
def workflow_start():
    """Start a new iterative workflow.
    Body: { target, task, objectives?, max_iterations?, auto_approve?, enabled_steps? }
    """
    try:
        df = _df()
        if df is None:
            return _err("No dataset loaded", 404)

        store = _store()
        body = request.get_json(silent=True) or {}

        target = body.get("target") or store.get("target")
        task = body.get("task") or store.get("task", "classification")
        objectives = body.get("objectives", "")
        max_iterations = int(body.get("max_iterations", 5))
        auto_approve = body.get("auto_approve", True)
        enabled_steps = body.get("enabled_steps")

        if not target or target not in df.columns:
            return _err("Valid target column is required")

        # Save target/task in session
        store["target"] = target
        store["task"] = task

        sid = _sid()
        logger.info("Workflow start: session=%s, target=%s, task=%s, objectives='%s'",
                     sid, target, task, objectives[:100])

        engine = WorkflowEngine(
            df=df,
            target_column=target,
            task_type=task,
            objectives=objectives,
            max_iterations=max_iterations,
            auto_approve=auto_approve,
            enabled_steps=enabled_steps,
        )

        WORKFLOW_STORE[sid] = engine
        state = engine.start()
        return _ok(state)

    except Exception as e:
        logger.error("Workflow start failed: %s", e, exc_info=True)
        return _err(f"Workflow start failed: {e}", 500)


@app.route("/api/workflow/status")
def workflow_status():
    """Get current workflow state."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _ok({"status": "none", "message": "No active workflow"})
    return _ok(engine.get_state())


@app.route("/api/workflow/step", methods=["POST"])
def workflow_run_step():
    """Execute the next pending step in the current iteration."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    try:
        state = engine.run_next_step()
        # Sync the working DataFrame back to session
        _store()["df"] = engine.get_dataframe()
        return _ok(state)
    except Exception as e:
        logger.error("Workflow step failed: %s", e, exc_info=True)
        return _err(f"Step execution failed: {e}", 500)


@app.route("/api/workflow/run-iteration", methods=["POST"])
def workflow_run_iteration():
    """Execute all pending steps in the current iteration."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    try:
        state = engine.run_full_iteration()
        _store()["df"] = engine.get_dataframe()
        return _ok(state)
    except Exception as e:
        logger.error("Workflow iteration failed: %s", e, exc_info=True)
        return _err(f"Iteration failed: {e}", 500)


@app.route("/api/workflow/continue", methods=["POST"])
def workflow_continue():
    """Continue to the next iteration (plan + start)."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    try:
        state = engine.continue_workflow()
        return _ok(state)
    except Exception as e:
        logger.error("Workflow continue failed: %s", e, exc_info=True)
        return _err(f"Continue failed: {e}", 500)


@app.route("/api/workflow/run-all", methods=["POST"])
def workflow_run_all():
    """Run the entire workflow end-to-end (all iterations)."""
    try:
        df = _df()
        if df is None:
            return _err("No dataset loaded", 404)

        store = _store()
        body = request.get_json(silent=True) or {}

        target = body.get("target") or store.get("target")
        task = body.get("task") or store.get("task", "classification")
        objectives = body.get("objectives", "")
        max_iterations = int(body.get("max_iterations", 5))
        enabled_steps = body.get("enabled_steps")

        if not target or target not in df.columns:
            return _err("Valid target column is required")

        store["target"] = target
        store["task"] = task

        sid = _sid()
        logger.info("Workflow run-all: session=%s, target=%s, max_iter=%d",
                     sid, target, max_iterations)

        engine = WorkflowEngine(
            df=df,
            target_column=target,
            task_type=task,
            objectives=objectives,
            max_iterations=max_iterations,
            auto_approve=True,
            enabled_steps=enabled_steps,
        )

        WORKFLOW_STORE[sid] = engine
        state = engine.run_all()
        _store()["df"] = engine.get_dataframe()

        logger.info("Workflow run-all complete: status=%s, iterations=%d",
                     state.get("status"), state.get("current_iteration", 0))
        return _ok(state)

    except Exception as e:
        logger.error("Workflow run-all failed: %s", e, exc_info=True)
        return _err(f"Workflow failed: {e}", 500)


@app.route("/api/workflow/approve", methods=["POST"])
def workflow_approve():
    """Approve the current iteration plan and resume."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)
    return _ok(engine.approve_iteration())


@app.route("/api/workflow/abort", methods=["POST"])
def workflow_abort():
    """Abort the current workflow."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    state = engine.abort()
    logger.info("Workflow aborted: session=%s", sid)
    return _ok(state)


@app.route("/api/workflow/skip-step", methods=["POST"])
def workflow_skip_step():
    """Skip a specific step."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    body = request.get_json(silent=True) or {}
    step_id = body.get("step_id")
    if not step_id:
        return _err("step_id is required")

    return _ok(engine.skip_step(step_id))


@app.route("/api/workflow/defer-step", methods=["POST"])
def workflow_defer_step():
    """Defer a pending step — move it to the deferred queue for later."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    body = request.get_json(silent=True) or {}
    step_id = body.get("step_id")
    if not step_id:
        return _err("step_id is required")

    return _ok(engine.defer_step(step_id))


@app.route("/api/workflow/recall-step", methods=["POST"])
def workflow_recall_step():
    """Move a deferred step back into the active queue."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    body = request.get_json(silent=True) or {}
    step_id = body.get("step_id")
    position = body.get("position")  # optional insert position
    if not step_id:
        return _err("step_id is required")

    return _ok(engine.recall_step(step_id, position))


@app.route("/api/workflow/run-step-by-id", methods=["POST"])
def workflow_run_step_by_id():
    """Execute a specific step by its ID (any status — pending, deferred, completed, failed)."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    body = request.get_json(silent=True) or {}
    step_id = body.get("step_id")
    if not step_id:
        return _err("step_id is required")

    try:
        state = engine.run_step_by_id(step_id)
        _store()["df"] = engine.get_dataframe()
        return _ok(state)
    except Exception as e:
        logger.error("Workflow run-step-by-id failed: %s", e, exc_info=True)
        return _err(f"Step execution failed: {e}", 500)


@app.route("/api/workflow/reorder-steps", methods=["POST"])
def workflow_reorder_steps():
    """Move a pending step up or down in the active queue.
    Body: { step_id, direction: 'up' | 'down' }"""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    body = request.get_json(silent=True) or {}
    step_id = body.get("step_id")
    direction = body.get("direction", "up")
    if not step_id:
        return _err("step_id is required")
    if direction not in ("up", "down"):
        return _err("direction must be 'up' or 'down'")

    return _ok(engine.reorder_steps(step_id, direction))


@app.route("/api/workflow/rerun-step", methods=["POST"])
def workflow_rerun_step():
    """Re-run a completed or failed step. Restores DataFrame to pre-step state."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    body = request.get_json(silent=True) or {}
    step_id = body.get("step_id")
    if not step_id:
        return _err("step_id is required")

    try:
        state = engine.rerun_step(step_id)
        _store()["df"] = engine.get_dataframe()
        return _ok(state)
    except Exception as e:
        logger.error("Workflow rerun-step failed: %s", e, exc_info=True)
        return _err(f"Step re-run failed: {e}", 500)


@app.route("/api/workflow/finish-iteration", methods=["POST"])
def workflow_finish_iteration():
    """Force-finish the current iteration even if deferred steps remain.
    Deferred steps are left as-is (skipped for this iteration)."""
    sid = _sid()
    engine = WORKFLOW_STORE.get(sid)
    if not engine:
        return _err("No active workflow", 404)

    try:
        import time as _time
        iteration_dict = engine._current_iteration()
        if not iteration_dict:
            return _err("No active iteration")

        # Mark any remaining deferred steps as skipped for this iteration
        for s in iteration_dict.get("deferred_steps", []):
            s["status"] = "skipped"
            iteration_dict["steps"].append(s)
        iteration_dict["deferred_steps"] = []

        if not iteration_dict.get("completed_at"):
            iteration_dict["completed_at"] = _time.time()
        engine._evaluate_iteration(iteration_dict)
        engine.state.updated_at = _time.time()

        _store()["df"] = engine.get_dataframe()
        return _ok(engine.get_state())
    except Exception as e:
        logger.error("Workflow finish-iteration failed: %s", e, exc_info=True)
        return _err(f"Finish iteration failed: {e}", 500)


@app.route("/api/workflow/reset", methods=["POST"])
def workflow_reset():
    """Clear the active workflow for this session."""
    sid = _sid()
    WORKFLOW_STORE.pop(sid, None)
    logger.info("Workflow reset: session=%s", sid)
    return _ok(message="Workflow reset")


# ────────────────────────────────────────────────────────────────────
#  Chat  –  multi-turn conversational AI with rich pipeline context
# ────────────────────────────────────────────────────────────────────
# Per-session chat history  { sid: [ {role, content}, ... ] }
CHAT_HISTORY: dict[str, list] = {}
MAX_CHAT_HISTORY = 40  # keep last N messages per session


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Multi-turn chat endpoint.  Accepts:
      - message (str): the user's latest message
      - include_context (dict, optional): which context to attach
            { logs: bool, data_summary: bool, tuning: bool, evaluation: bool, user_context: str }
    Returns: { reply: str, history_length: int }
    """
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return _err("message is required")

    sid = _sid()
    include = body.get("include_context", {})
    logger.info("Chat message (session=%s, len=%d, ctx_keys=%s)", sid, len(message), list(include.keys()))

    # ── Build rich context dict ──────────────────────────────────
    context: dict = {}
    store = _store()
    df = _df()

    if include.get("data_summary") and df is not None:
        info = DataLoader.get_info(df)
        context["data_summary"] = {
            "rows": info.get("rows"),
            "columns": info.get("columns"),
            "column_names": info.get("column_names"),
            "dtypes": info.get("dtypes"),
            "numeric_columns": info.get("numeric_columns"),
            "categorical_columns": info.get("categorical_columns"),
            "target": store.get("target"),
            "task": store.get("task"),
        }

    if include.get("logs"):
        context["recent_logs"] = get_recent_logs(n=30, min_level="INFO")

    if include.get("tuning") and store.get("split"):
        # Gather info about trained models and any tuning results
        tuning_ctx: dict = {}
        models = store.get("models", {})
        if models:
            tuning_ctx["trained_models"] = list(models.keys())
        context["tuning_parameters"] = tuning_ctx

    if include.get("evaluation"):
        # Include the last evaluation results if stored by frontend
        eval_data = body.get("evaluation_snapshot")
        if eval_data:
            context["evaluation_results"] = eval_data

    user_context = include.get("user_context")
    if user_context and isinstance(user_context, str) and user_context.strip():
        context["user_provided_context"] = user_context.strip()

    # ── Conversation history ─────────────────────────────────────
    history = CHAT_HISTORY.setdefault(sid, [])
    history.append({"role": "user", "content": message})

    # Trim to keep context window manageable
    if len(history) > MAX_CHAT_HISTORY:
        history[:] = history[-MAX_CHAT_HISTORY:]

    # ── Call LLM ─────────────────────────────────────────────────
    try:
        reply = LLMAnalyzer.chat(history, context if context else None)
    except Exception as e:
        logger.error("Chat LLM call failed: %s", e, exc_info=True)
        reply = f"Sorry, I encountered an error: {e}"

    history.append({"role": "assistant", "content": reply})
    logger.info("Chat reply delivered (session=%s, reply_len=%d, history=%d)",
                sid, len(reply), len(history))

    return _ok({"reply": reply, "history_length": len(history)})


@app.route("/api/chat/history")
def chat_history():
    """Return the current session's chat history."""
    sid = _sid()
    history = CHAT_HISTORY.get(sid, [])
    return _ok({"messages": history})


@app.route("/api/chat/clear", methods=["POST"])
def chat_clear():
    """Clear chat history for the current session."""
    sid = _sid()
    CHAT_HISTORY.pop(sid, None)
    logger.info("Chat history cleared (session=%s)", sid)
    return _ok(message="Chat history cleared")


@app.route("/api/logs/recent")
def recent_logs():
    """Return recent in-memory log entries for display in the chat panel."""
    n = request.args.get("n", 30, type=int)
    level = request.args.get("level", "INFO")
    return _ok(get_recent_logs(n=n, min_level=level))


# ────────────────────────────────────────────────────────────────────
#  Reset
# ────────────────────────────────────────────────────────────────────
@app.route("/api/reset", methods=["POST"])
def reset_session():
    sid = _sid()
    DATA_STORE.pop(sid, None)
    CHAT_HISTORY.pop(sid, None)
    WORKFLOW_STORE.pop(sid, None)
    logger.info("Session reset: %s", sid)
    return _ok(message="Session reset")


# ────────────────────────────────────────────────────────────────────
#  Available Models (categorized)
# ────────────────────────────────────────────────────────────────────
@app.route("/api/models")
def available_models():
    task = request.args.get("task", "classification")
    return _ok(ModelTrainer.get_available_models(task))


@app.route("/api/models/categorized")
def available_models_categorized():
    """Return models grouped by category with metadata."""
    task = request.args.get("task", "classification")
    categories = Config.MODEL_CATEGORIES.get(task, {})
    return _ok(categories)


@app.route("/api/models/recommend", methods=["POST"])
def recommend_models():
    """AI-based model recommendation based on dataset characteristics."""
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)

    body = request.get_json(silent=True) or {}
    target = body.get("target") or _store().get("target")
    task = body.get("task") or _store().get("task", "classification")

    try:
        n_rows, n_cols = df.shape
        n_numeric = len(df.select_dtypes(include=["number"]).columns)
        n_cat = len(df.select_dtypes(include=["object", "category"]).columns)
        n_classes = df[target].nunique() if target and target in df.columns else 0
        has_missing = int(df.isnull().any().sum())
        is_imbalanced = False

        if task == "classification" and target and target in df.columns:
            vc = df[target].value_counts(normalize=True)
            is_imbalanced = vc.min() < 0.15

        # Rule-based recommendations
        recommendations = []
        all_models = Config.MODEL_CATEGORIES.get(task, {})

        if task in ("classification", "regression"):
            # Always recommend: strong baseline
            if task == "classification":
                recommendations.append({
                    "key": "random_forest_clf",
                    "name": "Random Forest",
                    "reason": "Robust, handles mixed features well, minimal tuning needed. Best starting model for most datasets.",
                    "priority": 1,
                })
                if n_rows > 1000:
                    recommendations.append({
                        "key": "xgboost_clf",
                        "name": "XGBoost",
                        "reason": f"Industry standard for tabular data. Your {n_rows:,} rows + {n_cols} features is a good fit.",
                        "priority": 2,
                    })
                if n_rows < 5000 and n_cols < 50:
                    recommendations.append({
                        "key": "logistic_regression",
                        "name": "Logistic Regression",
                        "reason": "Fast, interpretable baseline. Good for understanding feature importance.",
                        "priority": 3,
                    })
                if n_rows > 5000 and n_cols > 20:
                    recommendations.append({
                        "key": "gradient_boosting_clf",
                        "name": "Gradient Boosting",
                        "reason": "Strong performance on complex data with many features.",
                        "priority": 3,
                    })
                if n_rows < 2000:
                    recommendations.append({
                        "key": "knn_clf",
                        "name": "K-Nearest Neighbors",
                        "reason": "Simple and effective for smaller datasets. Easy to interpret.",
                        "priority": 4,
                    })
            else:  # regression
                recommendations.append({
                    "key": "random_forest_reg",
                    "name": "Random Forest",
                    "reason": "Robust baseline. Handles non-linear relationships without extensive tuning.",
                    "priority": 1,
                })
                if n_rows > 1000:
                    recommendations.append({
                        "key": "xgboost_reg",
                        "name": "XGBoost",
                        "reason": f"Excellent for tabular regression. {n_rows:,} rows is sufficient for XGBoost.",
                        "priority": 2,
                    })
                recommendations.append({
                    "key": "linear_regression",
                    "name": "Linear Regression",
                    "reason": "Fast interpretable baseline. Reveals linear relationships.",
                    "priority": 3,
                })
                if n_rows > 5000:
                    recommendations.append({
                        "key": "gradient_boosting_reg",
                        "name": "Gradient Boosting",
                        "reason": "Powerful for complex patterns. Often top performer on tabular data.",
                        "priority": 3,
                    })

        elif task == "unsupervised":
            recommendations.append({
                "key": "kmeans",
                "name": "K-Means Clustering",
                "reason": f"Fast, intuitive grouping. Works well with {n_numeric} numeric features.",
                "priority": 1,
            })
            if n_rows > 500:
                recommendations.append({
                    "key": "dbscan",
                    "name": "DBSCAN",
                    "reason": "Finds clusters of arbitrary shape. Handles noise well.",
                    "priority": 2,
                })
            recommendations.append({
                "key": "isolation_forest",
                "name": "Isolation Forest",
                "reason": "Useful for anomaly/outlier detection before or after clustering.",
                "priority": 3,
            })

        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"])

        summary = {
            "task": task,
            "dataset_info": {
                "rows": n_rows,
                "columns": n_cols,
                "numeric_cols": n_numeric,
                "categorical_cols": n_cat,
                "n_classes": n_classes,
                "has_missing": has_missing,
                "is_imbalanced": is_imbalanced,
            },
            "recommendations": recommendations,
            "recommended_keys": [r["key"] for r in recommendations[:3]],
        }

        logger.info("Model recommendations: %s for task=%s, shape=%s",
                    [r["key"] for r in recommendations], task, df.shape)
        return _ok(summary)

    except Exception as e:
        logger.error("Model recommendation failed: %s", e, exc_info=True)
        return _err(f"Recommendation failed: {str(e)}")


# ────────────────────────────────────────────────────────────────────
#  Visualizations
# ────────────────────────────────────────────────────────────────────
@app.route("/api/viz/histogram", methods=["POST"])
def viz_histogram():
    logger.info("Histogram visualization requested (session=%s)", _sid())
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    
    body = request.get_json(silent=True) or {}
    column = body.get("column")
    bins = body.get("bins", 30)
    
    if not column or column not in df.columns:
        return _err(f"Invalid column: {column}")
    
    try:
        result = DataVisualizer.histogram(df, column, bins)
        return _ok(result)
    except Exception as e:
        logger.error("Histogram generation failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


@app.route("/api/viz/box-plot", methods=["POST"])
def viz_box_plot():
    logger.info("Box plot visualization requested (session=%s)", _sid())
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    
    body = request.get_json(silent=True) or {}
    columns = body.get("columns")
    
    try:
        result = DataVisualizer.box_plot(df, columns)
        return _ok(result)
    except Exception as e:
        logger.error("Box plot generation failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


@app.route("/api/viz/scatter", methods=["POST"])
def viz_scatter():
    logger.info("Scatter plot visualization requested (session=%s)", _sid())
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    
    body = request.get_json(silent=True) or {}
    x_col = body.get("x_column")
    y_col = body.get("y_column")
    hue_col = body.get("hue_column")
    
    if not x_col or not y_col:
        return _err("Both x_column and y_column are required")
    
    try:
        result = DataVisualizer.scatter_plot(df, x_col, y_col, hue_col)
        return _ok(result)
    except Exception as e:
        logger.error("Scatter plot generation failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


@app.route("/api/viz/correlation", methods=["POST"])
def viz_correlation():
    logger.info("Correlation heatmap visualization requested (session=%s)", _sid())
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    
    body = request.get_json(silent=True) or {}
    columns = body.get("columns")
    
    try:
        result = DataVisualizer.correlation_heatmap(df, columns)
        return _ok(result)
    except Exception as e:
        logger.error("Correlation heatmap generation failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


@app.route("/api/viz/distribution", methods=["POST"])
def viz_distribution():
    logger.info("Distribution comparison visualization requested (session=%s)", _sid())
    df = _df()
    if df is None:
        return _err("No dataset loaded", 404)
    
    body = request.get_json(silent=True) or {}
    column = body.get("column")
    group_by = body.get("group_by")
    
    if not column or column not in df.columns:
        return _err(f"Invalid column: {column}")
    
    try:
        result = DataVisualizer.distribution_comparison(df, column, group_by)
        return _ok(result)
    except Exception as e:
        logger.error("Distribution visualization failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


@app.route("/api/viz/confusion-matrix", methods=["POST"])
def viz_confusion_matrix():
    logger.info("Confusion matrix visualization requested (session=%s)", _sid())
    store = _store()
    
    body = request.get_json(silent=True) or {}
    model_key = body.get("model_key")
    
    if not model_key or model_key not in store.get("models", {}):
        return _err(f"Model not found: {model_key}")
    
    model_data = store["models"][model_key]
    if "evaluation" not in model_data:
        return _err("Model has not been evaluated yet")
    
    eval_data = model_data["evaluation"]
    y_true = np.array(eval_data.get("y_true", []))
    y_pred = np.array(eval_data.get("y_pred", []))
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return _err("Evaluation data not available")
    
    try:
        labels = eval_data.get("class_labels")
        result = DataVisualizer.confusion_matrix(y_true, y_pred, labels)
        return _ok(result)
    except Exception as e:
        logger.error("Confusion matrix generation failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


@app.route("/api/viz/roc-curve", methods=["POST"])
def viz_roc_curve():
    logger.info("ROC curve visualization requested (session=%s)", _sid())
    store = _store()
    
    body = request.get_json(silent=True) or {}
    model_key = body.get("model_key")
    
    if not model_key or model_key not in store.get("models", {}):
        return _err(f"Model not found: {model_key}")
    
    model_data = store["models"][model_key]
    if "evaluation" not in model_data:
        return _err("Model has not been evaluated yet")
    
    eval_data = model_data["evaluation"]
    y_true = np.array(eval_data.get("y_true", []))
    y_prob = np.array(eval_data.get("y_prob", []))
    
    if len(y_true) == 0 or len(y_prob) == 0:
        return _err("Probability predictions not available for ROC curve")
    
    try:
        labels = eval_data.get("class_labels")
        result = DataVisualizer.roc_curve(y_true, y_prob, labels)
        return _ok(result)
    except Exception as e:
        logger.error("ROC curve generation failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


@app.route("/api/viz/feature-importance", methods=["POST"])
def viz_feature_importance():
    logger.info("Feature importance visualization requested (session=%s)", _sid())
    store = _store()
    
    body = request.get_json(silent=True) or {}
    model_key = body.get("model_key")
    top_n = body.get("top_n", 20)
    
    if not model_key or model_key not in store.get("models", {}):
        return _err(f"Model not found: {model_key}")
    
    model_data = store["models"][model_key]
    model = model_data.get("model")
    
    if model is None:
        return _err("Model not available")
    
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        return _err("This model type does not support feature importance")
    
    try:
        df = _df()
        if df is None:
            return _err("No dataset loaded")
        target = store.get("target")
        feature_names = [c for c in df.columns if c != target]
        importances = model.feature_importances_
        
        result = DataVisualizer.feature_importance(feature_names, importances, top_n)
        return _ok(result)
    except Exception as e:
        logger.error("Feature importance visualization failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


@app.route("/api/viz/model-comparison", methods=["POST"])
def viz_model_comparison():
    logger.info("Model comparison visualization requested (session=%s)", _sid())
    store = _store()
    
    body = request.get_json(silent=True) or {}
    model_results = body.get("model_results")
    
    if not model_results:
        # Use stored comparison results if available
        model_results = store.get("last_comparison_results", [])
    
    if not model_results:
        return _err("No model results to compare")
    
    try:
        result = DataVisualizer.model_comparison(model_results)
        return _ok(result)
    except Exception as e:
        logger.error("Model comparison visualization failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


@app.route("/api/viz/residual-plot", methods=["POST"])
def viz_residual_plot():
    logger.info("Residual plot visualization requested (session=%s)", _sid())
    store = _store()
    
    body = request.get_json(silent=True) or {}
    model_key = body.get("model_key")
    
    if not model_key or model_key not in store.get("models", {}):
        return _err(f"Model not found: {model_key}")
    
    model_data = store["models"][model_key]
    if "evaluation" not in model_data:
        return _err("Model has not been evaluated yet")
    
    eval_data = model_data["evaluation"]
    y_true = np.array(eval_data.get("y_true", []))
    y_pred = np.array(eval_data.get("y_pred", []))
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return _err("Evaluation data not available")
    
    # Check if task is regression
    if store.get("task") != "regression":
        return _err("Residual plots are only for regression tasks")
    
    try:
        result = DataVisualizer.residual_plot(y_true, y_pred)
        return _ok(result)
    except Exception as e:
        logger.error("Residual plot generation failed: %s", e, exc_info=True)
        return _err(f"Visualization failed: {e}")


# ────────────────────────────────────────────────────────────────────
#  Settings Management
# ────────────────────────────────────────────────────────────────────
@app.route("/settings")
def settings_page():
    """Render settings page."""
    return render_template("settings.html")


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Get current application settings."""
    try:
        from config import load_properties, BASE_DIR
        
        # Load current properties from file
        props = load_properties(os.path.join(BASE_DIR, 'config.properties'))
        
        # Include all configurable settings with current values
        settings = {
            # Application
            "SECRET_KEY": Config.SECRET_KEY,
            "DEBUG": str(Config.DEBUG),
            
            # File Upload
            "MAX_FILE_UPLOAD_SIZE_MB": str(Config.MAX_FILE_UPLOAD_SIZE_MB),
            "CHUNK_SIZE_MB": str(Config.CHUNK_SIZE_MB),
            "LARGE_FILE_THRESHOLD_MB": str(Config.LARGE_FILE_THRESHOLD_MB),
            "USE_DB_FOR_LARGE_FILES": str(Config.USE_DB_FOR_LARGE_FILES),
            "DB_FALLBACK_THRESHOLD_MB": str(Config.DB_FALLBACK_THRESHOLD_MB),
            
            # LLM Configuration
            "LLM_PROVIDER": Config.LLM_PROVIDER,
            "OPENAI_API_KEY": Config.OPENAI_API_KEY,
            "OPENAI_MODEL": Config.OPENAI_MODEL,
            "OPENAI_MAX_TOKENS": str(Config.OPENAI_MAX_TOKENS),
            "OPENAI_TEMPERATURE": str(Config.OPENAI_TEMPERATURE),
            "AZURE_OPENAI_ENDPOINT": Config.AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_KEY": Config.AZURE_OPENAI_KEY,
            "AZURE_OPENAI_DEPLOYMENT": Config.AZURE_OPENAI_DEPLOYMENT,
            "AZURE_OPENAI_API_VERSION": Config.AZURE_OPENAI_API_VERSION,
            "ANTHROPIC_API_KEY": Config.ANTHROPIC_API_KEY,
            "ANTHROPIC_MODEL": Config.ANTHROPIC_MODEL,
            "DEEPSEEK_API_KEY": Config.DEEPSEEK_API_KEY,
            "DEEPSEEK_MODEL": Config.DEEPSEEK_MODEL,
            "DEEPSEEK_BASE_URL": Config.DEEPSEEK_BASE_URL,
            "DEEPSEEK_MAX_TOKENS": str(Config.DEEPSEEK_MAX_TOKENS),
            "DEEPSEEK_TEMPERATURE": str(Config.DEEPSEEK_TEMPERATURE),
            
            # Cloud GPU
            "CLOUD_GPU_ENABLED": str(Config.CLOUD_GPU_ENABLED),
            "CLOUD_GPU_PROVIDER": Config.CLOUD_GPU_PROVIDER,
            "AWS_REGION": Config.AWS_REGION,
            "AWS_ACCESS_KEY_ID": Config.AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": Config.AWS_SECRET_ACCESS_KEY,
            "AWS_SAGEMAKER_ROLE_ARN": Config.AWS_SAGEMAKER_ROLE_ARN,
            "AWS_SAGEMAKER_INSTANCE_TYPE": Config.AWS_SAGEMAKER_INSTANCE_TYPE,
            "AWS_SAGEMAKER_S3_BUCKET": Config.AWS_SAGEMAKER_S3_BUCKET,
            "AZURE_SUBSCRIPTION_ID": Config.AZURE_SUBSCRIPTION_ID,
            "AZURE_RESOURCE_GROUP": Config.AZURE_RESOURCE_GROUP,
            "AZURE_WORKSPACE_NAME": Config.AZURE_WORKSPACE_NAME,
            "AZURE_TENANT_ID": Config.AZURE_TENANT_ID,
            "AZURE_CLIENT_ID": Config.AZURE_CLIENT_ID,
            "AZURE_CLIENT_SECRET": Config.AZURE_CLIENT_SECRET,
            "AZURE_COMPUTE_TARGET": Config.AZURE_COMPUTE_TARGET,
            "AZURE_VM_SIZE": Config.AZURE_VM_SIZE,
            "GCP_PROJECT_ID": Config.GCP_PROJECT_ID,
            "GCP_REGION": Config.GCP_REGION,
            "GCP_SERVICE_ACCOUNT_KEY_PATH": Config.GCP_SERVICE_ACCOUNT_KEY_PATH,
            "GCP_MACHINE_TYPE": Config.GCP_MACHINE_TYPE,
            "GCP_ACCELERATOR_TYPE": Config.GCP_ACCELERATOR_TYPE,
            "GCP_ACCELERATOR_COUNT": str(Config.GCP_ACCELERATOR_COUNT),
            "CUSTOM_GPU_ENDPOINT": Config.CUSTOM_GPU_ENDPOINT,
            "CUSTOM_GPU_API_KEY": Config.CUSTOM_GPU_API_KEY,
            "CUSTOM_GPU_AUTH_TOKEN": Config.CUSTOM_GPU_AUTH_TOKEN,
            "CUSTOM_GPU_USERNAME": Config.CUSTOM_GPU_USERNAME,
            "CUSTOM_GPU_PASSWORD": Config.CUSTOM_GPU_PASSWORD,
            "GPU_JOB_TIMEOUT": str(Config.GPU_JOB_TIMEOUT),
            "GPU_FALLBACK_TO_LOCAL": str(Config.GPU_FALLBACK_TO_LOCAL),
            "GPU_MAX_RETRIES": str(Config.GPU_MAX_RETRIES),
        }
        
        return jsonify({"success": True, "settings": settings})
    except Exception as e:
        logger.error("Failed to get settings: %s", e, exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update application settings."""
    try:
        from config import save_properties, BASE_DIR
        
        settings = request.get_json()
        if not settings:
            return jsonify({"success": False, "message": "No settings provided"}), 400
        
        # Save to config.properties file
        config_path = os.path.join(BASE_DIR, 'config.properties')
        
        if save_properties(config_path, settings):
            logger.info("Settings updated successfully")
            return jsonify({
                "success": True, 
                "message": "Settings saved successfully. Restart the application for changes to take effect."
            })
        else:
            return jsonify({"success": False, "message": "Failed to save settings"}), 500
            
    except Exception as e:
        logger.error("Failed to update settings: %s", e, exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/settings/reset", methods=["POST"])
def reset_settings():
    """Reset settings to defaults from config.properties.example."""
    try:
        from config import load_properties, save_properties, BASE_DIR
        
        # Load defaults from example file
        example_path = os.path.join(BASE_DIR, 'config.properties.example')
        defaults = load_properties(example_path)
        
        # Save to config.properties
        config_path = os.path.join(BASE_DIR, 'config.properties')
        
        if save_properties(config_path, defaults):
            logger.info("Settings reset to defaults")
            return jsonify({
                "success": True, 
                "message": "Settings reset to defaults. Restart the application for changes to take effect."
            })
        else:
            return jsonify({"success": False, "message": "Failed to reset settings"}), 500
            
    except Exception as e:
        logger.error("Failed to reset settings: %s", e, exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


# ────────────────────────────────────────────────────────────────────
#  Error Handlers
# ────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def handle_404(e):
    """Handle 404 Not Found errors (like missing favicon.ico)."""
    logger.debug("404 Not Found: %s", request.url)
    return jsonify({"status": "error", "message": "Resource not found"}), 404


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unexpected exceptions, but let HTTP exceptions pass through."""
    # Don't catch HTTPException (werkzeug exceptions like 404, 405, etc.)
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        logger.debug("HTTP exception: %s %s", e.code, e.description)
        return e
    
    # Log and return actual server errors
    logger.critical("Unhandled exception: %s", e, exc_info=True)
    return _err(f"Internal error: {e}", 500)


# ────────────────────────────────────────────────────────────────────
#  Run
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Detect if running in a container
    import platform
    is_container = os.path.exists('/.dockerenv') or os.environ.get('CONTAINER') == 'true'
    
    # Bind to 0.0.0.0 in containers for accessibility, 127.0.0.1 locally
    host = '0.0.0.0' if is_container else '127.0.0.1'
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("Starting %s v%s on http://%s:%d", Config.APP_NAME, Config.APP_VERSION, host, port)
    print(f"\n{'='*50}")
    print(f"  {Config.APP_NAME} v{Config.APP_VERSION}")
    print(f"  http://{host}:{port}")
    print(f"  Environment: {'Container' if is_container else 'Local'}")
    print(f"{'='*50}\n")
    app.run(debug=Config.DEBUG, host=host, port=port)

