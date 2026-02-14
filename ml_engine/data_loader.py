"""
AceML Studio – Data Loader
===========================
Load datasets from CSV, Excel, JSON, Parquet. Provide preview and metadata.
Supports memory-efficient loading for large files via dtype optimisation,
chunked CSV reading, and optional sampling.
"""

import os
import gc
import logging
import pandas as pd
import numpy as np
from config import Config

logger = logging.getLogger("aceml.data_loader")


class DataLoader:
    """Handles loading, previewing, and basic info for uploaded datasets."""

    LOADERS = {
        "csv": lambda p, **kw: pd.read_csv(p, **kw),
        "xlsx": lambda p, **kw: pd.read_excel(p, engine="openpyxl", **kw),
        "xls": lambda p, **kw: pd.read_excel(p, **kw),
        "json": lambda p, **kw: pd.read_json(p, **kw),
        "parquet": lambda p, **kw: pd.read_parquet(p, **kw),
    }

    @staticmethod
    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    # ───────────────────── dtype optimisation ─────────────────────
    @staticmethod
    def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast numeric dtypes and convert low-cardinality strings to
        categoricals to reduce memory footprint — often 50-80 % savings."""
        before_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        for col in df.columns:
            col_dtype = df[col].dtype

            # Downcast integers
            if pd.api.types.is_integer_dtype(col_dtype):
                df[col] = pd.to_numeric(df[col], downcast="integer")

            # Downcast floats
            elif pd.api.types.is_float_dtype(col_dtype):
                df[col] = pd.to_numeric(df[col], downcast="float")

            # Low-cardinality object → category (if unique ratio < 50 %)
            elif col_dtype == object:
                nunique = df[col].nunique()
                if nunique / max(len(df), 1) < 0.5:
                    df[col] = df[col].astype("category")

        after_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info("dtype optimisation: %.2f MB → %.2f MB (%.0f%% reduction)",
                    before_mb, after_mb,
                    (1 - after_mb / max(before_mb, 0.01)) * 100)
        return df

    # ───────────────────── chunked CSV reader ─────────────────────
    @classmethod
    def _load_csv_chunked(cls, filepath: str, max_rows: int | None = None,
                          chunksize: int | None = None, **kwargs) -> pd.DataFrame:
        """Read a large CSV in chunks, optionally capping at *max_rows*.
        Each chunk is dtype-optimised before being concatenated so peak
        memory stays manageable."""
        chunksize = chunksize or Config.LOW_MEMORY_CSV_CHUNKSIZE
        chunks: list[pd.DataFrame] = []
        rows_read = 0

        logger.info("Chunked CSV read started (chunksize=%d, max_rows=%s)",
                    chunksize, max_rows)
        for chunk in pd.read_csv(filepath, chunksize=chunksize,
                                 low_memory=True, **kwargs):
            chunk = cls.optimise_dtypes(chunk)
            chunks.append(chunk)
            rows_read += len(chunk)
            if max_rows and rows_read >= max_rows:
                break

        df = pd.concat(chunks, ignore_index=True)
        if max_rows and len(df) > max_rows:
            df = df.iloc[:max_rows]
        del chunks
        gc.collect()
        logger.info("Chunked CSV read done — %d rows loaded", len(df))
        return df

    # ───────────────────── file size helper ────────────────────────
    @staticmethod
    def _file_size_mb(filepath: str) -> float:
        return os.path.getsize(filepath) / (1024 * 1024)

    # ───────────────────── main load method ────────────────────────
    @classmethod
    def load(cls, filepath: str, optimise_memory: bool = True,
             sample: int | None = None, **kwargs) -> pd.DataFrame:
        """Load a file into a DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the data file.
        optimise_memory : bool
            Automatically downcast dtypes for large files.
        sample : int | None
            If set, randomly sample this many rows after loading.
        **kwargs
            Extra arguments forwarded to the underlying pandas reader.
        """
        ext = filepath.rsplit(".", 1)[1].lower()
        loader = cls.LOADERS.get(ext)
        if loader is None:
            logger.error("Unsupported file type: .%s (file=%s)", ext, filepath)
            raise ValueError(f"Unsupported file type: .{ext}")

        file_mb = cls._file_size_mb(filepath)
        logger.info("Loading file: %s (type=.%s, size=%.1f MB)",
                    os.path.basename(filepath), ext, file_mb)

        # --- Strategy selection based on file size ---
        large_file = file_mb >= Config.LARGE_FILE_THRESHOLD_MB

        if ext == "csv" and large_file:
            # Use chunked reader for big CSVs
            max_rows = sample or (Config.MAX_IN_MEMORY_ROWS if file_mb > Config.MEMORY_OPTIMIZE_THRESHOLD_MB * 2 else None)
            df = cls._load_csv_chunked(filepath, max_rows=max_rows, **kwargs)
        else:
            df = loader(filepath, **kwargs)

        # Auto-optimise dtypes for anything above threshold
        if optimise_memory and (large_file or
                                df.memory_usage(deep=True).sum() / (1024**2) > Config.MEMORY_OPTIMIZE_THRESHOLD_MB):
            df = cls.optimise_dtypes(df)

        # Sample if dataset is still too large
        if sample and len(df) > sample:
            logger.info("Sampling %d rows from %d", sample, len(df))
            df = df.sample(n=sample, random_state=Config.DEFAULT_RANDOM_STATE).reset_index(drop=True)
        elif len(df) > Config.MAX_IN_MEMORY_ROWS:
            n = Config.SAMPLE_SIZE_FOR_VERY_LARGE
            logger.warning("Dataset has %d rows (> %d limit) — auto-sampling to %d rows",
                           len(df), Config.MAX_IN_MEMORY_ROWS, n)
            df = df.sample(n=n, random_state=Config.DEFAULT_RANDOM_STATE).reset_index(drop=True)

        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info("Loaded %d rows x %d cols (%.2f MB in memory)",
                    df.shape[0], df.shape[1], mem_mb)
        return df

    @staticmethod
    def get_info(df: pd.DataFrame) -> dict:
        """Return summary metadata about a DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        boolean_cols = df.select_dtypes(include=["bool"]).columns.tolist()

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "boolean_columns": boolean_cols,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }

    @staticmethod
    def get_preview(df: pd.DataFrame, n: int = 20) -> dict:
        """Return head rows as dict for JSON serialization."""
        preview_df = df.head(n).copy()
        # Convert all values to JSON-safe types
        for col in preview_df.columns:
            if preview_df[col].dtype == "datetime64[ns]":
                preview_df[col] = preview_df[col].astype(str)
            preview_df[col] = preview_df[col].where(preview_df[col].notna(), None)  # type: ignore
        return {
            "columns": preview_df.columns.tolist(),
            "data": preview_df.values.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in preview_df.dtypes.items()},
        }

    @staticmethod
    def get_statistics(df: pd.DataFrame) -> dict:
        """Return descriptive statistics."""
        desc = df.describe(include="all").fillna("").to_dict()
        # Ensure JSON-serializable
        clean = {}
        for col, stats in desc.items():
            clean[col] = {}
            for stat, val in stats.items():
                if isinstance(val, (np.integer,)):
                    clean[col][stat] = int(val)
                elif isinstance(val, (np.floating,)):
                    clean[col][stat] = round(float(val), 4)
                else:
                    clean[col][stat] = str(val) if val != "" else None
        return clean
