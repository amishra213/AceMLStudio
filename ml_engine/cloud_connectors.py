"""
AceML Studio – Cloud Storage & Database Connectors
====================================================
Provides unified connectors for:

  Cloud Storage
  ─────────────
  • AWS S3           (boto3)
  • Azure Blob       (azure-storage-blob)
  • Google Cloud     (google-cloud-storage)

  Databases
  ─────────
  • SQLite           (built-in via sqlalchemy)
  • PostgreSQL       (sqlalchemy + psycopg2 / psycopg2-binary)
  • MySQL / MariaDB  (sqlalchemy + pymysql)
  • SQL Server       (sqlalchemy + pyodbc)
  • Generic          (any sqlalchemy connection string)

All connectors are optional-import safe: if the required library is not
installed the connector raises a clear ImportError at connection time so
the rest of the application keeps working.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Any, TYPE_CHECKING

import pandas as pd

logger = logging.getLogger("aceml.cloud_connectors")

# ─── optional-dependency sentinels ───────────────────────────────────────────
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError as BotoClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from azure.storage.blob import BlobServiceClient
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

try:
    from google.cloud import storage as gcs_storage  # type: ignore
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    from sqlalchemy import create_engine, text, inspect
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

if TYPE_CHECKING:
    import boto3  # type: ignore
    from azure.storage.blob import BlobServiceClient  # type: ignore
    from google.cloud import storage  # noqa: F401  # type: ignore
    gcs_storage = storage  # type: ignore
    from sqlalchemy import create_engine, text, inspect  # type: ignore


# ─── helpers ─────────────────────────────────────────────────────────────────

def _check(flag: bool, package: str) -> None:
    if not flag:
        raise ImportError(
            f"Required package '{package}' is not installed. "
            f"Run: pip install {package}"
        )


def _read_df_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Detect format from filename and parse bytes into a DataFrame."""
    ext = filename.rsplit(".", 1)[-1].lower()
    buf = io.BytesIO(data)
    if ext == "csv":
        return pd.read_csv(buf)
    if ext in ("xlsx", "xls"):
        return pd.read_excel(buf, engine="openpyxl" if ext == "xlsx" else None)
    if ext == "parquet":
        return pd.read_parquet(buf)
    if ext == "json":
        return pd.read_json(buf)
    raise ValueError(f"Unsupported file extension: .{ext}")


# ══════════════════════════════════════════════════════════════════════════════
#  AWS S3 Connector
# ══════════════════════════════════════════════════════════════════════════════

class S3Connector:
    """Connect to AWS S3, list objects, and download files as DataFrames."""

    SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls", "parquet", "json"}

    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        _check(HAS_BOTO3, "boto3")
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            endpoint_url=endpoint_url or None,
        )
        logger.info("S3Connector initialised (region=%s)", region_name)

    # ── connection test ────────────────────────────────────────────────────
    def test_connection(self) -> dict:
        """List buckets to verify credentials and connectivity."""
        try:
            resp = self.client.list_buckets()
            buckets = [b["Name"] for b in resp.get("Buckets", [])]
            return {"ok": True, "buckets": buckets, "count": len(buckets)}
        except Exception as exc:
            logger.error("S3 connection test failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    # ── list files ────────────────────────────────────────────────────────
    def list_files(self, bucket: str, prefix: str = "", max_keys: int = 500) -> list[dict]:
        """Return a list of file objects in *bucket* whose keys match *prefix*."""
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=bucket, Prefix=prefix,
                PaginationConfig={"MaxItems": max_keys}
            )
            files = []
            for page in pages:
                for obj in page.get("Contents", []):
                    key: str = obj["Key"]
                    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
                    if ext in self.SUPPORTED_EXTENSIONS:
                        files.append({
                            "key": key,
                            "size_bytes": obj["Size"],
                            "size_kb": round(obj["Size"] / 1024, 1),
                            "last_modified": str(obj["LastModified"]),
                            "extension": ext,
                        })
            return files
        except Exception as exc:
            logger.error("S3 list_files failed: %s", exc)
            raise

    def list_buckets(self) -> list[str]:
        resp = self.client.list_buckets()
        return [b["Name"] for b in resp.get("Buckets", [])]

    # ── load DataFrame ────────────────────────────────────────────────────
    def load_dataframe(self, bucket: str, key: str) -> pd.DataFrame:
        """Download *key* from *bucket* and return as a DataFrame."""
        logger.info("S3: downloading s3://%s/%s", bucket, key)
        obj = self.client.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        filename = key.split("/")[-1]
        df = _read_df_from_bytes(data, filename)
        logger.info("S3: loaded %s → shape=%s", key, df.shape)
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  Azure Blob Storage Connector
# ══════════════════════════════════════════════════════════════════════════════

class AzureBlobConnector:
    """Connect to Azure Blob Storage, browse containers and blobs."""

    SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls", "parquet", "json"}

    def __init__(
        self,
        connection_string: str | None = None,
        account_name: str | None = None,
        account_key: str | None = None,
        sas_token: str | None = None,
    ) -> None:
        _check(HAS_AZURE, "azure-storage-blob")
        if connection_string:
            self.client = BlobServiceClient.from_connection_string(connection_string)
        elif account_name and account_key:
            url = f"https://{account_name}.blob.core.windows.net"
            self.client = BlobServiceClient(account_url=url, credential=account_key)
        elif account_name and sas_token:
            url = f"https://{account_name}.blob.core.windows.net"
            self.client = BlobServiceClient(account_url=url, credential=sas_token)
        else:
            raise ValueError(
                "Provide either connection_string, "
                "(account_name + account_key), or (account_name + sas_token)."
            )
        logger.info("AzureBlobConnector initialised")

    def test_connection(self) -> dict:
        try:
            containers = [c["name"] for c in self.client.list_containers()]
            return {"ok": True, "containers": containers, "count": len(containers)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def list_containers(self) -> list[str]:
        return [c["name"] for c in self.client.list_containers()]

    def list_files(self, container: str, prefix: str = "", max_results: int = 500) -> list[dict]:
        cc = self.client.get_container_client(container)
        files = []
        for blob in cc.list_blobs(name_starts_with=prefix or None):
            name: str = blob.name
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext in self.SUPPORTED_EXTENSIONS:
                files.append({
                    "name": name,
                    "size_bytes": blob.size,
                    "size_kb": round((blob.size or 0) / 1024, 1),
                    "last_modified": str(blob.last_modified),
                    "extension": ext,
                    "container": container,
                })
            if len(files) >= max_results:
                break
        return files

    def load_dataframe(self, container: str, blob_name: str) -> pd.DataFrame:
        logger.info("Azure: downloading %s/%s", container, blob_name)
        cc = self.client.get_container_client(container)
        data = cc.get_blob_client(blob_name).download_blob().readall()
        filename = blob_name.split("/")[-1]
        df = _read_df_from_bytes(data, filename)
        logger.info("Azure: loaded %s → shape=%s", blob_name, df.shape)
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  Google Cloud Storage Connector
# ══════════════════════════════════════════════════════════════════════════════

class GCSConnector:
    """Connect to Google Cloud Storage using a service-account key file or ADC."""

    SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls", "parquet", "json"}

    def __init__(
        self,
        credentials_json: str | None = None,
        project: str | None = None,
    ) -> None:
        _check(HAS_GCS, "google-cloud-storage")
        if credentials_json:
            import json as _json
            from google.oauth2 import service_account
            info = _json.loads(credentials_json)
            creds = service_account.Credentials.from_service_account_info(info)
            self.client = gcs_storage.Client(credentials=creds, project=project or info.get("project_id"))
        else:
            # Application Default Credentials
            self.client = gcs_storage.Client(project=project)
        logger.info("GCSConnector initialised (project=%s)", project)

    def test_connection(self) -> dict:
        try:
            buckets = [b.name for b in self.client.list_buckets()]
            return {"ok": True, "buckets": buckets, "count": len(buckets)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def list_buckets(self) -> list[str]:
        return [b.name for b in self.client.list_buckets()]

    def list_files(self, bucket: str, prefix: str = "", max_results: int = 500) -> list[dict]:
        blobs = self.client.list_blobs(bucket, prefix=prefix or None, max_results=max_results)
        files = []
        for blob in blobs:
            name: str = blob.name
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext in self.SUPPORTED_EXTENSIONS:
                files.append({
                    "name": name,
                    "size_bytes": blob.size,
                    "size_kb": round((blob.size or 0) / 1024, 1),
                    "updated": str(blob.updated),
                    "extension": ext,
                    "bucket": bucket,
                })
        return files

    def load_dataframe(self, bucket: str, blob_name: str) -> pd.DataFrame:
        logger.info("GCS: downloading gs://%s/%s", bucket, blob_name)
        blob = self.client.bucket(bucket).blob(blob_name)
        data = blob.download_as_bytes()
        filename = blob_name.split("/")[-1]
        df = _read_df_from_bytes(data, filename)
        logger.info("GCS: loaded %s → shape=%s", blob_name, df.shape)
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  Database Connector  (SQLAlchemy)
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseConnector:
    """
    Connect to any SQLAlchemy-supported database and load tables as DataFrames.

    Convenience factory methods:
      DatabaseConnector.sqlite(path)
      DatabaseConnector.postgres(host, port, db, user, password)
      DatabaseConnector.mysql(host, port, db, user, password)
      DatabaseConnector.from_url(url)
    """

    def __init__(self, connection_url: str) -> None:
        _check(HAS_SQLALCHEMY, "sqlalchemy")
        self.url = connection_url
        try:
            self.engine = create_engine(connection_url, pool_pre_ping=True)
        except ModuleNotFoundError as e:
            pkg = str(e).replace("No module named ", "").strip("'")
            raise ImportError(
                f"Required database driver '{pkg}' is not installed. "
                f"Install it with: pip install {pkg}"
            ) from e
        logger.info("DatabaseConnector initialised: %s", self._safe_url())

    def _safe_url(self) -> str:
        """Return URL with password masked."""
        try:
            from sqlalchemy.engine import make_url
            u = make_url(self.url)
            return u.render_as_string(hide_password=True)
        except Exception:
            return "<url>"

    # ── factory helpers ───────────────────────────────────────────────────
    @classmethod
    def sqlite(cls, db_path: str) -> "DatabaseConnector":
        if db_path == ":memory:":
            return cls("sqlite:///:memory:")
        return cls(f"sqlite:///{os.path.abspath(db_path)}")

    @classmethod
    def postgres(
        cls,
        host: str,
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        ssl: bool = False,
    ) -> "DatabaseConnector":
        ssl_suffix = "?sslmode=require" if ssl else ""
        return cls(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}{ssl_suffix}")

    @classmethod
    def mysql(
        cls,
        host: str,
        port: int = 3306,
        database: str = "",
        user: str = "root",
        password: str = "",
    ) -> "DatabaseConnector":
        return cls(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

    @classmethod
    def sqlserver(
        cls,
        host: str,
        port: int = 1433,
        database: str = "",
        user: str = "sa",
        password: str = "",
    ) -> "DatabaseConnector":
        return cls(f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server")

    @classmethod
    def from_url(cls, url: str) -> "DatabaseConnector":
        return cls(url)

    # ── introspection ─────────────────────────────────────────────────────
    def test_connection(self) -> dict:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            tables = self.list_tables()
            return {"ok": True, "tables": tables, "count": len(tables)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def list_tables(self) -> list[str]:
        """Return all table names in the current schema."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def list_views(self) -> list[str]:
        inspector = inspect(self.engine)
        return inspector.get_view_names()

    def get_table_info(self, table_name: str) -> dict:
        """Return columns and row count for *table_name*."""
        inspector = inspect(self.engine)
        columns = [
            {"name": col["name"], "type": str(col["type"])}
            for col in inspector.get_columns(table_name)
        ]
        with self.engine.connect() as conn:
            count = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
        return {"table": table_name, "columns": columns, "row_count": int(count or 0)}

    # ── load ──────────────────────────────────────────────────────────────
    def load_table(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        """Load an entire table (or the first *limit* rows) as a DataFrame."""
        q = f'SELECT * FROM "{table_name}"'
        if limit:
            q += f" LIMIT {int(limit)}"
        logger.info("DB: loading table=%s (limit=%s)", table_name, limit)
        df = pd.read_sql(text(q), self.engine)
        logger.info("DB: loaded %s → shape=%s", table_name, df.shape)
        return df

    def load_query(self, sql: str) -> pd.DataFrame:
        """Execute *sql* and return the result as a DataFrame."""
        logger.info("DB: running custom query (%d chars)", len(sql))
        df = pd.read_sql(text(sql), self.engine)
        logger.info("DB: query returned shape=%s", df.shape)
        return df

    def close(self) -> None:
        self.engine.dispose()
        logger.info("DatabaseConnector closed")


# ══════════════════════════════════════════════════════════════════════════════
#  Availability report  (used by the API to tell the frontend what's installed)
# ══════════════════════════════════════════════════════════════════════════════

def get_availability() -> dict:
    """Return which optional packages are installed."""
    return {
        "s3":       HAS_BOTO3,
        "azure":    HAS_AZURE,
        "gcs":      HAS_GCS,
        "database": HAS_SQLALCHEMY,
        "packages": {
            "boto3":               HAS_BOTO3,
            "azure-storage-blob":  HAS_AZURE,
            "google-cloud-storage":HAS_GCS,
            "sqlalchemy":          HAS_SQLALCHEMY,
        },
    }
