"""
Database Storage for Large Datasets
====================================
When RAM is insufficient, store datasets in SQLite database.
"""

import sqlite3
import pandas as pd
import json
import os
from typing import Optional, Union, Iterator
from logging_config import get_logger

logger = get_logger("db_storage")


class DataFrameDBStorage:
    """Store and retrieve DataFrames from SQLite database."""
    
    def __init__(self, db_path: str):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist."""
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  # Only create directory if path includes a directory
                os.makedirs(db_dir, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Create metadata table for temporary session data
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dataset_metadata (
                        session_id TEXT PRIMARY KEY,
                        table_name TEXT NOT NULL,
                        filename TEXT,
                        rows INTEGER,
                        columns INTEGER,
                        dtypes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create table for saved/persistent datasets
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS saved_datasets (
                        dataset_id TEXT PRIMARY KEY,
                        dataset_name TEXT NOT NULL UNIQUE,
                        table_name TEXT NOT NULL,
                        description TEXT,
                        original_filename TEXT,
                        rows INTEGER,
                        columns INTEGER,
                        column_names TEXT,
                        dtypes TEXT,
                        size_bytes INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        tags TEXT
                    )
                """)
                
                conn.commit()
            logger.info("Database initialized at %s", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize database: %s", e, exc_info=True)
            raise
    
    def store_dataframe(self, session_id: str, df: pd.DataFrame, 
                       filename: Optional[str] = None) -> bool:
        """Store a DataFrame in the database.
        
        Args:
            session_id: Unique session identifier
            df: DataFrame to store
            filename: Original filename (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            table_name = f"data_{session_id.replace('-', '_')}"
            
            with sqlite3.connect(self.db_path) as conn:
                # Store the DataFrame
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                # Store metadata
                metadata = {
                    'session_id': session_id,
                    'table_name': table_name,
                    'filename': filename or 'unknown',
                    'rows': len(df),
                    'columns': len(df.columns),
                    'dtypes': json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()})
                }
                
                conn.execute("""
                    INSERT OR REPLACE INTO dataset_metadata 
                    (session_id, table_name, filename, rows, columns, dtypes)
                    VALUES (:session_id, :table_name, :filename, :rows, :columns, :dtypes)
                """, metadata)
                
                conn.commit()
            
            logger.info("Stored DataFrame for session %s: %d rows x %d cols → %s",
                       session_id, len(df), len(df.columns), table_name)
            return True
            
        except Exception as e:
            logger.error("Failed to store DataFrame for session %s: %s", 
                        session_id, e, exc_info=True)
            return False
    
    def retrieve_dataframe(self, session_id: str,
                          chunksize: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame], None]:
        """Retrieve a DataFrame from the database.
        
        Args:
            session_id: Unique session identifier
            chunksize: If specified, return iterator of chunks instead of full DataFrame
            
        Returns:
            DataFrame or iterator if successful, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get metadata
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT table_name, dtypes FROM dataset_metadata WHERE session_id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    logger.warning("No data found for session %s", session_id)
                    return None
                
                table_name, dtypes_json = row
                dtypes = json.loads(dtypes_json)
                
                # Retrieve the DataFrame
                if chunksize:
                    logger.info("Retrieving DataFrame for session %s in chunks of %d",
                               session_id, chunksize)
                    return pd.read_sql_query(
                        f"SELECT * FROM {table_name}",
                        conn,
                        chunksize=chunksize
                    )
                else:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    logger.info("Retrieved DataFrame for session %s: %d rows x %d cols",
                               session_id, len(df), len(df.columns))
                    return df
                    
        except Exception as e:
            logger.error("Failed to retrieve DataFrame for session %s: %s",
                        session_id, e, exc_info=True)
            return None
    
    def delete_dataframe(self, session_id: str) -> bool:
        """Delete a DataFrame from the database.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table name
                cursor.execute(
                    "SELECT table_name FROM dataset_metadata WHERE session_id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    table_name = row[0]
                    # Drop the data table
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    # Delete metadata
                    cursor.execute(
                        "DELETE FROM dataset_metadata WHERE session_id = ?",
                        (session_id,)
                    )
                    conn.commit()
                    logger.info("Deleted data for session %s", session_id)
                    return True
                else:
                    logger.warning("No data to delete for session %s", session_id)
                    return False
                    
        except Exception as e:
            logger.error("Failed to delete data for session %s: %s",
                        session_id, e, exc_info=True)
            return False
    
    def get_metadata(self, session_id: str) -> Optional[dict]:
        """Get metadata for a stored DataFrame.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dictionary with metadata or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT filename, rows, columns, created_at
                    FROM dataset_metadata WHERE session_id = ?
                """, (session_id,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'filename': row[0],
                        'rows': row[1],
                        'columns': row[2],
                        'created_at': row[3],
                        'stored_in_db': True
                    }
                return None
                
        except Exception as e:
            logger.error("Failed to get metadata for session %s: %s",
                        session_id, e, exc_info=True)
            return None
    
    def cleanup_old_data(self, days: int = 7) -> int:
        """Delete data older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get old sessions
                cursor.execute("""
                    SELECT session_id, table_name FROM dataset_metadata
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                """, (days,))
                
                old_sessions = cursor.fetchall()
                count = 0
                
                for session_id, table_name in old_sessions:
                    try:
                        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                        cursor.execute(
                            "DELETE FROM dataset_metadata WHERE session_id = ?",
                            (session_id,)
                        )
                        count += 1
                    except Exception as e:
                        logger.warning("Failed to cleanup session %s: %s", session_id, e)
                        continue
                
                conn.commit()
                logger.info("Cleaned up %d old sessions (older than %d days)", count, days)
                return count
                
        except Exception as e:
            logger.error("Failed to cleanup old data: %s", e, exc_info=True)
            return 0

    # ================================================================
    #  SAVED DATASETS - Persistent Storage
    # ================================================================
    
    def save_dataset(self, dataset_name: str, df: pd.DataFrame,
                     description: str = "", original_filename: str = "",
                     tags: list[str] = None) -> bool:
        """Save a dataset permanently with a name.
        
        Args:
            dataset_name: Unique name for the dataset
            df: DataFrame to save
            description: Optional description
            original_filename: Original file name
            tags: Optional list of tags for categorization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import uuid
            dataset_id = str(uuid.uuid4())
            table_name = f"saved_{dataset_name.lower().replace(' ', '_').replace('-', '_')}"
            
            # Ensure table name is safe
            table_name = ''.join(c for c in table_name if c.isalnum() or c == '_')
            
            with sqlite3.connect(self.db_path) as conn:
                # Check if dataset name already exists
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT dataset_id FROM saved_datasets WHERE dataset_name = ?",
                    (dataset_name,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing dataset
                    dataset_id = existing[0]
                    # Drop old table
                    old_table = cursor.execute(
                        "SELECT table_name FROM saved_datasets WHERE dataset_id = ?",
                        (dataset_id,)
                    ).fetchone()[0]
                    cursor.execute(f"DROP TABLE IF EXISTS {old_table}")
                    logger.info("Updating existing dataset '%s'", dataset_name)
                
                # Store the DataFrame
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                # Calculate size estimate
                size_bytes = df.memory_usage(deep=True).sum()
                
                # Store metadata
                metadata = {
                    'dataset_id': dataset_id,
                    'dataset_name': dataset_name,
                    'table_name': table_name,
                    'description': description,
                    'original_filename': original_filename or 'unknown',
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': json.dumps(df.columns.tolist()),
                    'dtypes': json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()}),
                    'size_bytes': int(size_bytes),
                    'tags': json.dumps(tags or [])
                }
                
                if existing:
                    # Update
                    conn.execute("""
                        UPDATE saved_datasets 
                        SET table_name = :table_name,
                            description = :description,
                            original_filename = :original_filename,
                            rows = :rows,
                            columns = :columns,
                            column_names = :column_names,
                            dtypes = :dtypes,
                            size_bytes = :size_bytes,
                            updated_at = CURRENT_TIMESTAMP,
                            tags = :tags
                        WHERE dataset_id = :dataset_id
                    """, metadata)
                else:
                    # Insert
                    conn.execute("""
                        INSERT INTO saved_datasets 
                        (dataset_id, dataset_name, table_name, description, original_filename,
                         rows, columns, column_names, dtypes, size_bytes, tags)
                        VALUES (:dataset_id, :dataset_name, :table_name, :description, 
                                :original_filename, :rows, :columns, :column_names, 
                                :dtypes, :size_bytes, :tags)
                    """, metadata)
                
                conn.commit()
            
            logger.info("Saved dataset '%s': %d rows x %d cols (%d bytes)",
                       dataset_name, len(df), len(df.columns), size_bytes)
            return True
            
        except Exception as e:
            logger.error("Failed to save dataset '%s': %s", dataset_name, e, exc_info=True)
            return False
    
    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load a saved dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            DataFrame if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT table_name, dtypes FROM saved_datasets WHERE dataset_name = ?",
                    (dataset_name,)
                )
                row = cursor.fetchone()
                
                if not row:
                    logger.warning("Dataset '%s' not found", dataset_name)
                    return None
                
                table_name, dtypes_json = row
                
                # Load the DataFrame
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                
                # Update access time
                cursor.execute("""
                    UPDATE saved_datasets 
                    SET accessed_at = CURRENT_TIMESTAMP 
                    WHERE dataset_name = ?
                """, (dataset_name,))
                conn.commit()
                
                logger.info("Loaded dataset '%s': %d rows x %d cols",
                           dataset_name, len(df), len(df.columns))
                return df
                    
        except Exception as e:
            logger.error("Failed to load dataset '%s': %s", dataset_name, e, exc_info=True)
            return None
    
    def list_datasets(self, search: str = None, tags: list[str] = None) -> list[dict]:
        """List all saved datasets.
        
        Args:
            search: Optional search term to filter by name or description
            tags: Optional list of tags to filter by
            
        Returns:
            List of dataset metadata dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                    SELECT dataset_id, dataset_name, description, original_filename,
                           rows, columns, column_names, size_bytes, 
                           created_at, updated_at, accessed_at, tags
                    FROM saved_datasets
                """
                params = []
                conditions = []
                
                if search:
                    conditions.append("(dataset_name LIKE ? OR description LIKE ?)")
                    search_term = f"%{search}%"
                    params.extend([search_term, search_term])
                
                if tags:
                    # Check if any of the specified tags exist in the tags JSON
                    for tag in tags:
                        conditions.append("tags LIKE ?")
                        params.append(f"%{tag}%")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY updated_at DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                datasets = []
                for row in rows:
                    datasets.append({
                        'dataset_id': row['dataset_id'],
                        'dataset_name': row['dataset_name'],
                        'description': row['description'],
                        'original_filename': row['original_filename'],
                        'rows': row['rows'],
                        'columns': row['columns'],
                        'column_names': json.loads(row['column_names']) if row['column_names'] else [],
                        'size_mb': round(row['size_bytes'] / (1024 * 1024), 2) if row['size_bytes'] else 0,
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'accessed_at': row['accessed_at'],
                        'tags': json.loads(row['tags']) if row['tags'] else []
                    })
                
                logger.info("Listed %d saved datasets", len(datasets))
                return datasets
                
        except Exception as e:
            logger.error("Failed to list datasets: %s", e, exc_info=True)
            return []
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a saved dataset.
        
        Args:
            dataset_name: Name of the dataset to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table name
                cursor.execute(
                    "SELECT table_name FROM saved_datasets WHERE dataset_name = ?",
                    (dataset_name,)
                )
                row = cursor.fetchone()
                
                if row:
                    table_name = row[0]
                    # Drop the data table
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    # Delete metadata
                    cursor.execute(
                        "DELETE FROM saved_datasets WHERE dataset_name = ?",
                        (dataset_name,)
                    )
                    conn.commit()
                    logger.info("Deleted dataset '%s'", dataset_name)
                    return True
                else:
                    logger.warning("Dataset '%s' not found for deletion", dataset_name)
                    return False
                    
        except Exception as e:
            logger.error("Failed to delete dataset '%s': %s", dataset_name, e, exc_info=True)
            return False
    
    def get_dataset_info(self, dataset_name: str) -> Optional[dict]:
        """Get detailed information about a saved dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM saved_datasets WHERE dataset_name = ?
                """, (dataset_name,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'dataset_id': row['dataset_id'],
                        'dataset_name': row['dataset_name'],
                        'description': row['description'],
                        'original_filename': row['original_filename'],
                        'rows': row['rows'],
                        'columns': row['columns'],
                        'column_names': json.loads(row['column_names']) if row['column_names'] else [],
                        'dtypes': json.loads(row['dtypes']) if row['dtypes'] else {},
                        'size_mb': round(row['size_bytes'] / (1024 * 1024), 2) if row['size_bytes'] else 0,
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'accessed_at': row['accessed_at'],
                        'tags': json.loads(row['tags']) if row['tags'] else []
                    }
                return None
                
        except Exception as e:
            logger.error("Failed to get info for dataset '%s': %s", dataset_name, e, exc_info=True)
            return None
