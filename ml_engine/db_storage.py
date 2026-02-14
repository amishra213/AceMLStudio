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
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Create metadata table
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
