"""
AceML Studio – Database Connector
====================================
Connect to various databases and extract data for ML training.
Supports Azure SQL Server, PostgreSQL, MySQL, SQLite, Oracle, etc.

Features:
  • Multiple database support
  • Connection pooling
  • Column selection and filtering
  • Date range queries
  • Scheduled extraction
  • Feedback loop for dropped columns
"""

import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from cryptography.fernet import Fernet
import base64

import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, select, inspect,
    text, Column, String, Integer, DateTime, Boolean
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from config import Config

logger = logging.getLogger("aceml.db_connector")

# ═══════════════════════════════════════════════════════════════════
#  Database Connection Manager
# ═══════════════════════════════════════════════════════════════════

class DatabaseConnector:
    """
    Connect to various databases and extract data for ML training.
    """
    
    SUPPORTED_DATABASES = {
        'azure_sql': 'mssql+pyodbc',
        'sql_server': 'mssql+pyodbc',
        'postgresql': 'postgresql+psycopg2',
        'mysql': 'mysql+pymysql',
        'sqlite': 'sqlite',
        'oracle': 'oracle+cx_oracle',
        'snowflake': 'snowflake'
    }
    
    def __init__(self):
        self.engine = None
        self.connection_config = None
        self.metadata = None
        self.session = None
        
    def connect(self, 
                db_type: str,
                host: str | None = None,
                port: int | None = None,
                database: str | None = None,
                username: str | None = None,
                password: str | None = None,
                driver: str | None = None,
                additional_params: Dict[str, str] | None = None) -> bool:
        """
        Establish database connection.
        
        Args:
            db_type: Database type ('azure_sql', 'postgresql', 'mysql', etc.)
            host: Database host/server
            port: Database port
            database: Database name
            username: Username
            password: Password
            driver: ODBC driver (for SQL Server)
            additional_params: Additional connection parameters
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Build connection string
            connection_string = self._build_connection_string(
                db_type, host, port, database, username, password, driver, additional_params
            )
            
            # Create engine with connection pooling
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Test connections before using
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
            self.connection_config = {
                'db_type': db_type,
                'host': host,
                'port': port,
                'database': database,
                'username': username,
                'connected_at': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully connected to {db_type} database: {database}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def _build_connection_string(self,
                                 db_type: str,
                                 host: str | None,
                                 port: int | None,
                                 database: str | None,
                                 username: str | None,
                                 password: str | None,
                                 driver: str | None,
                                 additional_params: Dict[str, str] | None) -> str:
        """Build database connection string based on type."""
        
        if db_type not in self.SUPPORTED_DATABASES:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        dialect = self.SUPPORTED_DATABASES[db_type]
        
        # Azure SQL Server / SQL Server
        if db_type in ['azure_sql', 'sql_server']:
            driver = driver or 'ODBC Driver 17 for SQL Server'
            conn_str = f"{dialect}://{username}:{password}@{host}"
            if port:
                conn_str += f":{port}"
            conn_str += f"/{database}?driver={driver}"
            if additional_params:
                for key, value in additional_params.items():
                    conn_str += f"&{key}={value}"
            return conn_str
        
        # PostgreSQL
        elif db_type == 'postgresql':
            port = port or 5432
            return f"{dialect}://{username}:{password}@{host}:{port}/{database}"
        
        # MySQL
        elif db_type == 'mysql':
            port = port or 3306
            return f"{dialect}://{username}:{password}@{host}:{port}/{database}"
        
        # SQLite
        elif db_type == 'sqlite':
            return f"sqlite:///{database}"
        
        # Oracle
        elif db_type == 'oracle':
            port = port or 1521
            return f"{dialect}://{username}:{password}@{host}:{port}/{database}"
        
        # Snowflake
        elif db_type == 'snowflake':
            return f"snowflake://{username}:{password}@{host}/{database}"
        
        else:
            raise ValueError(f"Connection string builder not implemented for: {db_type}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test current database connection and return status."""
        if not self.engine:
            return {'success': False, 'message': 'No active connection'}
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            return {
                'success': True,
                'message': 'Connection successful',
                'database': self.connection_config.get('database') if self.connection_config else None,
                'db_type': self.connection_config.get('db_type') if self.connection_config else None
            }
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return {'success': False, 'message': str(e)}
    
    def list_tables(self) -> List[str]:
        """List all tables in the connected database."""
        if not self.engine:
            logger.warning("No active database connection")
            return []
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            logger.info(f"Found {len(tables)} tables in database")
            return sorted(tables)
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a specific table.
        
        Returns:
            List of dicts with column info: name, type, nullable, primary_key
        """
        if not self.engine:
            logger.warning("No active database connection")
            return []
        
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            # Get primary keys
            pk_constraint = inspector.get_pk_constraint(table_name)
            primary_keys = pk_constraint.get('constrained_columns', [])
            
            result = []
            for col in columns:
                result.append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col['nullable'],
                    'primary_key': col['name'] in primary_keys,
                    'default': str(col.get('default', ''))
                })
            
            logger.info(f"Retrieved {len(result)} columns for table: {table_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting columns for {table_name}: {e}")
            return []
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get approximate row count for a table."""
        try:
            if not self.engine:
                logger.error("Database engine not initialized. Call connect() first.")
                return 0
            
            query = text(f"SELECT COUNT(*) as count FROM {table_name}")
            with self.engine.connect() as conn:
                result = conn.execute(query)
                row = result.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {e}")
            return 0
    
    def extract_data(self,
                    table_name: str,
                    columns: List[str] | None = None,
                    date_column: str | None = None,
                    start_date: str | None = None,
                    end_date: str | None = None,
                    where_clause: str | None = None,
                    limit: int | None = None,
                    excluded_columns: List[str] | None = None) -> Optional[pd.DataFrame]:
        """
        Extract data from database table.
        
        Args:
            table_name: Name of the table
            columns: List of column names to extract (None = all columns)
            date_column: Column to use for date filtering
            start_date: Start date for filtering (ISO format)
            end_date: End date for filtering (ISO format)
            where_clause: Additional WHERE clause
            limit: Maximum number of rows to fetch
            excluded_columns: Columns to exclude (from feedback loop)
            
        Returns:
            DataFrame with extracted data or None if error
        """
        try:
            if not self.engine:
                logger.error("Database engine not initialized. Call connect() first.")
                return None
            
            # Build SELECT clause
            if columns:
                # Filter out excluded columns
                if excluded_columns:
                    columns = [col for col in columns if col not in excluded_columns]
                
                if not columns:
                    logger.warning("All selected columns were excluded by feedback loop")
                    return None
                
                select_clause = ", ".join([f'"{col}"' for col in columns])
            else:
                select_clause = "*"
            
            # Build query
            query = f"SELECT {select_clause} FROM {table_name}"
            
            # Add WHERE conditions
            conditions = []
            
            if date_column and (start_date or end_date):
                if start_date:
                    conditions.append(f"{date_column} >= '{start_date}'")
                if end_date:
                    conditions.append(f"{date_column} <= '{end_date}'")
            
            if where_clause:
                conditions.append(f"({where_clause})")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Add LIMIT
            if limit:
                query += f" LIMIT {limit}"
            
            logger.info(f"Executing query: {query}")
            
            # Execute query and load into DataFrame
            df = pd.read_sql(query, self.engine)
            
            logger.info(f"Extracted {len(df)} rows with {len(df.columns)} columns from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting data from {table_name}: {e}")
            return None
    
    def get_date_range(self, table_name: str, date_column: str) -> Dict[str, Any]:
        """Get min and max dates from a date column."""
        try:
            query = text(f"""
                SELECT 
                    MIN({date_column}) as min_date,
                    MAX({date_column}) as max_date
                FROM {table_name}
            """)
            
            if not self.engine:
                return {'min_date': None, 'max_date': None}
            
            with self.engine.connect() as conn:
                result = conn.execute(query)
                row = result.fetchone()
                
                if row:
                    return {
                        'min_date': row[0].isoformat() if row[0] else None,
                        'max_date': row[1].isoformat() if row[1] else None
                    }
            
            return {'min_date': None, 'max_date': None}
            
        except Exception as e:
            logger.error(f"Error getting date range: {e}")
            return {'min_date': None, 'max_date': None}
    
    def disconnect(self):
        """Close database connection."""
        try:
            if self.session:
                self.session.close()
            if self.engine:
                self.engine.dispose()
            
            self.engine = None
            self.session = None
            self.metadata = None
            self.connection_config = None
            
            logger.info("Database connection closed")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")


# ═══════════════════════════════════════════════════════════════════
#  Database Configuration Manager
# ═══════════════════════════════════════════════════════════════════

class DatabaseConfigManager:
    """
    Manage saved database connections and extraction configurations.
    Encrypts sensitive credentials.
    """
    
    def __init__(self, config_file: str = "db_connections.json"):
        self.config_file = config_file
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        self.connections = self._load_connections()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for credentials."""
        key_file = "db_encryption.key"
        try:
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    return f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                logger.info("Created new encryption key for DB credentials")
                return key
        except Exception as e:
            logger.error(f"Error with encryption key: {e}")
            # Fallback to hardcoded key (not secure for production!)
            return base64.urlsafe_b64encode(hashlib.sha256(b"aceml_default_key").digest())
    
    def _load_connections(self) -> Dict[str, Any]:
        """Load saved connections from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading connections: {e}")
            return {}
    
    def _save_connections(self):
        """Save connections to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.connections, f, indent=2)
            logger.info(f"Saved {len(self.connections)} database connections")
        except Exception as e:
            logger.error(f"Error saving connections: {e}")
    
    def encrypt_password(self, password: str) -> str:
        """Encrypt password."""
        try:
            return self.cipher.encrypt(password.encode()).decode()
        except Exception as e:
            logger.error(f"Error encrypting password: {e}")
            return password
    
    def decrypt_password(self, encrypted_password: str) -> str:
        """Decrypt password."""
        try:
            return self.cipher.decrypt(encrypted_password.encode()).decode()
        except Exception as e:
            logger.error(f"Error decrypting password: {e}")
            return encrypted_password
    
    def save_connection(self,
                       connection_id: str,
                       name: str,
                       db_type: str,
                       host: str,
                       port: int,
                       database: str,
                       username: str,
                       password: str,
                       driver: str | None = None) -> bool:
        """Save database connection configuration."""
        try:
            encrypted_password = self.encrypt_password(password)
            
            self.connections[connection_id] = {
                'name': name,
                'db_type': db_type,
                'host': host,
                'port': port,
                'database': database,
                'username': username,
                'password': encrypted_password,
                'driver': driver,
                'created_at': datetime.now().isoformat(),
                'last_used': None
            }
            
            self._save_connections()
            logger.info(f"Saved connection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving connection: {e}")
            return False
    
    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection configuration by ID."""
        config = self.connections.get(connection_id)
        if config:
            # Decrypt password
            config_copy = config.copy()
            config_copy['password'] = self.decrypt_password(config['password'])
            return config_copy
        return None
    
    def list_connections(self) -> List[Dict[str, Any]]:
        """List all saved connections (without passwords)."""
        result = []
        for conn_id, config in self.connections.items():
            result.append({
                'id': conn_id,
                'name': config['name'],
                'db_type': config['db_type'],
                'database': config['database'],
                'host': config['host'],
                'created_at': config['created_at'],
                'last_used': config.get('last_used')
            })
        return result
    
    def delete_connection(self, connection_id: str) -> bool:
        """Delete a saved connection."""
        if connection_id in self.connections:
            del self.connections[connection_id]
            self._save_connections()
            logger.info(f"Deleted connection: {connection_id}")
            return True
        return False
    
    def update_last_used(self, connection_id: str):
        """Update last used timestamp."""
        if connection_id in self.connections:
            self.connections[connection_id]['last_used'] = datetime.now().isoformat()
            self._save_connections()


# ═══════════════════════════════════════════════════════════════════
#  Extraction Query Manager
# ═══════════════════════════════════════════════════════════════════

class ExtractionQueryManager:
    """
    Manage saved extraction queries and schedules.
    Implements feedback loop for dropped columns.
    """
    
    def __init__(self, config_file: str = "extraction_queries.json"):
        self.config_file = config_file
        self.queries = self._load_queries()
        self.feedback_file = "column_feedback.json"
        self.dropped_columns = self._load_dropped_columns()
    
    def _load_queries(self) -> Dict[str, Any]:
        """Load saved queries from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            return {}
    
    def _save_queries(self):
        """Save queries to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.queries, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving queries: {e}")
    
    def _load_dropped_columns(self) -> Dict[str, List[str]]:
        """Load feedback of dropped columns."""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading dropped columns: {e}")
            return {}
    
    def _save_dropped_columns(self):
        """Save dropped columns feedback."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.dropped_columns, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving dropped columns: {e}")
    
    def save_query(self,
                  query_id: str,
                  name: str,
                  connection_id: str,
                  table_name: str,
                  columns: List[str],
                  date_column: str | None = None,
                  start_date: str | None = None,
                  end_date: str | None = None,
                  where_clause: str | None = None,
                  schedule_enabled: bool = False,
                  schedule_interval_minutes: int = 60) -> bool:
        """Save extraction query configuration."""
        try:
            self.queries[query_id] = {
                'name': name,
                'connection_id': connection_id,
                'table_name': table_name,
                'columns': columns,
                'date_column': date_column,
                'start_date': start_date,
                'end_date': end_date,
                'where_clause': where_clause,
                'schedule_enabled': schedule_enabled,
                'schedule_interval_minutes': schedule_interval_minutes,
                'created_at': datetime.now().isoformat(),
                'last_run': None,
                'last_success': None,
                'run_count': 0
            }
            
            self._save_queries()
            logger.info(f"Saved query: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving query: {e}")
            return False
    
    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get query configuration by ID."""
        return self.queries.get(query_id)
    
    def list_queries(self) -> List[Dict[str, Any]]:
        """List all saved queries."""
        result = []
        for query_id, config in self.queries.items():
            result.append({
                'id': query_id,
                'name': config['name'],
                'table_name': config['table_name'],
                'schedule_enabled': config.get('schedule_enabled', False),
                'schedule_interval_minutes': config.get('schedule_interval_minutes'),
                'last_run': config.get('last_run'),
                'run_count': config.get('run_count', 0)
            })
        return result
    
    def delete_query(self, query_id: str) -> bool:
        """Delete a saved query."""
        if query_id in self.queries:
            del self.queries[query_id]
            self._save_queries()
            logger.info(f"Deleted query: {query_id}")
            return True
        return False
    
    def update_run_stats(self, query_id: str, success: bool = True):
        """Update query run statistics."""
        if query_id in self.queries:
            self.queries[query_id]['last_run'] = datetime.now().isoformat()
            self.queries[query_id]['run_count'] = self.queries[query_id].get('run_count', 0) + 1
            if success:
                self.queries[query_id]['last_success'] = datetime.now().isoformat()
            self._save_queries()
    
    def record_dropped_column(self, query_id: str, column_name: str):
        """Record that a column was dropped in the pipeline (feedback loop)."""
        if query_id not in self.dropped_columns:
            self.dropped_columns[query_id] = []
        
        if column_name not in self.dropped_columns[query_id]:
            self.dropped_columns[query_id].append(column_name)
            self._save_dropped_columns()
            logger.info(f"Recorded dropped column '{column_name}' for query {query_id}")
    
    def get_excluded_columns(self, query_id: str) -> List[str]:
        """Get list of columns to exclude for a query (from feedback loop)."""
        return self.dropped_columns.get(query_id, [])
    
    def clear_dropped_columns(self, query_id: str):
        """Clear feedback for a query."""
        if query_id in self.dropped_columns:
            del self.dropped_columns[query_id]
            self._save_dropped_columns()
            logger.info(f"Cleared dropped columns for query {query_id}")


import os
