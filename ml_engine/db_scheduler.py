"""
AceML Studio – Database Extraction Scheduler
==============================================
Schedule periodic data extraction from databases for ML training.

Features:
  • APScheduler integration
  • Configurable intervals
  • Background job execution
  • Error handling and retry logic
  • Integration with existing pipeline
"""

import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

from .db_connector import DatabaseConnector, DatabaseConfigManager, ExtractionQueryManager

logger = logging.getLogger("aceml.db_scheduler")

# Lazy import to avoid circular dependencies
def _get_retraining_pipeline():
    from .retraining_engine import get_retraining_pipeline  # noqa: PLC0415
    return get_retraining_pipeline()


def _get_retraining_job_manager():
    from .retraining_engine import get_retraining_job_manager  # noqa: PLC0415
    return get_retraining_job_manager()

# ═══════════════════════════════════════════════════════════════════
#  Database Extraction Scheduler
# ═══════════════════════════════════════════════════════════════════

class DatabaseExtractionScheduler:
    """
    Schedule and manage periodic data extraction from databases.
    """
    
    def __init__(self):
        # Configure APScheduler
        jobstores = {
            'default': MemoryJobStore()
        }
        executors = {
            'default': ThreadPoolExecutor(max_workers=3)
        }
        job_defaults = {
            'coalesce': True,  # Combine missed runs
            'max_instances': 1,  # One instance per job
            'misfire_grace_time': 300  # 5 minutes grace
        }
        
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )
        
        self.config_manager = DatabaseConfigManager()
        self.query_manager = ExtractionQueryManager()
        
        # Set up uploads folder (default to ./uploads)
        self.uploads_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        os.makedirs(self.uploads_folder, exist_ok=True)
        
        self.active_jobs = {}
        self.extraction_history = []
        
        logger.info("DatabaseExtractionScheduler initialized")
    
    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")
            
            # Load and schedule saved queries with schedules enabled
            self._load_scheduled_queries()
    
    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler stopped")
    
    def _load_scheduled_queries(self):
        """Load queries with schedules enabled and add them to scheduler."""
        queries = self.query_manager.list_queries()
        
        for query_info in queries:
            query_id = query_info['id']
            query_config = self.query_manager.get_query(query_id)
            
            if query_config and query_config.get('schedule_enabled'):
                interval_minutes = query_config.get('schedule_interval_minutes', 60)
                
                self.schedule_extraction(
                    query_id=query_id,
                    interval_minutes=interval_minutes
                )
                
                logger.info(f"Loaded scheduled query: {query_config['name']} (every {interval_minutes} min)")
    
    def schedule_extraction(self,
                          query_id: str,
                          interval_minutes: int = 60,
                          start_immediately: bool = False) -> Optional[str]:
        """
        Schedule periodic data extraction.
        
        Args:
            query_id: ID of the extraction query
            interval_minutes: Extraction interval in minutes
            start_immediately: Whether to run extraction immediately
            
        Returns:
            Job ID if successful, None otherwise
        """
        try:
            query_config = self.query_manager.get_query(query_id)
            if not query_config:
                logger.error(f"Query not found: {query_id}")
                return None
            
            # Remove existing job if any
            if query_id in self.active_jobs:
                self.cancel_extraction(query_id)
            
            # Create trigger
            trigger = IntervalTrigger(minutes=interval_minutes)
            
            # Add job to scheduler
            job = self.scheduler.add_job(
                func=self._execute_extraction,
                trigger=trigger,
                args=[query_id],
                id=query_id,
                name=f"Extract: {query_config['name']}",
                replace_existing=True
            )
            
            self.active_jobs[query_id] = {
                'job_id': job.id,
                'query_name': query_config['name'],
                'interval_minutes': interval_minutes,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'scheduled_at': datetime.now().isoformat()
            }
            
            logger.info(f"Scheduled extraction for query '{query_config['name']}' every {interval_minutes} minutes")
            
            # Run immediately if requested
            if start_immediately:
                self._execute_extraction(query_id)
            
            return job.id
            
        except Exception as e:
            logger.error(f"Error scheduling extraction: {e}")
            return None
    
    def _execute_extraction(self, query_id: str):
        """
        Execute data extraction for a query.
        This is called by the scheduler.
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting scheduled extraction for query: {query_id}")
            
            # Get query configuration
            query_config = self.query_manager.get_query(query_id)
            if not query_config:
                logger.error(f"Query configuration not found: {query_id}")
                return
            
            # Get connection configuration
            connection_config = self.config_manager.get_connection(query_config['connection_id'])
            if not connection_config:
                logger.error(f"Connection configuration not found: {query_config['connection_id']}")
                return
            
            # Connect to database
            connector = DatabaseConnector()
            connected = connector.connect(
                db_type=connection_config['db_type'],
                host=connection_config['host'],
                port=connection_config['port'],
                database=connection_config['database'],
                username=connection_config['username'],
                password=connection_config['password'],
                driver=connection_config.get('driver')
            )
            
            if not connected:
                logger.error("Failed to connect to database")
                self._record_extraction_result(query_id, False, "Connection failed", start_time)
                return
            
            # Get excluded columns from feedback loop
            excluded_columns = self.query_manager.get_excluded_columns(query_id)
            if excluded_columns:
                logger.info(f"Excluding columns from feedback loop: {excluded_columns}")
            
            # Extract data
            df = connector.extract_data(
                table_name=query_config['table_name'],
                columns=query_config.get('columns'),
                date_column=query_config.get('date_column'),
                start_date=query_config.get('start_date'),
                end_date=query_config.get('end_date'),
                where_clause=query_config.get('where_clause'),
                excluded_columns=excluded_columns
            )
            
            connector.disconnect()
            
            if df is None or df.empty:
                logger.warning(f"No data extracted for query: {query_id}")
                self._record_extraction_result(query_id, False, "No data extracted", start_time)
                return
            
            # Save extracted data to uploads directory
            file_id = uuid.uuid4().hex[:8]
            filename = f"{file_id}_{query_config['table_name']}_scheduled.csv"
            filepath = os.path.join(self.uploads_folder, filename)
            
            df.to_csv(filepath, index=False)
            logger.info(f"Saved extracted data to: {filepath}")
            
            # Update query run statistics
            self.query_manager.update_run_stats(query_id, success=True)
            self.config_manager.update_last_used(query_config['connection_id'])
            
            # Record success
            elapsed = (datetime.now() - start_time).total_seconds()
            self._record_extraction_result(
                query_id, 
                True, 
                f"Extracted {len(df)} rows, {len(df.columns)} columns",
                start_time,
                rows=len(df),
                columns=len(df.columns),
                file_path=filepath,
                elapsed_seconds=elapsed
            )
            
            logger.info(f"Extraction completed successfully in {elapsed:.2f}s: {filename}")

            # ── Trigger linked retraining jobs ──────────────────────────────
            self._trigger_retraining_jobs(query_id, filepath)
            
        except Exception as e:
            logger.error(f"Error during extraction execution: {e}", exc_info=True)
            self._record_extraction_result(query_id, False, str(e), start_time)
    
    def _record_extraction_result(self,
                                  query_id: str,
                                  success: bool,
                                  message: str,
                                  start_time: datetime,
                                  rows: int = 0,
                                  columns: int = 0,
                                  file_path: str | None = None,
                                  elapsed_seconds: float = 0):
        """Record extraction result in history."""
        result = {
            'query_id': query_id,
            'timestamp': start_time.isoformat(),
            'success': success,
            'message': message,
            'rows': rows,
            'columns': columns,
            'file_path': file_path,
            'elapsed_seconds': elapsed_seconds
        }
        
        self.extraction_history.append(result)
        
        # Keep only last 100 results
        if len(self.extraction_history) > 100:
            self.extraction_history = self.extraction_history[-100:]
    
    def _trigger_retraining_jobs(self, query_id: str, data_file_path: str):
        """
        After a successful extraction, find all retraining jobs linked to this
        query and execute them in the same background thread.
        """
        try:
            job_manager = _get_retraining_job_manager()
            linked_jobs = job_manager.list_jobs_for_query(query_id)

            enabled_jobs = [j for j in linked_jobs if j.get("schedule_enabled", False)]
            if not enabled_jobs:
                return

            logger.info(
                "Triggering %d retraining job(s) for extraction query %s",
                len(enabled_jobs), query_id,
            )

            pipeline = _get_retraining_pipeline()
            for job in enabled_jobs:
                job_id = job["job_id"]
                try:
                    logger.info("  → Starting retraining job '%s' (%s)", job["name"], job_id)
                    result = pipeline.run(job_id=job_id, data_file_path=data_file_path)
                    status = result.get("status", "unknown")
                    version = result.get("new_model_version", "?")
                    promoted = result.get("promoted", False)
                    logger.info(
                        "  ✓ Retraining job '%s' finished – status=%s version=%s promoted=%s",
                        job["name"], status, version, promoted,
                    )
                except Exception as exc:
                    logger.error(
                        "  ✗ Retraining job '%s' (%s) raised an exception: %s",
                        job["name"], job_id, exc, exc_info=True,
                    )
        except Exception as exc:
            logger.error("_trigger_retraining_jobs error for query %s: %s", query_id, exc, exc_info=True)

    def schedule_retraining(self, retrain_job_id: str, interval_minutes: int = 60) -> Optional[str]:
        """
        Schedule a standalone (not extraction-linked) periodic retraining job.

        The scheduled function runs the full pipeline against the LAST extracted
        file for the associated extraction query.

        Args:
            retrain_job_id:   Retraining job ID (from RetrainingJobManager)
            interval_minutes: How often to retrain

        Returns:
            APScheduler job ID, or None on failure
        """
        try:
            job_manager = _get_retraining_job_manager()
            retrain_job = job_manager.get_job(retrain_job_id)
            if not retrain_job:
                logger.error("Retraining job not found: %s", retrain_job_id)
                return None

            sched_id = f"retrain_{retrain_job_id}"

            # Remove existing schedule first
            try:
                existing = self.scheduler.get_job(sched_id)
                if existing:
                    self.scheduler.remove_job(sched_id)
            except Exception:
                pass

            trigger = IntervalTrigger(minutes=interval_minutes)
            job = self.scheduler.add_job(
                func=self._run_retraining_standalone,
                trigger=trigger,
                args=[retrain_job_id],
                id=sched_id,
                name=f"Retrain: {retrain_job['name']}",
                replace_existing=True,
            )

            logger.info(
                "Standalone retraining schedule: '%s' every %d min",
                retrain_job["name"], interval_minutes,
            )
            return job.id

        except Exception as exc:
            logger.error("schedule_retraining error: %s", exc)
            return None

    def _run_retraining_standalone(self, retrain_job_id: str):
        """
        Standalone scheduler callback: find the most recent extracted file for
        the job's linked query and run the retraining pipeline against it.
        """
        try:
            job_manager = _get_retraining_job_manager()
            retrain_job = job_manager.get_job(retrain_job_id)
            if not retrain_job:
                logger.error("Standalone retrain: job not found: %s", retrain_job_id)
                return

            query_id = retrain_job.get("query_id", "")

            # Find the most recent extraction file for this query
            latest_file = self._find_latest_extraction_file(query_id)
            if not latest_file:
                logger.warning(
                    "Standalone retrain: no extraction file found for query %s – "
                    "run 'Extract Now' first or link to an active extraction schedule",
                    query_id,
                )
                return

            pipeline = _get_retraining_pipeline()
            result = pipeline.run(job_id=retrain_job_id, data_file_path=latest_file)
            logger.info(
                "Standalone retrain '%s' done – status=%s version=%s promoted=%s",
                retrain_job["name"],
                result.get("status"),
                result.get("new_model_version"),
                result.get("promoted"),
            )
        except Exception as exc:
            logger.error("_run_retraining_standalone error: %s", exc, exc_info=True)

    def _find_latest_extraction_file(self, query_id: str) -> Optional[str]:
        """Return the path of the most recently extracted file for a given query."""
        # Check history records first (most reliable source)
        history = self.get_extraction_history(query_id=query_id, limit=20)
        for record in history:
            fp = record.get("file_path")
            if fp and os.path.exists(fp):
                return fp
        # Fallback: scan uploads folder for files matching the query's table name
        query_config = self.query_manager.get_query(query_id)
        if query_config:
            table = query_config.get("table_name", "")
            candidates = [
                os.path.join(self.uploads_folder, f)
                for f in os.listdir(self.uploads_folder)
                if table in f and f.endswith(".csv")
            ]
            if candidates:
                return max(candidates, key=os.path.getmtime)
        return None

    def cancel_extraction(self, query_id: str) -> bool:
        """Cancel scheduled extraction."""
        try:
            if query_id in self.active_jobs:
                self.scheduler.remove_job(query_id)
                del self.active_jobs[query_id]
                logger.info(f"Cancelled scheduled extraction: {query_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling extraction: {e}")
            return False
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of active scheduled jobs."""
        result = []
        
        for query_id, job_info in self.active_jobs.items():
            # Get next run time from scheduler
            job = self.scheduler.get_job(query_id)
            if job:
                job_info_copy = job_info.copy()
                job_info_copy['next_run'] = job.next_run_time.isoformat() if job.next_run_time else None
                result.append(job_info_copy)
        
        return result
    
    def get_extraction_history(self, query_id: str | None = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get extraction history, optionally filtered by query_id."""
        if query_id:
            filtered = [h for h in self.extraction_history if h['query_id'] == query_id]
        else:
            filtered = self.extraction_history
        
        # Return most recent first
        return sorted(filtered, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def run_extraction_now(self, query_id: str) -> bool:
        """Run extraction immediately (outside of schedule)."""
        try:
            self._execute_extraction(query_id)
            return True
        except Exception as e:
            logger.error(f"Error running extraction: {e}")
            return False
    
    def update_schedule(self, query_id: str, interval_minutes: int) -> bool:
        """Update the schedule interval for an extraction."""
        try:
            if query_id in self.active_jobs:
                # Reschedule with new interval
                return self.schedule_extraction(query_id, interval_minutes) is not None
            return False
        except Exception as e:
            logger.error(f"Error updating schedule: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════
#  Global Scheduler Instance
# ═══════════════════════════════════════════════════════════════════

# Singleton instance
_scheduler_instance = None

def get_scheduler() -> DatabaseExtractionScheduler:
    """Get the global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = DatabaseExtractionScheduler()
        _scheduler_instance.start()
    return _scheduler_instance
