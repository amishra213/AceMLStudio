"""
AceML Studio – Logging Configuration
======================================
Centralized logging setup with console + file handlers.
Provides structured logging for debugging, observability, and alerting.

Log Levels Used:
  - DEBUG   : Detailed internal state (data shapes, param values, timing)
  - INFO    : Normal operational events (request received, model trained, file loaded)
  - WARNING : Recoverable anomalies (missing config, fallback used, high missing %)
  - ERROR   : Operation failures (file load error, model training failure)
  - CRITICAL: System-level failures (LLM unreachable, config file missing)

Task-Level Structured Logging (for LLM chat context)
------------------------------------------------------
Every major pipeline step emits a TaskEvent dict captured in a per-session
PIPELINE_LOG.  The chat endpoint forwards the full pipeline log to the LLM
so it can give personalised, data-aware guidance.

Usage (in ml_engine or app.py):
    from logging_config import task_event

    with task_event(session_id, "data_cleaning", inputs={"rows": 1000}) as te:
        df = DataCleaner.apply_operations(df, ops)
        te["outputs"] = {"rows": len(df), "ops_applied": len(ops)}

If an exception is raised inside the `with` block, the event is automatically
marked status="error" and the traceback is stored in te["error"].
"""

import os
import time
import logging
import logging.handlers
import traceback
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone, UTC
from typing import Generator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── In-memory ring buffer for recent key log messages ────────────
# Stores the last N formatted log records so the chat assistant
# can include pipeline context without reading log files.
_MAX_BUFFER = 200
_log_buffer: deque[dict] = deque(maxlen=_MAX_BUFFER)

# ── Per-session structured pipeline event log ─────────────────────
# Maps session_id → deque of TaskEvent dicts.
# Each TaskEvent captures one pipeline task (upload, clean, train …)
# with rich metadata for the LLM to reason about.
_MAX_PIPELINE_EVENTS = 100
PIPELINE_LOG: dict[str, deque] = {}


def _session_pipeline_log(session_id: str) -> deque:
    """Return (or create) the pipeline event deque for *session_id*."""
    if session_id not in PIPELINE_LOG:
        PIPELINE_LOG[session_id] = deque(maxlen=_MAX_PIPELINE_EVENTS)
    return PIPELINE_LOG[session_id]


@contextmanager
def task_event(
    session_id: str,
    task_name: str,
    inputs: dict | None = None,
) -> Generator[dict, None, None]:
    """
    Context manager that records a structured TaskEvent for a pipeline step.

    Yields a mutable *event* dict so the caller can attach outputs::

        with task_event(sid, "model_training", inputs={"model": "rf"}) as ev:
            model, info = ModelTrainer.train(...)
            ev["outputs"] = {"train_score": info["train_score"], ...}

    The event is appended to the session's PIPELINE_LOG and emitted as an
    INFO log record when the block exits (success or error).
    """
    event: dict = {
        "task": task_name,
        "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "running",
        "inputs": inputs or {},
        "outputs": {},
        "duration_sec": None,
        "error": None,
        "warnings": [],
    }
    _t0 = time.perf_counter()
    _logger = logging.getLogger(f"aceml.task.{task_name}")

    try:
        yield event
        event["duration_sec"] = round(time.perf_counter() - _t0, 3)
        event["status"] = "success"
        _logger.info(
            "TASK %-28s | status=%-7s | duration=%.3fs | inputs=%s | outputs=%s",
            task_name, "success", event["duration_sec"],
            _compact(event["inputs"]), _compact(event["outputs"]),
        )
    except Exception as exc:
        event["duration_sec"] = round(time.perf_counter() - _t0, 3)
        event["status"] = "error"
        event["error"] = str(exc)
        event["traceback"] = traceback.format_exc(limit=6)
        _logger.error(
            "TASK %-28s | status=%-7s | duration=%.3fs | error=%s",
            task_name, "error", event["duration_sec"], exc,
        )
        raise
    finally:
        _session_pipeline_log(session_id).append(event)


def _compact(d: dict) -> str:
    """Return a short string representation of a dict for log lines."""
    if not d:
        return "{}"
    parts = []
    for k, v in list(d.items())[:6]:
        if isinstance(v, (list, dict)):
            parts.append(f"{k}={type(v).__name__}[{len(v)}]")
        else:
            parts.append(f"{k}={v!r}"[:60])
    return "{" + ", ".join(parts) + ("…" if len(d) > 6 else "") + "}"


def get_pipeline_log(session_id: str, n: int = 30) -> list[dict]:
    """
    Return the most recent *n* pipeline events for *session_id*.

    Each entry is a rich dict suitable for inclusion in an LLM system prompt.
    """
    log = _session_pipeline_log(session_id)
    events = list(log)[-n:]
    # Return lightweight copies (strip large tracebacks for LLM prompts)
    result = []
    for ev in events:
        copy = dict(ev)
        if copy.get("traceback"):
            # Keep only last 3 lines of traceback to save tokens
            lines = copy["traceback"].strip().splitlines()
            copy["traceback"] = "\n".join(lines[-3:])
        result.append(copy)
    return result


def clear_pipeline_log(session_id: str) -> None:
    """Remove all pipeline events for *session_id* (e.g. on session reset)."""
    PIPELINE_LOG.pop(session_id, None)


class _BufferHandler(logging.Handler):
    """Captures INFO+ log records into an in-memory ring buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_buffer.append({
                "ts": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            })
        except Exception:
            pass  # never break the app because of buffering


def get_recent_logs(n: int = 50, min_level: str = "INFO") -> list[dict]:
    """Return the most recent *n* log entries at or above *min_level*."""
    min_val = getattr(logging, min_level.upper(), logging.INFO)
    level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    out = []
    for entry in reversed(_log_buffer):
        if level_map.get(entry["level"], 0) >= min_val:
            out.append(entry)
            if len(out) >= n:
                break
    out.reverse()
    return out


def setup_logging(level: str = "DEBUG") -> None:
    """
    Configure root logger with console and rotating file handlers.
    Call once at application startup.
    """
    log_level = getattr(logging, level.upper(), logging.DEBUG)

    # ── Root logger ──────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(log_level)

    # Prevent duplicate handlers on re-init
    if root.handlers:
        return

    # ── Format ───────────────────────────────────────────────────────
    detailed_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Console Handler ──────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # ── Rotating File Handler (all logs) ─────────────────────────────
    all_log = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "aceml_studio.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    all_log.setLevel(logging.DEBUG)
    all_log.setFormatter(detailed_fmt)
    root.addHandler(all_log)

    # ── Error-only File Handler (alerts / critical) ──────────────────
    error_log = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "aceml_errors.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    error_log.setLevel(logging.ERROR)
    error_log.setFormatter(detailed_fmt)
    root.addHandler(error_log)

    # ── In-memory Buffer Handler (for chat context) ────────────────
    buf_handler = _BufferHandler()
    buf_handler.setLevel(logging.INFO)
    buf_handler.setFormatter(console_fmt)
    root.addHandler(buf_handler)

    # ── Structured JSONL Task Log (task.* loggers only) ──────────────
    # Each task_event writes a single INFO log line; this handler appends
    # full enriched records to tasks.jsonl for offline inspection.
    task_file = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "tasks.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    task_file.setLevel(logging.DEBUG)
    task_file.setFormatter(detailed_fmt)
    task_file.addFilter(lambda r: r.name.startswith("aceml.task."))
    root.addHandler(task_file)

    # ── Startup banner ───────────────────────────────────────────────
    logging.getLogger("aceml.startup").info(
        "Logging initialised  |  level=%s  |  log_dir=%s", level, LOG_DIR
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the aceml namespace."""
    return logging.getLogger(f"aceml.{name}")
