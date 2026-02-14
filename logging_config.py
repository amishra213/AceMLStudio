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
"""

import os
import logging
import logging.handlers
from collections import deque
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── In-memory ring buffer for recent key log messages ────────────
# Stores the last N formatted log records so the chat assistant
# can include pipeline context without reading log files.
_MAX_BUFFER = 200
_log_buffer: deque[dict] = deque(maxlen=_MAX_BUFFER)


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

    # ── Startup banner ───────────────────────────────────────────────
    logging.getLogger("aceml.startup").info(
        "Logging initialised  |  level=%s  |  log_dir=%s", level, LOG_DIR
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the aceml namespace."""
    return logging.getLogger(f"aceml.{name}")
