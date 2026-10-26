"""Central logging: rotating files, console output, and an in-memory error ring buffer.

Three sinks:
  * ``data/logs/kage.log`` - everything at INFO and above, rotated at 2 MB x5.
  * ``data/logs/errors.log``      - WARNING and above only, so problems are easy to find.
  * an in-process ring buffer     - served by ``GET /api/logs`` and the Diagnostics screen.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR, env

LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAIN_LOG = LOG_DIR / "kage.log"
ERROR_LOG = LOG_DIR / "errors.log"

LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False
_lock = threading.Lock()


class RingBufferHandler(logging.Handler):
    """Keeps the most recent records in memory for the diagnostics endpoint."""

    def __init__(self, capacity: int = 500):
        super().__init__()
        self.records: deque[dict] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "line": record.lineno,
                "message": record.getMessage(),
            }
            if record.exc_info:
                entry["traceback"] = "".join(traceback.format_exception(*record.exc_info))[-3000:]
            self.records.append(entry)
        except Exception:  # a logging handler must never raise
            pass

    def tail(self, limit: int = 100, min_level: str = "DEBUG") -> list[dict]:
        order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        threshold = order.get(min_level.upper(), 10)
        selected = [r for r in self.records if order.get(r["level"], 0) >= threshold]
        return selected[-limit:][::-1]  # newest first


RING = RingBufferHandler()


def setup_logging(level: str | None = None, console: bool = True) -> logging.Logger:
    """Idempotent: safe to call from the API, the CLI and the test suite."""
    global _configured
    with _lock:
        if _configured:
            return logging.getLogger("kage")

        level_name = (level or env("LOG_LEVEL") or "INFO").upper()
        resolved = getattr(logging, level_name, logging.INFO)

        root = logging.getLogger()
        root.setLevel(min(resolved, logging.INFO))
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

        main_handler = logging.handlers.RotatingFileHandler(
            MAIN_LOG, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        main_handler.setLevel(resolved)
        main_handler.setFormatter(formatter)
        root.addHandler(main_handler)

        error_handler = logging.handlers.RotatingFileHandler(
            ERROR_LOG, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        root.addHandler(error_handler)

        RING.setLevel(logging.DEBUG)
        root.addHandler(RING)

        if console:
            stream = logging.StreamHandler(sys.stdout)
            stream.setLevel(resolved)
            stream.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
            root.addHandler(stream)

        # These are noisy at INFO and drown out the app's own messages.
        for noisy in ("watchfiles", "urllib3", "PIL", "multipart", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _configured = True
        logging.getLogger("kage").info(
            "Logging ready - level=%s main=%s errors=%s", level_name, MAIN_LOG, ERROR_LOG
        )
        return logging.getLogger("kage")


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"kage.{name}")


def read_log_file(path: Path, limit: int = 200) -> list[str]:
    """Tail a log file without loading the whole thing."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in deque(handle, maxlen=limit)]
    except OSError:
        return []
