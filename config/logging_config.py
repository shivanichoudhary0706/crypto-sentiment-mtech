"""
Centralized logging setup.

Every module should call get_logger(__name__) instead of configuring
its own logging — this guarantees a consistent format across the
entire pipeline (ingestion, streaming, sentiment, analytics, api)
and writes to both console and a rotating log file.
"""

import logging
import os
import sys
from pathlib import Path
from config.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _init_root_logger() -> None:
    global _initialized
    if _initialized:
        return

    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotates at 5MB, keeps 5 backups, so logs/ doesn't grow unbounded
    # during a 24-48 hour continuous run (Week 11 integration testing)
        # One file per process: on Windows several processes cannot rotate one shared file
    # (WinError 32). No rotation; a 3-day run stays small.
    arg0 = sys.argv[0] if sys.argv else ""
    stem = Path(arg0).stem if arg0 not in ("", "-c") else "proc"
    log_file = log_dir / f"{stem}_{os.getpid()}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Usage in any module:
        from config.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("message")
    """
    _init_root_logger()
    return logging.getLogger(name)