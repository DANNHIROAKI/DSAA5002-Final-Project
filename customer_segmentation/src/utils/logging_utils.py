"""Logging configuration utilities for experiments and scripts.

Design goals
------------
- Stable logs for both CLI and notebooks (avoid duplicate handlers).
- Optional file logging for reproducibility.
- Minimal surprises: no implicit file creation unless ``log_file`` is provided.

Used by
-------
- ``customer_segmentation/run_all_experiments.py``
- ``customer_segmentation/src/experiments/*.py``

"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    logger_name: Optional[str] = None,
    *,
    force: bool = True,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure and return a logger.

    Parameters
    ----------
    level:
        Log level (default: INFO).
    log_file:
        Optional path to a log file. If a directory is provided, the log file
        name defaults to ``<logger_name or root>.log``.
    logger_name:
        Name of the logger to configure. ``None`` configures the root logger.
    force:
        If True (default), remove existing handlers to prevent duplicate logs.
    capture_warnings:
        If True (default), route Python warnings through logging.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Console handler (stderr)
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)

        # If user passes a directory, auto-generate file name.
        if log_path.exists() and log_path.is_dir():
            name = (logger_name or "root").replace("/", "_")
            log_path = log_path / f"{name}.log"
        elif str(log_path).endswith(("/", "\\")):
            name = (logger_name or "root").replace("/", "_")
            log_path = log_path / f"{name}.log"

        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid propagating to ancestor loggers (esp. root) to prevent duplicates.
    logger.propagate = False

    if capture_warnings:
        logging.captureWarnings(True)

    return logger


__all__ = ["configure_logging", "DEFAULT_LOG_FORMAT"]
