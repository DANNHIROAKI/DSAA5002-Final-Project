"""Logging configuration utilities for experiments and scripts.

Design goals (aligned with the new methodology):
- Stable, non-duplicated logs for both CLI and notebooks.
- Optional file logging for reproducibility.
- Minimal surprises: no implicit file creation unless log_file is provided.

Usage
-----
>>> from customer_segmentation.src.utils.logging_utils import configure_logging
>>> logger = configure_logging()
>>> logger.info("hello")
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
    """Configure and return a logger used across experiment entrypoints.

    This helper clears existing handlers (when force=True) to avoid duplicated logs
    when running notebooks or re-importing modules, wires a console handler, and
    optionally writes logs to `log_file`.

    Parameters
    ----------
    level :
        Desired log level (INFO by default).
    log_file :
        Optional path to a log file. Parent directories are created when needed.
        If given as a directory path, the log file name defaults to "<logger_name or root>.log".
    logger_name :
        Name of the logger to configure. None configures the root logger.
    force :
        If True (default), remove any existing handlers from the target logger
        before adding new ones (prevents duplicate outputs).
    capture_warnings :
        If True (default), route Python warnings through logging.

    Returns
    -------
    logging.Logger
        The configured logger instance.
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
