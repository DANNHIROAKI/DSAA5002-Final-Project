"""Logging configuration utilities for experiments and scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger used across experiment entrypoints.

    This helper removes existing handlers (to avoid duplicated logs when
    running notebooks or re-importing modules), wires a console handler,
    and optionally writes logs to ``log_file``.

    Parameters
    ----------
    level :
        Desired log level (INFO by default).
    log_file :
        Optional path to a log file. Parent directories are created when needed.
    logger_name :
        Name of the logger to configure. ``None`` configures the root logger.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate outputs in notebooks/CLI.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid propagating to ancestor loggers (esp. the root) to prevent duplicates.
    logger.propagate = False
    return logger


__all__ = ["configure_logging", "DEFAULT_LOG_FORMAT"]
