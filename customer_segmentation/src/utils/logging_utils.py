"""Logging configuration utilities."""
import logging


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a root logger."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    return logging.getLogger()
