"""
Centralized structured logger for Future Alignment Studio.
"""

from __future__ import annotations
import logging
import sys

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get or create a named console logger with formatting."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.propagate = False
    _LOGGERS[name] = logger
    return logger

