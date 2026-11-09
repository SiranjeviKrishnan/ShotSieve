"""Logging setup."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int | str = logging.INFO, logfile: str | None = None) -> logging.Logger:
    """Configure the `shotsieve` logger (idempotent) and return it."""
    logger = logging.getLogger("shotsieve")
    logger.setLevel(level)
    if not getattr(logger, "_shotsieve_configured", False):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        logger._shotsieve_configured = True  # type: ignore[attr-defined]
    if logfile and not any(
        isinstance(h, logging.FileHandler) and h.baseFilename.endswith(logfile)
        for h in logger.handlers
    ):
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fh)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the `shotsieve` namespace."""
    return logging.getLogger(f"shotsieve.{name}")
