"""
Configurable logging utility.

Provides a project-wide logger with console and optional file output.
"""

import logging
import os
import sys
from typing import Optional


_LOGGERS: dict[str, logging.Logger] = {}

_DEFAULT_FMT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
_DEFAULT_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "engine",
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = _DEFAULT_FMT,
    date_fmt: str = _DEFAULT_DATE_FMT,
    console: bool = True,
) -> logging.Logger:
    """Return a configured logger, reusing existing loggers by name.

    Parameters
    ----------
    name : str
        Logger name (usually module or component name).
    level : str
        Logging level — DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_file : str, optional
        If provided, also write logs to this file path.
    fmt : str
        Log message format string.
    date_fmt : str
        Timestamp format string.
    console : bool
        Whether to attach a console (stderr) handler.

    Returns
    -------
    logging.Logger
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger
