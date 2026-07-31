"""
Logging Configuration Module
=============================
Replaces print() with structured Python logging.
Outputs to both console (INFO) and file (DEBUG).

Authors: Sanman Kadam, Varsha Gupta
"""

import logging
import os
from src.config import LOGS_DIR


def setup_logging(log_level=logging.INFO):
    """
    Configures root logger with console and file handlers.

    Parameters:
        log_level: Console output level (default INFO).

    Returns:
        Configured logger instance.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, 'pipeline.log')

    # Create formatter
    console_fmt = logging.Formatter(
        fmt='%(message)s'
    )
    file_fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_fmt)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplicates on re-import
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger('fraud_detection')


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger for the given module."""
    return logging.getLogger(f'fraud_detection.{name}')
