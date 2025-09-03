"""Logging helpers with console and file handlers.

Resolves log level from config and ensures a logs directory exists.
"""
import logging
import sys
from pathlib import Path

from config import config

# Resolve level name dynamically to appease static analyzers.
LEVEL_NAME = str(getattr(config, "BT_SERVANT_LOG_LEVEL", "info")).upper()
LOG_LEVEL = getattr(logging, LEVEL_NAME, logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
log_file_path = LOGS_DIR / "bt_servant.log"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given name.

    Adds stream and file handlers once per logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    if not logger.hasHandlers():
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Stream to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Optional: write to file too
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
