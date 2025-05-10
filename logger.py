import logging
import sys
import os
from pathlib import Path

LOG_LEVEL = os.getenv("BT_SERVANT_LOG_LEVEL", "DEBUG").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
log_file_path = LOGS_DIR / "bt_servant.log"


def get_logger(name: str) -> logging.Logger:
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
