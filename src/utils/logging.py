"""
Structured logging for ATLAS.
"""
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
import config


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler (human-readable)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    # File handler (JSON)
    log_file = config.LOG_DIR / "atlas.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_agent_output(agent_name: str, date: str, output: dict) -> None:
    """Append agent output to daily log file."""
    log_file = config.LOG_DIR / f"agents_{date}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps({"agent": agent_name, "date": date, "output": output}) + "\n")
