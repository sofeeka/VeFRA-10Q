import json
import sys
from pathlib import Path

from loguru import logger

# Get project root to store logs in the main directory
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def serialize(record):
    """
    Custom serializer to format log records into a JSON structure.
    This makes logs machine-readable and easy to query.
    """
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "name": record["name"],
        "function": record["function"],
        "line": record["line"],
        # Add context variables to the log record
        "context": record["extra"],
    }

    if record["exception"]:
        subset["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": True,
        }
    return json.dumps(subset)


def formatter(record):
    """
    This function creates the final log string.
    We serialize the record and add a newline character.
    """
    record["extra"]["serialized"] = serialize(record)
    return "{extra[serialized]}\n"


def setup_logging():
    """
    Configures the Loguru logger for the entire application.
    """
    logger.remove()  # Remove the default handler

    # Add a handler for console logging (useful for development)
    # This uses the default Loguru format for better readability in the terminal
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    # Add a handler for file logging (essential for production)
    # This uses our custom JSON formatter
    log_file_path = LOGS_DIR / "vefra_app_{time}.jsonl"
    logger.add(
        log_file_path,
        level="DEBUG",  # Log everything to the file
        rotation="10 MB",  # Rotate the log file when it reaches 10 MB
        retention="30 days",  # Keep logs for 30 days
        enqueue=True,  # Make logging asynchronous
        backtrace=True,  # Show full stack traces
        diagnose=True,  # Add exception data
        format=formatter,  # Use our custom formatter
    )
    logger.info("Logger configured successfully.")
