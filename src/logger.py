import sys
from loguru import logger

# === CONFIGURATION ===
log = logger

# === LOGGER SETUP ===
def setup_logger():
    """Configure the logger with custom format and level."""
    log.remove()
    log.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )

# Auto-setup logger when module is imported
setup_logger()