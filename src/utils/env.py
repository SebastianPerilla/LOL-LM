"""Environment variable loading from .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv
from logger import log


def load_env_file(env_path: Path | None = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, searches for .env in project root.
    """
    if env_path is None:
        # Find project root (where .env should be)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        env_path = project_root / ".env"
    
    env_path = Path(env_path)
    
    if env_path.exists():
        load_dotenv(env_path, override=False)  # Don't override existing env vars
        log.debug(f"Loaded environment variables from: {env_path}")
    else:
        log.debug(f".env file not found at: {env_path} (using system environment variables)")


# Auto-load .env file when this module is imported
load_env_file()

