import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure root logging once, honoring LOG_LEVEL env var if provided.
    """

    if logging.getLogger().handlers:
        return  # already configured elsewhere

    env_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, env_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
