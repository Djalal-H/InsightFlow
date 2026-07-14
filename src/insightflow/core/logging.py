"""Standard-library logging configuration."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure consistent process-level logging."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

