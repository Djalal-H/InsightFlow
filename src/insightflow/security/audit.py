"""The security audit log: decisions and their reasons, never request or answer content.

Lines go to the `insightflow.audit` logger, so they carry the run context (request ID, thread,
the user's hash) and pass through the redacting formatter like every other line.
"""

import logging
from typing import Any

audit_logger = logging.getLogger("insightflow.audit")


def audit(event: str, **fields: Any) -> None:
    """Log one security event with structured fields (codes, names, and counts only)."""
    audit_logger.info(event, extra={"fields": {"event": event, **fields}})
