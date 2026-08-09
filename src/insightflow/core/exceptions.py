"""Application-specific exception types."""

from typing import Literal


class InsightFlowError(Exception):
    """Base exception for expected application failures."""


class ProviderError(InsightFlowError):
    """A hosted model provider request failed or returned invalid data."""


class ProviderConfigurationError(ProviderError):
    """The hosted model provider is missing or incorrectly configured."""


class ProviderTimeoutError(ProviderError):
    """The hosted model provider did not respond before its timeout."""


class ProviderRateLimitError(ProviderError):
    """The hosted model provider rejected a request due to rate limiting."""


class StorageError(InsightFlowError):
    """A vector-store operation failed."""


DocumentRejectionReason = Literal[
    "unsupported_format",
    "encrypted_pdf",
    "scanned_pdf",
    "textless_pdf",
    "empty_document",
    "conversion_failed",
]

class DocumentRejectedError(InsightFlowError):
    """A source document cannot enter the supported ingestion pipeline."""

    def __init__(
        self,
        *,
        reason: DocumentRejectionReason,
        source_identifier: str,
    ) -> None:
        self.reason = reason
        self.source_identifier = source_identifier
        super().__init__(f"Document rejected ({reason}): {source_identifier}")
