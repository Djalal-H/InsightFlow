"""Application-specific exception types."""


class InsightFlowError(Exception):
    """Base exception for expected application failures."""


class ProviderError(InsightFlowError):
    """A hosted model provider request failed or returned invalid data."""


class StorageError(InsightFlowError):
    """A vector-store operation failed."""

