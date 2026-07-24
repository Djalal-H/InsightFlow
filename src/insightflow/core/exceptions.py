"""Application-specific exception types."""


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
