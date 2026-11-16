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


class ToolError(InsightFlowError):
    """A tool server could not be reached or listed."""


class ToolTimeoutError(ToolError):
    """A tool server did not respond before its timeout."""


class ThreadNotFoundError(InsightFlowError):
    """A conversation thread has no stored checkpoint."""


class AuthenticationError(InsightFlowError):
    """The request has no valid bearer token. `reason` is a code for the audit log only."""

    def __init__(self, reason: str = "invalid_token") -> None:
        self.reason = reason
        super().__init__(reason)


class AuthorizationError(InsightFlowError):
    """The token is valid but does not grant the operation."""

    def __init__(self, missing_scope: str = "") -> None:
        self.missing_scope = missing_scope
        super().__init__(missing_scope)


class RateLimitExceededError(InsightFlowError):
    """The user sent requests faster than the configured rate."""

    def __init__(self, retry_after_seconds: int, *, limit: int = 0) -> None:
        self.retry_after_seconds = max(1, retry_after_seconds)
        self.limit = limit
        super().__init__("rate limited")


class QuotaExceededError(InsightFlowError):
    """The user used the daily request or token quota."""

    def __init__(self, retry_after_seconds: int, *, quota: str) -> None:
        self.retry_after_seconds = max(1, retry_after_seconds)
        self.quota = quota
        super().__init__(f"{quota} quota exceeded")


class RateLimiterUnavailableError(InsightFlowError):
    """The rate-limit store cannot be reached and the limiter fails closed."""


class InputRejectedError(InsightFlowError):
    """A guardrail refused the request input. `reason` is a code for the audit log only."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ApprovalNotFoundError(InsightFlowError):
    """The thread has no pending tool approval with this identifier."""


class BudgetExceededError(InsightFlowError):
    """A run reached its provider call or token budget (evaluation runs only)."""


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
