"""XRouter error hierarchy. Used to decide retry/fallback/circuit-breaker
behavior without any provider-specific branching in core code."""
from __future__ import annotations


class XRouterError(Exception):
    """Base class for all XRouter errors."""


class ConfigError(XRouterError):
    """Bad/missing configuration."""


# --- Provider-level errors -------------------------------------------------

class ProviderError(XRouterError):
    """Base class for errors raised by a provider adapter."""

    retryable = False

    def __init__(self, message: str, *, provider_id: str | None = None):
        super().__init__(message)
        self.provider_id = provider_id


class ProviderTimeoutError(ProviderError):
    retryable = True


class ProviderConnectionError(ProviderError):
    retryable = True


class ProviderRateLimitError(ProviderError):
    """HTTP 429 or equivalent. Retryable, but callers should honor
    retry_after and reduce routing weight rather than hammer the provider."""

    retryable = True

    def __init__(self, message: str, *, provider_id: str | None = None, retry_after: float | None = None):
        super().__init__(message, provider_id=provider_id)
        self.retry_after = retry_after


class ProviderAuthError(ProviderError):
    """Invalid/missing API key. Never retryable."""

    retryable = False


class ProviderInvalidRequestError(ProviderError):
    """Bad request (4xx other than 429/401/403). Never retryable."""

    retryable = False


class ProviderServerError(ProviderError):
    """5xx from the provider. Retryable a bounded number of times."""

    retryable = True


class ProviderUnavailableError(ProviderError):
    """Provider disabled, not configured, or circuit open."""

    retryable = False


# --- Routing / execution errors --------------------------------------------

class CircuitOpenError(ProviderUnavailableError):
    pass


class QuotaExceededError(ProviderUnavailableError):
    pass


class NoAvailableModelError(XRouterError):
    """Every candidate in the fallback chain failed or was unavailable."""

    def __init__(self, message: str, *, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = attempts or []
