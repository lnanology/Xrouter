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


# --- Tool-execution errors (Phase 3: XRouter-executed tools, e.g. web_search) --
# Distinct from ProviderError: these come from app/tools/*, never from a
# provider adapter, and only ever surface inside app/execution/tool_loop.py,
# which catches ToolError and feeds the message back to the model as the
# tool's own result rather than failing the whole node.

class ToolError(XRouterError):
    """Base class for errors raised by an XRouter-executed tool (as
    opposed to a client-supplied tool, which XRouter never executes on
    the client's behalf)."""


class ToolUnavailableError(ToolError):
    """The tool is disabled or missing its configuration (e.g. no API
    key) -- same graceful-degradation rule as an unconfigured provider."""


class ToolExecutionError(ToolError):
    """The tool ran but failed: bad arguments, an HTTP error, a timeout."""


# --- Orchestrator errors (Phase 3: Dynamic Agent Team, app/agents/orchestrator.py) --

class OrchestrationError(XRouterError):
    """The team ran (no NoAvailableModelError/PlannerError/
    DagValidationError along the way) but every node in the resulting DAG
    still failed, leaving nothing to synthesize into an answer -- distinct
    from those other errors, which mean a *stage itself* couldn't run at
    all. app/api/orchestrator.py maps this to a 503, the same "nothing
    could even answer" contract every other XRouter entrypoint uses."""
