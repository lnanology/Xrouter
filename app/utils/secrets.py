from __future__ import annotations

import os


def read_secret(env_var: str | None) -> str | None:
    """Read an API key from the environment. Never returns a hardcoded value,
    never logs the value. Returns None if unset/empty so callers can
    gracefully disable the provider instead of crashing."""
    if not env_var:
        return None
    value = os.environ.get(env_var)
    return value if value else None


def redact(value: str | None, keep: int = 4) -> str:
    """Redact a secret for logging: keep only the last `keep` chars."""
    if not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return "*" * (len(value) - keep) + value[-keep:]
