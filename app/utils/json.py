from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any) -> str:
    """Deterministic hash for cache keys: sorted keys, no whitespace."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
