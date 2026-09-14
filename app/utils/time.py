from __future__ import annotations

import time


def now() -> float:
    return time.time()


def now_ms() -> float:
    return time.time() * 1000
