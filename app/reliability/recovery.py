"""Recovery is driven by CircuitBreaker.allow_request()'s OPEN->HALF_OPEN
transition (time-based) combined with the HealthMonitor's periodic
provider.health() probes, which naturally act as HALF_OPEN trial checks.
This module holds the small policy decision of how confidently to trust a
just-recovered provider before restoring it to full routing weight."""
from __future__ import annotations

RECOVERY_GRACE_SUCCESSES = 2
"""Number of consecutive successful requests required after a HALF_OPEN ->
CLOSED transition before a provider is treated as fully trusted again. The
scorer already down-weights degraded/half-open providers via availability
score, so this constant is advisory metadata surfaced on /admin/providers
rather than a second gate — avoids duplicating state machines."""
