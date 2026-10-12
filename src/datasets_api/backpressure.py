"""`Backpressure`, the base of every refusal a full executor or permit raises, and the
`Retry-After` it carries."""

from __future__ import annotations

from types import MappingProxyType

from dfo_platform.problem import Refusal

# Seconds a client waits before retrying a full instance. A materialise permit wait ends at
# `materialise_wait_seconds` (30), so a mint refused for want of a permit has already waited
# out the capacity, and its retry queues for a permit again. Every other door refuses at once,
# and a read, a head hit or a work thread frees within a request. One second is the shortest
# `Retry-After` and matches the operation backlog's.
BACKPRESSURE_RETRY_AFTER_SECONDS = 1


class Backpressure(Refusal):
    """Capacity is full. Each subclass maps to retryable `service_unavailable` (503) and carries
    `Retry-After`; the base names no code of its own, so it is no refusal by itself."""

    retryable = True
    headers = MappingProxyType({"Retry-After": str(BACKPRESSURE_RETRY_AFTER_SECONDS)})
