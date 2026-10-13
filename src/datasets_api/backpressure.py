"""`Backpressure`, the base of every refusal a full executor or permit raises, and the
`Retry-After` it carries."""

from __future__ import annotations

from dfo_platform.problem import BACKPRESSURE_HEADERS, Refusal


class Backpressure(Refusal):
    """Capacity is full. Each subclass maps to retryable `service_unavailable` (503) and carries
    `Retry-After`; the base names no code of its own, so it is no refusal by itself."""

    retryable = True
    # A materialise permit wait ends at `materialise_wait_seconds` (30), so a mint refused for
    # want of a permit has already waited out the capacity, and its retry queues for a permit
    # again. One second matches the operation backlog's `Retry-After`.
    headers = BACKPRESSURE_HEADERS
