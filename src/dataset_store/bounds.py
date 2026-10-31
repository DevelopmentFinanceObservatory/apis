"""The time bounds a `LiveDatasetStore` transaction runs under by default."""

from __future__ import annotations

BORROW_TIMEOUT_SECONDS = 30.0
"""How long `LiveDatasetStore.transaction` waits for a pooled connection."""

CONNECT_TIMEOUT_SECONDS = 30.0
"""The bound on dialling a connection and on each read or write on its socket."""

TRANSACTION_STALL_SECONDS = BORROW_TIMEOUT_SECONDS + 2 * CONNECT_TIMEOUT_SECONDS
"""How long one transaction can stall under those bounds: a wait for a connection, a dial, and
a round trip whose reply does not come. Each reply restarts the socket's timer, so the bound is
on a stall, not on a transaction of many slow statements."""
