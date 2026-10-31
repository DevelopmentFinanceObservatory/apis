"""The time bounds a `LiveDatasetStore` transaction runs under."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

BORROW_TIMEOUT_SECONDS = 30.0
"""How long `LiveDatasetStore.transaction` waits for a pooled connection by default."""

CONNECT_TIMEOUT_SECONDS = 30.0
"""The default bound on dialling a connection and on each read or write on its socket
(`ConnectionSettings.connect_timeout`)."""


@dataclass(frozen=True)
class TransactionBounds:
    """A store's configured bounds, in seconds: `borrow`, the wait for a pooled connection, and
    `connect`, the bound on a dial and on each read or write on a connection's socket."""

    borrow: float = BORROW_TIMEOUT_SECONDS
    connect: float = CONNECT_TIMEOUT_SECONDS

    @property
    def round_trip(self) -> timedelta:
        """The longest one statement waits for its reply."""
        return timedelta(seconds=self.connect)

    @property
    def stall(self) -> timedelta:
        """The longest a transaction waits at one step: for a connection, a dial and a reply.
        Each reply restarts the socket's timer, so a transaction of several slow statements
        takes longer."""
        return timedelta(seconds=self.borrow + 2 * self.connect)


DEFAULT_TRANSACTION_BOUNDS = TransactionBounds()
