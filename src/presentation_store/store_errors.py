"""Store exceptions. SQLSTATE parsing and safe diagnostics come from `dfo_platform.store_errors`."""

from __future__ import annotations

from dataset_store.connection import default_is_broken
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics


def is_unique_violation(exc: Exception) -> bool:
    """Return whether the error has Postgres unique-violation SQLSTATE 23505."""
    return extract_sqlstate(exc) == "23505"


def violated_constraint(exc: Exception) -> str | None:
    """Return the constraint or index name pg8000 reports in `args[0]["n"]`, or `None`."""
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("n")
    return None


class GeneratedIdCollision(Exception):
    """A freshly generated ID collided on SQLSTATE 23505. Retrying the create mints another ID.
    A collision on the component write-token index raises `assembler.ports.WriteTokenInUse`
    instead. Other class-23 failures become non-retryable `PresentationStoreUnavailable` errors.
    """

    def __init__(self, *, subject_kind: str) -> None:
        self.subject_kind = subject_kind
        super().__init__(f"Generated {subject_kind} ID collision. Retry the request.")


class PresentationStoreUnavailable(Exception):
    """Wrap a driver failure with safe diagnostics and its SQLSTATE. Wrap sites raise this
    after leaving `except`, so driver text and exception context stay out of the public error.
    A missing SQLSTATE includes failures with no underlying driver exception. Permanent failures
    map to non-retryable 500. Transient failures map to retryable 503. `connection_broken`
    records whether `dataset_store.connection.default_is_broken` judged the driver failure a
    dead connection; `is_broken` reads it so the pool discards that connection.
    """

    def __init__(
        self, message: str, *, sqlstate: str | None = None, connection_broken: bool = False
    ) -> None:
        self.sqlstate = sqlstate
        self.connection_broken = connection_broken
        super().__init__(message)

    @classmethod
    def from_driver_failure(
        cls, action_detail: str, exc: Exception
    ) -> PresentationStoreUnavailable:
        """Build the scrubbed error for `exc`. Build it inside `except` and raise it after
        leaving, so the raised error carries no `__context__`.
        """
        return cls(
            f"{action_detail}: {safe_diagnostics(exc)}",
            sqlstate=extract_sqlstate(exc),
            connection_broken=default_is_broken(exc),
        )


def is_broken(exc: BaseException) -> bool:
    """The presentation pool's `is_broken`: `default_is_broken`, plus a
    `PresentationStoreUnavailable` translated from a dead connection inside a borrow.
    """
    if isinstance(exc, PresentationStoreUnavailable):
        return exc.connection_broken
    return default_is_broken(exc)
