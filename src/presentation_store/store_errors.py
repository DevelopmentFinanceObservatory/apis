"""Store exceptions. SQLSTATE parsing and safe diagnostics come from `dfo_platform.store_errors`."""

from __future__ import annotations

from dfo_platform.store_errors import extract_sqlstate


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
    map to non-retryable 500. Transient failures map to retryable 503.
    """

    def __init__(self, message: str, *, sqlstate: str | None = None) -> None:
        self.sqlstate = sqlstate
        super().__init__(message)
