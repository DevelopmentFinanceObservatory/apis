"""Shared SQLSTATE classification and safe summaries for store errors.

Summaries omit exception messages, which may contain query text or parameter values. This
module has no FastAPI dependency."""

from __future__ import annotations


def extract_sqlstate(exc: BaseException) -> str | None:
    """Return pg8000's SQLSTATE from `args[0]["C"]`, or `None` for other exception shapes."""
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("C")
    return None


# Permanent SQLSTATE classes are 42 (syntax/access), 22 (invalid data), 23
# (constraint violation), and 0A (unsupported feature). Retrying a constraint violation repeats it.
# All others are transient,
# including 08 (connection), 53 (resources), 57 (operator intervention), and 40
# (transaction rollback, including serialization and deadlock aborts).
_PERMANENT_STORE_FAILURE_SQLSTATE_CLASSES = frozenset({"42", "22", "23", "0A"})


def store_failure_is_permanent(sqlstate: str | None) -> bool:
    """Return whether the SQLSTATE class is permanent.

    Missing SQLSTATEs are transient, allowing a retry when classification is uncertain."""
    if sqlstate is None:
        return False
    return sqlstate[:2] in _PERMANENT_STORE_FAILURE_SQLSTATE_CLASSES


def safe_diagnostics(exc: Exception) -> str:
    """Return the qualified exception class and SQLSTATE, omitting the message."""
    cls = type(exc)
    qualified_name = f"{cls.__module__}.{cls.__qualname__}"
    sqlstate = extract_sqlstate(exc)
    return f"{qualified_name} sqlstate={sqlstate}" if sqlstate else qualified_name
