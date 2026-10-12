"""Typed registry errors.

Store failures omit raw driver messages.
"""

from __future__ import annotations

from dfo_platform.problem import Refusal


class SpecRegistryError(Exception):
    """Base for every spec_registry exception."""


class SpecIdMismatch(SpecRegistryError, Refusal):
    """Supplied and recomputed ids differ. Wire error is `spec_id_mismatch`."""

    code = "spec_id_mismatch"

    def __init__(self, *, supplied: str, computed: str) -> None:
        super().__init__(
            f"supplied spec_id {supplied!r} does not match the server-recomputed id {computed!r}"
        )
        self.supplied = supplied
        self.computed = computed


class IdentityCollision(SpecRegistryError, Refusal):
    """Canonical bytes differ at `subject_id`.

    Both versions are quarantined before this raises. Wire error is `identity_collision`,
    fail-closed and non-retryable.
    """

    code = "identity_collision"

    def __init__(self, *, subject_id: str) -> None:
        super().__init__(
            f"identity collision at {subject_id!r}: byte-different content already held at this id"
        )
        self.subject_id = subject_id


class IdentityQuarantined(SpecRegistryError, Refusal):
    """`subject_id` is quarantined, so neither stored identity is authoritative.

    Wire error is `identity_quarantined`, fail-closed and non-retryable.
    """

    code = "identity_quarantined"

    def __init__(self, *, subject_id: str, id_kind: str) -> None:
        super().__init__(
            f"{id_kind} id {subject_id!r} is collision-quarantined; registration is "
            "refused until the quarantine is resolved"
        )
        self.subject_id = subject_id
        self.id_kind = id_kind


class SpecRegistryStoreError(SpecRegistryError):
    """Registry store failure with safe diagnostics only.

    Store-error text carries safe diagnostics without driver messages or exception context.
    Logs include the exception class and SQLSTATE when available.
    """


class SpecRegistryUnavailable(SpecRegistryError, Refusal):
    """Pool borrow timed out before the write began. Wire error is retryable
    `service_unavailable`."""

    code = "service_unavailable"
    retryable = True
    detail = "the registry write pool had no connection free right now; retry"
