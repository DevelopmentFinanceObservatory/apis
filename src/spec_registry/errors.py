"""Typed registry errors.

Store failures omit raw driver messages.
"""

from __future__ import annotations


class SpecRegistryError(Exception):
    """Base for every spec_registry exception."""


class SpecIdMismatch(SpecRegistryError):
    """Supplied and recomputed ids differ.

    Wire error is `spec_id_mismatch` (422).

    `position` is the failing collection member's zero-based index, or `None` for a
    standalone registration.
    """

    def __init__(self, *, supplied: str, computed: str, position: int | None = None) -> None:
        detail = (
            f"supplied spec_id {supplied!r} does not match the server-recomputed id {computed!r}"
        )
        if position is not None:
            detail += f" (member at position {position})"
        super().__init__(detail)
        self.supplied = supplied
        self.computed = computed
        self.position = position


class IdentityCollision(SpecRegistryError):
    """Canonical bytes differ at `subject_id`.

    Both versions are quarantined before this raises. Wire error is `identity_collision`
    (500), fail-closed and non-retryable.
    """

    def __init__(self, *, subject_id: str) -> None:
        super().__init__(
            f"identity collision at {subject_id!r}: byte-different content already held at this id"
        )
        self.subject_id = subject_id


class IdentityQuarantined(SpecRegistryError):
    """`subject_id` is quarantined, so neither stored identity is authoritative.

    Wire error is `identity_quarantined` (500), fail-closed and non-retryable.
    """

    def __init__(self, *, subject_id: str, id_kind: str) -> None:
        super().__init__(
            f"{id_kind} id {subject_id!r} is collision-quarantined; registration is "
            "refused until the quarantine is resolved"
        )
        self.subject_id = subject_id
        self.id_kind = id_kind


class EmptyCollectionMembers(SpecRegistryError):
    """Raised when a collection has no member spec ids."""


class SpecRegistryStoreError(SpecRegistryError):
    """Registry store failure with safe diagnostics only.

    Store-error text carries safe diagnostics without driver messages or exception context.
    Logs include the exception class and SQLSTATE when available.
    """


class SpecRegistryUnavailable(SpecRegistryError):
    """Pool borrow timed out before the write began.

    Wire error is retryable `service_unavailable` (503).
    """
