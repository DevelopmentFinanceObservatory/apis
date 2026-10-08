"""Reads one stored revision of a component for the readback, render, and export operations."""

from __future__ import annotations

from assembler.errors import EditConflict, RevisionNotFound
from assembler.ports import RevisionRecord, RevisionStore


def read_stored_revision(
    *, component_id: str, revision: int, revision_store: RevisionStore
) -> RevisionRecord:
    """Return revision `revision` of `component_id`.

    The head is read only when the revision is missing, to name it in the refusal.

    Raises:
        DraftNotFound: The store holds no such component.
        RevisionNotFound: The component has a head and no revision with that number.
        EditConflict: The component has no head.
    """
    record = revision_store.read_revision(component_id=component_id, revision=revision)
    if record is not None:
        return record
    head = revision_store.read_current(component_id=component_id)
    if head is None:
        raise EditConflict
    raise RevisionNotFound(
        component_id=component_id,
        revision=revision,
        detail=(
            f"component {component_id} has no revision {revision}; "
            f"the current head is revision {head.revision}"
        ),
    )
