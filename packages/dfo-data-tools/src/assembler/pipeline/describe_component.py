"""The `describe_component` readback: `get_component`, called by
`assembler.pipeline`'s facade and, through it, by the components door.
Rebuilds a stored revision's fact summary from its resolved payload and
persisted `series_coordinates`, formatted with the formatter profile stored on
it. Read and write replies use `assembler.reply.build_fact_summary` for the
same member summary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from assembler.errors import EditConflict
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import RevisionStore
from assembler.reply import build_fact_summary

# -- get_component --------------------------------------------------------


def get_component(
    *,
    component_id: str,
    revision: int | None = None,
    revision_store: RevisionStore,
) -> Mapping[str, object]:
    """Returns the stored intent, summary, envelope, and composition membership for one revision.

    Membership is fixed at creation. `composition_revision` reports the
    composition head at read time. The summary uses the formatter profile stored on the
    revision, so a revision minted under any release reads back. A stored payload whose shape
    the summary builder does not recognise raises `ReleaseMismatch`.

    An unknown `component_id` raises `DraftNotFound`. A missing requested
    revision raises `RevisionNotFound` if a head exists, otherwise
    `EditConflict`. An unscoped read with no head also raises `EditConflict`."""
    if revision is not None:
        record = read_stored_revision(
            component_id=component_id, revision=revision, revision_store=revision_store
        )
    else:
        record = revision_store.read_current(component_id=component_id)
        if record is None:
            raise EditConflict

    component = cast("Mapping[str, object]", record.resolved_payload["component"])
    member = cast(str, component["type"])
    fact_summary = build_fact_summary(
        component=component,
        member=member,
        stored_intent=record.stored_intent,
        formatter_profile=record.formatter_profile,
        series_coordinates=record.series_coordinates,
        component_id=record.component_id,
        revision=record.revision,
        changed=(),
        catalog_version=record.catalog_version,
    )
    membership = revision_store.composition_of(component_id=record.component_id)

    return {
        "component_id": record.component_id,
        "revision": record.revision,
        "catalog_version": record.catalog_version,
        "composition_id": membership.composition_id if membership is not None else None,
        "composition_revision": membership.revision if membership is not None else None,
        "stored_intent": record.stored_intent,
        "fact_summary": fact_summary,
        "resolved_payload": record.resolved_payload,
    }
