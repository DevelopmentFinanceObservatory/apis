"""Serialise `WriteReply` for the components door. Undisclosed array/object changes emit only
`field`, while scalar changes emit `prior` and `new`, including explicit nulls. Create replies
add an `affordance` and the component's `write_token`, plus `composition_write_token` when
it minted a composition. Revise replies carry none of these. `serialize_read_reply` gives a
rendered revision the shape a component read has, less its `retention`.
"""

from __future__ import annotations

from collections.abc import Mapping

from assembler.reply import WriteReply

__all__ = ["serialize_create_reply", "serialize_read_reply", "serialize_write_reply"]


def serialize_write_reply(reply: WriteReply) -> dict[str, object]:
    return {
        "component_id": reply.component_id,
        "revision": reply.revision,
        "catalog_version": reply.catalog_version,
        "composition_id": reply.composition_id,
        "composition_revision": reply.composition_revision,
        "changed": [
            {"field": entry.field}
            if not entry.disclosed
            else {"field": entry.field, "prior": entry.prior, "new": entry.new}
            for entry in reply.changed
        ],
        "fact_summary": reply.fact_summary,
        "resolved_payload": reply.resolved_payload,
    }


def serialize_create_reply(reply: WriteReply) -> dict[str, object]:
    body = serialize_write_reply(reply)
    body["affordance"] = (
        f"Revise this component with PATCH /api/presentation/components/{reply.component_id}, "
        "sending only the fields to change and this reply's write_token as x-dfo-write-token, "
        f"or read it back with GET /api/presentation/components/{reply.component_id}."
    )
    body["write_token"] = reply.write_token
    if reply.composition_write_token is not None:
        body["composition_write_token"] = reply.composition_write_token
    return body


def serialize_read_reply(
    reply: WriteReply, *, stored_intent: Mapping[str, object]
) -> dict[str, object]:
    """A rendering of a stored revision as a component read states it: the revision's own
    `stored_intent` beside the rendering's summary and envelope."""
    return {
        "component_id": reply.component_id,
        "revision": reply.revision,
        "catalog_version": reply.catalog_version,
        "composition_id": reply.composition_id,
        "composition_revision": reply.composition_revision,
        "stored_intent": stored_intent,
        "fact_summary": reply.fact_summary,
        "resolved_payload": reply.resolved_payload,
    }
