"""Checks changed-field disclosure and verbatim resolved-payload serialization."""

from __future__ import annotations

from assembler.reply import ChangedField, WriteReply
from presentation_api.reply import serialize_write_reply

_ENVELOPE = {
    "schema_version": 1,
    "catalog_version": "2026.8.0",
    "component_id": "cmp_x",
    "revision": 2,
    "provenance": {"generated_at": "2026-09-28T00:00:00Z", "inputs": []},
    "component": {"type": "note_narration", "text": "Disbursements rose."},
}


def _reply(changed: tuple[ChangedField, ...]) -> WriteReply:
    return WriteReply(
        component_id="cmp_x",
        revision=2,
        catalog_version="2026.8.0",
        composition_id=None,
        composition_revision=None,
        changed=changed,
        fact_summary={},
        resolved_payload=_ENVELOPE,
    )


def test_a_disclosed_scalar_change_carries_prior_and_new():
    body = serialize_write_reply(_reply((ChangedField(field="scale", prior="linear", new="log"),)))
    assert body["changed"] == [{"field": "scale", "prior": "linear", "new": "log"}]


def test_a_scalar_cleared_to_null_on_both_sides_still_carries_the_keys():
    body = serialize_write_reply(
        _reply((ChangedField(field="title", prior=None, new=None, disclosed=True),))
    )
    assert body["changed"] == [{"field": "title", "prior": None, "new": None}]


def test_an_undisclosed_array_or_object_change_omits_prior_and_new_entirely():
    body = serialize_write_reply(
        _reply((ChangedField(field="select", prior=None, new=None, disclosed=False),))
    )
    assert body["changed"] == [{"field": "select"}]
    assert "prior" not in body["changed"][0]
    assert "new" not in body["changed"][0]


def test_the_resolved_payload_passes_through_verbatim():
    assert serialize_write_reply(_reply(()))["resolved_payload"] == _ENVELOPE
