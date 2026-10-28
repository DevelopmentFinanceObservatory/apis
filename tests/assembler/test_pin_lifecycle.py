"""Writes pin nothing, a share pins a card's frame, each revision keeps its frame's
rendering, and concurrent rebuilds."""

from __future__ import annotations

import dataclasses

import pytest

from assembler.errors import RevisionNotFound
from assembler.pipeline import write_component
from presentation_store import pin_card_frames

from ._pipeline_support import _malaria_intent

_PLAIN = {"base_content_version": "cv_plain", "minting": None}
_CUT = {"base_content_version": "cv_plain", "minting": {"without_party_totals": True}}


class _FailingRevisionStore:
    """Wrap a revision store whose `begin_write` raises the injected error."""

    def __init__(self, *, inner, exc):
        self._inner = inner
        self._exc = exc

    def read_current(self, **kwargs):
        return self._inner.read_current(**kwargs)

    def read_revision(self, **kwargs):
        return self._inner.read_revision(**kwargs)

    def prior_create(self, **kwargs):
        return self._inner.prior_create(**kwargs)

    def allocate_component_id(self):
        return self._inner.allocate_component_id()

    def begin_write(self, **kwargs):
        raise self._exc


def _create(frame_store, revision_store, release_registry, intent):
    return write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )


# -- writes pin nothing ------------------------------------------------------------


def test_create_and_revise_pin_nothing(frame_store, revision_store, release_registry):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    created = _create(frame_store, revision_store, release_registry, intent)

    write_component(
        intent={"title": "New title"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert frame_store.pins == ()


def test_a_failed_create_pins_nothing(frame_store, revision_store, release_registry):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    failing_store = _FailingRevisionStore(
        inner=revision_store, exc=RuntimeError("write transaction failed")
    )

    with pytest.raises(RuntimeError, match="write transaction failed"):
        _create(frame_store, failing_store, release_registry, intent)

    assert frame_store.pins == ()


# -- a share pins a card's frame ---------------------------------------------------------


def test_pin_card_frames_pins_the_revision_frame_for_the_component(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    created = _create(frame_store, revision_store, release_registry, intent)

    first = pin_card_frames(
        component_id=created.component_id,
        revision=1,
        revision_store=revision_store,
        frame_store=frame_store,
    )
    again = pin_card_frames(
        component_id=created.component_id,
        revision=1,
        revision_store=revision_store,
        frame_store=frame_store,
    )

    assert (first, again) == (True, False)
    assert frame_store.pins == ((frame.frame_ref, created.component_id),)


def test_pin_card_frames_refuses_a_revision_the_component_lacks(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    created = _create(frame_store, revision_store, release_registry, intent)

    with pytest.raises(RevisionNotFound):
        pin_card_frames(
            component_id=created.component_id,
            revision=2,
            revision_store=revision_store,
            frame_store=frame_store,
        )
    assert frame_store.pins == ()


# -- each revision keeps its frame's rendering ---------------------------------------------


def test_a_revise_keeps_the_rendering_the_create_recorded(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=dataclasses.replace(frame, rendering=_CUT))
    created = _create(frame_store, revision_store, release_registry, intent)
    # A later fetch links a plain file under the same version, so its records now read plain.
    frame_store.put(resolved_frame=dataclasses.replace(frame, rendering=_PLAIN))

    write_component(
        intent={"title": "New title"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    stored = [
        revision_store.read_revision(component_id=created.component_id, revision=n) for n in (1, 2)
    ]
    assert [record.frame_rendering for record in stored] == [_CUT, _CUT]


# -- concurrent-writer rebuild --------------------------------------------------


def test_in_transaction_rebuild_resolves_against_fresh_state_not_the_stale_pre_read(
    frame_store, revision_store, release_registry, monkeypatch
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    # Commit a concurrent revise between this call's pre-read and write.
    concurrent = write_component(
        intent={"title": "Concurrent title"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert concurrent.revision == 2

    stale_current = revision_store.read_revision(component_id=created.component_id, revision=1)
    monkeypatch.setattr(revision_store, "read_current", lambda *, component_id: stale_current)

    result = write_component(
        intent={"window": {"start": "2016", "end": "2024"}},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    # The locked reread sees revision 2, so revision 3 keeps the concurrent title.
    assert result.revision == 3
    rev3 = revision_store.read_revision(component_id=created.component_id, revision=3)
    assert rev3.stored_intent["title"] == "Concurrent title"
    assert rev3.stored_intent["window"] == {"start": "2016", "end": "2024"}
