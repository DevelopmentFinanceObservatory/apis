"""Pin write ordering, failed-create reconciliation, and concurrent rebuilds."""

from __future__ import annotations

import logging

import pytest

from assembler.pipeline import ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER, write_component

from ._pipeline_support import _malaria_intent

# -- pin ordering ---------------------------------------------------------------


class _OrderingFrameStore:
    """Wraps a real `FrameStore` fake, recording `pin` into a shared event
    log so a test can assert its position relative to `begin_write`."""

    def __init__(self, *, inner, events):
        self._inner = inner
        self._events = events

    def load(self, *, frame_ref):
        return self._inner.load(frame_ref=frame_ref)

    def pin(self, *, frame_ref, component_id):
        self._events.append("pin")
        return self._inner.pin(frame_ref=frame_ref, component_id=component_id)


class _OrderingRevisionStore:
    """Wraps a real `RevisionStore` fake, recording `begin_write` into the
    same shared event log as `_OrderingFrameStore`."""

    def __init__(self, *, inner, events):
        self._inner = inner
        self._events = events

    def read_current(self, **kwargs):
        return self._inner.read_current(**kwargs)

    def read_revision(self, **kwargs):
        return self._inner.read_revision(**kwargs)

    def allocate_component_id(self):
        return self._inner.allocate_component_id()

    def prior_create(self, **kwargs):
        return self._inner.prior_create(**kwargs)

    def begin_write(self, **kwargs):
        self._events.append("begin_write")
        return self._inner.begin_write(**kwargs)


class _LegacyFrameStore:
    """Adapter whose `pin()` method does not report whether the reference changed."""

    def __init__(self, *, inner):
        self._inner = inner

    def load(self, *, frame_ref):
        return self._inner.load(frame_ref=frame_ref)

    def pin(self, *, frame_ref, component_id):
        self._inner.pin(frame_ref=frame_ref, component_id=component_id)

    def release_pin(self, *, frame_ref, component_id):
        self._inner.release_pin(frame_ref=frame_ref, component_id=component_id)


def test_create_pins_before_begin_write_is_entered(frame_store, revision_store, release_registry):
    events: list[str] = []
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)

    write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=_OrderingFrameStore(inner=frame_store, events=events),
        revision_store=_OrderingRevisionStore(inner=revision_store, events=events),
        release_registry=release_registry,
    )
    assert events == ["pin", "begin_write"]
    assert len(frame_store.pins) == 1


def test_revise_never_pins(frame_store, revision_store, release_registry):
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
    pins_after_create = len(frame_store.pins)
    assert pins_after_create == 1

    write_component(
        intent={"title": "New title"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert len(frame_store.pins) == pins_after_create


# -- pin reconciliation after a post-pin write failure -----------------------


class _FailingRevisionStore:
    """Wrap a revision store whose `begin_write` raises the injected error."""

    def __init__(self, *, inner, exc, component_id=None):
        self._inner = inner
        self._exc = exc
        self._component_id = component_id

    def read_current(self, **kwargs):
        return self._inner.read_current(**kwargs)

    def read_revision(self, **kwargs):
        return self._inner.read_revision(**kwargs)

    def write_token_digest(self, **kwargs):
        return self._inner.write_token_digest(**kwargs)

    def prior_create(self, **kwargs):
        return self._inner.prior_create(**kwargs)

    def allocate_component_id(self):
        return self._component_id or self._inner.allocate_component_id()

    def begin_write(self, **kwargs):
        raise self._exc


def test_create_write_failure_preserves_pin_for_reconciliation(
    frame_store, revision_store, release_registry, caplog
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    failing_store = _FailingRevisionStore(
        inner=revision_store, exc=RuntimeError("write transaction failed")
    )

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError, match="write transaction failed"),
    ):
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=failing_store,
            release_registry=release_registry,
        )

    assert len(frame_store.pins) == 1
    assert ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER in caplog.text


def test_unreported_pin_result_is_preserved_for_reconciliation(
    frame_store, revision_store, release_registry, caplog
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    failing_store = _FailingRevisionStore(
        inner=revision_store, exc=RuntimeError("write transaction failed")
    )

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError, match="write transaction failed"),
    ):
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=_LegacyFrameStore(inner=frame_store),
            revision_store=failing_store,
            release_registry=release_registry,
        )

    assert len(frame_store.pins) == 1
    assert ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER in caplog.text


def test_write_failure_preserves_an_existing_idempotent_pin(
    frame_store, revision_store, release_registry, caplog
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    component_id = "cmp_existing"
    assert frame_store.pin(frame_ref=frame.frame_ref, component_id=component_id) is True
    failing_store = _FailingRevisionStore(
        inner=revision_store,
        exc=RuntimeError("generated id collided"),
        component_id=component_id,
    )

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="generated id collided"):
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=failing_store,
            release_registry=release_registry,
        )

    assert frame_store.pins == ((frame.frame_ref, component_id),)
    assert ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER not in caplog.text


def test_generated_id_collision_preserves_a_new_shared_pin(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    component_id = "cmp_shared"
    failing_store = _FailingRevisionStore(
        inner=revision_store,
        exc=RuntimeError("generated id collided"),
        component_id=component_id,
    )

    with pytest.raises(RuntimeError, match="generated id collided"):
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=failing_store,
            release_registry=release_registry,
        )

    assert frame_store.pins == ((frame.frame_ref, component_id),)


def test_revise_write_failure_never_attempts_pin_removal(
    frame_store, revision_store, release_registry
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
    pins_before = frame_store.pins

    # A failed revision preserves the frame pins created at component creation.
    frame_store.release_pin_error = RuntimeError("must not be called")
    failing_store = _FailingRevisionStore(
        inner=revision_store, exc=RuntimeError("write transaction failed")
    )

    with pytest.raises(RuntimeError, match="write transaction failed"):
        write_component(
            intent={"title": "New title"},
            component_id=created.component_id,
            write_token=created.write_token,
            composition_id=None,
            frame_store=frame_store,
            revision_store=failing_store,
            release_registry=release_registry,
        )
    assert frame_store.pins == pins_before


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
