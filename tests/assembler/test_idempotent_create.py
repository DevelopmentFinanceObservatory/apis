"""A create that carries a client-minted write token is keyed by it.

A retry with the same token and request returns the first create's reply without resolving or
pinning a frame. The same token with another request is refused, and a create that loses a race
to another create with its token returns the winner's reply.
"""

from __future__ import annotations

import dataclasses
import logging
import secrets
import threading
from contextlib import contextmanager

import pytest

from assembler.errors import WriteRefused
from assembler.pipeline import ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER, write_component
from assembler.write_token import CreateRequestInvalid, write_token_digest

from ._pipeline_support import _malaria_intent


def _client_token() -> str:
    return f"cmw_{secrets.token_urlsafe(16)}"


class _CountingFrameStore:
    """Wraps a `FrameStore` fake and counts `load` and `pin` calls."""

    def __init__(self, *, inner):
        self._inner = inner
        self.loads = 0
        self.pins = 0

    def load(self, *, frame_ref):
        self.loads += 1
        return self._inner.load(frame_ref=frame_ref)

    def pin(self, *, frame_ref, component_id):
        self.pins += 1
        return self._inner.pin(frame_ref=frame_ref, component_id=component_id)


@pytest.fixture
def frame_store(frame_store):
    frame, _intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    return _CountingFrameStore(inner=frame_store)


def _create(
    *,
    frame_store,
    revision_store,
    release_registry,
    create_write_token,
    intent=None,
    composition_id=None,
    write_token=None,
):
    return write_component(
        intent=intent if intent is not None else _malaria_intent()[1],
        component_id=None,
        composition_id=composition_id,
        write_token=write_token,
        create_write_token=create_write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )


@pytest.fixture
def create(frame_store, revision_store, release_registry):
    def _call(**kwargs):
        return _create(
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
            **kwargs,
        )

    return _call


# -- first create --------------------------------------------------------------


def test_a_client_token_becomes_the_components_write_token(create, revision_store):
    token = _client_token()

    created = create(create_write_token=token)

    assert created.write_token == token
    assert revision_store.write_token_digest(
        subject_kind="component", subject_id=created.component_id
    ) == write_token_digest(token)


# -- replay --------------------------------------------------------------------


def test_a_retried_create_returns_the_first_reply_without_frame_work(
    create, frame_store, revision_store
):
    token = _client_token()
    first = create(create_write_token=token)
    loads, pins = frame_store.loads, frame_store.pins

    again = create(create_write_token=token)

    assert again == first
    assert again.write_token == token
    assert (frame_store.loads, frame_store.pins) == (loads, pins)
    head = revision_store.read_current(component_id=first.component_id)
    assert head is not None
    assert head.revision == 1


def test_a_replay_matches_an_intent_with_its_keys_reordered(create):
    token = _client_token()
    intent = _malaria_intent()[1]
    first = create(create_write_token=token, intent=intent)

    again = create(create_write_token=token, intent=dict(reversed(list(intent.items()))))

    assert again.component_id == first.component_id


def test_a_replay_after_a_revise_returns_revision_one(
    create, frame_store, revision_store, release_registry
):
    token = _client_token()
    first = create(create_write_token=token)
    write_component(
        intent={"title": "A later title"},
        component_id=first.component_id,
        write_token=token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    again = create(create_write_token=token)

    assert again.revision == 1
    assert again.resolved_payload == first.resolved_payload


def test_a_replay_into_a_composition_names_it_and_its_head(create):
    starter = create(create_write_token=None, composition_id="new")
    token = _client_token()
    first = create(
        create_write_token=token,
        composition_id=starter.composition_id,
        write_token=starter.composition_write_token,
    )

    again = create(
        create_write_token=token,
        composition_id=starter.composition_id,
        write_token=starter.composition_write_token,
    )

    assert again.component_id == first.component_id
    assert again.composition_id == starter.composition_id
    assert again.composition_revision == first.composition_revision


def test_the_composition_token_check_runs_before_the_replay(create):
    starter = create(create_write_token=None, composition_id="new")
    token = _client_token()
    create(
        create_write_token=token,
        composition_id=starter.composition_id,
        write_token=starter.composition_write_token,
    )

    with pytest.raises(WriteRefused):
        create(create_write_token=token, composition_id=starter.composition_id)


# -- refusals ------------------------------------------------------------------


def test_the_same_token_with_a_different_request_is_refused(create, revision_store):
    token = _client_token()
    first = create(create_write_token=token)

    with pytest.raises(CreateRequestInvalid) as refused:
        create(create_write_token=token, intent=_malaria_intent(title="Another title")[1])

    assert refused.value.pointer == "/write_token"
    assert "different request" in refused.value.detail
    prior = revision_store.prior_create(write_token_sha256=write_token_digest(token))
    assert prior is not None
    assert prior.component_id == first.component_id


def test_a_token_held_by_a_component_created_without_a_request_hash_is_refused(
    create, revision_store
):
    token = _client_token()
    template = create(create_write_token=None)
    record = revision_store.read_revision(component_id=template.component_id, revision=1)
    assert record is not None
    with revision_store.begin_write(
        component_id="cmp_legacy",
        is_create=True,
        composition_id=None,
        write_token_sha256=write_token_digest(token),
    ) as ctx:
        ctx.append(
            record=dataclasses.replace(record, component_id="cmp_legacy"), composition_entry=None
        )

    with pytest.raises(CreateRequestInvalid) as refused:
        create(create_write_token=token)

    assert refused.value.pointer == "/write_token"


def test_a_token_with_a_new_composition_is_refused_before_any_write(
    create, frame_store, revision_store
):
    token = _client_token()

    with pytest.raises(CreateRequestInvalid) as refused:
        create(create_write_token=token, composition_id="new")

    assert refused.value.pointer == "/write_token"
    assert (frame_store.loads, frame_store.pins) == (0, 0)
    assert revision_store.prior_create(write_token_sha256=write_token_digest(token)) is None


@pytest.mark.parametrize(
    "token",
    [
        "cmw_short",
        "cpw_" + "A" * 22,
        "cmw_" + "A" * 21,
        "cmw_" + "A" * 23,
        "cmw_" + "A" * 21 + "+",
        "cmw_" + "A" * 21 + "=",
        " cmw_" + "A" * 22,
    ],
)
def test_a_malformed_token_is_refused_before_any_write(create, frame_store, token):
    with pytest.raises(CreateRequestInvalid) as refused:
        create(create_write_token=token)

    assert refused.value.pointer == "/write_token"
    assert (frame_store.loads, frame_store.pins) == (0, 0)


def test_a_token_on_a_revise_is_refused(create, frame_store, revision_store, release_registry):
    created = create(create_write_token=None)

    with pytest.raises(CreateRequestInvalid) as refused:
        write_component(
            intent={"title": "A new title"},
            component_id=created.component_id,
            write_token=created.write_token,
            create_write_token=_client_token(),
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    assert refused.value.pointer == "/write_token"


def test_an_intent_outside_i_json_is_refused_at_intent(create):
    with pytest.raises(CreateRequestInvalid) as refused:
        create(create_write_token=_client_token(), intent=_malaria_intent(title=2**60)[1])

    assert refused.value.pointer == "/intent"


# -- race ------------------------------------------------------------------------


class _RivalCommitsFirstRevisionStore:
    """Wraps a `FakeRevisionStore`. The first `begin_write` runs `rival` on another thread
    once its own write is open, so the rival commits first and this write fails at commit."""

    def __init__(self, *, inner, rival):
        self._inner = inner
        self._rival = rival

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @contextmanager
    def begin_write(self, **kwargs):
        with self._inner.begin_write(**kwargs) as ctx:
            rival, self._rival = self._rival, None
            if rival is not None:
                thread = threading.Thread(target=rival)
                thread.start()
                thread.join()
            yield ctx


def test_a_create_that_loses_the_race_returns_the_winners_reply(
    frame_store, revision_store, release_registry, caplog
):
    token = _client_token()
    rival_replies = []

    def rival():
        rival_replies.append(
            _create(
                frame_store=frame_store,
                revision_store=revision_store,
                release_registry=release_registry,
                create_write_token=token,
            )
        )

    with caplog.at_level(logging.ERROR, logger="assembler.pipeline.write"):
        loser = _create(
            frame_store=frame_store,
            revision_store=_RivalCommitsFirstRevisionStore(inner=revision_store, rival=rival),
            release_registry=release_registry,
            create_write_token=token,
        )

    [winner] = rival_replies
    assert loser == winner
    prior = revision_store.prior_create(write_token_sha256=write_token_digest(token))
    assert prior is not None
    assert prior.component_id == winner.component_id
    # The loser pinned a frame for a component that never committed.
    assert ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER in caplog.text
