"""The write token grants revise, and the component id grants only read.

A create returns a `cmw_` token for the component, plus a `cpw_` token when it starts a
composition. A revise needs the component's token. A create into an existing composition needs
that composition's token. The store keeps only SHA-256 digests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from assembler.errors import DraftNotFound, WriteRefused
from assembler.pipeline import get_component, write_component

from ._pipeline_support import _malaria_intent, _stat_card_intent

_TOKEN_PREFIXES = ("cmw_", "cpw_")


@pytest.fixture
def stores(frame_store, revision_store, release_registry):
    frame, _intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    return {
        "frame_store": frame_store,
        "revision_store": revision_store,
        "release_registry": release_registry,
    }


def _create(stores, *, intent=None, composition_id=None, write_token=None):
    return write_component(
        intent=intent if intent is not None else _malaria_intent()[1],
        component_id=None,
        composition_id=composition_id,
        write_token=write_token,
        **stores,
    )


def _revise(stores, *, component_id, write_token, title="A new title"):
    return write_component(
        intent={"title": title},
        component_id=component_id,
        write_token=write_token,
        **stores,
    )


def _assert_no_token_in(value: object) -> None:
    serialised = json.dumps(value, default=str)
    for prefix in _TOKEN_PREFIXES:
        assert prefix not in serialised


# -- minting ------------------------------------------------------------------


def test_a_standalone_create_returns_a_component_write_token_only(stores):
    created = _create(stores)

    assert created.write_token is not None
    assert created.write_token.startswith("cmw_")
    assert len(created.write_token) > len("cmw_") + 20
    assert created.composition_write_token is None


def test_a_create_starting_a_composition_returns_both_tokens(stores):
    created = _create(stores, composition_id="new")

    assert created.write_token is not None
    assert created.write_token.startswith("cmw_")
    assert created.composition_write_token is not None
    assert created.composition_write_token.startswith("cpw_")


def test_two_creates_mint_different_tokens(stores):
    assert _create(stores).write_token != _create(stores).write_token


def test_the_store_keeps_only_the_sha256_digest(stores, revision_store):
    created = _create(stores, composition_id="new")

    component_digest = revision_store.write_token_digest(
        subject_kind="component", subject_id=created.component_id
    )
    composition_digest = revision_store.write_token_digest(
        subject_kind="composition", subject_id=created.composition_id
    )
    assert component_digest == hashlib.sha256(created.write_token.encode()).digest()
    assert composition_digest == hashlib.sha256(created.composition_write_token.encode()).digest()


def test_no_token_reaches_the_stored_revision_or_a_read(stores, revision_store):
    created = _create(stores, composition_id="new")
    revised = _revise(stores, component_id=created.component_id, write_token=created.write_token)

    _assert_no_token_in(created.resolved_payload)
    _assert_no_token_in(created.fact_summary)
    _assert_no_token_in(dataclasses.asdict(revised))
    for revision in (1, 2):
        record = revision_store.read_revision(component_id=created.component_id, revision=revision)
        assert record is not None
        _assert_no_token_in(dataclasses.asdict(record))
        _assert_no_token_in(
            get_component(
                component_id=created.component_id,
                revision=revision,
                revision_store=stores["revision_store"],
            )
        )


# -- revise -------------------------------------------------------------------


def test_a_revise_with_the_components_token_succeeds_and_mints_no_new_token(stores):
    created = _create(stores)

    revised = _revise(stores, component_id=created.component_id, write_token=created.write_token)

    assert revised.revision == 2
    assert revised.write_token is None
    assert revised.composition_write_token is None


@pytest.mark.parametrize("case", ["missing", "wrong", "another_components"])
def test_a_revise_without_the_components_token_is_refused_and_writes_nothing(
    stores, revision_store, case
):
    created = _create(stores)
    other = _create(stores)
    write_token = {
        "missing": None,
        "wrong": "cmw_" + "x" * 22,
        "another_components": other.write_token,
    }[case]

    with pytest.raises(WriteRefused) as excinfo:
        _revise(stores, component_id=created.component_id, write_token=write_token)

    assert excinfo.value.subject_kind == "component"
    assert excinfo.value.subject_id == created.component_id
    assert "create a new component" in excinfo.value.fix
    current = revision_store.read_current(component_id=created.component_id)
    assert current is not None
    assert current.revision == 1


def test_a_component_with_no_stored_digest_refuses_every_revise(stores, revision_store):
    """A component created before write tokens stays readable and cannot be revised."""
    created = _create(stores)
    revision_store.forget_write_token(subject_kind="component", subject_id=created.component_id)

    with pytest.raises(WriteRefused) as excinfo:
        _revise(stores, component_id=created.component_id, write_token=created.write_token)

    assert "predates write tokens" in excinfo.value.detail
    assert (
        get_component(
            component_id=created.component_id,
            revision_store=stores["revision_store"],
        )["revision"]
        == 1
    )


def test_a_revise_of_an_unknown_component_is_still_draft_not_found(stores):
    with pytest.raises(DraftNotFound):
        _revise(stores, component_id="cmp_never_created", write_token="cmw_anything")


# -- compositions -------------------------------------------------------------


def test_an_append_into_an_existing_composition_needs_the_composition_token(stores):
    first = _create(stores, composition_id="new")
    _frame, stat_card = _stat_card_intent()

    appended = _create(
        stores,
        intent=stat_card,
        composition_id=first.composition_id,
        write_token=first.composition_write_token,
    )

    assert appended.composition_id == first.composition_id
    assert appended.composition_revision == 2
    assert appended.write_token is not None
    assert appended.write_token.startswith("cmw_")
    assert appended.composition_write_token is None


@pytest.mark.parametrize("case", ["missing", "the_components_token", "wrong"])
def test_an_append_without_the_composition_token_is_refused_before_any_pin(
    stores, frame_store, case
):
    first = _create(stores, composition_id="new")
    write_token = {
        "missing": None,
        "the_components_token": first.write_token,
        "wrong": "cpw_" + "x" * 22,
    }[case]
    pins_before = frame_store.pins
    _frame, stat_card = _stat_card_intent()

    with pytest.raises(WriteRefused) as excinfo:
        _create(
            stores,
            intent=stat_card,
            composition_id=first.composition_id,
            write_token=write_token,
        )

    assert excinfo.value.subject_kind == "composition"
    assert excinfo.value.subject_id == first.composition_id
    assert frame_store.pins == pins_before


def test_a_composition_with_no_stored_digest_refuses_every_append(stores, revision_store):
    first = _create(stores, composition_id="new")
    revision_store.forget_write_token(subject_kind="composition", subject_id=first.composition_id)

    with pytest.raises(WriteRefused):
        _create(
            stores,
            composition_id=first.composition_id,
            write_token=first.composition_write_token,
        )


def test_revising_a_composition_member_needs_only_the_components_token(stores):
    first = _create(stores, composition_id="new")

    revised = _revise(stores, component_id=first.component_id, write_token=first.write_token)

    assert revised.revision == 2
    assert revised.composition_id == first.composition_id
    assert revised.composition_revision == 2


def test_the_composition_token_does_not_revise_a_member(stores):
    first = _create(stores, composition_id="new")

    with pytest.raises(WriteRefused):
        _revise(stores, component_id=first.component_id, write_token=first.composition_write_token)
