"""A description record is written once, by the first call that can build it.

A mint that finds its version recorded leaves the record as it is. A mint or a republish that
finds a version with no record writes one, built from the stored version, so a writer that lost
the race to record a version takes the winner's citation and preparation date, never its own.
A mint served from a current head writes nothing, a record included, so the mint rows age the
head and the second mint rebuilds onto the stored version (`_rebuilt`); the backfill describes
versions made before records.
A field left `null` fills once, from evidence of the same version, and the record notes when it
arrived. A filled field never changes, and a republish carrying no steps leaves recorded steps
as they are.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from artifact_store import LiveArtifactInventory
from description.facts import DescriptionRecord

from ._description_doubles import (
    DENOMINATOR,
    NUMERATOR,
    OCV,
    OPS_ID,
    datasets_client,
    forget_record,
    mint_definition,
    publish,
    publish_body,
    quotient_steps,
    read_record,
    served_steps,
    store_as_frame1_era,
)
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door_currency import _FakeHeadStore, _RenderableArtifactStore

# Past every configured head maximum age, so the next mint rebuilds.
_STALE = timedelta(days=30)


def _rebuilt(client, heads: _FakeHeadStore, spec_id: str) -> dict[str, Any]:
    """A second mint of the `definition` fixture after its head has aged, so it materialises
    again and writes onto the version the first mint stored. A head hit writes no record."""
    heads.age_head(spec_id, age=_STALE)
    manifest = mint_definition(client)
    assert manifest["trace"]["served_from"] != "head"
    return manifest


def test_a_mint_that_finds_its_version_recorded_leaves_the_record_as_it_is() -> None:
    store = FakeVersionRecordDatasetStore()
    heads = _FakeHeadStore()
    client = datasets_client(store, head_store=heads)
    first = mint_definition(client)
    spec_id, content_version = first["definition"]["spec_id"], first["content_version"]
    written = read_record(store, subject_id=spec_id, content_version=content_version)
    assert written is not None

    second = _rebuilt(client, heads, spec_id)

    assert second["content_version"] == content_version
    assert read_record(store, subject_id=spec_id, content_version=content_version) == written


def test_a_mint_that_finds_its_version_without_a_record_writes_one() -> None:
    store = FakeVersionRecordDatasetStore()
    heads = _FakeHeadStore()
    client = datasets_client(store, head_store=heads)
    first = mint_definition(client)
    spec_id, content_version = first["definition"]["spec_id"], first["content_version"]
    forget_record(store, subject_id=spec_id, content_version=content_version)

    _rebuilt(client, heads, spec_id)

    record = read_record(store, subject_id=spec_id, content_version=content_version)
    assert record is not None
    assert record.basis == "mint"


# A version recorded before frame hashes, holding a partial frame1 document whose digest is
# not its frame's hash, matches a republish by the hash of the frame read.
_ERAS = pytest.mark.parametrize("frame1_era", [False, True], ids=["hashed", "frame1-era"])


@_ERAS
def test_a_republish_of_a_frame_without_a_record_writes_one(frame1_era: bool) -> None:
    store = FakeVersionRecordDatasetStore()
    artifacts = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=store))
    client = datasets_client(store, artifact_store=artifacts)
    steps = quotient_steps()
    assert publish(client, publish_body(steps=steps)).status_code == 200
    forget_record(store, subject_id=OPS_ID, content_version=OCV)
    if frame1_era:
        store_as_frame1_era(store, artifacts, subject_id=OPS_ID, content_version=OCV)

    response = publish(client, publish_body(steps=steps))

    assert response.status_code == 200, response.text
    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert record.basis == "publish"
    assert served_steps(record) == steps


_WINNER_PREPARED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
_WINNER_SOURCE_NAME = "The winner's source name"


def test_a_writer_that_lost_the_race_records_the_winners_citation_and_date() -> None:
    """The version row holds what the first writer stored, here a date and a source name no
    mint of this test would produce. A later mint that rebuilds the same version, finding no
    record, builds it from the stored version and not from its own trace."""
    store = FakeVersionRecordDatasetStore()
    heads = _FakeHeadStore()
    client = datasets_client(store, head_store=heads)
    first = mint_definition(client)
    spec_id, content_version = first["definition"]["spec_id"], first["content_version"]
    forget_record(store, subject_id=spec_id, content_version=content_version)
    citation = first["citation"]
    citation["sources"][0]["label"] = _WINNER_SOURCE_NAME
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET prepared_at = %s, citation = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (_WINNER_PREPARED, citation, spec_id, content_version),
        )

    loser = _rebuilt(client, heads, spec_id)

    assert loser["content_version"] == content_version
    record = read_record(store, subject_id=spec_id, content_version=content_version)
    assert record is not None
    assert record.evidence.prepared == date(2026, 1, 2)
    assert [source.name for source in record.evidence.sources] == [_WINNER_SOURCE_NAME]


def _published_with_steps_first() -> DescriptionRecord:
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    assert publish(client, publish_body(steps=quotient_steps())).status_code == 200
    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    return record


@_ERAS
def test_null_steps_fill_once_from_a_matching_republish_and_are_noted_as_late(
    frame1_era: bool,
) -> None:
    """A record first written without steps lists its lineage's inputs with unknown ids and
    holds the caveats and labels its steps state as unknown. A republish with steps fills them
    and replaces the inputs with those the steps retrieve and the series facts with those
    counted under the continuations they carry, so the record states what a first publish with
    those steps states."""
    store = FakeVersionRecordDatasetStore()
    artifacts = _RenderableArtifactStore(inventory=LiveArtifactInventory(store=store))
    client = datasets_client(store, artifact_store=artifacts)
    assert publish(client, publish_body()).status_code == 200
    first = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert first is not None
    assert first.computed is not None
    assert (first.steps, first.caveats, first.evidence.labels) == (None, None, None)
    assert [(held.id, held.content_version) for held in first.computed.inputs] == [
        (None, NUMERATOR[1]),
        (None, DENOMINATOR[1]),
    ]
    assert first.late_fields == []
    if frame1_era:
        store_as_frame1_era(store, artifacts, subject_id=OPS_ID, content_version=OCV)
    steps = quotient_steps()

    response = publish(client, publish_body(steps=steps))

    assert response.status_code == 200, response.text
    filled = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert filled is not None
    assert served_steps(filled) == steps
    assert sorted((late.field, late.basis) for late in filled.late_fields) == [
        ("caveats", "publish"),
        ("computed.inputs", "publish"),
        ("evidence.labels", "publish"),
        ("series", "publish"),
        ("steps", "publish"),
    ]
    assert filled.computed is not None
    assert [(held.id, held.content_version) for held in filled.computed.inputs] == [
        NUMERATOR,
        DENOMINATOR,
    ]
    direct = _published_with_steps_first()
    assert direct.caveats
    assert direct.computed is not None
    assert (filled.caveats, filled.evidence.labels, filled.computed.inputs) == (
        direct.caveats,
        direct.evidence.labels,
        direct.computed.inputs,
    )
    assert filled.basis == first.basis
    assert filled.evidence.prepared == first.evidence.prepared
    assert filled.series == first.series


def test_a_chained_frame_first_published_without_steps_takes_its_roots_as_inputs() -> None:
    """`ai` lists the immediate input frames, here a filtered frame by the `cv_` of its
    bytes. The late steps retrieve the roots, which replace that lineage as the inputs."""
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    filtered = "cv_" + "f0" * 16
    lineage = {"input_content_versions": [filtered, DENOMINATOR[1]]}
    steps = [
        {
            "id": "s1",
            "op": "retrieve",
            "input": {"id": NUMERATOR[0], "content_version": NUMERATOR[1]},
        },
        {"id": "s2", "op": "filter", "input": "s1", "params": {}},
        {
            "id": "s3",
            "op": "retrieve",
            "input": {"id": DENOMINATOR[0], "content_version": DENOMINATOR[1]},
        },
        {"id": "s4", "op": "quotient", "inputs": {"numerator": "s2", "denominator": "s3"}},
    ]
    assert publish(client, publish_body() | lineage).status_code == 200
    first = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert first is not None
    assert first.computed is not None
    assert [held.content_version for held in first.computed.inputs] == [filtered, DENOMINATOR[1]]

    response = publish(client, publish_body(steps=steps) | lineage)

    assert response.status_code == 200, response.text
    filled = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert filled is not None
    assert filled.computed is not None
    assert [(held.id, held.content_version) for held in filled.computed.inputs] == [
        NUMERATOR,
        DENOMINATOR,
    ]


def test_filled_steps_never_change_when_a_later_publish_words_them_differently() -> None:
    """A republish whose steps carry the same facts under other labels is accepted, and the
    record keeps the steps and the late-field note of the publish that filled them."""
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    assert publish(client, publish_body()).status_code == 200
    steps = quotient_steps()
    assert publish(client, publish_body(steps=steps)).status_code == 200
    filled = read_record(store, subject_id=OPS_ID, content_version=OCV)

    response = publish(
        client, publish_body(steps=quotient_steps(numerator_label="ODA, gross disbursements"))
    )

    assert response.status_code == 200, response.text
    kept = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert kept == filled
    assert kept is not None
    assert served_steps(kept) == steps


def test_a_republish_without_steps_over_recorded_steps_changes_nothing() -> None:
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    assert publish(client, publish_body(steps=quotient_steps())).status_code == 200
    recorded = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert recorded is not None

    response = publish(client, publish_body())

    assert response.status_code == 200, response.text
    assert read_record(store, subject_id=OPS_ID, content_version=OCV) == recorded


def test_a_republish_with_empty_steps_over_a_record_without_steps_is_refused() -> None:
    """Empty steps retrieve no version, so they would replace the lineage's inputs with none."""
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    assert publish(client, publish_body()).status_code == 200
    recorded = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert recorded is not None

    response = publish(client, publish_body(steps=[]))

    assert response.status_code == 400, response.text
    assert response.json()["pointer"] == "/steps"
    assert read_record(store, subject_id=OPS_ID, content_version=OCV) == recorded
