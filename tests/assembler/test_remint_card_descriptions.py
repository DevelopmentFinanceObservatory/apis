"""`datasets-maintenance remint-card-descriptions` revises the cards composed from a record
`backfill-within-sets` filled a set in, whose descriptions still carry the old words.

A card's head revision is selected when the record of its frame's version, or a record of an
input of its computed version, holds a `within_set` fill with basis `backfill`, whenever the
card was written. A dry run lists each card that would change with its old and new words and
writes nothing; an apply revises it through the owner's revise path, so its description names
today's records. A card written after the fill from a record a Presentation instance cached
before it is revised too; one written after the fill from today's record is selected and
unchanged. A card over a record the backfill never filled keeps its head, even where its
record now words it differently, and a second run revises nothing.

Each fresh record here adds the fixed value a stale one lacks, standing in for the set a
record written before sets were recorded leaves out: either rewords the title a card's
citation and a computed card's input labels carry.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from collections.abc import Sequence
from typing import Any, cast

import pytest

from assembler.fakes import FakeFrameStore
from assembler.ports import FrameRef, ResolvedFrame
from dataset_store import DatasetStore
from datasets_api.maintenance import run_remint_card_descriptions
from description.facts import DescriptionRecord, LateField
from presentation_store.store import LivePresentationStore
from tests.presentation_store.conftest import FakePresentationStore

from .test_accept_card_descriptions import (
    _computed_frame,
    _described,
    _frame,
    _input_records,
    _intent,
    _quotient_record,
    _record,
    _vocabulary,
    _vocabulary_source,
    _write,
)

_OTHER = FrameRef(spec_id="spec_" + "7" * 32, content_version="cv_" + "8" * 32)
_STALE_TITLE = "Malaria grants"
"""The title a stale record's mint composed, which a card over its version falls back to."""
_STALE_INPUT = "Malaria grant disbursements"
_FRESH_INPUT = "Malaria grant disbursements for disbursement"


class _Descriptions:
    """The description records Datasets holds, by `(id, content_version)`."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], DescriptionRecord] = {}

    def put(self, record: DescriptionRecord) -> None:
        self.records[(record.id, record.content_version)] = record

    def read_description(self, *, id: str, content_version: str) -> DescriptionRecord | None:
        return self.records.get((id, content_version))

    def read_descriptions(
        self, *, versions: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], DescriptionRecord]:
        return {version: self.records[version] for version in versions if version in self.records}


def _stale(record: DescriptionRecord) -> DescriptionRecord:
    """`record` as written before the backfill: without the value its title names last, and
    titled at mint without it."""
    assert record.variables is not None
    return record.model_copy(
        update={
            "variables": record.variables.model_copy(update={"fixed": []}),
            "evidence": record.evidence.model_copy(update={"title": _STALE_TITLE}),
        }
    )


def _filled(record: DescriptionRecord, *, at: datetime.datetime) -> DescriptionRecord:
    """`record` once `backfill-within-sets --apply` filled the set of its first dimension."""
    return record.model_copy(
        update={
            "late_fields": [
                LateField(field="variables.dimensions.0.within_set", arrived=at, basis="backfill")
            ]
        }
    )


@dataclasses.dataclass
class _World:
    frame_store: FakeFrameStore
    presentation: DatasetStore
    revision_store: LivePresentationStore
    descriptions: _Descriptions
    release_registry: Any
    cards: dict[str, str]

    def write(self, frame: ResolvedFrame, **fields: object) -> str:
        return _write(
            _intent(frame, "line_chart", **fields),
            vocabulary=_vocabulary(),
            frame_store=self.frame_store,
            revision_store=self.revision_store,
            release_registry=self.release_registry,
        ).component_id

    def hold(self, frame: ResolvedFrame, record: DescriptionRecord, **fields: Any) -> None:
        """Serve `frame` with `record` from the frame store and Datasets alike."""
        self.frame_store.put(
            resolved_frame=dataclasses.replace(_described(frame, record), **fields)
        )
        self.descriptions.put(record)

    def head(self, name: str) -> dict[str, Any]:
        record = self.revision_store.read_current(component_id=self.cards[name])
        assert record is not None
        description = cast("dict[str, Any]", record.resolved_payload["description"])
        return {"revision": record.revision, **description}

    def run(
        self, capsys: pytest.CaptureFixture[str], *, apply: bool, every: bool = False
    ) -> list[dict[str, Any]]:
        capsys.readouterr()
        failed = run_remint_card_descriptions(
            presentation=self.presentation,
            revision_store=self.revision_store,
            frame_store=self.frame_store,
            release_registry=self.release_registry,
            vocabulary_source=_vocabulary_source(_vocabulary()),
            descriptions=self.descriptions,
            apply=apply,
            every=every,
        )
        assert failed == 0
        return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


@pytest.fixture
def world(release_registry) -> _World:
    """Five cards: one over the malaria version and one over a quotient computed from it, both
    issued before the backfill filled the malaria record; one over another version whose record
    was reworded with no fill; and two over the malaria version issued after the fill, one from
    the record a Presentation instance cached before it and one from today's record."""
    # The fake answers the statements the live store issues.
    presentation = cast("DatasetStore", FakePresentationStore())
    world = _World(
        frame_store=FakeFrameStore(),
        presentation=presentation,
        revision_store=LivePresentationStore(store=presentation),
        descriptions=_Descriptions(),
        release_registry=release_registry,
        cards={},
    )
    malaria, other, quotient = _frame(), _frame(frame_ref=_OTHER), _computed_frame()
    fresh_malaria, gni = _input_records()
    stale_malaria = _stale(fresh_malaria)
    quotient_labels = {
        **quotient.labels,
        "MalariaGrantDisbursements": "Malaria grant disbursements",
        "GNI": "Gross national income",
    }
    world.descriptions.put(gni)

    world.hold(malaria, stale_malaria)
    world.hold(other, _stale(_record(other)))
    world.hold(
        quotient,
        _quotient_record(quotient),
        input_records=(stale_malaria, gni),
        labels=quotient_labels,
    )
    world.cards["malaria"] = world.write(malaria)
    world.cards["quotient"] = world.write(quotient, title="Malaria grants per dollar of GNI")
    world.cards["setless"] = world.write(other)

    filled = _filled(fresh_malaria, at=datetime.datetime.now(datetime.UTC))
    world.descriptions.put(filled)
    world.cards["cached"] = world.write(malaria)
    world.hold(malaria, filled)
    world.hold(other, _record(other))
    world.hold(
        quotient, _quotient_record(quotient), input_records=(filled, gni), labels=quotient_labels
    )
    world.cards["fresh"] = world.write(malaria)
    return world


def _input_labels(description: dict[str, Any]) -> list[str]:
    return [source["label"] for source in description["sources"] if "content_version" in source]


def test_a_dry_run_lists_the_stale_cards_and_writes_nothing(
    world: _World, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = world.run(capsys, apply=False)

    planned = {
        line["component_id"]: line
        for line in lines[:-1]
        if line["event"] == "presentation_card_remint_planned"
    }
    assert set(planned) == {world.cards[name] for name in ("malaria", "quotient", "cached")}
    malaria = planned[world.cards["malaria"]]
    assert malaria["event"] == "presentation_card_remint_planned"
    assert (malaria["revision"], malaria["new_revision"]) == (1, 2)
    assert malaria["old_title"].startswith(f"{_STALE_TITLE},")
    assert not malaria["new_title"].startswith(f"{_STALE_TITLE},")
    quotient = planned[world.cards["quotient"]]
    assert quotient["old_title"] != quotient["new_title"]
    counts = ("applied", "selected", "revised", "unchanged", "failed")
    assert {key: lines[-1][key] for key in counts} == {
        "applied": False,
        "selected": 4,
        "revised": 3,
        "unchanged": 1,
        "failed": 0,
    }
    assert {name: world.head(name)["revision"] for name in world.cards} == dict.fromkeys(
        world.cards, 1
    )


@pytest.mark.parametrize(
    ("every", "revised"),
    [
        pytest.param(False, {"malaria", "quotient", "cached"}, id="cards-over-filled-records"),
        pytest.param(
            True, {"malaria", "quotient", "cached", "setless"}, id="all-selects-every-card"
        ),
    ],
)
def test_an_apply_revises_the_stale_cards_to_todays_records_and_a_second_run_revises_none(
    world: _World, capsys: pytest.CaptureFixture[str], every: bool, revised: set[str]
) -> None:
    """`--all` also revises the card over a record reworded with no fill, as a composer
    change rewords every card's record."""
    before = {name: world.head(name) for name in world.cards}
    assert _input_labels(before["quotient"])[0] == _STALE_INPUT

    lines = world.run(capsys, apply=True, every=every)

    events = {line["component_id"]: line["event"] for line in lines[:-1]}
    selected = {name for name in world.cards if every or name != "setless"}
    assert events == {
        world.cards[name]: "presentation_card_reminted"
        if name in revised
        else "presentation_card_remint_unchanged"
        for name in selected
    }
    after = {name: world.head(name) for name in world.cards}
    assert before["cached"]["title"] == before["malaria"]["title"]
    for name in ("malaria", "cached"):
        assert after[name]["revision"] == 2
        assert after[name]["title"] == before["fresh"]["title"]
        assert not after[name]["title"].startswith(f"{_STALE_TITLE},")
    assert after["quotient"]["revision"] == 2
    assert _input_labels(after["quotient"])[0] == _FRESH_INPUT
    if every:
        assert after["setless"]["revision"] == 2
        assert after["setless"]["title"] != before["setless"]["title"]
    else:
        assert after["setless"] == before["setless"], "no fill reached the record it is drawn from"
    assert after["fresh"] == before["fresh"], "issued from today's record"

    again = world.run(capsys, apply=True, every=every)
    assert {line["event"] for line in again[:-1]} == {"presentation_card_remint_unchanged"}
    counts = ("selected", "revised", "unchanged")
    n = len(selected)
    assert {key: again[-1][key] for key in counts} == {"selected": n, "revised": 0, "unchanged": n}
    assert {name: world.head(name) for name in world.cards} == after
