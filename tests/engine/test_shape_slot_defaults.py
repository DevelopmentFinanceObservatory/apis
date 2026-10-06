"""`bind_repair.apply_shape_slot_defaults`: an open `medicalStatus` slot on a births shape
binds `LiveBirth`, so "birth rate" reads the crude birth rate."""

from __future__ import annotations

import logging

from search.engine.bind import DraftBindingEvidence, DraftBindingKind, SlotBindingDraft
from search.engine.bind_repair import SHAPE_SLOT_DEFAULTS, apply_shape_slot_defaults
from search.engine.errors import GraphInfraError
from search.engine.shape import ShapeDraft, shape_draft_from
from tests.engine.test_gap_fill import (
    _BIRTH_RATE_QUERY,
    _BIRTHS_PER_CAPITA,
    _CRUDE_BIRTH_RATE,
    _ETH,
    _FACET,
    _FERTILITY_RATE,
    _UN_TOTAL_FERTILITY_RATE,
    _birth_rate_graph,
    _constrained_sv_node,
    _gap_fill_lines,
    _leg,
    _menu_named,
    _RecordedBirthRateLLM,
    _resolve,
)
from tests.fixtures import FakeGraph


def _shape(
    *, population_type: str = "BirthEvent", taxonomy: tuple[str, ...] = ("LiveBirth",)
) -> ShapeDraft:
    """A shape with one `medicalStatus` slot whose taxonomy holds `taxonomy`."""
    return shape_draft_from(
        shape_id="births_per_capita",
        label="Count Of Birth Event (Per Capita)",
        pop_type_dcid=population_type,
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid="Count_Person",
        constraint_props=["medicalStatus"],
        prop_labels={"medicalStatus": "medicalStatus"},
        prop_observed_values={"medicalStatus": list(taxonomy)},
        sv_arc_facts={},
        slot_taxonomy={"how:medicalStatus": list(taxonomy)},
        place_available=True,
        value_labels=None,
        recalled=False,
    )


def _status(
    kind: DraftBindingKind = "unbound",
    *,
    value: str | None = None,
    evidence: DraftBindingEvidence = "mentioned",
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid="medicalStatus",
        kind=kind,
        value_dcids=[] if value is None else [value],
        evidence=evidence,
    )


def test_births_declare_live_birth_as_the_medical_status_default() -> None:
    assert [(d.population_type, d.property_dcid, d.value_dcid) for d in SHAPE_SLOT_DEFAULTS] == [
        ("BirthEvent", "medicalStatus", "LiveBirth")
    ]


def test_an_unbound_status_on_a_births_shape_binds_live_birth() -> None:
    draft = _status()

    repaired, defaulted = apply_shape_slot_defaults([draft], _shape())

    assert defaulted == [draft]
    assert [(b.property_dcid, b.kind, b.value_dcids) for b in repaired] == [
        ("medicalStatus", "value", ["LiveBirth"])
    ]


def test_a_status_drafted_absent_binds_live_birth() -> None:
    repaired, defaulted = apply_shape_slot_defaults([_status("absent")], _shape())

    assert len(defaulted) == 1
    assert repaired[0].kind == "value"
    assert repaired[0].value_dcids == ["LiveBirth"]


def test_a_status_the_binder_bound_keeps_its_value() -> None:
    bound = _status("value", value="Stillbirth")

    repaired, defaulted = apply_shape_slot_defaults(
        [bound], _shape(taxonomy=("LiveBirth", "Stillbirth"))
    )

    assert defaulted == []
    assert repaired == [bound]


def test_a_status_the_query_asked_for_in_full_stays_open() -> None:
    requested = _status(evidence="requested_all")

    repaired, defaulted = apply_shape_slot_defaults([requested], _shape())

    assert defaulted == []
    assert repaired == [requested]


def test_a_place_where_no_live_birth_member_has_data_keeps_the_slot_open() -> None:
    draft = _status()

    repaired, defaulted = apply_shape_slot_defaults([draft], _shape(taxonomy=()))

    assert defaulted == []
    assert repaired == [draft]


def test_another_population_type_keeps_its_open_status() -> None:
    draft = _status()

    repaired, defaulted = apply_shape_slot_defaults(
        [draft], _shape(population_type="MortalityEvent")
    )

    assert defaulted == []
    assert repaired == [draft]


def test_birth_rate_ethiopia_answers_the_crude_birth_rate_beside_the_fertility_rate(
    caplog,
) -> None:
    """The reads and the first-pass bind recorded for "birth rate Ethiopia": the model elects
    births per capita with medicalStatus unbound and nominates the fertility rate."""
    llm = _RecordedBirthRateLLM()
    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(
            _birth_rate_graph(un_score=0.73), query="birth rate", entities=["Ethiopia"], llm=llm
        )

    assert result.status == "candidates"
    assert [[sv.ref.dcid for sv in spec.resolved.members] for spec in result.specs] == [
        [_CRUDE_BIRTH_RATE],
        [_FERTILITY_RATE],
    ]
    crude_birth_rate = result.specs[0]
    assert [
        (f.key.property.dcid, [ref.dcid for ref in f.refs], f.provenance)
        for f in crude_birth_rate.resolved.trace.slot_filters
        if f.key.property is not None
    ] == [("medicalStatus", ["LiveBirth"], "declared_default")]
    assert not _menu_named(llm, _UN_TOTAL_FERTILITY_RATE), "a second pass ran"
    (line,) = _gap_fill_lines(caplog)
    assert "pass1=candidates/None" in line
    assert "pass2=skipped:answered" in line


class _ProbeFailsGraph(FakeGraph):
    """A graph whose place-availability probe, the first batch read, raises, so the shape's
    taxonomy lists every member and `place_available` is unknown. Later reads succeed."""

    _probed = False

    def observation_facets_batch(self, *args, **kwargs):
        if not self._probed:
            self._probed = True
            raise GraphInfraError("simulated availability probe failure")
        return super().observation_facets_batch(*args, **kwargs)


def test_a_failed_availability_probe_leaves_the_status_open_and_the_total_answers() -> None:
    """Only the births total has data at Ethiopia. With the probe down the taxonomy still
    lists `LiveBirth`, and a default bound from it would read the crude birth rate: no data."""
    births = {"population_type": "BirthEvent", "meas_prop": "count", "denominator": "Count_Person"}
    graph = _ProbeFailsGraph(
        nodes={
            _ETH: {"label": "Ethiopia", "type": "Country"},
            _BIRTHS_PER_CAPITA: _constrained_sv_node(_BIRTHS_PER_CAPITA, constraints={}, **births),
            _CRUDE_BIRTH_RATE: _constrained_sv_node(
                _CRUDE_BIRTH_RATE, constraints={"medicalStatus": "LiveBirth"}, **births
            ),
        },
        obs={f"{_BIRTHS_PER_CAPITA}|{_ETH}": [_FACET]},
        detect={
            _BIRTH_RATE_QUERY: _leg((_BIRTHS_PER_CAPITA, 1.0), (_CRUDE_BIRTH_RATE, 1.0)),
        },
        un_detect={},
        resolve={"Ethiopia": _ETH},
    )

    result = _resolve(
        graph, query=_BIRTH_RATE_QUERY, entities=["Ethiopia"], llm=_RecordedBirthRateLLM()
    )

    assert result.status != "no_data"
    assert [[sv.ref.dcid for sv in spec.resolved.members] for spec in result.specs] == [
        [_BIRTHS_PER_CAPITA]
    ]
