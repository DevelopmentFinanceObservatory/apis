"""Search keeps a member's facets to its family's elected dataset by the mint's test, and
refuses a member a mint of its answer serves where the mint refuses it, by the mint's own
method (`materialise.facets.ElectedDataset.published`)."""

from __future__ import annotations

import pytest

from dc_graph import Facet
from materialise.errors import DatasetDisagreementError
from materialise.facets import ElectedDataset
from search.engine.retrieval.elected_dataset import check_elected_datasets, published

_SV = "sv"
_ELECTED = "provenance/Elected"
_OTHER = "provenance/Other"
_UNRECORDED = "provenance/Unrecorded"
_DATASETS = {_SV: ElectedDataset(provenance=_ELECTED, member_provenances=(_ELECTED, _OTHER))}


def _facet(facet_id: str, provenance: str) -> Facet:
    return Facet(
        obs_count=1,
        earliest_date="2020",
        latest_date="2020",
        facet_id=facet_id,
        provenance_id=provenance,
    )


def test_a_member_keeps_only_its_elected_datasets_facets():
    candidates = [_facet("f1", _OTHER), _facet("f2", _ELECTED), _facet("f3", _UNRECORDED)]

    assert published(_DATASETS, _SV, candidates) == [candidates[1]]


def test_a_reader_filters_without_refusing():
    """A reader of one window or one place keeps what the elected dataset publishes there;
    the refusal is the resolver's, over the member's facets in every year."""
    assert published(_DATASETS, _SV, [_facet("f3", _UNRECORDED)]) == []


def test_a_member_under_another_recorded_dataset_alone_is_no_disagreement():
    check_elected_datasets(_DATASETS, {_SV: [_facet("f1", _OTHER)]})


def test_a_member_only_an_unrecorded_dataset_publishes_raises_the_mints_error():
    with pytest.raises(DatasetDisagreementError) as raised:
        check_elected_datasets(_DATASETS, {_SV: [_facet("f1", _OTHER), _facet("f3", _UNRECORDED)]})

    assert raised.value.stat_var == _SV
    assert raised.value.dataset == _ELECTED


def test_a_member_its_elected_dataset_publishes_anywhere_is_no_disagreement():
    """The elected dataset publishes the member at one entity, so an unrecorded dataset at
    another is no disagreement, as the mint checks across the bound entities."""
    check_elected_datasets(_DATASETS, {_SV: [_facet("f3", _UNRECORDED), _facet("f2", _ELECTED)]})


def test_a_member_with_no_elected_dataset_is_not_checked():
    check_elected_datasets({}, {_SV: [_facet("f3", _UNRECORDED)]})


# ---------------------------------------------------------------------------
# A standard answer checks only the members its mint serves.
# ---------------------------------------------------------------------------

_P = "dc/base/Elected"
_Q = "dc/base/Unrecorded"
_TOTAL = "Count_Person"
_FEMALE = "Count_Person_Female"
_MALE = "Count_Person_Male"
_KEN = "country/KEN"


def _population_row(dcid: str, constraints: dict[str, str]):
    from code_index.types import StatVarRow

    return StatVarRow(
        dcid=dcid,
        population_type="Person",
        measured_property="count",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("observationAbout",),
        constraint_properties=constraints,
        provenances=(_P,),
    )


def _population_arcs(constraints: dict[str, str]) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


def _resolve_population(monkeypatch, members: dict[str, tuple[dict[str, str], str]]):
    """An index family of `members`, each dcid to its constraints and the provenance of its
    one facet, electing `_P` at Kenya with gender unbound."""
    from code_index.members import DatasetFamily
    from search.engine.bind import SlotBindingDraft
    from search.engine.retrieval import bound_values
    from search.engine.retrieval.standard import standard_resolve
    from search.engine.shape import ShapeDraft
    from tests.fixtures import FakeGraph

    family = DatasetFamily(
        provenance=_P,
        rows=tuple(_population_row(sv, constraints) for sv, (constraints, _) in members.items()),
    )
    monkeypatch.setattr(bound_values, "index_family", lambda _index, _query: family)
    shape = ShapeDraft(
        shape_id="population",
        label="Population",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            sv: _population_arcs(constraints) for sv, (constraints, _) in members.items()
        },
        representative_score=0.9,
    )

    def point(provenance: str) -> list[dict]:
        return [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": provenance,
            }
        ]

    return standard_resolve(
        shape=shape,
        bindings=[
            SlotBindingDraft(axis="how", property_dcid="gender", kind="unbound", value_dcids=[])
        ],
        entities={"observationAbout": [_KEN]},
        graph=FakeGraph(
            nodes={},
            obs={f"{sv}|{_KEN}": point(provenance) for sv, (_, provenance) in members.items()},
            detect={},
            resolve={},
        ),
        place_dcids=[_KEN],
    )


@pytest.mark.parametrize("female_provenance", [_P, _Q])
def test_a_member_the_mint_does_not_serve_is_not_checked(monkeypatch, female_provenance):
    """Gender is unbound, so the total dominates the female member and the mint serves the
    total alone. The female member is not checked, whichever dataset publishes it."""
    from search.engine.retrieval.retrieve import Materialised

    result = _resolve_population(
        monkeypatch, {_TOTAL: ({}, _P), _FEMALE: ({"gender": "Female"}, female_provenance)}
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_TOTAL]


def test_a_member_the_mint_serves_is_checked(monkeypatch):
    with pytest.raises(DatasetDisagreementError) as raised:
        _resolve_population(monkeypatch, {_TOTAL: ({}, _Q), _FEMALE: ({"gender": "Female"}, _P)})

    assert raised.value.stat_var == _TOTAL


def test_a_kept_member_after_the_elected_one_is_still_checked(monkeypatch):
    """With no total the mint serves both members, so the male member is checked although
    the female member, read first under one StatVar per read, is elected."""
    from search.engine.retrieval import standard

    monkeypatch.setattr(standard, "_FACET_PROBE_MAX_PAIRS_PER_POST", 1)
    with pytest.raises(DatasetDisagreementError) as raised:
        _resolve_population(
            monkeypatch,
            {_FEMALE: ({"gender": "Female"}, _P), _MALE: ({"gender": "Male"}, _Q)},
        )

    assert raised.value.stat_var == _MALE


def test_an_alternate_the_mint_refuses_is_dropped_and_the_others_stand(monkeypatch):
    """An alternate whose member the mint refuses is dropped, as one whose identity the
    mint's walk refuses is; the other alternate is materialised and the call returns."""
    from search.engine.region import candidates as candidates_module
    from search.engine.retrieval.retrieve import Materialised
    from search.engine.shape import ShapeDraft
    from search.models import CoverageBare
    from tests.engine._builders import (
        bound_place_context,
        materialise_alternates,
        subject_place_roles,
    )
    from tests.engine._fake_coverage import empty_coverage_snapshot
    from tests.engine._fake_sdmx import FakeSdmxClient
    from tests.fixtures import FakeGraph

    def alternate(shape_id: str, stat_var: str) -> ShapeDraft:
        return ShapeDraft(
            shape_id=shape_id,
            label=shape_id,
            pop_type_dcid="Person",
            meas_prop_dcid="count",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            slot_keys=(),
            sv_arc_facts={stat_var: _population_arcs({})},
            representative_score=0.9,
        )

    def materialise(shape, *_args, **_kwargs):
        (stat_var,) = shape.sv_arc_facts
        if shape.shape_id == "stale":
            raise DatasetDisagreementError("stale", dataset=_P, stat_var=stat_var)
        return Materialised(
            sv_dcids=[stat_var], has_data=True, coverage=CoverageBare(has_data=True)
        )

    monkeypatch.setattr(candidates_module, "materialise", materialise)

    results, unsupported = materialise_alternates(
        [alternate("stale", "Count_Stale"), alternate("sound", "Count_Sound")],
        places=[bound_place_context(subject_place_roles(_KEN), _KEN)] * 2,
        elected_bindings=[],
        elected_has_data=False,
        named_place_dcids=[_KEN],
        graph=FakeGraph(nodes={}, obs={}, detect={}, resolve={}),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    assert unsupported is None
    assert [candidate.shape.shape_id for candidate in results] == ["sound"]
