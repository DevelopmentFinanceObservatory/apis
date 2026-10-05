"""Tests for how the dataset a variable asks for reaches its candidate shapes, its
bindings, its slots and its sources.

`dataset_mention` reads what a mention asks for; shape candidacy keeps the shapes of the
dataset it names, or drops every opt-in dataset's shapes when it names none;
`region.named_source` writes the source into the bindings a family is elected from, and
`spec_build` binds the source slot from them.
"""

from __future__ import annotations

import dataclasses

import pytest

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.dataset_mention import dataset_mentions
from search.engine.qualifier_match import CodedQualifier
from search.engine.region import candidates as candidates_module
from search.engine.region.named_source import (
    with_mention,
    with_named_source,
    without_dataset_drafts,
)
from search.engine.region.phase_rank_bind import _dataset_candidates, _read_mention
from search.engine.region.place_context import empty_place_context
from search.engine.retrieval.bound_values import (
    bound_source,
    shape_query,
    source_binding,
    synthetic_shape_for_identity,
)
from search.engine.retrieval.retrieve import NoDataDraft
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_build import build_slot, elected_sources, pair_slot_drafts
from search.engine.spec_id import _facets_member, mint_identity
from search.models import BaseIdentity, BindingUnbound, GraphRef, IdentityShapeSchema, ProvenanceRef
from tests.code_index.aid_identity import (
    AID_TYPE,
    CRS_PROVENANCE,
    IMPUTED_MULTILATERAL,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
)
from tests.engine._builders import materialise_alternates
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_INDEX = get_index()
_COMMITMENTS = "FinancialObligation"


def _aid_family_shape(identity: dict | None = None) -> ShapeDraft:
    """The shape of `identity`, the education-aid identity by default, with every slot key
    it declares."""
    identity = identity if identity is not None else education_aid_identity()
    _, definition = mint_identity(identity)
    assert isinstance(definition.shape, IdentityShapeSchema)
    return dataclasses.replace(
        synthetic_shape_for_identity(definition.shape),
        sv_arc_facts_complete=False,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=slot["property"], label=slot["property"])
            for slot in identity["slots"]
        ),
    )


def _imputed_family_shape() -> ShapeDraft:
    """The imputed multilateral table's shape, on its own `attributedAmount` five-tuple."""
    return _aid_family_shape(imputed_education_aid_identity())


def _population_shape() -> ShapeDraft:
    """A shape no declared dataset publishes."""
    return dataclasses.replace(
        _aid_family_shape(), pop_type_dcid="Person", meas_prop_dcid="count", meas_qual_dcid=None
    )


def _commitments_shape() -> ShapeDraft:
    """A second shape the CRS publishes, on its commitments five-tuple."""
    rows = [
        row
        for row in _INDEX.statvars
        if row.population_type == _COMMITMENTS and row.measurement_denominator is None
    ]
    first = rows[0]
    rows = [
        row
        for row in rows
        if (row.measured_property, row.stat_type, row.measurement_qualifier)
        == (first.measured_property, first.stat_type, first.measurement_qualifier)
    ]
    assert not any(IMPUTED_PROVENANCE in row.provenances for row in rows)
    return dataclasses.replace(
        _aid_family_shape(),
        pop_type_dcid=_COMMITMENTS,
        meas_prop_dcid=first.measured_property,
        stat_type_dcid=first.stat_type,
        meas_qual_dcid=first.measurement_qualifier,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=prop, label=prop)
            for prop in sorted({prop for row in rows for prop in row.constraint_properties})
        ),
    )


def _mention(variable: str, index=_INDEX):
    (mention,) = dataset_mentions(variable, index)
    return mention


def _purpose(code: str) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid="crsPurpose", kind="value", value_dcids=[code]
    )


@pytest.mark.parametrize(
    ("sidecar", "key"),
    [
        (("imputed multilateral",), IMPUTED_PROVENANCE),
        (("purpose code 110", "CRS"), CRS_PROVENANCE),
        (("purpose code 110",), None),
        (("imputed multilateral education aid",), None),
        ((), None),
    ],
)
def test_a_variable_reads_its_mention_from_a_sidecar_phrase_that_is_a_mention_alone(sidecar, key):
    qualifiers = tuple(CodedQualifier(phrase=phrase, spans=(), matches=()) for phrase in sidecar)

    mention = _read_mention(qualifiers)

    assert (mention.provenance if mention is not None else None) == key
    assert _read_mention(None) is None


# ---------------------------------------------------------------------------
# The mention in the bindings
# ---------------------------------------------------------------------------


def test_a_mention_binds_the_source():
    asked = with_mention([], _mention("CRS aid"))

    assert asked == [source_binding(CRS_PROVENANCE)]
    assert shape_query(_aid_family_shape(), asked).source == CRS_PROVENANCE


def test_every_drafted_binding_on_the_source_is_removed():
    purpose = _purpose("CRSPurposeCode110")
    drafts = [purpose, source_binding("provenance/Other")]

    assert without_dataset_drafts(drafts) == [purpose]
    assert with_mention(drafts, _mention("CRS aid")) == [
        purpose,
        source_binding(CRS_PROVENANCE),
    ]


def test_with_named_source_names_a_dataset_only_for_a_shape_it_publishes():
    drafted = source_binding("provenance/Other")

    assert with_named_source([drafted], _imputed_family_shape(), IMPUTED_PROVENANCE) == [
        source_binding(IMPUTED_PROVENANCE)
    ]
    assert with_named_source([drafted], _imputed_family_shape(), None) == []
    assert with_named_source([], _aid_family_shape(), IMPUTED_PROVENANCE) == []
    assert with_named_source([], _commitments_shape(), CRS_PROVENANCE) == [
        source_binding(CRS_PROVENANCE)
    ]
    assert with_named_source([], _population_shape(), CRS_PROVENANCE) == []


# ---------------------------------------------------------------------------
# Shape candidacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "kept"),
    [
        (None, ["aid", "commitments", "population"]),
        ("imputed multilateral aid", ["imputed"]),
        ("CRS aid", ["aid", "commitments"]),
    ],
    ids=["no-mention", "opt-in-mention", "default-mention"],
)
def test_a_part_elects_from_the_shapes_of_the_dataset_its_mention_names(variable, kept):
    """A part with no mention drops the opt-in dataset's shapes alone. A part with a
    mention keeps its dataset's shapes alone, and so drops a shape no dataset publishes."""
    shapes = {
        "aid": _aid_family_shape(),
        "imputed": _imputed_family_shape(),
        "commitments": _commitments_shape(),
        "population": _population_shape(),
    }
    mention = _mention(variable) if variable is not None else None

    candidates = _dataset_candidates(list(shapes.values()), mention)

    assert [name for name, shape in shapes.items() if shape in candidates] == kept


@pytest.mark.parametrize(
    ("bindings", "source"),
    [
        ([], None),
        ([source_binding(CRS_PROVENANCE)], CRS_PROVENANCE),
        ([SlotBindingDraft(axis="source", property_dcid=None, kind="unbound")], None),
        (
            [
                SlotBindingDraft(
                    axis="source",
                    property_dcid=None,
                    kind="set",
                    value_dcids=[CRS_PROVENANCE, IMPUTED_PROVENANCE],
                )
            ],
            None,
        ),
        ([source_binding(CRS_PROVENANCE), source_binding(IMPUTED_PROVENANCE)], None),
    ],
    ids=["none", "one-value", "unbound", "set", "two-values"],
)
def test_bound_source_reads_one_source_axis_value_binding(bindings, source):
    assert bound_source(bindings) == source
    assert shape_query(_aid_family_shape(), bindings).source == source


# ---------------------------------------------------------------------------
# Slots, candidates and sources
# ---------------------------------------------------------------------------


def test_an_alternate_shape_is_materialised_under_the_elected_bindings_source(monkeypatch):
    """A candidate's bindings name the dataset the elected bindings name when that dataset
    publishes the candidate's shape, so its family and its source slot follow it."""
    captured: list[list[SlotBindingDraft]] = []

    def _capture(shape, bindings, entities, **_kwargs):
        captured.append(list(bindings))
        return NoDataDraft(reason="no_observations")

    monkeypatch.setattr(candidates_module, "materialise", _capture)
    imputed = dataclasses.replace(_imputed_family_shape(), sv_arc_facts=_imputed_arcs())
    population = dataclasses.replace(_population_shape(), sv_arc_facts=_aid_shape().sv_arc_facts)

    materialise_alternates(
        [imputed, population],
        elected_bindings=[source_binding(IMPUTED_PROVENANCE)],
        elected_has_data=False,
        places=[empty_place_context(), empty_place_context()],
        named_place_dcids=(),
        graph=FakeGraph(nodes={}, detect={}, resolve={}, obs={}),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    assert [bound_source(bindings) for bindings in captured] == [IMPUTED_PROVENANCE, None]


_SOURCE_KEY = SlotKeyDraft(axis="source", property_dcid=None, label="data source")
_WHERE_KEY = SlotKeyDraft(axis="where", property_dcid=None, label="place")
_WHEN_KEY = SlotKeyDraft(axis="when", property_dcid=None, label="time window")


def test_a_source_binding_reaches_the_source_slot_alone():
    purpose = SlotBindingDraft(
        axis="how", property_dcid="crsPurpose", kind="value", value_dcids=["CRSPurposeCode110"]
    )
    place = SlotBindingDraft(
        axis="where", property_dcid=None, kind="value", value_dcids=["country/KEN"]
    )
    source = source_binding(CRS_PROVENANCE)

    pairs = pair_slot_drafts(
        (_PURPOSE_SLOT, _AID_TYPE_SLOT, _WHERE_KEY, _WHEN_KEY, _SOURCE_KEY),
        [purpose, place, source],
    )

    assert [draft for _key, draft in pairs] == [purpose, None, place, None, source]


def test_build_slot_binds_the_source_slot_to_the_dataset_its_draft_names():
    slot = build_slot(_SOURCE_KEY, source_binding(CRS_PROVENANCE), [])

    assert slot.binding.kind == "value"
    assert slot.binding.value.value_kind == "source"
    assert slot.binding.value.ref == GraphRef(
        dcid=CRS_PROVENANCE, label=_INDEX.datasets[CRS_PROVENANCE].label
    )
    assert _facets_member([slot]) == {"provenance": {"state": "value", "ref": CRS_PROVENANCE}}


@pytest.mark.parametrize(
    "draft",
    [
        None,
        SlotBindingDraft(axis="source", property_dcid=None, kind="unbound"),
        SlotBindingDraft(
            axis="source",
            property_dcid=None,
            kind="set",
            value_dcids=[CRS_PROVENANCE, IMPUTED_PROVENANCE],
        ),
    ],
    ids=["no-draft", "unbound", "set"],
)
def test_build_slot_leaves_the_source_slot_unbound_without_one_named_dataset(draft):
    assert build_slot(_SOURCE_KEY, draft, []).binding == BindingUnbound()


def _provenance(dcid: str) -> ProvenanceRef:
    return ProvenanceRef(dcid=dcid, label=f"catalogue name of {dcid}")


@pytest.mark.parametrize(
    ("identity", "elected"),
    [
        (education_aid_identity(), CRS_PROVENANCE),
        (imputed_education_aid_identity(), IMPUTED_PROVENANCE),
        (imputed_education_aid_identity(source=IMPUTED_PROVENANCE), IMPUTED_PROVENANCE),
    ],
    ids=["crs", "imputed", "imputed-source-bound"],
)
def test_the_sources_of_an_index_family_are_its_elected_dataset_alone(identity: dict, elected: str):
    observed = [_provenance(CRS_PROVENANCE), _provenance(IMPUTED_PROVENANCE)]

    sources = elected_sources(observed, BaseIdentity.model_validate(identity))

    assert sources == [ProvenanceRef(dcid=elected, label=_INDEX.datasets[elected].label)]


def test_an_answer_that_served_nothing_from_its_elected_dataset_names_no_source():
    """The read serves the elected dataset's facets alone, so a source list without it
    names no source the answer read."""
    sources = elected_sources(
        [_provenance(IMPUTED_PROVENANCE)], BaseIdentity.model_validate(education_aid_identity())
    )

    assert sources == []


def test_sources_pass_through_when_none_is_observed_or_the_index_elects_no_dataset():
    population = BaseIdentity.model_validate(
        {
            "identity_version": "spec1",
            "shape": {
                "kind": "statvar_schema",
                "population_type": "Person",
                "measured_property": "count",
                "stat_type": "measuredValue",
            },
            "slots": [],
            "entities": {"observationAbout": ["country/KEN"]},
            "requested_window": {"kind": "all"},
        }
    )
    observed = [_provenance("provenance/A"), _provenance("provenance/B")]

    assert elected_sources(observed, population) == observed
    assert elected_sources([], BaseIdentity.model_validate(education_aid_identity())) == []


# ---------------------------------------------------------------------------
# The bind prompt
# ---------------------------------------------------------------------------

_PURPOSE_SLOT = SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="sector")
_AID_TYPE_SLOT = SlotKeyDraft(axis="how", property_dcid=AID_TYPE, label="aid type")


def _constraint_arcs(constraints: dict[str, str]) -> dict:
    return {
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


def _imputed_arcs() -> dict[str, dict]:
    """The arcs of the imputed table's education members, from the packaged index."""
    education = _INDEX.descendants("CRSPurposeCode110")
    return {
        row.dcid: _constraint_arcs(dict(row.constraint_properties))
        for row in _INDEX.statvars
        if IMPUTED_PROVENANCE in row.provenances
        and row.constraint_properties.get("crsPurpose") in education
    }


def _aid_shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="aid",
        label="aid",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(_PURPOSE_SLOT, _AID_TYPE_SLOT),
        slot_taxonomy={
            "how:crsPurpose": ["CRSPurposeCode110"],
            f"how:{AID_TYPE}": [IMPUTED_MULTILATERAL],
        },
        sv_arc_facts={
            "sv_plain": _constraint_arcs({"crsPurpose": "CRSPurposeCode110"}),
            "sv_imputed": _constraint_arcs(
                {"crsPurpose": "CRSPurposeCode110", AID_TYPE: IMPUTED_MULTILATERAL}
            ),
        },
        representative_score=0.8,
    )
