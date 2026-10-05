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

from code_index.members import canonical_identity, family_for_five_tuple, invariant_bindings
from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import CodeSystemDeclaration, DatasetDeclaration, IndexStamp, StatVarRow
from search.engine.bind import SlotBindingDraft, _build_universal_bind_prompt
from search.engine.dataset_mention import DatasetMention, dataset_mentions
from search.engine.qualifier_match import CodedQualifier
from search.engine.region import candidates as candidates_module
from search.engine.region import phase_rank_bind as phase_rank_bind_module
from search.engine.region.named_source import (
    with_mention,
    with_named_source,
    without_dataset_drafts,
)
from search.engine.region.phase_rank_bind import _dataset_candidates, _read_mention
from search.engine.region.place_context import empty_place_context
from search.engine.retrieval import bound_values as bound_values_module
from search.engine.retrieval.bound_values import (
    bind_shared_values,
    bound_source,
    canonical_pin_bindings,
    five_tuple_member_facts,
    shape_invariants,
    shape_pins,
    shape_query,
    source_binding,
    synthetic_shape_for_identity,
)
from search.engine.retrieval.retrieve import NoDataDraft
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_build import build_slot, elected_sources, pair_slot_drafts
from search.engine.spec_id import _facets_member, build_spec_identity, mint_identity
from search.heading import pins as heading_pins_module
from search.heading.pins import identity_pinned_properties
from search.models import (
    BaseIdentity,
    BindingUnbound,
    GraphRef,
    IdentityShapeSchema,
    ProvenanceRef,
)
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
    assert shape_query(_aid_family_shape(), asked, ()).source == CRS_PROVENANCE


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

    # Candidacy stamps the datasets the part may answer from and the one it names, which
    # enter equality.
    unstamped = [
        dataclasses.replace(shape, datasets=None, named_dataset=None) for shape in candidates
    ]
    assert [name for name, shape in shapes.items() if shape in unstamped] == kept


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
    assert shape_query(_aid_family_shape(), bindings, ()).source == source


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


def test_an_alternate_whose_bindings_hold_the_pins_mints_the_canonical_spec_id(monkeypatch):
    """An alternate takes its representative member's constraint values as bindings, so an
    imputed alternate's member holds the aid type and the flow its table declares. Its
    bindings keep both (`canonical_pin_bindings`), so its slots bind them, and its identity
    (`build_spec_identity`) hashes to the canonical spec_id, which the same identity posted
    with both unbound reaches too."""
    captured: list[list[SlotBindingDraft]] = []

    def _capture(shape, bindings, entities, **_kwargs):
        captured.append(list(bindings))
        return NoDataDraft(reason="no_observations")

    monkeypatch.setattr(candidates_module, "materialise", _capture)
    imputed = dataclasses.replace(_imputed_family_shape(), sv_arc_facts=_imputed_arcs())
    materialise_alternates(
        [imputed],
        elected_bindings=[source_binding(IMPUTED_PROVENANCE)],
        elected_has_data=False,
        places=[empty_place_context()],
        named_place_dcids=(),
        graph=FakeGraph(nodes={}, detect={}, resolve={}, obs={}),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )
    (bindings,) = captured
    member = next(iter(_imputed_arcs().values()))
    pins = {AID_TYPE: IMPUTED_MULTILATERAL, "crsTypeOfFlow": "CRSTypeOfFlow10"}
    assert {prop: member[prop]["nodes"][0]["dcid"] for prop in pins} == pins
    assert {
        b.property_dcid: b.value_dcids[0]
        for b in bindings
        if b.kind == "value" and b.property_dcid in pins
    } == pins

    slots = [
        build_slot(
            key,
            draft,
            [GraphRef(dcid=value, label=value) for value in draft.value_dcids] if draft else [],
            property_ref=GraphRef(dcid=key.property_dcid, label=key.property_dcid)
            if key.property_dcid is not None
            else None,
        )
        for key, draft in pair_slot_drafts((*imputed.slot_keys, _SOURCE_KEY), bindings)
    ]
    spec_id, identity = build_spec_identity(
        {
            "kind": "statvar_schema",
            "population_type": imputed.pop_type_dcid,
            "measured_property": imputed.meas_prop_dcid,
            "stat_type": imputed.stat_type_dcid,
            "measurement_qualifier": imputed.meas_qual_dcid,
        },
        slots,
    )

    document = identity.model_dump(mode="json", exclude_none=True)
    assert {
        slot["property"]: slot["binding"] for slot in document["slots"] if slot["property"] in pins
    } == {prop: {"state": "value", "ref": value} for prop, value in pins.items()}
    assert canonical_identity(_INDEX, document) == document
    for slot in document["slots"]:
        if slot["property"] in pins:
            slot["binding"] = {"state": "unbound"}
    assert mint_identity(document)[0] != spec_id
    assert mint_identity(canonical_identity(_INDEX, document))[0] == spec_id


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


# ---------------------------------------------------------------------------
# Several datasets publishing one five-tuple
# ---------------------------------------------------------------------------

_WIDE = "provenance/Wide"
_OTHER = "provenance/Other"
_NARROW = "provenance/Narrow"
_ONLY = "provenance/Only"

_SHARED_FIVE_TUPLE = ("TestFlow", "amount")
_OWN_FIVE_TUPLE = ("TestFlow", "lineAmount")


def _publisher_row(
    dcid: str, constraints: dict[str, str], provenance: str, *, five_tuple=_SHARED_FIVE_TUPLE
) -> StatVarRow:
    population_type, measured_property = five_tuple
    return StatVarRow(
        dcid=dcid,
        population_type=population_type,
        measured_property=measured_property,
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=(),
        constraint_properties=constraints,
        provenances=(provenance,),
    )


def _publisher(
    provenance: str, *, rank: int, opt_in: bool, invariants: dict[str, str]
) -> DatasetDeclaration:
    return DatasetDeclaration(
        provenance=provenance,
        label=provenance,
        synonyms=(),
        opt_in=opt_in,
        absent_observation="zero",
        rank=rank,
        invariants=invariants,
    )


def _code_system(code_system: str) -> CodeSystemDeclaration:
    return CodeSystemDeclaration(
        code_system=code_system,
        hierarchical=False,
        disjoint_from_children=False,
        summable=True,
        absent_reading="not_declared",
        plain_name=code_system,
    )


def _two_publisher_index() -> CodeIndex:
    """The shared five-tuple is published by `_WIDE` and `_OTHER`, which serve by default,
    and by the opt-in `_NARROW`, which alone carries `line`. Every row holds `basis` G,
    and each dataset declares it and its own `flow` as invariants. The opt-in `_ONLY` alone
    publishes its own five-tuple."""
    rows = (
        _publisher_row("wide_a", {"basis": "G", "flow": "F1", "sector": "A"}, _WIDE),
        _publisher_row("wide_b", {"basis": "G", "flow": "F1", "sector": "B"}, _WIDE),
        _publisher_row("other_b", {"basis": "G", "flow": "F2", "sector": "B"}, _OTHER),
        _publisher_row("other_d", {"basis": "G", "flow": "F2", "sector": "D"}, _OTHER),
        _publisher_row(
            "narrow_a", {"basis": "G", "flow": "F1", "line": "L1", "sector": "A"}, _NARROW
        ),
        _publisher_row(
            "narrow_c", {"basis": "G", "flow": "F1", "line": "L1", "sector": "C"}, _NARROW
        ),
        _publisher_row("only_a", {"sector": "A"}, _ONLY, five_tuple=_OWN_FIVE_TUPLE),
    )
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=tuple(_code_system(prop) for prop in ("basis", "flow", "line", "sector")),
        codes=(),
        statvars=rows,
        datasets=(
            _publisher(_WIDE, rank=0, opt_in=False, invariants={"basis": "G", "flow": "F1"}),
            _publisher(_OTHER, rank=1, opt_in=False, invariants={"basis": "G", "flow": "F2"}),
            _publisher(
                _NARROW,
                rank=2,
                opt_in=True,
                invariants={"basis": "G", "flow": "F1", "line": "L1"},
            ),
            _publisher(_ONLY, rank=3, opt_in=True, invariants={"sector": "A"}),
        ),
    )


@pytest.fixture
def two_publishers(monkeypatch) -> CodeIndex:
    """`_two_publisher_index` in place of the packaged index wherever Search reads it."""
    index = _two_publisher_index()
    monkeypatch.setattr(bound_values_module, "get_index", lambda: index)
    monkeypatch.setattr(phase_rank_bind_module, "get_index", lambda: index)
    monkeypatch.setattr(heading_pins_module, "get_index", lambda: index)
    return index


def _publisher_shape(
    five_tuple: tuple[str, str] = _SHARED_FIVE_TUPLE, shape_id: str = "shared"
) -> ShapeDraft:
    """A shape as `derive_shapes` builds it on `five_tuple`, with the slot keys and taxonomy
    of every row of the five-tuple, whatever dataset publishes it."""
    population_type, measured_property = five_tuple
    rows = [
        row
        for row in _two_publisher_index().statvars
        if (row.population_type, row.measured_property) == five_tuple
    ]
    props = sorted({prop for row in rows for prop in row.constraint_properties})
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=population_type,
        meas_prop_dcid=measured_property,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=tuple(SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in props),
        slot_taxonomy={
            f"how:{prop}": list(
                dict.fromkeys(
                    row.constraint_properties[prop]
                    for row in rows
                    if prop in row.constraint_properties
                )
            )
            for prop in props
        },
        slot_taxonomy_index_sourced=True,
        sv_arc_facts_complete=False,
    )


def _source_mention(source: str | None) -> DatasetMention | None:
    if source is None:
        return None
    return DatasetMention(
        provenance=source, start=0, end=len(source), text=source, name=source, measure_words=()
    )


def _candidate(source: str | None) -> ShapeDraft:
    (shape,) = _dataset_candidates([_publisher_shape()], _source_mention(source))
    return shape


def _sector(code: str) -> SlotBindingDraft:
    return _value("sector", code)


def _value(prop: str, code: str) -> SlotBindingDraft:
    return SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[code])


@pytest.mark.parametrize(
    ("source", "kept", "datasets"),
    [
        (None, ["shared", "unindexed"], [(_WIDE, _OTHER), ()]),
        (_WIDE, ["shared"], [(_WIDE,)]),
        (_NARROW, ["shared"], [(_NARROW,)]),
        (_ONLY, ["own"], [(_ONLY,)]),
    ],
    ids=["unnamed", "named-default", "named-opt-in-on-shared", "named-opt-in-alone"],
)
def test_a_part_keeps_the_shapes_its_datasets_publish_on_a_shared_five_tuple(
    two_publishers, source, kept, datasets
):
    """An unnamed part keeps a shape with a publisher that is not opt-in, and one the index
    does not carry. A named part keeps the shapes its dataset publishes, beside others or
    alone. Each kept shape carries the datasets the part may answer it from."""
    shapes = [
        _publisher_shape(),
        _publisher_shape(_OWN_FIVE_TUPLE, "own"),
        dataclasses.replace(_publisher_shape(), shape_id="unindexed", pop_type_dcid="Person"),
    ]

    candidates = _dataset_candidates(shapes, _source_mention(source))

    assert [shape.shape_id for shape in candidates] == kept
    assert [shape.datasets for shape in candidates] == datasets


@pytest.mark.parametrize(
    ("source", "sectors", "lines", "members"),
    [
        (None, ["A", "B", "D"], None, {"wide_a", "wide_b", "other_b", "other_d"}),
        (_WIDE, ["A", "B"], None, {"wide_a", "wide_b"}),
        (_NARROW, ["A", "C"], ["L1"], {"narrow_a", "narrow_c"}),
    ],
    ids=["unnamed", "named-default", "named-opt-in"],
)
def test_the_bind_model_sees_the_codes_of_every_dataset_the_part_may_answer_from(
    two_publishers, source, sectors, lines, members
):
    """The taxonomy and the pre-bind member facts hold the rows of the named dataset, else
    of every eligible publisher. The slot keys stay the whole five-tuple's."""
    shape = _candidate(source)
    taxonomy = shape.slot_taxonomy or {}

    assert taxonomy["how:sector"] == sectors
    assert taxonomy.get("how:line") == lines
    assert shape.slot_keys == _publisher_shape().slot_keys
    assert set(five_tuple_member_facts(shape)[0]) == members


def test_a_named_and_an_unnamed_part_on_one_five_tuple_keep_their_own_vocabularies(
    two_publishers,
):
    """One derived shape kept by an unnamed part and by a part naming the opt-in dataset is
    two shapes, so a mapping keyed on them holds both vocabularies and both pin sets."""
    derived = _publisher_shape()
    unnamed = _dataset_candidates([derived], None)[0]
    named = _dataset_candidates([derived], _source_mention(_NARROW))[0]

    by_shape = {unnamed: unnamed.slot_taxonomy, named: named.slot_taxonomy}

    assert unnamed != named
    assert by_shape[unnamed] == {
        "how:basis": ["G"],
        "how:flow": ["F1", "F2"],
        "how:sector": ["A", "B", "D"],
    }
    assert by_shape[named] == {
        "how:basis": ["G"],
        "how:flow": ["F1"],
        "how:line": ["L1"],
        "how:sector": ["A", "C"],
    }
    assert shape_pins(unnamed) != shape_pins(named)


@pytest.mark.parametrize(
    ("source", "pins"),
    [
        (None, {}),
        (_WIDE, {"basis": "G", "flow": "F1"}),
        (_NARROW, {"basis": "G", "flow": "F1", "line": "L1"}),
    ],
    ids=["unnamed", "named-default", "named-opt-in"],
)
def test_the_bind_prompt_withholds_the_pins_every_dataset_of_the_part_holds(
    two_publishers, source, pins
):
    """`_WIDE` and `_OTHER` pin `flow` to different values, so an unnamed part withholds
    only `basis`, and a named part withholds every pin of its dataset. The election implies a
    withheld pin, so nothing binds it."""
    shape = _candidate(source)

    prompt = _build_universal_bind_prompt("aid", [shape])

    assert shape_invariants(shape) == pins
    assert {
        prop for prop in ("basis", "flow", "line", "sector") if f"property_dcid={prop} " in prompt
    } == {"basis", "flow", "line", "sector"} - pins.keys()
    assert bind_shared_values(shape, []) == ([], [])


@pytest.mark.parametrize(
    ("source", "sector", "elected", "pins"),
    [
        (None, "B", _WIDE, {"basis": "G", "flow": "F1"}),
        (None, "D", _OTHER, {"basis": "G", "flow": "F2"}),
        (None, "C", None, {}),
        (_NARROW, "A", _NARROW, {"basis": "G", "flow": "F1", "line": "L1"}),
    ],
    ids=["rank", "only-other", "only-opt-in", "named"],
)
@pytest.mark.parametrize("model_binds", [True, False], ids=["drafted", "open"])
def test_the_identity_binds_the_invariants_only_of_the_dataset_its_part_names(
    two_publishers, source, sector, elected, pins, model_binds
):
    """A draft that binds the elected dataset's invariants keeps them. One that leaves them
    open keeps them open when the part names no dataset, since a later export may elect
    another, and binds each when the part names the dataset. The query the other doors read
    elects the same dataset and family as the one binding every pin."""
    shape = _candidate(source)
    named = [source_binding(source)] if source else []
    pinned = [_value(prop, code) for prop, code in pins.items()]
    drafted = [_sector(sector), *(pinned if model_binds else []), *named]
    role_keys = {"observationAbout"}

    bindings = canonical_pin_bindings(shape, drafted, role_keys)
    query = shape_query(shape, bindings, role_keys)

    assert {
        b.property_dcid: b.value_dcids[0]
        for b in bindings
        if b.kind == "value" and b.property_dcid is not None
    } == {"sector": sector, **(pins if model_binds or source else {})}
    assert canonical_pin_bindings(shape, bindings, role_keys) == bindings
    assert family_for_five_tuple(two_publishers, query).provenance == elected
    assert family_for_five_tuple(two_publishers, query) == family_for_five_tuple(
        two_publishers, shape_query(shape, [_sector(sector), *pinned, *named], role_keys)
    )
    assert invariant_bindings(two_publishers, query) == {}


@pytest.mark.parametrize(
    ("slots", "source", "pinned"),
    [
        ({"sector": "D"}, None, {"basis", "flow"}),
        ({"sector": "A"}, _NARROW, {"basis", "flow", "line"}),
        ({"sector": "C"}, None, {"basis"}),
    ],
    ids=["elected-other", "named", "none-elected"],
)
def test_an_identity_states_the_pins_of_its_elected_dataset(two_publishers, slots, source, pinned):
    """With no elected family, the pins are those every eligible dataset holds alike."""
    identity: dict = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "TestFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
        },
        "slots": [
            {
                "property": prop,
                "binding": (
                    {"state": "value", "ref": slots[prop]}
                    if prop in slots
                    else {"state": "unbound"}
                ),
            }
            for prop in ("basis", "flow", "line", "sector")
        ],
        "entities": {"observationAbout": ["country/KEN"]},
        "requested_window": {"kind": "all"},
    }
    if source is not None:
        identity["facets"] = {"provenance": {"state": "value", "ref": source}}

    assert identity_pinned_properties(mint_identity(identity)[1]) == pinned


@pytest.mark.parametrize(
    ("source", "named"), [(_NARROW, True), (_OTHER, True), (_ONLY, False), (None, False)]
)
def test_with_named_source_names_any_dataset_publishing_a_shared_five_tuple(
    two_publishers, source, named
):
    assert with_named_source([], _publisher_shape(), source) == (
        [source_binding(source)] if named and source else []
    )
