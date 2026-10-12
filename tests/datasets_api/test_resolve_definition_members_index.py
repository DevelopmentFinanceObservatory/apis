"""An indexed five-tuple resolves members from `CodeIndex` without graph reads or a
member-count ceiling. Tests compare it with the walked result and cover tuples absent from
the index.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from code_index import default_index_path, load_index
from code_index.query import CodeIndex
from code_index.types import DatasetDeclaration, IndexStamp, StatVarRow
from datasets_api.adapter import (
    MemberReadBounds,
    ResolvedMembers,
    resolve_definition_members,
    to_materialisation_request,
)
from datasets_api.codes_materialise import (
    MemberNotResolvable,
    SpecNotExportable,
)
from datasets_api.kinds.spec_id import _resolve_members
from datasets_api.mint_models import Scope
from datasets_api.shape import StatVarShape
from materialise.errors import FacetNotAvailableError
from materialise.facets import ElectedDataset
from materialise.graph import Facet
from materialise.group_facets import select_group_facets
from statvar_members import MemberEnumerator
from tests.code_index.aid_identity import (
    AID_TYPE,
    CRS_MEMBER_COUNT,
    CRS_MEMBERS_WITHOUT_MODALITY,
    CRS_PROVENANCE,
    EDUCATION_AID_SPEC_ID,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_MULTILATERAL,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
)

from ._graph_doubles import FakeMintGraphReader
from ._member_bounds import mint_bounds

_DAC1_PROVENANCE = "provenance/OECDDAC1"
_IDS_PROVENANCE = "provenance/WorldBankInternationalDebtStatistics"

_SPEC_ID = "spec_test"

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def _empty_code_index() -> CodeIndex:
    return CodeIndex(stamp=_STAMP, declarations=(), codes=(), statvars=())


# ---------------------------------------------------------------------------
# Index-backed resolution accepts 701 members without graph access.
# ---------------------------------------------------------------------------

_OVER_OLD_CAP_COUNT = 701


def _over_cap_index(count: int) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=tuple(
            StatVarRow(
                dcid=f"OverCapStatVar_{i}",
                population_type="TestPop",
                measured_property="testAmount",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("recipient",),
                constraint_properties={"scheme": f"Code{i}"},
                provenances=("provenance/Test",),
            )
            for i in range(count)
        ),
    )


def _over_cap_identity() -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "TestPop",
            "measured_property": "testAmount",
            "stat_type": "measuredValue",
        },
        "slots": [{"property": "scheme", "binding": {"state": "unbound"}}],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


def test_a_count_past_the_old_member_cap_resolves_without_refusing():
    code_index = _over_cap_index(_OVER_OLD_CAP_COUNT)
    graph = FakeMintGraphReader(arcs={})

    shapes = resolve_definition_members(
        _over_cap_identity(),
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=code_index,
        bounds=mint_bounds(),
    ).shapes

    assert len(shapes) == _OVER_OLD_CAP_COUNT
    # A zero call count proves the index path skipped graph enumeration.
    assert sum(graph.calls.values()) == 0


# ---------------------------------------------------------------------------
# Under cap: the index-derived result equals the walked result, and a
# five-tuple the index does not carry keeps the walk.
# ---------------------------------------------------------------------------


def _person_family_index() -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=(
            StatVarRow(
                dcid="Person",
                population_type="Person",
                measured_property="count",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=(),
                constraint_properties={},
                provenances=("provenance/Test",),
            ),
            StatVarRow(
                dcid="Person_Male",
                population_type="Person",
                measured_property="count",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=(),
                constraint_properties={"gender": "Male"},
                provenances=("provenance/Test",),
            ),
            StatVarRow(
                dcid="Person_Female",
                population_type="Person",
                measured_property="count",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=(),
                constraint_properties={"gender": "Female"},
                provenances=("provenance/Test",),
            ),
        ),
    )


def _person_family_graph(*, extra_genders: tuple[str, ...] = ()) -> FakeMintGraphReader:
    def _arcs(gender: str | None) -> dict[str, tuple[str, ...]]:
        return {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "constraintProperties": ("gender",) if gender else (),
            "observationProperties": (),
            **({"gender": (gender,)} if gender else {}),
        }

    return FakeMintGraphReader(
        arcs={
            "Person": _arcs(None),
            "Person_Male": _arcs("Male"),
            "Person_Female": _arcs("Female"),
            **{f"Person_{gender}": _arcs(gender) for gender in extra_genders},
        }
    )


def _gender_family_identity() -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "Person",
            "measured_property": "count",
            "stat_type": "measuredValue",
        },
        "slots": [{"property": "gender", "binding": {"state": "unbound"}}],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


def test_under_cap_indexed_spec_derives_members_equal_to_the_walk():
    identity = _gender_family_identity()

    walked_graph = _person_family_graph()
    walked = resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=walked_graph,
        enumerator=MemberEnumerator(),
        # The empty index sends this five-tuple through the graph walk.
        code_index=_empty_code_index(),
        bounds=mint_bounds(),
    )
    assert sum(walked_graph.calls.values()) > 0

    indexed_graph = FakeMintGraphReader(arcs={})
    indexed = resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=indexed_graph,
        enumerator=MemberEnumerator(),
        code_index=_person_family_index(),
        bounds=mint_bounds(),
    )
    assert sum(indexed_graph.calls.values()) == 0

    # An indexed shape carries the dataset elected for its family, which a walk does not read.
    assert frozenset(
        dataclasses.replace(shape, dataset=None) for shape in indexed.shapes
    ) == frozenset(walked.shapes)
    assert {shape.dataset for shape in indexed.shapes} == {
        ElectedDataset(provenance="provenance/Test", member_provenances=("provenance/Test",))
    }
    assert {shape.stat_var for shape in indexed.shapes} == {"dcid:Person"}
    # Both paths report the family before reduction, the members the root dominates included.
    assert indexed.family == walked.family == {"Person", "Person_Male", "Person_Female"}


# ---------------------------------------------------------------------------
# Indexed resolution refuses rows declaring `observationAbout` as a role.
# ---------------------------------------------------------------------------


def _degenerate_observation_properties_index() -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=(
            StatVarRow(
                dcid="DegenerateStatVar",
                population_type="TestPop",
                measured_property="testAmount",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("observationAbout",),
                constraint_properties={},
                provenances=("provenance/Test",),
            ),
        ),
    )


def _degenerate_identity() -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "TestPop",
            "measured_property": "testAmount",
            "stat_type": "measuredValue",
        },
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


def test_indexed_spec_refuses_a_row_naming_observation_about_in_observation_properties():
    code_index = _degenerate_observation_properties_index()
    graph = FakeMintGraphReader(arcs={})

    with pytest.raises(MemberNotResolvable) as excinfo:
        resolve_definition_members(
            _degenerate_identity(),
            spec_id=_SPEC_ID,
            graph=graph,
            enumerator=MemberEnumerator(),
            code_index=code_index,
            bounds=mint_bounds(),
        )

    assert excinfo.value.members == (_SPEC_ID,)
    assert sum(graph.calls.values()) == 0


def test_non_indexed_five_tuple_still_walks():
    graph = _person_family_graph()

    shapes = resolve_definition_members(
        _gender_family_identity(),
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=_empty_code_index(),
        bounds=mint_bounds(),
    ).shapes

    assert sum(graph.calls.values()) > 0
    assert {shape.stat_var for shape in shapes} == {"dcid:Person"}


def test_a_listed_family_resolves_the_walks_members_without_reading_the_graph():
    """A family the index lists outside its code systems (`CodeIndex.listed_statvars`) holds
    the walk's candidates, so the mint resolves the walk's members and family from it with no
    graph read."""
    identity = _gender_family_identity()
    walked = resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=_person_family_graph(),
        enumerator=MemberEnumerator(),
        code_index=_empty_code_index(),
        bounds=mint_bounds(),
    )
    listed_index = CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=(),
        listed_statvars=_person_family_index().statvars,
    )
    graph = FakeMintGraphReader(arcs={})

    listed = resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=listed_index,
        bounds=mint_bounds(),
    )

    assert sum(graph.calls.values()) == 0
    assert listed == walked


# ---------------------------------------------------------------------------
# Index-backed resolution marks the resulting requests as index-sourced.
# ---------------------------------------------------------------------------


def test_the_mint_style_resolve_then_adapt_chain_sets_index_sourced_on_the_request():
    identity = _gender_family_identity()
    code_index = _person_family_index()
    graph = FakeMintGraphReader(arcs={})

    shapes = resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=code_index,
        bounds=mint_bounds(),
    ).shapes
    requests = [
        to_materialisation_request(identity, shape=shape, code_index=code_index) for shape in shapes
    ]

    assert requests
    assert all(request.index_sourced for request in requests)


# ---------------------------------------------------------------------------
# The education-aid identity against the packaged index: a family is one
# dataset's members.
# ---------------------------------------------------------------------------


def _education_aid_members(identity: dict, code_index: CodeIndex) -> ResolvedMembers:
    graph = FakeMintGraphReader(arcs={})
    resolved = resolve_definition_members(
        identity,
        spec_id=EDUCATION_AID_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=code_index,
        bounds=mint_bounds(),
    )
    assert sum(graph.calls.values()) == 0
    return resolved


def _provenances(shapes: tuple[StatVarShape, ...], code_index: CodeIndex) -> set[tuple[str, ...]]:
    members = {shape.stat_var.removeprefix("dcid:") for shape in shapes}
    return {row.provenances for row in code_index.statvars if row.dcid in members}


def test_the_plain_aid_identity_resolves_the_crs_members():
    code_index = load_index(default_index_path())

    resolved = _education_aid_members(education_aid_identity(), code_index)

    assert len(resolved.shapes) == CRS_MEMBER_COUNT
    assert len(resolved.family) == CRS_MEMBER_COUNT
    assert _provenances(resolved.shapes, code_index) == {(CRS_PROVENANCE,)}
    assert not any(AID_TYPE in dict(shape.slots) for shape in resolved.shapes)


def test_the_plain_aid_identity_keeps_its_members_without_a_modality():
    code_index = load_index(default_index_path())

    shapes = _education_aid_members(education_aid_identity(), code_index).shapes

    without_modality = [s for s in shapes if "crsCooperationModality" not in dict(s.slots)]
    assert len(without_modality) == CRS_MEMBERS_WITHOUT_MODALITY


def test_the_imputed_identity_resolves_the_imputed_members():
    code_index = load_index(default_index_path())

    shapes = _education_aid_members(imputed_education_aid_identity(), code_index).shapes

    assert len(shapes) == IMPUTED_MEMBER_COUNT
    assert {dict(shape.slots)[AID_TYPE] for shape in shapes} == {IMPUTED_MULTILATERAL}
    assert _provenances(shapes, code_index) == {(IMPUTED_PROVENANCE,)}


def test_naming_the_imputed_provenance_on_its_own_five_tuple_changes_no_member():
    code_index = load_index(default_index_path())

    named = _education_aid_members(
        imputed_education_aid_identity(source=IMPUTED_PROVENANCE), code_index
    )
    plain = _education_aid_members(imputed_education_aid_identity(), code_index)

    assert len(named.shapes) == IMPUTED_MEMBER_COUNT
    assert named == plain


def test_a_named_identity_resolves_with_an_unbound_slot_its_dataset_does_not_carry():
    """Search states a property the graph members carry and the index rows do not as an
    unbound slot. The identity naming its dataset mints the dataset's members with it."""
    code_index = load_index(default_index_path())
    identity = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    identity["slots"].append({"property": "crsCooperationModality", "binding": _UNBOUND})

    resolved = _education_aid_members(identity, code_index)

    assert len(resolved.shapes) == IMPUTED_MEMBER_COUNT
    assert _provenances(resolved.shapes, code_index) == {(IMPUTED_PROVENANCE,)}


def test_a_stored_identity_binding_the_imputed_aid_type_on_amount_refuses():
    """An identity that binds the imputed aid type on the `amount` five-tuple enumerates no
    member and refuses. The mint does not translate it to `attributedAmount`."""
    code_index = load_index(default_index_path())

    with pytest.raises(SpecNotExportable):
        _education_aid_members(education_aid_identity(aid_type=IMPUTED_MULTILATERAL), code_index)


def _member_arcs(row: StatVarRow) -> dict[str, tuple[str, ...]]:
    """The graph arcs of the StatVar an index row describes."""
    measure = {
        "populationType": row.population_type,
        "measuredProperty": row.measured_property,
        "statType": row.stat_type,
        "measurementQualifier": row.measurement_qualifier,
    }
    return {
        "typeOf": ("StatisticalVariable",),
        **{prop: (value,) for prop, value in measure.items() if value is not None},
        "constraintProperties": tuple(row.constraint_properties),
        "observationProperties": tuple(row.observation_properties or ()),
        **{prop: (value,) for prop, value in row.constraint_properties.items()},
    }


def _imputed_member_graph(code_index: CodeIndex) -> FakeMintGraphReader:
    """A graph holding every imputed index row as a StatVar with its own arcs."""
    return FakeMintGraphReader(
        arcs={
            row.dcid: _member_arcs(row)
            for row in code_index.statvars
            if IMPUTED_PROVENANCE in row.provenances
        }
    )


def _aid_type_unbound(identity: dict) -> dict:
    """`identity` with its aid-type slot unbound."""
    slot = next(slot for slot in identity["slots"] if slot["property"] == AID_TYPE)
    slot["binding"] = {"state": "unbound"}
    return identity


def _purpose_no_row_holds(identity: dict) -> dict:
    """`identity` with its purpose bound to a code no index row holds."""
    slot = next(slot for slot in identity["slots"] if slot["property"] == "crsPurpose")
    slot["binding"] = {"state": "value", "ref": "CRSPurposeCodeNoRowHolds"}
    return identity


def _only_flow_type_bound(identity: dict) -> dict:
    """`identity` with every slot unbound but `financialFlowType`, which the rows of every
    dataset publishing its five-tuple carry."""
    for slot in identity["slots"]:
        if slot["property"] != "financialFlowType":
            slot["binding"] = {"state": "unbound"}
    return identity


def test_an_identity_that_leaves_a_pinned_slot_unbound_resolves_the_bound_identity_s_members():
    """Every imputed row holds one aid type, so the election implies it. The identity that
    leaves it unbound resolves, from the index and without a graph read, the members of the
    identity that binds it."""
    code_index = load_index(default_index_path())
    identity = _aid_type_unbound(imputed_education_aid_identity())
    graph = _imputed_member_graph(code_index)

    unbound = resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=code_index,
        bounds=mint_bounds(),
    )

    assert sum(graph.calls.values()) == 0
    assert unbound == _education_aid_members(imputed_education_aid_identity(), code_index)


def test_an_identity_that_omits_a_pinned_slot_refuses():
    """The identity that omits the aid type covers no index row, so the graph walks, and
    every imputed member asserts the aid type the identity does not key, so the walk
    confirms none."""
    code_index = load_index(default_index_path())
    identity = imputed_education_aid_identity()
    identity["slots"] = [slot for slot in identity["slots"] if slot["property"] != AID_TYPE]
    graph = _imputed_member_graph(code_index)

    with pytest.raises(SpecNotExportable):
        resolve_definition_members(
            identity,
            spec_id=_SPEC_ID,
            graph=graph,
            enumerator=MemberEnumerator(),
            code_index=code_index,
            bounds=mint_bounds(),
        )

    assert graph.calls["node_subjects"] > 0


@pytest.mark.parametrize(
    ("identity", "named", "publisher"),
    [
        (
            education_aid_identity(source=IMPUTED_PROVENANCE),
            IMPUTED_PROVENANCE,
            CRS_PROVENANCE,
        ),
        (
            imputed_education_aid_identity(source=CRS_PROVENANCE),
            CRS_PROVENANCE,
            IMPUTED_PROVENANCE,
        ),
        (
            _aid_type_unbound(imputed_education_aid_identity(source=CRS_PROVENANCE)),
            CRS_PROVENANCE,
            IMPUTED_PROVENANCE,
        ),
        (
            _purpose_no_row_holds(education_aid_identity(source=IMPUTED_PROVENANCE)),
            IMPUTED_PROVENANCE,
            CRS_PROVENANCE,
        ),
        (
            _only_flow_type_bound(education_aid_identity(source=_DAC1_PROVENANCE)),
            _DAC1_PROVENANCE,
            CRS_PROVENANCE,
        ),
    ],
    ids=[
        "imputed-on-amount",
        "crs-on-attributed-amount",
        "crs-leaving-the-imputed-pin-open",
        "imputed-on-amount-with-a-purpose-no-row-holds",
        "dac1-under-crs-slots-binding-a-property-both-carry",
    ],
)
def test_a_provenance_that_publishes_no_member_refuses_facet_not_available(
    identity: dict, named: str, publisher: str
):
    """A named dataset that does not publish the five-tuple pins nothing, so an identity
    leaving open a slot the publisher pins refuses for its provenance, not the pin. The
    refusal reads the five-tuple alone, so it holds when no row matches the slots either.
    DAC1 publishes the `amount` five-tuple under slots of its own, so it refuses under the
    CRS slots, whichever of them the identity binds."""
    code_index = load_index(default_index_path())

    with pytest.raises(FacetNotAvailableError) as refusal:
        _education_aid_members(identity, code_index)

    assert refusal.value.facet_pointer == "/facets/provenance"
    assert named in str(refusal.value)
    assert publisher in str(refusal.value)


def test_a_scope_naming_another_datasets_provenance_refuses():
    code_index = load_index(default_index_path())
    graph = FakeMintGraphReader(arcs={})
    resolve = {
        "spec_id": EDUCATION_AID_SPEC_ID,
        "graph": graph,
        "enumerator": MemberEnumerator(),
        "code_index": code_index,
        "bounds": MemberReadBounds(subject_ceiling=20_000, deadline=time.monotonic() + 3_600),
    }

    plain = _resolve_members(education_aid_identity(), **resolve)
    window_only = _resolve_members(
        education_aid_identity(),
        scope=Scope.model_validate({"window": {"kind": "latest"}}),
        **resolve,
    )
    assert len(plain) == CRS_MEMBER_COUNT
    assert window_only == plain

    with pytest.raises(FacetNotAvailableError) as refusal:
        _resolve_members(
            education_aid_identity(),
            scope=Scope.model_validate({"facets": {"provenance": {"ref": IMPUTED_PROVENANCE}}}),
            **resolve,
        )
    assert refusal.value.facet_pointer == "/facets/provenance"


def _debt_service_identity(**facets: object) -> dict:
    """Debt service to Ghana: a five-tuple CRS, DAC1, DAC2A and DAC2B publish under slot
    keys of their own and IDS publishes through rows the index lists
    (`CodeIndex.listed_statvars`) without declaring IDS a dataset."""
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": [
            {
                "property": "financialFlowType",
                "binding": {"state": "value", "ref": "ServicingPayment"},
            },
            *(
                {"property": prop, "binding": {"state": "unbound"}}
                for prop in (
                    "financingResource",
                    "guaranteeStatus",
                    "idsConcessionality",
                    "idsCreditorType",
                    "idsDebtorType",
                    "includesIMFCredit",
                    "maturityBand",
                )
            ),
        ],
        "entities": {"destination": ["country/GHA"]},
        "requested_window": {"kind": "all"},
        **({"facets": facets} if facets else {}),
    }


def test_a_pin_on_a_listed_provenance_resolves_the_unpinned_members():
    """A provenance the index does not declare as a dataset names no dataset: the pin
    scopes the reads alone, so members resolve as for the identity that pins nothing,
    though declared datasets publish the five-tuple."""
    code_index = load_index(default_index_path())
    assert _IDS_PROVENANCE not in code_index.datasets
    graph = FakeMintGraphReader(arcs={})
    resolve = {
        "spec_id": _SPEC_ID,
        "graph": graph,
        "enumerator": MemberEnumerator(),
        "code_index": code_index,
        "bounds": mint_bounds(),
    }

    unpinned = resolve_definition_members(_debt_service_identity(), **resolve)
    pinned = resolve_definition_members(
        _debt_service_identity(provenance={"state": "value", "ref": _IDS_PROVENANCE}), **resolve
    )

    assert pinned == unpinned
    assert {shape.stat_var for shape in pinned.shapes} == {
        "dcid:Amount_FinancialFlow_ServicingPayment_Nominal"
    }
    assert sum(graph.calls.values()) == 0


def _source_facet(shape: StatVarShape, code_index: CodeIndex) -> Facet:
    """The one facet the member's own dataset publishes."""
    row = next(
        row for row in code_index.statvars if row.dcid == shape.stat_var.removeprefix("dcid:")
    )
    (provenance,) = row.provenances
    return Facet(
        obs_count=10,
        earliest_date="2000",
        latest_date="2023",
        facet_id=f"facet_{provenance}",
        provenance_id=provenance,
        unit="CurrencyUSD",
    )


def _selected_provenance(
    identity: dict, shapes: tuple[StatVarShape, ...], code_index: CodeIndex
) -> str | None:
    requests = [
        to_materialisation_request(identity, shape=shape, code_index=code_index) for shape in shapes
    ]
    selections = select_group_facets(
        requests,
        [[_source_facet(shape, code_index)] for shape in shapes],
        graph=FakeMintGraphReader(arcs={}),
    ).selections
    return selections["provenance"].ref


def test_group_facet_selection_on_the_plain_family_picks_the_crs_provenance():
    code_index = load_index(default_index_path())
    identity = education_aid_identity()
    shapes = _education_aid_members(identity, code_index).shapes

    assert _selected_provenance(identity, shapes, code_index) == CRS_PROVENANCE


@pytest.mark.parametrize(
    ("identity", "provenance"),
    [
        (education_aid_identity(), CRS_PROVENANCE),
        (imputed_education_aid_identity(), IMPUTED_PROVENANCE),
        (imputed_education_aid_identity(source=IMPUTED_PROVENANCE), IMPUTED_PROVENANCE),
    ],
    ids=["plain", "imputed", "imputed-source-named"],
)
def test_every_request_of_an_index_family_carries_the_elected_dataset(
    identity: dict, provenance: str
):
    code_index = load_index(default_index_path())
    shapes = _education_aid_members(identity, code_index).shapes

    requests = [
        to_materialisation_request(identity, shape=shape, code_index=code_index) for shape in shapes
    ]

    assert {request.dataset for request in requests} == {
        ElectedDataset(provenance=provenance, member_provenances=(provenance,))
    }


def test_a_member_held_by_another_dataset_alone_does_not_refuse_the_plain_family():
    """One member, recorded under both datasets, holds observations at the bound entities
    under the imputed dataset alone. The family's provenance is the elected one, read before
    the members' candidates are compared, so that member is unpopulated."""
    code_index = load_index(default_index_path())
    identity = education_aid_identity()
    shapes = _education_aid_members(identity, code_index).shapes
    requests = [
        to_materialisation_request(identity, shape=shape, code_index=code_index) for shape in shapes
    ]
    requests[0] = dataclasses.replace(
        requests[0],
        dataset=ElectedDataset(
            provenance=CRS_PROVENANCE, member_provenances=(CRS_PROVENANCE, IMPUTED_PROVENANCE)
        ),
    )
    imputed_only = Facet(
        obs_count=99,
        earliest_date="2000",
        latest_date="2023",
        facet_id=f"facet_{IMPUTED_PROVENANCE}",
        provenance_id=IMPUTED_PROVENANCE,
        unit="CurrencyUSD",
    )
    candidates = [[_source_facet(shape, code_index)] for shape in shapes]
    candidates[0] = [imputed_only]

    selections = select_group_facets(
        requests, candidates, graph=FakeMintGraphReader(arcs={})
    ).selections

    assert selections["provenance"].ref == CRS_PROVENANCE


def test_group_facet_selection_refuses_a_family_spanning_both_datasets():
    """The members the mint resolved before the election: every row of both datasets."""
    code_index = load_index(default_index_path())
    identity = education_aid_identity()
    spanning = (
        *_education_aid_members(identity, code_index).shapes,
        *_education_aid_members(imputed_education_aid_identity(), code_index).shapes,
    )

    with pytest.raises(FacetNotAvailableError):
        _selected_provenance(identity, spanning, code_index)


# ---------------------------------------------------------------------------
# Synthetic indexes: a row with no provenance, an ambiguous election, and the
# gate that chooses between the index and the graph walk.
# ---------------------------------------------------------------------------

_PLAIN = "provenance/Plain"
_LINES = "provenance/Lines"
_LINE = "tableLine"


def _person_row(dcid: str, constraints: dict[str, str], *provenances: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Person",
        measured_property="count",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("recipient",),
        constraint_properties=constraints,
        provenances=provenances,
    )


def _dataset(provenance: str) -> DatasetDeclaration:
    return DatasetDeclaration(
        provenance=provenance,
        label=provenance,
        description="A test table.",
        opt_in=False,
        absent_observation="zero",
        account="provider",
        rank=0,
    )


def _person_identity(*slots: tuple[str, dict]) -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "Person",
            "measured_property": "count",
            "stat_type": "measuredValue",
        },
        "slots": [{"property": prop, "binding": binding} for prop, binding in slots],
        "entities": {},
        "requested_window": {"kind": "all"},
    }


_UNBOUND = {"state": "unbound"}


def test_a_row_with_no_provenance_is_in_no_family():
    code_index = CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=(
            _person_row("Covered_Male", {"gender": "Male"}, _PLAIN),
            _person_row("Uncovered_Female", {"gender": "Female"}),
        ),
        datasets=(_dataset(_PLAIN),),
    )
    graph = FakeMintGraphReader(arcs={})

    resolved = resolve_definition_members(
        _person_identity(("gender", _UNBOUND)),
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=code_index,
        bounds=mint_bounds(),
    )

    assert resolved.family == {"Covered_Male"}
    assert [shape.stat_var for shape in resolved.shapes] == ["dcid:Covered_Male"]

    only_uncovered = _person_identity(("gender", {"state": "value", "ref": "Female"}))
    with pytest.raises(SpecNotExportable):
        resolve_definition_members(
            only_uncovered,
            spec_id=_SPEC_ID,
            graph=graph,
            enumerator=MemberEnumerator(),
            code_index=code_index,
            bounds=mint_bounds(),
        )


def _population_line_index() -> CodeIndex:
    """A table of aggregate lines publishes a `Person`/`count` line, as such a table carries
    its population denominator."""
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=(_person_row("Lines_Population", {_LINE: "PopulationLine"}, _LINES),),
        datasets=(_dataset(_LINES),),
    )


def test_an_identity_whose_slot_keys_cover_no_index_row_walks_the_graph():
    """`Count_Person` carries no slot, and the index's only `Person`/`count` row carries the
    table's line property, so the index does not describe the population asked for."""
    code_index = _population_line_index()
    graph = _person_family_graph()
    identity = _person_identity()

    shapes = resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=code_index,
        bounds=mint_bounds(),
    ).shapes

    assert [shape.stat_var for shape in shapes] == ["dcid:Person"]
    assert graph.calls["node_subjects"] > 0
    request = to_materialisation_request(identity, shape=shapes[0], code_index=code_index)
    assert request.index_sourced is False


def test_an_identity_naming_the_line_resolves_from_the_index():
    code_index = _population_line_index()
    graph = FakeMintGraphReader(arcs={})
    identity = _person_identity((_LINE, {"state": "value", "ref": "PopulationLine"}))

    shapes = resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=code_index,
        bounds=mint_bounds(),
    ).shapes

    assert [shape.stat_var for shape in shapes] == ["dcid:Lines_Population"]
    assert sum(graph.calls.values()) == 0
    request = to_materialisation_request(identity, shape=shapes[0], code_index=code_index)
    assert request.index_sourced is True


def test_a_wrong_value_on_covered_slot_keys_refuses_on_the_index_path():
    code_index = load_index(default_index_path())
    identity = education_aid_identity()
    purpose = next(slot for slot in identity["slots"] if slot["property"] == "crsPurpose")
    purpose["binding"] = {"state": "value", "ref": "NoSuchPurposeCode"}
    graph = FakeMintGraphReader(arcs={})

    with pytest.raises(SpecNotExportable):
        resolve_definition_members(
            identity,
            spec_id=_SPEC_ID,
            graph=graph,
            enumerator=MemberEnumerator(),
            code_index=code_index,
            bounds=mint_bounds(),
        )

    assert sum(graph.calls.values()) == 0
