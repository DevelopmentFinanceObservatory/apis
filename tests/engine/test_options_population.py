"""Tests options-population derivation from the fixture index and graph enumeration.

Covers filtering, empty populations, truncation, and the graph branch's 700-member cap."""

from __future__ import annotations

import time

import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import IndexStamp, StatVarRow
from search.engine.errors import EngineInputError
from search.engine.options import population as population_module
from search.engine.options.population import options_population
from search.engine.retrieval import bound_values as bound_values_module
from search.models import (
    BaseIdentity,
    IdentityBindingSet,
    IdentityBindingUnbound,
    IdentityBindingValue,
    IdentityShapeDenominatorUnresolved,
    IdentityShapeSchema,
    IdentitySlot,
    IdentityWindowAll,
    RoleAxis,
)
from statvar_members import DEFAULT_MEMBER_CAP, MemberReadDeadline
from statvar_members.enumerate import (
    DegenerateObservationProperties,
    MultiValuedConstraint,
    NotAStatisticalVariable,
)
from statvar_members.types import UNRESOLVED_NODE
from tests.datasets_api._graph_doubles import FirstPageWithholdingGraphReader
from tests.statvar_members.conftest import (
    _KENYA_LIFE_EXPECTANCY_ARCS,
    FakeMemberArcReader,
    FlakyMemberArcReader,
    kenya_reader,
    male_female_only_reader,
)

# The CRS index has 3,013 members for this five-tuple, above the graph cap of 700.
_CRS_SHAPE = IdentityShapeSchema(
    population_type="FinancialFlow",
    measured_property="grantEquivalent",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)

# This Person/lifeExpectancy tuple is absent from the index and uses the graph branch.
_NON_INDEX_SHAPE = IdentityShapeSchema(
    population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
)

# The graph-source cases walk fake IDS series on IDS five-tuples, which the fixture index
# lists.
pytestmark = pytest.mark.usefixtures("unlisted_index")


def _crs_population_dcids() -> list[str]:
    """Returns the CRS family's index members for `_CRS_SHAPE`, independently of
    `options_population`."""
    index = get_index()
    target = ("FinancialFlow", "grantEquivalent", "measuredValue", "Nominal", None)
    return [
        row.dcid
        for row in index.statvars
        if (
            row.population_type or "",
            row.measured_property,
            row.stat_type or "",
            row.measurement_qualifier,
            row.measurement_denominator,
        )
        == target
        and "provenance/OECDCreditorReportingSystem" in row.provenances
    ]


def _identity(shape: IdentityShapeSchema, slots: list[IdentitySlot]) -> BaseIdentity:
    return BaseIdentity(shape=shape, slots=slots, requested_window=IdentityWindowAll())


def _crs_slots(**bound: str) -> list[IdentitySlot]:
    """Every constraint property `_CRS_SHAPE`'s index rows assert, as a CRS answer's
    identity carries them: unbound unless named in `bound`."""
    population = set(_crs_population_dcids())
    props = dict.fromkeys(
        prop
        for row in get_index().statvars
        if row.dcid in population
        for prop in row.constraint_properties
    )
    return [
        IdentitySlot(
            property=prop,
            binding=IdentityBindingValue(ref=bound[prop])
            if prop in bound
            else IdentityBindingUnbound(),
        )
        for prop in props
    ]


def test_a_crs_index_carried_shape_sources_the_full_index_population():
    """A broad ODA-style ask, nothing bound on any axis: the full CRS
    family comes back, not a recall-sized sample, and it is far larger
    than the non-index branch's own 700-member cap -- proving the index
    branch is never capped that way."""
    population_dcids = _crs_population_dcids()
    assert len(population_dcids) > 700, "expected a CRS family larger than member_cap=700"

    identity = _identity(_CRS_SHAPE, slots=_crs_slots())

    result = options_population(identity, graph=FakeMemberArcReader(arcs={}))

    assert result.source == "index"
    assert set(result.stat_var_dcids) == set(population_dcids)
    assert result.declared_roles == ("financingProvider", "financingRecipient")


def test_a_bound_crs_purpose_narrows_the_index_population():
    population_dcids = _crs_population_dcids()
    index = get_index()
    bound_purpose = next(
        row.constraint_properties["crsPurpose"]
        for row in index.statvars
        if row.dcid in population_dcids and "crsPurpose" in row.constraint_properties
    )
    identity = _identity(_CRS_SHAPE, slots=_crs_slots(crsPurpose=bound_purpose))

    result = options_population(identity, graph=FakeMemberArcReader(arcs={}))

    assert result.source == "index"
    assert 0 < len(result.stat_var_dcids) < len(population_dcids)


def test_an_identity_whose_slots_cover_no_index_row_enumerates_the_graph():
    """The index carries the five-tuple, and every row of it carries a property the identity
    declares no slot for. The index does not decide such an identity's membership, so the
    population is the graph's, as it is at the Datasets mint."""
    graph_member = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("FinancialFlow",),
        "measuredProperty": ("grantEquivalent",),
        "statType": ("measuredValue",),
        "measurementQualifier": ("Nominal",),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": ("financingProvider", "financingRecipient"),
    }
    identity = _identity(_CRS_SHAPE, slots=[])

    result = options_population(
        identity, graph=FakeMemberArcReader(arcs={"GrantEquivalent_Total": graph_member})
    )

    assert result.source == "graph"
    assert result.stat_var_dcids == ("GrantEquivalent_Total",)


def test_a_set_of_parent_codes_reaches_their_children():
    """Index rows assert leaf cooperation modalities such as `CRSCooperationModalityA02`. A set
    bound to the parents A and C widens to their descendants, as the datasets mint widens the
    same wire set, so the index serves every member under either parent."""
    parents = ["CRSCooperationModalityA", "CRSCooperationModalityC"]
    slots = [
        IdentitySlot(property=slot.property, binding=IdentityBindingSet(refs=parents))
        if slot.property == "crsCooperationModality"
        else slot
        for slot in _crs_slots()
    ]
    index = get_index()
    children = {code for parent in parents for code in index.descendants(parent)}

    result = options_population(
        _identity(_CRS_SHAPE, slots=slots), graph=FakeMemberArcReader(arcs={})
    )

    assert result.source == "index"
    assert result.stat_var_dcids
    modalities = {
        row.constraint_properties.get("crsCooperationModality")
        for row in index.statvars
        if row.dcid in set(result.stat_var_dcids)
    }
    assert modalities <= children
    assert not modalities & set(parents)


def test_a_parent_code_value_reaches_its_children():
    """A value bound to the parent `CRSCooperationModalityA`, which no index row asserts,
    widens to its descendants as the datasets mint widens it, so the index serves A's
    children."""
    parent = "CRSCooperationModalityA"
    slots = [
        IdentitySlot(property=slot.property, binding=IdentityBindingValue(ref=parent))
        if slot.property == "crsCooperationModality"
        else slot
        for slot in _crs_slots()
    ]
    index = get_index()

    result = options_population(
        _identity(_CRS_SHAPE, slots=slots), graph=FakeMemberArcReader(arcs={})
    )

    assert result.source == "index"
    assert result.stat_var_dcids
    modalities = {
        row.constraint_properties.get("crsCooperationModality")
        for row in index.statvars
        if row.dcid in set(result.stat_var_dcids)
    }
    assert modalities <= set(index.descendants(parent)) - {parent}


def _person_amount_arcs(denominator: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": denominator,
        "constraintProperties": (),
        "observationProperties": (),
    }


def test_an_unresolved_denominator_shape_names_the_marker_member():
    """An identity whose denominator is the unresolved marker denotes the member whose
    denominator arc names no dcid, never the member with no denominator at all."""
    reader = FakeMemberArcReader(
        arcs={
            "Amount_Person_PerMarker": _person_amount_arcs((UNRESOLVED_NODE,)),
            "Amount_Person": _person_amount_arcs(()),
        }
    )
    shape = IdentityShapeSchema(
        population_type="Person",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_denominator=IdentityShapeDenominatorUnresolved(),
    )

    result = options_population(_identity(shape, slots=[]), graph=reader)

    assert result.source == "graph"
    assert result.stat_var_dcids == ("Amount_Person_PerMarker",)


def test_an_index_miss_reads_as_empty_never_reaching_a_recall_pool():
    """An indexed shape with no matching `crsPurpose` yields an empty population: its slot
    keys fit the index rows, so the index is its source, and the graph is never read."""
    identity = _identity(_CRS_SHAPE, slots=_crs_slots(crsPurpose="CRSPurposeCodeDoesNotExist999"))
    graph = FakeMemberArcReader(arcs={})

    with pytest.raises(EngineInputError) as exc_info:
        options_population(identity, graph=graph)

    assert sum(graph.calls.values()) == 0

    assert exc_info.value.code == "population_empty"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/identity"


def test_a_non_index_shape_sources_via_statvar_members_enumerate_members():
    identity = _identity(
        _NON_INDEX_SHAPE, slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())]
    )

    result = options_population(identity, graph=kenya_reader())

    assert result.source == "graph"
    assert result.stat_var_dcids == ("LifeExpectancy_Person",)
    assert result.declared_roles == ("observationAbout",)


@pytest.mark.parametrize(
    ("slots", "expected"),
    [
        pytest.param(
            ("gender",),
            ("LifeExpectancy_Person_Female", "LifeExpectancy_Person_Male"),
            id="a_subset_identity_leaves_out_members_asserting_other_slots",
        ),
        pytest.param(
            ("age", "gender"),
            (
                "LifeExpectancy_Person_0Years",
                "LifeExpectancy_Person_Female",
                "LifeExpectancy_Person_Male",
            ),
            id="the_full_identity_keeps_each_signature",
        ),
    ],
)
def test_a_family_with_no_total_populates_the_slots_its_identity_carries(slots, expected):
    """A family with no total breaks down by sex and by age. An identity carrying the sex
    slot alone, as a breakdown decision mints it, denotes the members asserting sex alone,
    as a mint of it serves them."""
    no_total = {
        dcid: arcs
        for dcid, arcs in _KENYA_LIFE_EXPECTANCY_ARCS.items()
        if dcid != "LifeExpectancy_Person"
    }
    identity = _identity(
        _NON_INDEX_SHAPE,
        slots=[IdentitySlot(property=prop, binding=IdentityBindingUnbound()) for prop in slots],
    )

    result = options_population(identity, graph=FakeMemberArcReader(arcs=no_total))

    assert result.source == "graph"
    assert result.stat_var_dcids == expected


def test_an_empty_non_index_population_refuses():
    identity = _identity(
        _NON_INDEX_SHAPE, slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())]
    )

    with pytest.raises(EngineInputError) as exc_info:
        options_population(identity, graph=FakeMemberArcReader(arcs={}))

    assert exc_info.value.code == "population_empty"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/identity"


def test_a_truncated_non_index_population_with_a_confirmed_member_resolves():
    """On a truncated page, a confirmed member whose constraint keys match the unbound
    identity proves completeness."""
    identity = _identity(
        _NON_INDEX_SHAPE, slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())]
    )

    result = options_population(identity, graph=kenya_reader(force_truncated=True))

    assert result.source == "graph"
    assert result.stat_var_dcids == ("LifeExpectancy_Person",)
    # Its constraint keys match the unbound identity; unbound gender defaults to aggregate.
    # This lets the antichain prove completeness despite the truncated graph page.
    assert result.declared_roles == ("observationAbout",)


def test_a_deep_root_population_settles_from_the_whole_filtered_set():
    """The first page truncates and leaves out the root, so it cannot prove itself complete.
    The walk reads the whole filtered set, as the mint does, and the root dominates the
    gendered members."""
    arcs = {
        dcid: {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "constraintProperties": ("gender",) if gender else (),
            **({"gender": (gender,)} if gender else {}),
        }
        for dcid, gender in (
            ("Count_Person", None),
            ("Count_Person_Male", "Male"),
            ("Count_Person_Female", "Female"),
        )
    }
    shape = IdentityShapeSchema(
        population_type="Person", measured_property="count", stat_type="measuredValue"
    )
    identity = _identity(
        shape, slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())]
    )

    result = options_population(
        identity,
        graph=FirstPageWithholdingGraphReader(withheld=frozenset({"Count_Person"}), arcs=arcs),
    )

    assert result.stat_var_dcids == ("Count_Person",)


def test_a_graph_walk_past_its_deadline_raises_a_timeout():
    """The door answers a `TimeoutError` 504, so a walk past the request deadline never
    reads as an empty or truncated population."""
    identity = _identity(
        _NON_INDEX_SHAPE, slots=[IdentitySlot(property="gender", binding=IdentityBindingUnbound())]
    )

    with pytest.raises(MemberReadDeadline):
        options_population(identity, graph=kenya_reader(), deadline=time.monotonic() - 1)


_COUNT_SHAPE = IdentityShapeSchema(
    population_type="Person", measured_property="count", stat_type="measuredValue"
)


@pytest.mark.parametrize(
    ("shape", "binding", "graph"),
    [
        pytest.param(
            _NON_INDEX_SHAPE,
            IdentityBindingValue(ref="Nonbinary"),
            lambda: kenya_reader(force_truncated=True),
            id="no_confirmed_member",
        ),
        pytest.param(
            _COUNT_SHAPE,
            IdentityBindingUnbound(),
            lambda: male_female_only_reader(force_truncated=True),
            id="incomparable_members",
        ),
    ],
)
def test_a_population_past_the_subject_ceiling_reads_scope_too_broad(shape, binding, graph):
    """A filtered set the full read cannot finish under the subject ceiling refuses
    `scope_too_broad`, whatever its first page confirmed, as the mint refuses it."""
    identity = _identity(shape, slots=[IdentitySlot(property="gender", binding=binding)])

    with pytest.raises(EngineInputError) as exc_info:
        options_population(identity, graph=graph())

    assert exc_info.value.code == "scope_too_broad"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/identity"


def test_a_non_index_population_past_the_member_cap_refuses():
    arcs = {
        f"Count_Person_Bucket{i}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        }
        for i in range(DEFAULT_MEMBER_CAP + 1)
    }
    shape = IdentityShapeSchema(
        population_type="Person", measured_property="count", stat_type="measuredValue"
    )
    identity = _identity(shape, slots=[])

    with pytest.raises(EngineInputError) as exc_info:
        options_population(identity, graph=FakeMemberArcReader(arcs=arcs))

    assert exc_info.value.code == "scope_too_broad"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/identity"


_IDS_SHAPE = IdentityShapeSchema(
    population_type="FinancialPosition",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)
_IDS_CONSTRAINTS = {
    "financialPositionType": "AmountOutstanding",
    "guaranteeStatus": "PublicOrPubliclyGuaranteed",
}
_IDS_TOTAL = "PPG_Total"
_IDS_BY_CREDITOR = "PPG_ByCreditor"
_IDS_LENDER = "PPG_IBRD"
_IDS_LENDER_CONSTRAINTS = {
    "financialPositionType": "AmountOutstanding",
    "guaranteeStatus": "IBRDLoan",
}
_IDS_MEMBERS = {
    _IDS_TOTAL: (_IDS_CONSTRAINTS, ("financingRecipient",)),
    _IDS_BY_CREDITOR: (_IDS_CONSTRAINTS, ("financingProvider", "financingRecipient")),
}


def _ids_identity(
    entities: dict[str, list[str]],
    *,
    extra_slots: tuple[str, ...] = (),
    open_roles: tuple[RoleAxis, ...] | None = None,
) -> BaseIdentity:
    return BaseIdentity(
        shape=_IDS_SHAPE,
        slots=[
            IdentitySlot(property=name, binding=IdentityBindingUnbound())
            for name in (*_IDS_CONSTRAINTS, *extra_slots)
        ],
        entities=entities,
        open_roles=open_roles,
        requested_window=IdentityWindowAll(),
    )


def _ids_reader(members: dict[str, tuple[dict[str, str], tuple[str, ...]]]) -> FakeMemberArcReader:
    return FakeMemberArcReader(
        arcs={
            dcid: {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("FinancialPosition",),
                "measuredProperty": ("amount",),
                "statType": ("measuredValue",),
                "measurementQualifier": ("Nominal",),
                "measurementDenominator": (),
                "constraintProperties": tuple(constraints),
                "observationProperties": roles,
                **{name: (value,) for name, value in constraints.items()},
            }
            for dcid, (constraints, roles) in members.items()
        }
    )


def _ids_index(
    members: dict[str, tuple[dict[str, str], tuple[str, ...]]], *, listed: bool = False
) -> CodeIndex:
    """An index holding `members` as rows, or as listed rows (`CodeIndex.listed_statvars`)."""
    rows = tuple(
        StatVarRow(
            dcid=dcid,
            population_type="FinancialPosition",
            measured_property="amount",
            stat_type="measuredValue",
            measurement_qualifier="Nominal",
            measurement_denominator=None,
            observation_properties=roles,
            constraint_properties=constraints,
            provenances=("provenance/Test",),
        )
        for dcid, (constraints, roles) in members.items()
    )
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=(),
        codes=(),
        statvars=() if listed else rows,
        listed_statvars=rows if listed else (),
    )


def _use_index(monkeypatch: pytest.MonkeyPatch, index: CodeIndex) -> None:
    monkeypatch.setattr(bound_values_module, "get_index", lambda: index)
    monkeypatch.setattr(population_module, "get_index", lambda: index)


@pytest.mark.parametrize(
    ("entities", "expected"),
    [
        ({}, _IDS_TOTAL),
        ({"destination": ["country/KEN"]}, _IDS_TOTAL),
        ({"origin": ["country/CHN"], "destination": ["country/KEN"]}, _IDS_BY_CREDITOR),
    ],
)
def test_graph_role_twins_resolve_by_the_identitys_bound_roles(entities, expected):
    result = options_population(_ids_identity(entities), graph=_ids_reader(_IDS_MEMBERS))

    assert result.source == "graph"
    assert result.stat_var_dcids == (expected,)
    assert result.declared_roles == _IDS_MEMBERS[expected][1]


@pytest.mark.parametrize(
    ("entities", "expected"),
    [
        ({}, _IDS_TOTAL),
        ({"destination": ["country/KEN"]}, _IDS_TOTAL),
        ({"origin": ["country/CHN"], "destination": ["country/KEN"]}, _IDS_BY_CREDITOR),
    ],
)
def test_index_role_twins_resolve_by_the_identitys_bound_roles(monkeypatch, entities, expected):
    _use_index(monkeypatch, _ids_index(_IDS_MEMBERS))

    result = options_population(_ids_identity(entities), graph=FakeMemberArcReader(arcs={}))

    assert result.source == "index"
    assert result.stat_var_dcids == (expected,)
    assert result.declared_roles == _IDS_MEMBERS[expected][1]


@pytest.mark.parametrize("source", ["graph", "index"])
def test_an_open_creditor_resolves_the_role_twins_to_the_breakdown(monkeypatch, source):
    """The identity's open creditor joins the role keys on both sources, so the
    by-creditor twin covers them and the total does not."""
    if source == "index":
        _use_index(monkeypatch, _ids_index(_IDS_MEMBERS))
        reader = FakeMemberArcReader(arcs={})
    else:
        reader = _ids_reader(_IDS_MEMBERS)

    result = options_population(
        _ids_identity({"destination": ["country/KEN"]}, open_roles=("origin",)), graph=reader
    )

    assert result.source == source
    assert result.stat_var_dcids == (_IDS_BY_CREDITOR,)


_IDS_WITH_LENDER = {
    **_IDS_MEMBERS,
    _IDS_LENDER: (_IDS_LENDER_CONSTRAINTS, ("financingProvider", "financingRecipient")),
}


@pytest.mark.parametrize("source", ["graph", "index", "listed"])
def test_a_total_beside_a_lender_series_resolves_the_total(monkeypatch, source):
    """The lender series holds its own constraint values and declares the creditor beside
    the debtor. The identity names no creditor, so the population is the total, the
    series published without one. Listed rows elect the walk's population without a
    graph read."""
    if source == "graph":
        reader = _ids_reader(_IDS_WITH_LENDER)
    else:
        _use_index(monkeypatch, _ids_index(_IDS_WITH_LENDER, listed=source == "listed"))
        reader = FakeMemberArcReader(arcs={})

    result = options_population(_ids_identity({"destination": ["country/KEN"]}), graph=reader)

    assert result.source == ("graph" if source == "listed" else source)
    assert result.stat_var_dcids == (_IDS_TOTAL,)
    assert result.declared_roles == ("financingRecipient",)


_IDS_NOT_NESTED = {
    _IDS_TOTAL: (_IDS_CONSTRAINTS, ("financingRecipient",)),
    _IDS_LENDER: (_IDS_LENDER_CONSTRAINTS, ("financingProvider",)),
}


@pytest.mark.parametrize("source", ["graph", "index"])
def test_survivors_whose_role_sets_do_not_nest_refuse_role_signature_incompatible(
    monkeypatch, source
):
    """A debtor-only total beside a creditor-only series of other constraint values:
    neither role set holds the other, so no one role signature describes the population."""
    if source == "index":
        _use_index(monkeypatch, _ids_index(_IDS_NOT_NESTED))
        reader = FakeMemberArcReader(arcs={})
    else:
        reader = _ids_reader(_IDS_NOT_NESTED)

    with pytest.raises(EngineInputError) as exc_info:
        options_population(_ids_identity({}), graph=reader)

    assert exc_info.value.code == "role_signature_incompatible"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/identity"


def test_role_signature_refusal_names_five_members_and_counts_the_rest(monkeypatch):
    """A family of twelve members that disagree on roles names five of them and counts
    the other seven, so the refusal stays short whatever the family's size."""
    members = {
        f"dfo/ids/M{i:02d}": (
            _IDS_CONSTRAINTS,
            ("financingRecipient",) if i % 2 else ("financingProvider",),
        )
        for i in range(12)
    }
    _use_index(monkeypatch, _ids_index(members))

    with pytest.raises(EngineInputError) as exc_info:
        options_population(_ids_identity({}), graph=FakeMemberArcReader(arcs={}))

    assert exc_info.value.code == "role_signature_incompatible"
    message = str(exc_info.value)
    assert sum(dcid in message for dcid in members) == 5
    assert "7 more" in message


@pytest.mark.parametrize(
    "error",
    [
        MultiValuedConstraint(stat_var=_IDS_TOTAL, property="lendingEntity", values=("a", "b")),
        NotAStatisticalVariable(stat_var=_IDS_TOTAL, type_of=("Thing",)),
        NotAStatisticalVariable(stat_var=_IDS_TOTAL, type_of=()),
        DegenerateObservationProperties(
            stat_var=_IDS_TOTAL, observation_properties=("observationAbout",)
        ),
    ],
)
def test_an_unreadable_candidate_refuses_request_invalid_naming_it(monkeypatch, error):
    """A candidate the graph walk cannot read as a member refuses `request_invalid` at the
    identity, naming the candidate in a sentence that carries no Python repr."""

    class _Refusing:
        member_cap = DEFAULT_MEMBER_CAP

        def enumerate(self, **kwargs):
            raise error

    monkeypatch.setattr(population_module, "shared_enumerator", _Refusing)

    with pytest.raises(EngineInputError) as exc_info:
        options_population(_ids_identity({}), graph=_ids_reader(_IDS_NOT_NESTED))

    assert exc_info.value.code == "request_invalid"
    assert exc_info.value.status == 400
    assert exc_info.value.pointer == "/identity"
    message = str(exc_info.value)
    assert _IDS_TOTAL in message
    assert not set("'[]") & set(message.replace("identity's", ""))


def test_index_member_refining_another_over_an_unbound_constraint_is_dropped(monkeypatch):
    members = {
        _IDS_TOTAL: (_IDS_CONSTRAINTS, ("financingRecipient",)),
        "PPG_BySector": (
            {**_IDS_CONSTRAINTS, "sector": "Health"},
            ("financingProvider", "financingRecipient"),
        ),
    }
    _use_index(monkeypatch, _ids_index(members))

    result = options_population(
        _ids_identity({}, extra_slots=("sector",)), graph=FakeMemberArcReader(arcs={})
    )

    assert result.source == "index"
    assert result.stat_var_dcids == (_IDS_TOTAL,)
    assert result.declared_roles == ("financingRecipient",)


_IDS_BY_CREDITOR_SECTOR = "PPG_ByCreditorSector"
_IDS_ANTICHAIN_BEFORE_TWINS: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    _IDS_TOTAL: (_IDS_CONSTRAINTS, ("financingRecipient",)),
    _IDS_BY_CREDITOR_SECTOR: (
        {**_IDS_CONSTRAINTS, "sector": "Health"},
        ("financingProvider", "financingRecipient"),
    ),
}


@pytest.mark.parametrize(
    ("entities", "expected"),
    [
        ({"destination": ["country/KEN"]}, _IDS_TOTAL),
        ({"origin": ["country/CHN"], "destination": ["country/KEN"]}, _IDS_BY_CREDITOR_SECTOR),
    ],
)
def test_index_bound_roles_are_required_before_the_constraint_antichain(
    monkeypatch, entities, expected
):
    _use_index(monkeypatch, _ids_index(_IDS_ANTICHAIN_BEFORE_TWINS))

    result = options_population(
        _ids_identity(entities, extra_slots=("sector",)), graph=FakeMemberArcReader(arcs={})
    )

    assert result.source == "index"
    assert result.stat_var_dcids == (expected,)
    assert result.declared_roles == _IDS_ANTICHAIN_BEFORE_TWINS[expected][1]


def test_confirmed_roots_that_disagree_refuse_beside_an_unreadable_twin():
    """`Count_Person_Z` reads empty every time. `Count_Person_A` and `Count_Person_B` are
    confirmed and each places the destination with the one role it declares, so they
    disagree whatever Z declares, and no one role signature describes them."""

    def root(*roles: str) -> dict[str, tuple[str, ...]]:
        return {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": roles,
        }

    arcs = {
        "Count_Person_A": root("recipient"),
        "Count_Person_B": root("target"),
        "Count_Person_Z": root("recipient"),
    }
    shape = IdentityShapeSchema(
        population_type="Person", measured_property="count", stat_type="measuredValue"
    )

    with pytest.raises(EngineInputError) as exc_info:
        options_population(
            BaseIdentity(
                shape=shape,
                slots=[],
                entities={"destination": ["country/KEN"]},
                requested_window=IdentityWindowAll(),
            ),
            graph=FlakyMemberArcReader(arcs=arcs, empty_always=["Count_Person_Z"]),
        )

    assert exc_info.value.code == "role_signature_incompatible"
    assert exc_info.value.status == 422
    assert "Count_Person_B" in str(exc_info.value)
