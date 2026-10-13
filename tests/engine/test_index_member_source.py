"""Tests when `shape_member_facts` reads a shape's members from the packaged index.

Only incomplete recall facts widen, and only for a query whose membership the index decides. A
separate test checks that indexed five-tuples cannot form availability-sibling pairs.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from collections import defaultdict

import pytest

import search.engine.retrieval.bound_values as bound_values_module
from code_index import FamilyQuery, StatVarRow, default_index_path, parse_index
from code_index.shared import get_index, get_measure_index
from datasets_api.adapter import resolve_definition_members
from scripts.export_code_index import content_hash
from search.engine.bind import SlotBindingDraft
from search.engine.options.population import options_population
from search.engine.options.population_memo import _population_key
from search.engine.region.availability import _are_availability_siblings
from search.engine.region.grounding import _shape_with_taxonomy
from search.engine.retrieval.bound_values import (
    declaration_version,
    five_tuple_member_facts,
    identity_query,
    index_decides_membership,
    measure_family_partitions,
    shape_member_facts,
    shape_query,
    source_binding,
    synthetic_shape_for_identity,
)
from search.engine.retrieval.member_facts import ArcsMemberFacts, IndexMemberFacts
from search.engine.retrieval.retrieve import Materialised, materialise
from search.engine.retrieval.standard import (
    _admit_standard_members,
    _AdmittedStandardMembers,
    _elect_family,
)
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_id import mint_identity
from search.models import BaseIdentity, IdentityShapeSchema
from series_coverage.schema import RoleNames
from statvar_members import MemberEnumerator
from tests.code_index.aid_identity import (
    AID_TYPE,
    CRS_MEMBER_COUNT,
    CRS_PROVENANCE,
    EDUCATION_AID_SPEC_ID,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_MULTILATERAL,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
)
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._member_bounds import mint_bounds
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._index_slots import index_slot_keys, query_for
from tests.fixtures import FakeGraph

# A real CRS five-tuple this deployment's index carries hundreds of members for.
_POP_TYPE = "FinancialFlow"
_MEAS_PROP = "grantEquivalent"
_STAT_TYPE = "measuredValue"
_MEAS_QUAL = "Nominal"
_MEAS_DENOM = None


def _crs_shape(shape_id: str, sv_arc_facts: dict[str, dict], *, complete: bool) -> ShapeDraft:
    """A shape of the five-tuple, with a slot key for every constraint property its index
    rows carry, as `derive_shapes` builds it."""
    properties = sorted({prop for row in _crs_rows() for prop in row.constraint_properties})
    return ShapeDraft(
        shape_id=shape_id,
        label="test shape",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=_MEAS_QUAL,
        meas_denom_dcid=_MEAS_DENOM,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in properties
        ),
        sv_arc_facts=sv_arc_facts,
        sv_arc_facts_complete=complete,
    )


def _crs_population() -> list[str]:
    """List this five-tuple's index dcids independently for cross-checks."""
    return [row.dcid for row in _crs_rows()]


def _crs_rows() -> list[StatVarRow]:
    index = get_index()
    target = (_POP_TYPE, _MEAS_PROP, _STAT_TYPE, _MEAS_QUAL, _MEAS_DENOM)
    return [
        row
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


def test_a_broad_ask_sources_the_full_index_population_not_the_recall_pool():
    """One incomplete recall member widens to the full indexed population,
    regardless of sample size."""
    population = _crs_population()
    assert len(population) > 100, "expected a genuinely broad CRS five-tuple group"

    recalled_sv = population[0]
    shape = _crs_shape("test_broad_crs_ask", {recalled_sv: {}}, complete=False)

    # A broad ask: nothing bound on any axis.
    member_facts, index_sourced = shape_member_facts(shape, query_for(shape, {}))

    assert index_sourced is True
    assert set(member_facts) == set(population)
    assert len(member_facts) == len(population)
    assert len(member_facts) > 1

    # Injected, not fetched: every member's facts are an IndexMemberFacts
    # built straight from its own StatVarRow -- shape_member_facts takes no
    # graph/reader argument at all, so there is no second fetch to make.
    extra_sv = next(sv for sv in population if sv != recalled_sv)
    assert isinstance(member_facts[extra_sv], IndexMemberFacts)


def test_bound_value_narrows_the_index_population():
    population = _crs_population()
    shape = _crs_shape("test_narrowed_crs_ask", {population[0]: {}}, complete=False)
    index = get_index()
    bound_purpose = next(
        row.constraint_properties["crsPurpose"]
        for row in index.statvars
        if row.dcid in population and "crsPurpose" in row.constraint_properties
    )

    member_facts, index_sourced = shape_member_facts(
        shape, query_for(shape, {"crsPurpose": [bound_purpose]})
    )

    assert index_sourced is True
    assert 0 < len(member_facts) < len(population)
    assert all(
        mf.read_constraints().get("crsPurpose") == bound_purpose for mf in member_facts.values()
    )


def test_a_precise_small_posted_set_stays_on_the_recall_path():
    """A complete posted set stays exact even when its five-tuple exists in the index."""
    population = _crs_population()
    one_posted_sv = population[0]
    shape = _crs_shape("test_resubmit_promote_shape", {one_posted_sv: {}}, complete=True)

    member_facts, index_sourced = shape_member_facts(shape, query_for(shape, {}))

    assert index_sourced is False
    assert set(member_facts) == {one_posted_sv}
    assert isinstance(member_facts[one_posted_sv], ArcsMemberFacts)


def test_a_large_precise_posted_set_still_stays_on_the_recall_path():
    """A complete posted set of 40 members, short of the whole population, remains exact."""
    population = _crs_population()
    assert len(population) > 40
    posted_set = {key: {} for key in population[:40]}
    shape = _crs_shape("test_large_precise_posted_set", posted_set, complete=True)

    member_facts, index_sourced = shape_member_facts(shape, query_for(shape, {}))

    assert index_sourced is False
    assert set(member_facts) == set(posted_set)
    assert len(member_facts) == len(posted_set) < len(population)


def test_a_shape_the_index_does_not_carry_keeps_the_recall_pool_unchanged():
    """A demographic-shaped five-tuple: no row in the packaged index (CRS
    code systems alone) matches it, so the swap never fires."""
    arcs_a = {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "gender"}]},
        "gender": {"nodes": [{"dcid": "Female"}]},
    }
    shape = ShapeDraft(
        shape_id="test_demographic_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="unemploymentRate",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={"sv_female": arcs_a},
        sv_arc_facts_complete=False,
    )

    member_facts, index_sourced = shape_member_facts(shape, query_for(shape, {}))

    assert index_sourced is False
    assert set(member_facts) == {"sv_female"}
    assert isinstance(member_facts["sv_female"], ArcsMemberFacts)
    assert member_facts["sv_female"].read_constraints() == {"gender": "Female"}


def test_standard_resolve_confirms_an_index_sourced_member_never_in_the_recall_pool():
    """Materialise confirms an indexed member absent from the recall sample."""
    population = _crs_population()
    recalled_sv = population[0]
    extra_sv = next(sv for sv in population if sv != recalled_sv)
    # The recall pool confirmed only recalled_sv; it carries no observations
    # below, so the resolve can only succeed by reaching extra_sv through the
    # index-sourced population instead.
    shape = _crs_shape("test_broad_crs_resolve", {recalled_sv: {}}, complete=False)
    recipient = "country/KEN"
    graph = FakeGraph(
        nodes={},
        obs={
            f"{extra_sv}|{recipient}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
            ]
        },
        detect={},
        resolve={},
    )

    result = materialise(
        shape,
        [],
        {"observationAbout": [recipient]},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.has_data is True
    assert result.sv_dcids == [extra_sv]


def test_no_two_index_carried_five_tuples_form_an_availability_sibling_pair():
    """Indexed rows currently have `meas_denom_unresolved=False`, so this passes vacuously.

    The guard fails if an indexed five-tuple gains an unresolved denominator that pairs with a
    resolved one, which could bypass availability pruning during index widening.
    """
    index = get_index()
    groups: dict[tuple[str, str, str, str | None, str | None], set[str]] = defaultdict(set)
    for row in index.statvars:
        key = (
            row.population_type or "",
            row.measured_property,
            row.stat_type or "",
            row.measurement_qualifier,
            row.measurement_denominator,
        )
        groups[key].update(row.constraint_properties)

    assert len(groups) > 1, "expected the packaged index to carry more than one five-tuple"

    shapes = [
        ShapeDraft(
            shape_id=f"idx-five-tuple-{i}",
            label="index five-tuple",
            pop_type_dcid=pop_type,
            meas_prop_dcid=meas_prop,
            stat_type_dcid=stat_type,
            meas_qual_dcid=meas_qual,
            meas_denom_dcid=meas_denom,
            meas_denom_unresolved=False,
            slot_keys=tuple(
                SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in sorted(props)
            ),
        )
        for i, ((pop_type, meas_prop, stat_type, meas_qual, meas_denom), props) in enumerate(
            groups.items()
        )
    ]

    for a in shapes:
        for b in shapes:
            if a is b:
                continue
            assert not _are_availability_siblings(a, b), (a.shape_id, b.shape_id)


# ---------------------------------------------------------------------------
# The education-aid identities: Search and the mint resolve one family, of one
# dataset's members. The CRS publishes the `amount` five-tuple and the imputed
# multilateral table the `attributedAmount` one.
# ---------------------------------------------------------------------------


def _education_shape(identity: dict) -> tuple[ShapeDraft, FamilyQuery]:
    """The shape Search resolves `identity` through, and the identity's query."""
    _, definition = mint_identity(identity)
    assert isinstance(definition.shape, IdentityShapeSchema)
    shape = dataclasses.replace(
        synthetic_shape_for_identity(definition.shape), sv_arc_facts_complete=False
    )
    return shape, identity_query(definition)


def _search_family(identity: dict) -> set[str]:
    """The members Search resolves for `identity` from the packaged index."""
    member_facts, index_sourced = shape_member_facts(*_education_shape(identity))
    assert index_sourced is True
    return set(member_facts)


def _mint_family(identity: dict) -> set[str]:
    """The members the Datasets mint resolves for `identity`, as bare dcids."""
    graph = FakeMintGraphReader(arcs={})
    resolved = resolve_definition_members(
        identity,
        spec_id=mint_identity(identity)[0],
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
    )
    return {shape.stat_var.removeprefix("dcid:") for shape in resolved.shapes}


def _imputed_rows() -> set[str]:
    return {row.dcid for row in get_index().statvars if IMPUTED_PROVENANCE in row.provenances}


def test_shape_member_facts_matches_the_mint_family():
    plain = education_aid_identity()
    search_family = _search_family(plain)
    assert search_family == _mint_family(plain)
    assert len(search_family) == CRS_MEMBER_COUNT
    assert search_family.isdisjoint(_imputed_rows())

    imputed = imputed_education_aid_identity()
    imputed_family = _search_family(imputed)
    assert imputed_family == _mint_family(imputed)
    assert len(imputed_family) == IMPUTED_MEMBER_COUNT
    assert imputed_family <= _imputed_rows()


def test_shape_member_facts_matches_the_mint_family_for_a_named_source():
    named = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    search_family = _search_family(named)

    assert search_family == _mint_family(named)
    assert len(search_family) == IMPUTED_MEMBER_COUNT


@pytest.mark.parametrize(
    ("identity", "member_count"),
    [
        (education_aid_identity(), CRS_MEMBER_COUNT),
        (imputed_education_aid_identity(), IMPUTED_MEMBER_COUNT),
        (imputed_education_aid_identity(source=IMPUTED_PROVENANCE), IMPUTED_MEMBER_COUNT),
    ],
    ids=["crs", "imputed", "imputed-source-named"],
)
def test_standard_election_keeps_the_members_the_mint_serves(identity: dict, member_count: int):
    """`_elect_family` reduces the elected dataset's rows with the mint's reducer, so the
    members Search elects among are the mint's shapes."""
    member_facts, _ = shape_member_facts(*_education_shape(identity))
    stated = {
        slot["property"]: [slot["binding"]["ref"]]
        for slot in identity["slots"]
        if slot["binding"]["state"] == "value"
    }

    family = _elect_family(list(member_facts), member_facts, stated)

    assert set(family.order) == _mint_family(identity)
    assert len(family.order) == member_count


def test_a_shape_query_names_the_dataset_a_source_binding_binds():
    """A source-axis value binding carries the dataset in a bindings list, and the query it
    yields equals the query of the identity that names the same dataset."""
    identity = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    shape, query = _education_shape(identity)
    shape = dataclasses.replace(
        shape,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=slot["property"], label=slot["property"])
            for slot in identity["slots"]
        ),
    )
    bindings = [
        SlotBindingDraft(
            axis="how",
            property_dcid=slot["property"],
            kind="value",
            value_dcids=[slot["binding"]["ref"]],
        )
        for slot in identity["slots"]
        if slot["binding"]["state"] == "value"
    ]

    assert shape_query(shape, bindings, ()).source is None
    named = shape_query(
        shape, [*bindings, source_binding(IMPUTED_PROVENANCE)], identity["entities"]
    )
    assert named == query
    two_sources = SlotBindingDraft(
        axis="source",
        property_dcid=None,
        kind="set",
        value_dcids=[IMPUTED_PROVENANCE, CRS_PROVENANCE],
    )
    assert shape_query(shape, [*bindings, two_sources], ()).source is None


def test_the_resolver_reads_the_dataset_a_source_binding_names():
    """`retrieve.materialise` receives bindings and no identity. A source binding among them
    makes the resolver admit the family the mint resolves for the identity naming it."""
    identity = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    family = _mint_family(identity)
    shape, _ = _education_shape(identity)
    recalled = next(row for row in get_index().statvars if row.dcid in family)
    shape = dataclasses.replace(
        shape,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=slot["property"], label=slot["property"])
            for slot in identity["slots"]
        ),
        sv_arc_facts={recalled.dcid: _index_row_arcs(recalled)},
        declared_roles=("financingProvider", "financingRecipient"),
    )
    bindings = [
        SlotBindingDraft(
            axis="how",
            property_dcid=slot["property"],
            kind="value",
            value_dcids=[slot["binding"]["ref"]],
        )
        for slot in identity["slots"]
        if slot["binding"]["state"] == "value"
    ]
    coverage = build_coverage_index(
        [(dcid, "f1", "country/USA", "country/KEN", [2020]) for dcid in sorted(family)],
        role_names={
            dcid: RoleNames(donor="financingProvider", recipient="financingRecipient")
            for dcid in family
        },
    )

    result = materialise(
        shape,
        [*bindings, source_binding(IMPUTED_PROVENANCE)],
        {"financingRecipient": ["country/KEN"]},
        graph=FakeGraph(
            nodes={
                row.dcid: {
                    "label": row.dcid,
                    "type": "StatisticalVariable",
                    "arcs": _index_row_arcs(row),
                }
                for row in get_index().statvars
                if row.dcid in family
            },
            obs={},
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=FakeCoverageService(coverage).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.member_facts is not None
    assert set(result.member_facts) == family

    admitted = _admit_standard_members(
        dataclasses.replace(shape, declared_roles=None),
        [*bindings, source_binding(IMPUTED_PROVENANCE)],
        (),
    )
    assert isinstance(admitted, _AdmittedStandardMembers)
    assert set(admitted.member_facts) == family


def test_a_source_that_publishes_no_member_reads_an_empty_index_population():
    shape, query = _education_shape(imputed_education_aid_identity(source=CRS_PROVENANCE))

    member_facts, index_sourced = shape_member_facts(shape, query)

    assert member_facts == {}
    assert index_sourced is True


def test_options_population_reads_the_dataset_the_identity_names():
    plain = BaseIdentity.model_validate(education_aid_identity())
    named = BaseIdentity.model_validate(imputed_education_aid_identity(source=IMPUTED_PROVENANCE))
    graph = FakeMintGraphReader(arcs={})

    assert len(options_population(plain, graph=graph).stat_var_dcids) == CRS_MEMBER_COUNT
    named_population = options_population(named, graph=graph)
    assert len(named_population.stat_var_dcids) == IMPUTED_MEMBER_COUNT
    assert set(named_population.stat_var_dcids) <= _imputed_rows()
    assert _population_key(plain) != _population_key(named)


def test_a_set_pinned_identity_keys_its_options_population_apart_from_the_unpinned_one():
    plain = education_aid_identity()
    pinned = education_aid_identity()
    pinned["facets"] = {
        "provenance": {"state": "set", "refs": sorted([CRS_PROVENANCE, IMPUTED_PROVENANCE])}
    }
    pinned_identity = BaseIdentity.model_validate(pinned)

    assert identity_query(pinned_identity).scope is not None
    assert _population_key(BaseIdentity.model_validate(plain)) != _population_key(pinned_identity)


def test_the_measure_family_holds_one_datasets_rows():
    shape, query = _education_shape(education_aid_identity())
    index = get_index()

    partitions = measure_family_partitions(shape, index, get_measure_index().axes, query)

    members = {dcid for partition in partitions for dcid in partition.members}
    assert members
    assert members.isdisjoint(_imputed_rows())
    provenances = {row.provenances for row in index.statvars if row.dcid in members}
    assert provenances == {(CRS_PROVENANCE,)}


def test_five_tuple_member_facts_reads_every_row_of_the_shapes_family():
    """A shape with the CRS slot keys reads every CRS row of its five-tuple, and no DAC1 row
    of the same five-tuple."""
    shape, _ = _education_shape(education_aid_identity())
    shape = dataclasses.replace(
        shape,
        slot_keys=index_slot_keys(
            population_type="FinancialFlow",
            measured_property="amount",
            measurement_qualifier="Nominal",
        ),
    )
    five_tuple_rows = {
        row.dcid
        for row in get_index().statvars
        if (row.population_type, row.measured_property, row.measurement_qualifier)
        == ("FinancialFlow", "amount", "Nominal")
        and CRS_PROVENANCE in row.provenances
    }

    member_facts, index_sourced = five_tuple_member_facts(shape)

    assert index_sourced is True
    assert set(member_facts) == five_tuple_rows
    assert set(member_facts) > _search_family(education_aid_identity())
    assert five_tuple_rows.isdisjoint(_imputed_rows())


def test_slot_taxonomy_offers_the_one_value_of_a_slot_the_five_tuple_pins():
    shape, _ = _education_shape(imputed_education_aid_identity())
    shape = dataclasses.replace(
        shape, slot_keys=(SlotKeyDraft(axis="how", property_dcid=AID_TYPE, label="aid type"),)
    )

    taxonomy = _shape_with_taxonomy(shape).slot_taxonomy

    assert taxonomy is not None
    assert taxonomy[f"how:{AID_TYPE}"] == [IMPUTED_MULTILATERAL]


def test_the_plain_aid_identity_declares_no_aid_type_slot():
    """CRS rows declare no aid type, so the plain CRS identity has no slot for it."""
    spec_id, definition = mint_identity(education_aid_identity())

    assert spec_id == EDUCATION_AID_SPEC_ID
    assert AID_TYPE not in {slot.property for slot in definition.slots}


# ---------------------------------------------------------------------------
# The recall pool of a sample leaves out members of a dataset the query did
# not ask for.
# ---------------------------------------------------------------------------


def _arcs_carrying(constraints: dict[str, str]) -> dict:
    """Recall arcs for a demographic StatVar the packaged index does not carry."""
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


def _recall_only_shape(sv_arc_facts: dict[str, dict], *, complete: bool) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_recall_only_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="unemploymentRate",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts=sv_arc_facts,
        sv_arc_facts_complete=complete,
    )


_RECALL_POOL = {
    "sv_plain": _arcs_carrying({"gender": "Female"}),
    "sv_imputed": _arcs_carrying({"gender": "Female", AID_TYPE: IMPUTED_MULTILATERAL}),
}


@pytest.mark.parametrize(
    ("bound_values", "source"),
    [({AID_TYPE: [IMPUTED_MULTILATERAL]}, None), ({}, IMPUTED_PROVENANCE)],
    ids=["aid_type_bound", "source_named"],
)
def test_the_recall_pool_keeps_members_of_a_dataset_the_query_asks_for(
    bound_values: dict[str, list[str]], source: str | None
):
    shape = _recall_only_shape(_RECALL_POOL, complete=False)

    query = dataclasses.replace(query_for(shape, bound_values), source=source)

    member_facts, index_sourced = shape_member_facts(shape, query)

    assert index_sourced is False
    assert set(member_facts) == {"sv_plain", "sv_imputed"}


def test_an_empty_index_match_reads_no_recall_member():
    """The index decides the query's membership and matches no row, so the population is
    empty, as the mint's is for the same identity."""
    population = _crs_population()
    shape = _crs_shape("test_empty_index_match", {population[0]: {}}, complete=False)

    member_facts, index_sourced = shape_member_facts(
        shape, query_for(shape, {"crsPurpose": ["NoSuchPurposeCode"]})
    )

    assert member_facts == {}
    assert index_sourced is True


def _index_row_arcs(row: StatVarRow) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in row.constraint_properties]},
        "observationProperties": {"nodes": [{"dcid": role} for role in row.observation_properties]},
        **{prop: {"nodes": [{"dcid": v}]} for prop, v in row.constraint_properties.items()},
    }


def test_an_exact_posted_set_is_returned_as_posted():
    shape = _recall_only_shape(_RECALL_POOL, complete=True)

    member_facts, index_sourced = shape_member_facts(shape, query_for(shape, {}))

    assert index_sourced is False
    assert set(member_facts) == {"sv_plain", "sv_imputed"}


# ---------------------------------------------------------------------------
# The trace's declaration version follows the dataset declarations.
# ---------------------------------------------------------------------------


def _plain_aid_identity() -> BaseIdentity:
    """The aid identity with every code-system slot unbound: a plain CRS family."""
    document = education_aid_identity()
    for slot in document["slots"]:
        if slot["property"] in get_index().declarations:
            slot["binding"] = {"state": "unbound"}
    return BaseIdentity.model_validate(document)


def _index_with_an_edited_dataset_declaration():
    document = json.loads(gzip.decompress(default_index_path().read_bytes()))
    imputed = next(d for d in document["datasets"] if d["provenance"] == IMPUTED_PROVENANCE)
    imputed["label"] = f"{imputed['label']} (edited)"
    blocks = {key: value for key, value in document.items() if key != "stamp"}
    document["stamp"]["content_hash"] = content_hash(blocks)
    return parse_index(document)


def test_declaration_version_moves_with_a_dataset_declaration(monkeypatch: pytest.MonkeyPatch):
    edited = _index_with_an_edited_dataset_declaration()
    identity = BaseIdentity.model_validate(education_aid_identity())
    bindings = [
        SlotBindingDraft(
            axis="how", property_dcid="crsPurpose", kind="value", value_dcids=["CRSPurposeCode110"]
        )
    ]

    packaged_version = declaration_version(bindings, identity)
    monkeypatch.setattr(bound_values_module, "get_index", lambda: edited)
    edited_version = declaration_version(bindings, identity)

    assert packaged_version == get_index().stamp.content_hash
    assert edited_version is not None
    assert edited_version != packaged_version


def test_a_plain_identity_the_index_resolves_carries_the_declaration_version(
    monkeypatch: pytest.MonkeyPatch,
):
    """The dataset declarations elect the family of an identity that binds no code-system
    property, so its trace carries their version and the version moves with them."""
    identity = _plain_aid_identity()
    assert index_decides_membership(identity)

    packaged_version = declaration_version([], identity)
    monkeypatch.setattr(bound_values_module, "get_index", _index_with_an_edited_dataset_declaration)
    edited_version = declaration_version([], identity)

    assert packaged_version == get_index().stamp.content_hash
    assert edited_version is not None
    assert edited_version != packaged_version
