"""Search's first resolve elects the dataset, the family and the pins every identity door
reads back from the identity it mints.

The first resolve keeps the part's candidate shapes (`_dataset_candidates`), withholds the
invariants every dataset of the part declares (`shape_invariants`), takes the model's
codes, puts each slot an invariant of the elected dataset fixes in canonical form
(`canonical_pin_bindings`), reads
the family (`shape_member_facts`) and mints the identity from the bindings (`build_slot`,
`build_spec_identity`). The Datasets mint
(`resolve_definition_members`), the options door (`options_population`) and the resubmit
(`canonicalise_identity`, `refuse_source_outside_family`) then read that identity, and the
same identity with every pin bound. Each cell of the grid asserts the dataset, the members
and the pins it expects, and that every door agrees with the first resolve on all three.

The fixture index stands in for the aid tables: `_BROAD` (rank 0) has donor-and-recipient
rows; `_DETAIL` (rank 1) shares the five-tuple and one row with it, alone carries the
`channel` code system, and pins `flow`; `_DONORS` (rank 2) has donor-only rows on the same
five-tuple. The opt-in `_BESIDE` publishes beside them, and the opt-in `_ALONE` alone
publishes a second five-tuple. `_BROAD` alone publishes a third. On a fourth, `_BROAD`'s
donor-and-recipient rows and `_DONORS`' donor-only rows differ on the measure axis
`netting`. On a fifth, `_TOTALS` publishes donor-only rows without `sector` beside
`_BROAD`'s rows by sector and channel. The perturbation tests show which cells the rank
order, opt-in and the exact-role-set preference decide. The repair-phase tests drive
`_phase_repair_bindings` from places the question names. The drill-down tests at the end
read the answer's slot menus (`offers_slot`) and each option's coverage under the dataset
its election serves (`option_families`) on the fifth five-tuple.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import sys
from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

import pytest

from code_index import members as members_module
from code_index import shared as shared_module
from code_index.members import (
    FamilyQuery,
    canonical_identity,
    elected_pins,
    family_for_five_tuple,
    identity_family_query,
    invariant_bindings,
    option_families,
)
from code_index.query import CodeIndex
from code_index.types import (
    CodeRow,
    CodeSystemDeclaration,
    DatasetDeclaration,
    IndexStamp,
    MeasureAxis,
    MeasureIndex,
    StatVarRow,
)
from datasets_api.adapter import resolve_definition_members
from dfo_platform.roles import RoleAxis
from entity_resolution import Admission, EntityResolver
from materialise.facets import ElectedDataset
from search.engine.bind import SlotBindingDraft, _build_universal_bind_prompt, _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.dataset_mention import DatasetMention
from search.engine.interpret import Recall
from search.engine.options.population import options_population
from search.engine.qualifier_match import recognise_spans
from search.engine.region import option_coverage as option_coverage_module
from search.engine.region import phase_election as phase_election_module
from search.engine.region import phase_rank_bind as phase_rank_bind_module
from search.engine.region import related_read as related_read_module
from search.engine.region.alternate_guard import AlternateJudge, inadmissible_reason, query_pins
from search.engine.region.grounding import _ground_answer
from search.engine.region.option_coverage import (
    RoleServedFilter,
    _maybe_enrich_slot_options,
    option_axis_facts,
)
from search.engine.region.phase_election import _phase_repair_bindings
from search.engine.region.phase_rank_bind import (
    _dataset_candidates,
    _phase_prune_and_rank,
    _undeclared_qualifiers,
)
from search.engine.region.place_context import empty_place_context
from search.engine.region.place_roles import (
    PlaceRoles,
    bind_place_roles,
    bound_after_election,
    capped_place_dcids,
)
from search.engine.region.related_read import start_elected_related_read
from search.engine.region.sentence_events import _build_shape_event
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import (
    SeriesRows,
    bind_shared_values,
    canonical_pin_bindings,
    canonicalise_identity,
    elected_role_class_shape,
    entity_role_keys,
    identity_dataset,
    identity_query,
    refuse_source_outside_family,
    shape_invariants,
    shape_member_facts,
    shape_query,
    source_binding,
)
from search.engine.retrieval.member_facts import IndexMemberFacts, MemberFacts
from search.engine.retrieval.read_plan import PairPlan
from search.engine.retrieval.retrieve import Materialised
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft, SlotKeyDraft, identity_shape_for, is_role_bearing
from search.engine.spec_build import build_slot, pair_slot_drafts
from search.engine.spec_id import build_spec_identity, mint_identity
from search.models import (
    BaseIdentity,
    CoverageExact,
    Entity,
    GraphRef,
    IdentityBindingValue,
    SlotKey,
    SlotOption,
    SlotOptions,
    SlotValue,
    TimeWindow,
)
from search.models.entity import EntityRoleDirectional
from statvar_members import FiveTuple, MemberEnumerator
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.materialise.conftest import FakeGraphReader
from tests.statvar_members.conftest import FakeMemberArcReader

_BROAD = "provenance/BroadLedger"
_DETAIL = "provenance/DetailLedger"
_DONORS = "provenance/DonorLedger"
_ALONE = "provenance/AloneLedger"
_BESIDE = "provenance/BesideLedger"
_TOTALS = "provenance/TotalLedger"

_SHARED = ("TestFlow", "amount")
_OPT_IN_ONLY = ("TestFlow", "lineAmount")
_SOLE = ("TestFlow", "grantAmount")
_MEASURED = ("TestFlow", "netAmount")
_DRILLED = ("TestFlow", "drillAmount")
_PINNED = ("TestFlow", "pinnedAmount")

_DONOR = ("financingProvider",)
_DONOR_RECIPIENT = ("financingProvider", "financingRecipient")

_PROPERTIES = ("basis", "channel", "flow", "netting", "sector")
_SOURCE_KEY = SlotKeyDraft(axis="source", property_dcid=None, label="data source")


def _row(
    dcid: str,
    constraints: dict[str, str],
    *provenances: str,
    roles: tuple[str, ...] = _DONOR_RECIPIENT,
    five_tuple: tuple[str, str] = _SHARED,
) -> StatVarRow:
    population_type, measured_property = five_tuple
    return StatVarRow(
        dcid=dcid,
        population_type=population_type,
        measured_property=measured_property,
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=roles,
        constraint_properties=constraints,
        provenances=provenances,
    )


_ROWS = (
    # `_BROAD` holds `basis` G on every row, so it declares `basis` as an invariant.
    _row("broad_a", {"basis": "G", "flow": "F1", "sector": "A"}, _BROAD),
    _row("broad_b", {"basis": "G", "flow": "F2", "sector": "B"}, _BROAD),
    _row("both_c", {"basis": "G", "flow": "F3", "sector": "C"}, _BROAD, _DETAIL),
    # `_DETAIL` holds `flow` F3 on every row, so it declares `flow`; `channel` is its own.
    _row("detail_d_k1", {"basis": "H", "channel": "K1", "flow": "F3", "sector": "D"}, _DETAIL),
    _row("detail_d_k2", {"basis": "H", "channel": "K2", "flow": "F3", "sector": "D"}, _DETAIL),
    # `_DONORS` declares the donor alone and holds `basis` G and `flow` F1 on every row.
    _row("donors_a", {"basis": "G", "flow": "F1", "sector": "A"}, _DONORS, roles=_DONOR),
    _row("donors_b", {"basis": "G", "flow": "F1", "sector": "B"}, _DONORS, roles=_DONOR),
    # The opt-in `_BESIDE` alone carries sector E.
    _row("beside_a", {"basis": "G", "flow": "F9", "sector": "A"}, _BESIDE),
    _row("beside_e", {"basis": "G", "flow": "F9", "sector": "E"}, _BESIDE),
    _row("alone_a", {"basis": "G", "sector": "A"}, _ALONE, five_tuple=_OPT_IN_ONLY),
    _row("alone_b", {"basis": "G", "sector": "B"}, _ALONE, five_tuple=_OPT_IN_ONLY),
    _row("sole_a", {"basis": "G", "sector": "A"}, _BROAD, five_tuple=_SOLE),
    _row("sole_b", {"basis": "G", "sector": "B"}, _BROAD, five_tuple=_SOLE),
    # `_BROAD` reports `netting` gross, the measure vote's preferred value, and `_DONORS`
    # reports it net on its one row of the five-tuple.
    _row(
        "gross_a", {"basis": "G", "netting": "Gross", "sector": "A"}, _BROAD, five_tuple=_MEASURED
    ),
    _row(
        "gross_b", {"basis": "G", "netting": "Gross", "sector": "B"}, _BROAD, five_tuple=_MEASURED
    ),
    _row(
        "net_a",
        {"basis": "G", "flow": "F1", "netting": "Net", "sector": "A"},
        _DONORS,
        roles=_DONOR,
        five_tuple=_MEASURED,
    ),
    # `_TOTALS` publishes donor-only rows and carries no `sector`; `_BROAD` publishes the
    # same five-tuple by sector and channel for donor and recipient, and both record the
    # donor-only channel K3 row.
    _row("total", {}, _TOTALS, roles=_DONOR, five_tuple=_DRILLED),
    _row("drill_a", {"basis": "G", "channel": "K1", "sector": "A"}, _BROAD, five_tuple=_DRILLED),
    _row("drill_b", {"basis": "G", "channel": "K2", "sector": "B"}, _BROAD, five_tuple=_DRILLED),
    _row(
        "shared_k3",
        {"basis": "G", "channel": "K3"},
        _BROAD,
        _TOTALS,
        roles=_DONOR,
        five_tuple=_DRILLED,
    ),
    # `_BROAD` reports `flow` F2 alone and `_DETAIL` F3 alone, so each pins its own flow on
    # this five-tuple.
    _row("pinned_f2_a", {"basis": "G", "flow": "F2", "sector": "A"}, _BROAD, five_tuple=_PINNED),
    _row("pinned_f2_b", {"basis": "G", "flow": "F2", "sector": "B"}, _BROAD, five_tuple=_PINNED),
    _row("pinned_f3_a", {"basis": "G", "flow": "F3", "sector": "A"}, _DETAIL, five_tuple=_PINNED),
    _row("pinned_f3_b", {"basis": "G", "flow": "F3", "sector": "B"}, _DETAIL, five_tuple=_PINNED),
)

_CODES = {"channel": ("K1", "K2", "K3"), "sector": ("A", "B", "C", "D", "E")}

_RANKS = {_BROAD: 0, _DETAIL: 1, _DONORS: 2, _ALONE: 3, _BESIDE: 4, _TOTALS: 5}
_OPT_IN = frozenset({_ALONE, _BESIDE})


def _invariants(rows: tuple[StatVarRow, ...], provenance: str) -> dict[str, str]:
    """The constraint values every row of `provenance` in `rows` holds, which the fixture
    declares as the dataset's invariants (`DatasetDeclaration.invariants`)."""
    held = [row.constraint_properties for row in rows if provenance in row.provenances]
    if not held:
        return {}
    return {
        prop: value
        for prop, value in held[0].items()
        if all(constraints.get(prop) == value for constraints in held)
    }


def _index(
    *,
    ranks: Mapping[str, int] = _RANKS,
    opt_in: frozenset[str] = _OPT_IN,
    rows: tuple[StatVarRow, ...] = _ROWS,
    codes: Mapping[str, tuple[str, ...]] = _CODES,
    parents: Mapping[str, str] | None = None,
) -> CodeIndex:
    """The fixture index over `rows`, declaring `codes`. A code system with a code in
    `parents` is hierarchical, each such code a child of its parent. Each dataset declares
    as invariants the values every one of its rows holds (`_invariants`)."""
    parents = parents or {}
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=tuple(
            CodeSystemDeclaration(
                code_system=prop,
                hierarchical=any(code in parents for code in codes.get(prop, ())),
                disjoint_from_children=False,
                summable=True,
                absent_reading="not_declared",
                plain_name=prop,
            )
            for prop in _PROPERTIES
        ),
        codes=tuple(
            CodeRow(
                dcid=code,
                code_system=prop,
                code=code,
                label=code,
                synonyms=(),
                parent=parents.get(code),
            )
            for prop, system in codes.items()
            for code in system
        ),
        statvars=rows,
        datasets=tuple(
            DatasetDeclaration(
                provenance=provenance,
                label=provenance,
                synonyms=(),
                opt_in=provenance in opt_in,
                absent_observation="zero",
                rank=rank,
                invariants=_invariants(rows, provenance),
            )
            for provenance, rank in ranks.items()
        ),
    )


def _use_index(monkeypatch: pytest.MonkeyPatch, index: CodeIndex) -> None:
    """Serve `index` wherever a loaded module reads the packaged index."""
    packaged = shared_module.get_index
    for module in list(sys.modules.values()):
        if vars(module).get("get_index") is packaged:
            monkeypatch.setattr(module, "get_index", lambda: index)


@pytest.fixture
def index(monkeypatch: pytest.MonkeyPatch) -> CodeIndex:
    index = _index()
    _use_index(monkeypatch, index)
    return index


def _derived_shape(
    five_tuple: tuple[str, str], *, index_rows: tuple[StatVarRow, ...] = _ROWS
) -> ShapeDraft:
    """A shape as `derive_shapes` builds it on `five_tuple`: the slot keys and taxonomy of
    every row of `index_rows` on the five-tuple, whatever dataset publishes it, and a source
    slot."""
    population_type, measured_property = five_tuple
    rows = [row for row in index_rows if (row.population_type, row.measured_property) == five_tuple]
    props = sorted({prop for row in rows for prop in row.constraint_properties})
    return ShapeDraft(
        shape_id="_".join(five_tuple),
        label=measured_property,
        pop_type_dcid=population_type,
        meas_prop_dcid=measured_property,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            *(SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in props),
            _SOURCE_KEY,
        ),
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


def _mention(source: str | None) -> DatasetMention | None:
    if source is None:
        return None
    return DatasetMention(
        provenance=source, start=0, end=len(source), text=source, name=source, measure_words=()
    )


# The entities a question names, keyed as `ShapeDraft.entities_by_role` keys them.
_ENTITIES: dict[str, dict[str, list[str]]] = {
    "none": {},
    "donor": {"financingProvider": ["country/DEU"]},
    "donor-recipient": {
        "financingProvider": ["country/DEU"],
        "financingRecipient": ["country/KEN"],
    },
}
_AXES: dict[str, RoleAxis] = {"financingProvider": "origin", "financingRecipient": "destination"}


def _identity_entities(roles: str) -> list[Entity]:
    return [
        Entity(
            ref=GraphRef(dcid=dcid, label=dcid),
            role=EntityRoleDirectional(role=GraphRef(dcid=role, label=role), axis=_AXES[role]),
        )
        for role, dcids in _ENTITIES[roles].items()
        for dcid in dcids
    ]


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What a first resolve settles for one part."""

    datasets: tuple[str, ...] | None
    """The datasets candidacy stamps on the part's shape; None when it keeps no shape."""
    withheld: Mapping[str, str] = dataclasses.field(default_factory=dict)
    elected: str | None = None
    members: Mapping[str, ElectedDataset | None] = dataclasses.field(default_factory=dict)
    identity: BaseIdentity | None = None
    spec_id: str | None = None
    shape: ShapeDraft | None = None
    """The part's candidate shape, stamped with its datasets."""
    bindings: tuple[SlotBindingDraft, ...] = ()
    """The bindings the identity is minted from."""
    pins: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """The elected dataset's pins (`elected_pins`) on the identity's query."""


def _first_resolve(
    five_tuple: tuple[str, str],
    source: str | None,
    roles: str,
    model: Mapping[str, str],
    *,
    window: TimeWindow | None = None,
) -> Outcome:
    """Run the first resolve of one part on `five_tuple`, naming `source` or none, about the
    entities of `roles`, where the bind model binds `model`."""
    candidates = _dataset_candidates([_derived_shape(five_tuple)], _mention(source))
    if not candidates:
        return Outcome(datasets=None)
    (shape,) = candidates
    taxonomy = shape.slot_taxonomy or {}
    for prop, value in model.items():
        assert value in taxonomy[f"how:{prop}"], f"the bind model is not offered {prop}={value}"

    drafted = [
        SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])
        for prop, value in model.items()
    ]
    if source is not None:
        drafted.append(source_binding(source))
    bindings, _shared = bind_shared_values(shape, drafted)
    role_keys = entity_role_keys(shape, _ENTITIES[roles])
    bindings = canonical_pin_bindings(shape, bindings, role_keys)
    query = shape_query(shape, bindings, role_keys)
    facts, index_sourced = shape_member_facts(shape, query)
    assert index_sourced
    members = {dcid: fact.read_dataset() for dcid, fact in facts.items()}
    elected = {dataset.provenance for dataset in members.values() if dataset is not None}
    assert len(elected) <= 1

    slots = [
        build_slot(
            key,
            draft,
            [GraphRef(dcid=value, label=value) for value in draft.value_dcids]
            if draft is not None
            else [],
            property_ref=GraphRef(dcid=key.property_dcid, label=key.property_dcid)
            if key.property_dcid is not None
            else None,
        )
        for key, draft in pair_slot_drafts(shape.slot_keys, bindings)
    ]
    spec_id, identity = build_spec_identity(
        identity_shape_for(
            pop_type_dcid=shape.pop_type_dcid,
            meas_prop_dcid=shape.meas_prop_dcid,
            stat_type_dcid=shape.stat_type_dcid,
            meas_qual_dcid=None,
            meas_denom_dcid=None,
        ),
        slots,
        entities=_identity_entities(roles),
        requested_window=window,
    )
    assert identity_query(identity) == query
    return Outcome(
        datasets=shape.datasets,
        withheld=dict(shape_invariants(shape)),
        elected=next(iter(elected), None),
        members=members,
        identity=identity,
        spec_id=spec_id,
        shape=shape,
        bindings=tuple(bindings),
        pins=elected_pins(shared_module.get_index(), query),
    )


def _mint(identity: BaseIdentity, index: CodeIndex) -> dict[str, ElectedDataset | None]:
    graph = FakeMintGraphReader(arcs={})
    resolved = resolve_definition_members(
        identity.model_dump(mode="json", exclude_none=True),
        spec_id="spec_test",
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=index,
    )
    assert sum(graph.calls.values()) == 0
    return {shape.stat_var.removeprefix("dcid:"): shape.dataset for shape in resolved.shapes}


def _bound_slots(identity: BaseIdentity) -> dict[str, str]:
    return {
        slot.property: slot.binding.ref
        for slot in identity.slots
        if isinstance(slot.binding, IdentityBindingValue)
    }


def _with_pins_bound(identity: BaseIdentity, pins: Mapping[str, str | None]) -> BaseIdentity:
    """`identity` with each slot it declares that `pins` names bound to the pinned value, or
    unbound where `pins` maps it to `None`."""
    document = identity.model_dump(mode="json", exclude_none=True)
    for slot in document["slots"]:
        if slot["property"] in pins:
            value = pins[slot["property"]]
            slot["binding"] = (
                {"state": "unbound"} if value is None else {"state": "value", "ref": value}
            )
    return mint_identity(document)[1]


def _assert_doors_agree(outcome: Outcome, index: CodeIndex) -> None:
    """The mint, the options door and the resubmit read the first resolve's identity back
    to its dataset, members and pins. The identity is canonical. An identity that names its
    dataset, posted with every invariant of it unbound, canonicalises to it; one that names
    none, posted with every pin bound, keeps its own bindings. Either reaches the members
    at every door."""
    identity = outcome.identity
    assert identity is not None
    query = identity_query(identity)
    named = query.source is not None
    invariants = index.datasets[outcome.elected].invariants
    pinned = _with_pins_bound(
        identity,
        dict.fromkeys(invariants) if named else dict(outcome.pins),
    )
    for posted in (identity, pinned):
        assert identity_dataset(posted) == outcome.elected
        assert _mint(posted, index) == outcome.members
        population = options_population(posted, FakeMemberArcReader(arcs={}))
        assert population.source == "index"
        assert dict(population.datasets) == outcome.members
        assert canonicalise_identity(posted) == (identity if named else posted)
        refuse_source_outside_family(posted, pointer="/input/identity/facets/provenance")
    assert invariant_bindings(index, query) == {}
    assert elected_pins(index, query) == outcome.pins


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Expected:
    elected: str
    members: Mapping[str, tuple[str, ...]]
    """Each member mapped to the datasets that record it."""
    pins: Mapping[str, str]


@dataclasses.dataclass(frozen=True)
class Scenario:
    five_tuple: tuple[str, str]
    source: str | None
    sector: str
    """The sector code the bind model binds."""
    datasets: tuple[str, ...] | None
    withheld: Mapping[str, str]
    by_roles: Mapping[str, Expected]


_BROAD_PINS = {"basis": "G"}
_DONORS_PINS = {"basis": "G", "flow": "F1"}

# Keyed by (named, publishers, opt-in). An unnamed opt-in scenario binds a code an opt-in
# dataset also carries; a named one names the opt-in dataset.
_SCENARIOS: dict[tuple[str, str, str], Scenario] = {
    ("unnamed", "two", "not-opt-in"): Scenario(
        five_tuple=_SHARED,
        source=None,
        sector="C",
        datasets=(_BROAD, _DETAIL, _DONORS),
        withheld={},
        by_roles={
            # `_DONORS` has no sector C row, so neither role set places exactly and the
            # rank serves the shared row from `_BROAD`.
            "donor": Expected(_BROAD, {"both_c": (_BROAD, _DETAIL)}, _BROAD_PINS),
            "donor-recipient": Expected(_BROAD, {"both_c": (_BROAD, _DETAIL)}, _BROAD_PINS),
        },
    ),
    ("unnamed", "two", "opt-in"): Scenario(
        five_tuple=_SHARED,
        source=None,
        sector="A",
        datasets=(_BROAD, _DETAIL, _DONORS),
        withheld={},
        by_roles={
            "donor": Expected(_DONORS, {"donors_a": (_DONORS,)}, _DONORS_PINS),
            "donor-recipient": Expected(_BROAD, {"broad_a": (_BROAD,)}, _BROAD_PINS),
        },
    ),
    ("unnamed", "one", "not-opt-in"): Scenario(
        five_tuple=_SOLE,
        source=None,
        sector="A",
        datasets=(_BROAD,),
        withheld={},
        by_roles={
            "donor": Expected(_BROAD, {"sole_a": (_BROAD,)}, _BROAD_PINS),
            "donor-recipient": Expected(_BROAD, {"sole_a": (_BROAD,)}, _BROAD_PINS),
        },
    ),
    ("unnamed", "one", "opt-in"): Scenario(
        five_tuple=_OPT_IN_ONLY,
        source=None,
        sector="A",
        datasets=None,
        withheld={},
        by_roles={},
    ),
    ("named", "two", "not-opt-in"): Scenario(
        five_tuple=_SHARED,
        source=_DETAIL,
        sector="C",
        datasets=(_DETAIL,),
        withheld={"flow": "F3"},
        by_roles={
            "donor": Expected(_DETAIL, {"both_c": (_BROAD, _DETAIL)}, {"flow": "F3"}),
            "donor-recipient": Expected(_DETAIL, {"both_c": (_BROAD, _DETAIL)}, {"flow": "F3"}),
        },
    ),
    ("named", "two", "opt-in"): Scenario(
        five_tuple=_SHARED,
        source=_BESIDE,
        sector="A",
        datasets=(_BESIDE,),
        withheld={"basis": "G", "flow": "F9"},
        by_roles={
            "donor": Expected(_BESIDE, {"beside_a": (_BESIDE,)}, {"basis": "G", "flow": "F9"}),
            "donor-recipient": Expected(
                _BESIDE, {"beside_a": (_BESIDE,)}, {"basis": "G", "flow": "F9"}
            ),
        },
    ),
    ("named", "one", "not-opt-in"): Scenario(
        five_tuple=_SOLE,
        source=_BROAD,
        sector="A",
        datasets=(_BROAD,),
        withheld=_BROAD_PINS,
        by_roles={
            "donor": Expected(_BROAD, {"sole_a": (_BROAD,)}, _BROAD_PINS),
            "donor-recipient": Expected(_BROAD, {"sole_a": (_BROAD,)}, _BROAD_PINS),
        },
    ),
    ("named", "one", "opt-in"): Scenario(
        five_tuple=_OPT_IN_ONLY,
        source=_ALONE,
        sector="A",
        datasets=(_ALONE,),
        withheld=_BROAD_PINS,
        by_roles={
            "donor": Expected(_ALONE, {"alone_a": (_ALONE,)}, _BROAD_PINS),
            "donor-recipient": Expected(_ALONE, {"alone_a": (_ALONE,)}, _BROAD_PINS),
        },
    ),
}


@dataclasses.dataclass(frozen=True)
class Cell:
    id: str
    scenario: Scenario
    roles: str
    model_binds_pins: bool
    """Whether the bind model binds the elected dataset's pins itself."""

    @property
    def expected(self) -> Expected | None:
        return self.scenario.by_roles.get(self.roles)

    @property
    def model(self) -> dict[str, str]:
        pins = self.expected.pins if self.expected and self.model_binds_pins else {}
        return {"sector": self.scenario.sector, **pins}

    def resolve(self) -> Outcome:
        return _first_resolve(
            self.scenario.five_tuple, self.scenario.source, self.roles, self.model
        )


_CELLS = {
    cell.id: cell
    for (key, scenario), roles, pins in itertools.product(
        _SCENARIOS.items(), ("donor", "donor-recipient"), ("pins", "none")
    )
    for cell in [
        Cell(
            id="-".join((*key, roles, pins)),
            scenario=scenario,
            roles=roles,
            model_binds_pins=pins == "none",
        )
    ]
}


def _assert_cell(cell: Cell, outcome: Outcome) -> None:
    assert outcome.datasets == cell.scenario.datasets
    expected = cell.expected
    if expected is None:
        return
    assert outcome.withheld == cell.scenario.withheld
    assert outcome.elected == expected.elected
    assert outcome.members == {
        dcid: ElectedDataset(provenance=expected.elected, member_provenances=recorded)
        for dcid, recorded in expected.members.items()
    }
    assert outcome.identity is not None
    assert outcome.pins == expected.pins
    bound = _bound_slots(outcome.identity)
    if cell.scenario.source is None:
        # A part that names no dataset keeps the pins the model bound, and binds no other.
        assert {prop: bound.get(prop) for prop in expected.pins} == {
            prop: cell.model.get(prop) for prop in expected.pins
        }
    else:
        # The identity names the dataset, so it binds every invariant of it.
        assert {prop: bound.get(prop) for prop in expected.pins} == expected.pins
    facets = outcome.identity.model_dump(mode="json", exclude_none=True).get("facets", {})
    # The identity binds the dataset only when the part names it.
    assert facets.get("provenance", {}).get("ref") == cell.scenario.source


@pytest.mark.parametrize("cell", _CELLS.values(), ids=_CELLS.keys())
def test_the_first_resolve_and_every_identity_door_elect_one_dataset(index, cell: Cell):
    """An unnamed part never answers from an opt-in dataset beside a default one, and drops a
    five-tuple only opt-in datasets publish. A donor-only question reads the donor-only
    dataset where it has the row, and the rank decides otherwise. A named part reads its
    dataset's rows, with that dataset's pins withheld from the model and left unbound."""
    outcome = cell.resolve()

    _assert_cell(cell, outcome)
    if outcome.identity is not None:
        _assert_doors_agree(outcome, index)


@pytest.mark.parametrize(
    "scenario_roles",
    [(key, roles) for key, scenario in _SCENARIOS.items() for roles in scenario.by_roles],
    ids=lambda pair: "-".join((*pair[0], pair[1])),
)
def test_the_identity_is_one_whether_or_not_the_model_binds_the_pins(index, scenario_roles):
    """A part that names its dataset mints one identity whether or not the model binds the
    dataset's invariants. One that names none keeps what the model bound, so the two mint
    two identities that serve the same members."""
    key, roles = scenario_roles
    with_pins = _CELLS["-".join((*key, roles, "none"))].resolve()
    without = _CELLS["-".join((*key, roles, "pins"))].resolve()

    assert with_pins.members == without.members
    assert (with_pins.spec_id == without.spec_id) is (_SCENARIOS[key].source is not None)


def test_a_code_only_the_second_dataset_carries_elects_it_at_every_door(index):
    """The unnamed part's bind model sees `_DETAIL`'s own `channel` codes. Binding one
    elects `_DETAIL` over the first-ranked `_BROAD`, and every door serves `_DETAIL`. The part
    names no dataset, so the identity binds only what the model bound."""
    outcome = _first_resolve(_SHARED, None, "donor-recipient", {"channel": "K1"})

    assert outcome.elected == _DETAIL
    assert outcome.members == {
        "detail_d_k1": ElectedDataset(provenance=_DETAIL, member_provenances=(_DETAIL,))
    }
    assert outcome.identity is not None
    assert _bound_slots(outcome.identity) == {"channel": "K1"}
    assert outcome.pins == {"flow": "F3"}
    _assert_doors_agree(outcome, index)


@pytest.mark.parametrize(
    ("named", "refused"),
    [(_DONORS, True), (_DETAIL, False)],
    ids=["named-table-without-the-code", "named-table-with-the-code"],
)
def test_a_named_part_refuses_a_code_its_table_does_not_carry(index, named, refused):
    """`channel` K1 is a code only `_DETAIL`'s rows of `_SHARED` carry. The part naming
    `_DONORS` keeps the shared shape, whose slot keys are the whole five-tuple's, yet its
    table holds no `channel` code, so the term refuses `qualifier_not_bound` before bind.
    The part naming `_DETAIL` can bind it."""
    (shape,) = _dataset_candidates([_derived_shape(_SHARED)], _mention(named))
    spans = recognise_spans("K1 aid", index, MeasureIndex(axes=(), values=(), partition_labels=()))
    assert [row.code_system for span in spans for row in span.rows] == ["channel"]

    undeclared = _undeclared_qualifiers(spans, [shape])

    assert [record.text for record in undeclared] == (["K1"] if refused else [])


@pytest.mark.parametrize(
    ("named", "variable", "refused"),
    [
        (_DETAIL, "sector A aid", True),
        (_DETAIL, "B aid", True),
        (_BROAD, "B aid", False),
        (_BROAD, "D aid", True),
    ],
    ids=[
        "detail-sector-a",
        "detail-sector-only-broad-holds",
        "broad-own-sector",
        "broad-sector-only-detail-holds",
    ],
)
def test_a_named_part_refuses_a_code_only_another_table_holds_before_bind(
    index, monkeypatch, named, variable, refused
):
    """`_DETAIL` and `_BROAD` both carry `sector` on `_SHARED`, but no `_DETAIL` row holds
    sector A or B and no `_BROAD` row holds sector D. A part naming one table and a sector
    only the other holds refuses `qualifier_not_bound`, naming the sector, before the bind
    call. A part naming the table that holds the sector goes on to bind."""
    state = _repair_state(_derived_shape(_SHARED), {}, {})
    state.variable = variable
    state.shapes_found = [_derived_shape(_SHARED)]
    monkeypatch.setattr(phase_rank_bind_module, "_read_mention", lambda _q: _mention(named))
    monkeypatch.setattr(phase_rank_bind_module, "_record_shape_step", _no_record)

    result = asyncio.run(_phase_prune_and_rank(state))

    if refused:
        assert result is not None
        assert result.no_data_reason == "qualifier_not_bound"
        (record,) = result.unbound_qualifiers
        assert [ref.dcid for ref in record.named] == [variable.split()[-2]]
    else:
        assert result is None
        assert state.capped_shapes


async def _no_record(_state: ResolveState) -> None:
    return None


def test_a_donor_only_question_reads_the_donor_only_dataset_though_it_ranks_last(index):
    """`_BROAD` ranks first and carries sector A for donor and recipient. A question naming
    the donor alone reads `_DONORS`, whose rows declare exactly that role."""
    outcome = _first_resolve(_SHARED, None, "donor", {"sector": "A"})

    assert _RANKS[_DONORS] > _RANKS[_BROAD]
    assert outcome.elected == _DONORS
    assert set(outcome.members) == {"donors_a"}
    _assert_doors_agree(outcome, index)


def test_the_election_reads_no_observation(index):
    """No step of the election takes coverage or observations, so a requested year only
    `_DETAIL` might cover still elects `_BROAD` for the shared row. The fixture index
    carries no observations. The window changes the identity and nothing the election
    reads."""
    outcome = _first_resolve(
        _SHARED, None, "donor-recipient", {"sector": "C"}, window=TimeWindow(start_year=2030)
    )
    plain = _first_resolve(_SHARED, None, "donor-recipient", {"sector": "C"})

    assert outcome.elected == plain.elected == _BROAD
    assert outcome.members == plain.members
    assert outcome.spec_id != plain.spec_id
    _assert_doors_agree(outcome, index)


# ---------------------------------------------------------------------------
# Perturbations: each turns the cells it decides red
# ---------------------------------------------------------------------------


def _red_cells() -> set[str]:
    red = set()
    for cell_id, cell in _CELLS.items():
        try:
            _assert_cell(cell, cell.resolve())
        except AssertionError:
            red.add(cell_id)
    return red


def test_the_grid_is_green_on_the_fixture_index(index):
    assert _red_cells() == set()


def test_reversing_the_ranks_turns_the_rank_decided_cells_red(monkeypatch):
    """With the ranks reversed, the shared row of sector C is served by `_DETAIL`."""
    _use_index(monkeypatch, _index(ranks={p: len(_RANKS) - r for p, r in _RANKS.items()}))

    red = _red_cells()

    assert {
        "unnamed-two-not-opt-in-donor-pins",
        "unnamed-two-not-opt-in-donor-none",
        "unnamed-two-not-opt-in-donor-recipient-pins",
        "unnamed-two-not-opt-in-donor-recipient-none",
    } <= red
    reversed_outcome = _CELLS["unnamed-two-not-opt-in-donor-recipient-pins"].resolve()
    assert reversed_outcome.elected == _DETAIL


def test_treating_every_dataset_as_not_opt_in_turns_the_unnamed_cells_red(monkeypatch):
    """Every unnamed part on the shared five-tuple would answer from `_BESIDE`'s rows too,
    and an unnamed part would keep the five-tuple only `_ALONE` publishes."""
    _use_index(monkeypatch, _index(opt_in=frozenset()))

    red = _red_cells()

    assert {
        cell_id
        for cell_id in _CELLS
        if cell_id.startswith("unnamed-") and _CELLS[cell_id].scenario.five_tuple != _SOLE
    } <= red
    assert _CELLS["unnamed-one-opt-in-donor-pins"].resolve().datasets == (_ALONE,)


def test_dropping_the_exact_role_set_preference_turns_the_donor_only_cells_red(monkeypatch):
    """Without it a donor-only question reads `_BROAD`, which ranks first."""
    _use_index(monkeypatch, _index())
    monkeypatch.setattr(members_module, "_places_exactly", lambda facts, rows, role_keys: True)

    red = _red_cells()

    assert {"unnamed-two-opt-in-donor-pins", "unnamed-two-opt-in-donor-none"} <= red
    assert _CELLS["unnamed-two-opt-in-donor-pins"].resolve().elected == _BROAD


# ---------------------------------------------------------------------------
# The repair phase: role keys from the question's places
# ---------------------------------------------------------------------------

_NETTING = MeasureAxis(axis="netting", kind="constraint", role="free", prefer=("Gross", "Net"))

# Each question's places, keyed as `_ENTITIES` is, with the axis the door read for each.
_PLACES: dict[str, dict[str, RoleAxis]] = {
    "none": {},
    "donor": {"Germany": "origin"},
    "donor-recipient": {"Germany": "origin", "Kenya": "destination"},
}
_DCIDS = {"Germany": "country/DEU", "Kenya": "country/KEN"}


def _flow_shape(five_tuple: tuple[str, str], *, roles: tuple[str, ...]) -> ShapeDraft:
    """The unnamed part's candidate shape on `five_tuple`, declaring `roles`, of which the
    recipient is optional."""
    (shape,) = _dataset_candidates([_derived_shape(five_tuple)], None)
    return dataclasses.replace(
        shape,
        declared_roles=roles,
        optional_roles=frozenset(roles) & {"financingRecipient"},
    )


def _repair_state(
    shape: ShapeDraft, places: Mapping[str, RoleAxis], model: Mapping[str, str]
) -> ResolveState:
    """The state `_phase_repair_bindings` reads, for a question naming `places` whose bind
    model binds `model` on `shape`."""
    return ResolveState(
        variable="aid",
        entities=list(places),
        date_request=None,
        detect_query="aid",
        role_query="aid from Germany to Kenya",
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        source_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        pipeline_steps=[],
        timing={},
        warnings=[],
        rcl=Recall(
            candidates=[],
            resolutions={},
            resolved_entity_names={name: _DCIDS[name] for name in places},
        ),
        entity_roles={name: [DoorRole(axis=axis, span=name)] for name, axis in places.items()},
        bind_result=_UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])
                for prop, value in model.items()
            ],
            also_plausible=None,
            ask=None,
        ),
        shape_draft=shape,
        coverage=empty_coverage_snapshot(),
    )


def _repair(
    monkeypatch: pytest.MonkeyPatch, state: ResolveState, *axes: MeasureAxis
) -> list[SlotBindingDraft]:
    """Run `_phase_repair_bindings` under the measure `axes`, and return the bindings the
    qualifier dispositions read."""
    monkeypatch.setattr(
        phase_election_module,
        "get_measure_index",
        lambda: MeasureIndex(axes=axes, values=(), partition_labels=()),
    )
    disposed: list[list[SlotBindingDraft]] = []

    def _dispositions(state: ResolveState) -> list:
        disposed.append(list(state.bindings))
        return []

    monkeypatch.setattr(phase_election_module, "_resolve_query_qualifiers", _dispositions)
    assert asyncio.run(_phase_repair_bindings(state)) is None
    (bindings,) = disposed
    assert bindings == state.bindings
    return bindings


def _values(bindings: list[SlotBindingDraft]) -> dict[str, str]:
    return {
        binding.property_dcid: binding.value_dcids[0]
        for binding in bindings
        if binding.kind == "value" and binding.property_dcid is not None
    }


@pytest.mark.parametrize(
    ("places", "roles", "role_keys", "pins"),
    [
        ("donor", _DONOR_RECIPIENT, {"origin"}, _DONORS_PINS),
        ("donor-recipient", _DONOR_RECIPIENT, {"origin", "destination"}, _BROAD_PINS),
        # The recipient has no role on a donor-only shape, so the places refuse
        # `role_incompatible`, and the refusal's flagged identity binds no entity.
        ("donor-recipient", _DONOR, "role_incompatible", _BROAD_PINS),
        # A role-bearing shape takes no default scope, so a question naming no place binds
        # no entity and elects under no role key.
        ("none", _DONOR_RECIPIENT, set(), _BROAD_PINS),
    ],
    ids=["donor", "donor-recipient", "role-incompatible", "no-place"],
)
@pytest.mark.parametrize("model_binds_pins", [False, True], ids=["model-open", "model-pins"])
def test_the_repair_keeps_the_models_pins_before_the_qualifier_dispositions(
    index, monkeypatch, places, roles, role_keys, pins, model_binds_pins
):
    """`_DETAIL` holds `basis` H and `_DONORS` alone pins `flow`, so neither is a pin every
    dataset of the part shares. The part names no dataset, so the repair keeps the pins the
    model binds and binds none it leaves open, before qualifier dispositions read the
    bindings and before any refusal mints a flagged identity."""
    shape = _flow_shape(_SHARED, roles=roles)
    model = {"sector": "A", **(pins if model_binds_pins else {})}
    state = _repair_state(shape, _PLACES[places], model)
    bound = bind_place_roles(state, shape, places=capped_place_dcids(state)[0])
    if isinstance(role_keys, str):
        assert bound == role_keys
        role_keys = set()
    else:
        assert isinstance(bound, PlaceRoles)
        assert entity_role_keys(shape, bound.resolved_entities) == role_keys

    bindings = _repair(monkeypatch, state)

    assert _values(bindings) == {"sector": "A", **(pins if model_binds_pins else {})}
    query = shape_query(shape, bindings, role_keys)
    assert invariant_bindings(index, query) == {}
    assert elected_pins(index, query) == pins


def test_a_question_naming_no_place_elects_under_no_role_key_at_every_door(index):
    """A question naming no place mints an identity with no entity, so every door reads no
    role key. No row is subject-only, so every row stays in the election and the rank
    serves `_BROAD`: Search, the option coverage of the bound sector, the mint, the options
    door and a resubmit read the same dataset, family and pins, and the identity binds only
    the sector the question names."""
    outcome = _first_resolve(_SHARED, None, "none", {"sector": "A"})

    assert outcome.identity is not None
    query = identity_query(outcome.identity)
    assert query.role_keys == frozenset()
    assert outcome.elected == _BROAD
    assert outcome.members == {
        "broad_a": ElectedDataset(provenance=_BROAD, member_provenances=(_BROAD,))
    }
    assert _bound_slots(outcome.identity) == {"sector": "A"}
    assert outcome.pins == _BROAD_PINS
    families = option_families(index, query, "sector", ["A"])
    assert families is not None
    assert families["A"].provenance == _BROAD
    assert [row.dcid for row in families["A"].rows] == ["broad_a"]
    _assert_doors_agree(outcome, index)


def test_the_measure_vote_reads_the_dataset_the_role_keys_elect(index, monkeypatch):
    """`_BROAD` ranks first and reports gross. A donor-only question elects `_DONORS`, which
    reports net, so the vote forces net and the family stays `_DONORS`'. Electing by rank
    alone would force gross, which no `_DONORS` row holds, and gross would stay bound since
    releasing it elects `_DONORS`. `_DONORS`' one row of the five-tuple pins the netting and
    the sector beside its invariants, so no sentence states them, but the question names no
    dataset, so the identity keeps both bound."""
    shape = _flow_shape(_MEASURED, roles=_DONOR_RECIPIENT)
    state = _repair_state(shape, _PLACES["donor"], {"sector": "A"})

    bindings = _repair(monkeypatch, state, _NETTING)

    assert _values(bindings) == {"sector": "A", "netting": "Net"}
    assert state.measure_elected_partition is not None
    assert dict(state.measure_elected_partition.values) == {"netting": "Net"}
    query = shape_query(shape, bindings, {"origin"})
    family = family_for_five_tuple(index, query)
    assert family.provenance == _DONORS
    assert [row.dcid for row in family.rows] == ["net_a"]


def test_the_answer_reads_the_measure_partition_of_the_dataset_the_role_keys_elect(
    index, monkeypatch
):
    """The answer's measure partition, which the Spec and the related-measures read take
    (`start_elected_related_read`), is read under the role keys the identity binds its
    places under. A donor-only question serves `_DONORS`' net row. Read by rank alone, the
    family would be `_BROAD`'s, whose gross partitions hold no served member's values, and
    the answer would have no partition."""
    shape = _flow_shape(_MEASURED, roles=_DONOR_RECIPIENT)
    state = _repair_state(shape, _PLACES["donor"], {"sector": "A"})
    bindings = _repair(monkeypatch, state, _NETTING)
    roles = bind_place_roles(state, shape, places=capped_place_dcids(state)[0])
    assert isinstance(roles, PlaceRoles)
    query = shape_query(shape, bindings, entity_role_keys(shape, roles.resolved_entities))
    assert query.role_keys == {"origin"}
    state.shape_draft = elected_role_class_shape(shape, query).value
    state.place = dataclasses.replace(empty_place_context(), roles=roles)
    served, _ = shape_member_facts(state.shape_draft, query)
    monkeypatch.setattr(related_read_module, "start_related_read", lambda *_a, **_k: None)
    monkeypatch.setattr(
        related_read_module,
        "get_measure_index",
        lambda: MeasureIndex(axes=(_NETTING,), values=(), partition_labels=()),
    )

    start_elected_related_read(
        state, cast(Materialised, SimpleNamespace(member_facts=served, sv_dcids=list(served)))
    )

    partition, _siblings = state.measure_family
    assert partition is not None
    assert dict(partition.values) == {"netting": "Net"}
    assert partition.members == ("net_a",)


# ---------------------------------------------------------------------------
# Drill-down: slot menus and option coverage on a five-tuple two tables publish
# ---------------------------------------------------------------------------

_DONOR_PLACE = "country/DEU"

# Each `_DRILLED` member's facets at the donor, `_BROAD`'s and `_TOTALS`', by points held.
_DRILLED_POINTS = {
    "drill_a": {_BROAD: 2, _TOTALS: 9},
    "drill_b": {_BROAD: 4, _TOTALS: 8},
    "shared_k3": {_BROAD: 3, _TOTALS: 7},
    "total": {_TOTALS: 5},
}


def _arcs(row: StatVarRow) -> dict[str, dict[str, list[dict[str, str]]]]:
    """`row` as the recall reads its arcs."""
    return {
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in row.constraint_properties]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in row.constraint_properties.items()},
    }


class _Labels:
    """A graph that labels and types nothing, so labels come from the code index."""

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        return {}

    def node_type(self, dcid: str) -> None:
        return None


def _menus(outcome: Outcome) -> dict[str, set[str]]:
    """The values of the answer's slot menus for `outcome`'s identity, by property."""
    assert outcome.shape is not None
    grounded = _ground_answer(
        outcome.shape,
        list(outcome.bindings),
        sorted(outcome.members),
        {_DONOR_PLACE: "origin"},
        cast(EngineGraphClient, _Labels()),
    )
    slot_options = grounded[7]
    return {
        cast(GraphRef, menu.key.property).dcid: {
            cast(GraphRef, option.value.ref).dcid for option in menu.options
        }
        for menu in slot_options
    }


def test_a_bound_invariant_scopes_its_slot_menu_around_its_value(monkeypatch):
    """`flow` is hierarchical here, F3 a child of F. A part naming `_DETAIL` on `_PINNED`
    binds `_DETAIL`'s `flow` invariant F3, and its menu is the one the index scopes around
    F3 (`level_scoped_options`), not the unbound menu of every root and its children."""
    index = _index(codes={**_CODES, "flow": ("F", "F2", "F3")}, parents={"F2": "F", "F3": "F"})
    _use_index(monkeypatch, index)
    outcome = _first_resolve(_PINNED, _DETAIL, "donor", {})
    assert outcome.identity is not None
    assert _bound_slots(outcome.identity)["flow"] == "F3"
    assert outcome.pins["flow"] == "F3"

    menus = _menus(outcome)

    scoped = {option.dcid for option in index.level_scoped_options("flow", "F3")}
    assert menus["flow"] == scoped
    assert scoped != {option.dcid for option in index.level_scoped_options("flow", None)}


def _option_points(
    outcome: Outcome,
    *,
    rows: tuple[StatVarRow, ...] = _ROWS,
    points: Mapping[str, Mapping[str, int]] = _DRILLED_POINTS,
) -> dict[tuple[str, str], int]:
    """Each option's points at the donor, by property and value, as the trailing coverage
    probe states them over the recalled members of `_DRILLED` among `rows`, each holding
    `points`."""
    from materialise.graph import Facet as MaterialiseFacet

    assert outcome.shape is not None
    shape = dataclasses.replace(
        outcome.shape,
        sv_arc_facts={
            row.dcid: _arcs(row) for row in rows if row.measured_property == "drillAmount"
        },
    )
    reader = FakeGraphReader(
        facets={
            (sv, _DONOR_PLACE): [
                MaterialiseFacet(
                    facet_id=f"{sv}-{provenance}",
                    provenance_id=provenance,
                    measurement_method=None,
                    unit=None,
                    obs_count=points,
                    earliest_date="2020",
                    latest_date="2021",
                )
                for provenance, points in by_provenance.items()
            ]
            for sv, by_provenance in points.items()
        }
    )
    enriched, warning = _maybe_enrich_slot_options(
        shape,
        list(outcome.bindings),
        [
            SlotOptions(
                key=SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop),
                options=[
                    SlotOption(
                        value=SlotValue(
                            ref=GraphRef(dcid=code, label=code), value_kind="enum_value"
                        )
                    )
                    for code in codes
                ],
                truncated=False,
                source="observed",
            )
            for prop, codes in _CODES.items()
        ],
        entities=_ENTITIES["donor"],
        date_request=None,
        materialise_graph=reader,
        place_dcids=[_DONOR_PLACE],
        include_option_coverage=True,
        facets_pinned=False,
    )
    assert warning is None
    return {
        (cast(GraphRef, menu.key.property).dcid, cast(GraphRef, option.value.ref).dcid): (
            option.coverage.observation_count
        )
        for menu in enriched
        for option in menu.options
        if isinstance(option.coverage, CoverageExact)
    }


def test_the_donor_total_offers_the_menus_of_the_table_a_bound_option_elects(index):
    """An unnamed donor total elects `_TOTALS`, whose rows carry no sector. Its answer still
    offers the sector menu, since `_BROAD` serves once a sector is bound."""
    outcome = _first_resolve(_DRILLED, None, "donor", {})

    assert outcome.elected == _TOTALS
    assert set(outcome.members) == {"total", "shared_k3"}
    assert _menus(outcome) == {"channel": set(_CODES["channel"]), "sector": set(_CODES["sector"])}


def test_a_named_donor_total_offers_no_menu_its_table_does_not_carry(index):
    """Named, `_TOTALS` serves every value bound, and its rows carry channel alone."""
    outcome = _first_resolve(_DRILLED, _TOTALS, "donor", {})

    assert outcome.elected == _TOTALS
    assert _menus(outcome) == {"channel": set(_CODES["channel"])}


class _EchoLabels:
    """A graph that labels every node with its dcid."""

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        return {dcid: dcid for dcid in dcids}


@pytest.mark.parametrize(
    ("source", "menus"),
    [(None, {"channel", "sector"}), (_TOTALS, {"channel"})],
    ids=["unnamed", "named"],
)
def test_the_shape_event_offers_the_menus_of_the_datasets_the_part_answers_from(
    index, source, menus
):
    """Before bind, a part offers a menu when the datasets it may answer from carry the
    slot's property."""
    (shape,) = _dataset_candidates([_derived_shape(_DRILLED)], _mention(source))

    event = asyncio.run(
        _build_shape_event(shape, variable_index=0, graph=cast(EngineGraphClient, _EchoLabels()))
    )

    assert {cast(GraphRef, menu.key.property).dcid for menu in event.slot_options} == menus


# A `_TOTALS` donor row of channel K1, beside `_BROAD`'s `drill_a`.
_TOTALS_K1 = _row("totals_k1", {"channel": "K1"}, _TOTALS, roles=_DONOR, five_tuple=_DRILLED)


@pytest.mark.parametrize(
    ("extra", "k1"),
    [((), 2), ((_TOTALS_K1,), 1)],
    ids=["broad-serves-k1", "totals-serves-k1"],
)
def test_each_option_counts_the_facets_of_the_table_its_election_serves(monkeypatch, extra, k1):
    """Each option of the donor total counts only the points of the table that serves it
    once bound, `_BROAD`. Channel K3, which both tables record, counts `_BROAD`'s too, where
    the answer's own family serves it from `_TOTALS`. Where `_TOTALS` records its own K1
    row, K1 elects `_TOTALS` with that row alone, and `drill_a`, recalled with K1 but
    outside the family, counts nothing."""
    rows = (*_ROWS, *extra)
    _use_index(monkeypatch, _index(rows=rows))
    outcome = _first_resolve(_DRILLED, None, "donor", {})

    assert _option_points(
        outcome, rows=rows, points={**_DRILLED_POINTS, "totals_k1": {_TOTALS: 1}}
    ) == {
        ("channel", "K1"): k1,
        ("channel", "K2"): 4,
        ("channel", "K3"): 3,
        ("sector", "A"): 2,
        ("sector", "B"): 4,
    }


@pytest.mark.parametrize(
    ("prop", "value", "members"),
    [
        ("sector", "A", {"drill_a": (_BROAD,)}),
        ("sector", "B", {"drill_b": (_BROAD,)}),
        ("channel", "K1", {"drill_a": (_BROAD,)}),
        ("channel", "K3", {"shared_k3": (_BROAD, _TOTALS)}),
    ],
)
def test_resubmitting_a_picked_option_elects_the_table_its_coverage_read(
    index, prop, value, members
):
    """The family an option's coverage reads (`option_families`) is the one every door
    serves once the option is bound."""
    total = _first_resolve(_DRILLED, None, "donor", {})
    families = option_families(
        index, identity_query(cast(BaseIdentity, total.identity)), prop, [value]
    )

    picked = _first_resolve(_DRILLED, None, "donor", {prop: value})

    assert families is not None
    assert families[value].provenance == picked.elected == _BROAD
    assert picked.members == {
        dcid: ElectedDataset(provenance=_BROAD, member_provenances=recorded)
        for dcid, recorded in members.items()
    }
    assert {row.dcid: row.provenances for row in families[value].rows} == members
    _assert_doors_agree(picked, index)


def test_option_families_share_a_memo_across_the_slot_s_own_binding_alone(index):
    """`option_families` memoises per query less the slot's binding, so a query with sector B
    picked reads the family the open query elected for sector A. Role keys key their own
    entry, so a donor-only query elects `_DONORS` for sector A where a donor-and-recipient
    query elects `_BROAD`."""
    query = identity_query(
        cast(BaseIdentity, _first_resolve(_SHARED, None, "donor-recipient", {}).identity)
    )
    donor = dataclasses.replace(query, role_keys=frozenset({"origin"}))
    picked = dataclasses.replace(query, slots={**query.slots, "sector": ("value", "B")})

    families = {
        name: option_families(index, family_query, "sector", ["A"])
        for name, family_query in (("open", query), ("donor", donor), ("picked", picked))
    }

    assert {name: found and found["A"].provenance for name, found in families.items()} == {
        "open": _BROAD,
        "donor": _DONORS,
        "picked": _BROAD,
    }
    assert families["picked"] is not None and families["open"] is not None
    assert families["picked"]["A"] is families["open"]["A"]


def test_an_option_another_table_serves_under_another_pinned_value_elects_that_table(index):
    """An unnamed donor-and-recipient answer elects `_BROAD`, which pins `basis` G. Sector D
    is `_DETAIL`'s alone, under `basis` H. The answer names no dataset, so its identity
    leaves `basis` unbound, and the option's coverage, a resubmit that binds sector D and a
    fresh question binding it all elect `_DETAIL`."""
    answer = _first_resolve(_SHARED, None, "donor-recipient", {})
    assert answer.identity is not None
    families = option_families(index, identity_query(answer.identity), "sector", ["D"])
    document = answer.identity.model_dump(mode="json", exclude_none=True)
    for slot in document["slots"]:
        if slot["property"] == "sector":
            slot["binding"] = {"state": "value", "ref": "D"}
    resubmitted = canonicalise_identity(mint_identity(document)[1])

    fresh = _first_resolve(_SHARED, None, "donor-recipient", {"sector": "D"})

    assert answer.elected == _BROAD
    assert families is not None
    assert families["D"].provenance == identity_dataset(resubmitted) == fresh.elected == _DETAIL
    assert [row.dcid for row in families["D"].rows] == sorted(fresh.members)
    assert resubmitted == fresh.identity
    _assert_doors_agree(fresh, index)


# `_BROAD` and `_DETAIL` both publish `collide_shared`, a donor row. `_BROAD` alone
# publishes `collide_broad`, a donor-and-recipient row of sector A, so a pick of flow F1
# elects `_BROAD` and a pick of sector A elects `_DETAIL`, whose one role class places the
# donor exactly.
_COLLIDE = ("TestFlow", "collideAmount")
_COLLIDE_ROWS = (
    _row(
        "collide_shared",
        {"flow": "F1", "sector": "A"},
        _BROAD,
        _DETAIL,
        roles=_DONOR,
        five_tuple=_COLLIDE,
    ),
    _row("collide_broad", {"sector": "A"}, _BROAD, five_tuple=_COLLIDE),
)
# The rows each dataset's facets hold of any series.
_DATASET_ROWS = {_BROAD: 1, _DETAIL: 2}


@dataclasses.dataclass(frozen=True)
class _DatasetRowsFilter:
    """A donor's served test that serves each member one series, with the rows
    `_DATASET_ROWS` gives the dataset the member is read from."""

    probe_entities: Mapping[str, list[str]] = dataclasses.field(
        default_factory=lambda: {"financingProvider": [_DONOR_PLACE]}
    )

    def served(self, member_facts: Mapping[str, MemberFacts]) -> dict:
        return {
            sv: {
                (sv, _DONOR_PLACE): SeriesRows(
                    rows=_DATASET_ROWS[cast(ElectedDataset, facts.read_dataset()).provenance],
                    exact=True,
                )
            }
            for sv, facts in member_facts.items()
        }


def test_each_pick_counts_the_facets_of_its_own_election_where_picks_share_a_row(monkeypatch):
    """Flow F1 and sector A both read `collide_shared`, from `_BROAD` and from `_DETAIL`.
    Each option counts the facets of the dataset its own pick elects."""
    _use_index(monkeypatch, _index(rows=_COLLIDE_ROWS, ranks={_BROAD: 0, _DETAIL: 1}))
    shape = dataclasses.replace(
        _derived_shape(_COLLIDE, index_rows=_COLLIDE_ROWS), declared_roles=_DONOR
    )

    facts = option_axis_facts(
        shape,
        [],
        menus={"flow": ["F1"], "sector": ["A"]},
        plan=PairPlan(members=(), years=None),
        member_facts={},
        served_filter=cast(RoleServedFilter, _DatasetRowsFilter()),
        facets_pinned=False,
    )

    assert {
        (fact.property, value): fact.rows(value).rows for fact in facts for value in fact.served
    } == {
        ("flow", "F1"): _DATASET_ROWS[_BROAD],
        ("sector", "A"): _DATASET_ROWS[_DETAIL],
    }


_SUBJECT = ("observationAbout",)
_PLACE_AMOUNT = ("TestFlow", "placeAmount")
_PLACE_ROWS = (
    _row("place_a", {"sector": "A"}, _BROAD, roles=_SUBJECT, five_tuple=_PLACE_AMOUNT),
    _row("place_b", {"sector": "B"}, _BROAD, roles=_SUBJECT, five_tuple=_PLACE_AMOUNT),
)
# Sector D, which `_DETAIL` alone carries, so a pick of it elects `_DETAIL`.
_PLACE_D = _row("place_d", {"sector": "D"}, _DETAIL, roles=_SUBJECT, five_tuple=_PLACE_AMOUNT)


@pytest.mark.parametrize(
    ("extra", "counted"),
    [((), {"A": 5, "B": 5, "D": 0}), ((_PLACE_D,), None)],
    ids=["one-publisher", "two-publishers"],
)
def test_a_subject_place_menu_is_counted_from_the_plan_only_when_one_dataset_can_serve_it(
    monkeypatch, extra, counted
):
    """An unnamed subject-place answer elects `_BROAD` and its plan serves sectors A and B.
    Where `_DETAIL` also publishes the five-tuple, a pick of sector D elects `_DETAIL`,
    whose series the plan does not hold, so the menu gets no fact and stays for the probe,
    which counts each option over the dataset its election serves."""
    rows = (*_PLACE_ROWS, *extra)
    _use_index(monkeypatch, _index(rows=rows, ranks={_BROAD: 0, _DETAIL: 1}))
    monkeypatch.setattr(
        option_coverage_module,
        "_plan_series",
        lambda _plan, **_kw: {
            row.dcid: {(row.dcid, "geoId/1"): SeriesRows(rows=5, exact=True)} for row in _PLACE_ROWS
        },
    )

    facts = option_axis_facts(
        _derived_shape(_PLACE_AMOUNT, index_rows=rows),
        [],
        menus={"sector": ["A", "B", "D"]},
        plan=PairPlan(members=(), years=None),
        member_facts={row.dcid: IndexMemberFacts(row=row, dataset=_BROAD) for row in _PLACE_ROWS},
        served_filter=None,
        facets_pinned=False,
    )

    counts = {value: facts[0].rows(value).rows for value in ("A", "B", "D")} if facts else None
    assert counts == counted


# ---------------------------------------------------------------------------
# Alternate guard: an alternate is judged by the rows of the table that serves it
# ---------------------------------------------------------------------------

_GUARDED = ("TestFlow", "guardAmount")


# The elected shape reads sector alone. On its real-terms sibling `_BROAD` reports each
# sector by channel, and a second table reports sector alone.
def _guard_rows(second: str) -> tuple[StatVarRow, ...]:
    return (
        _row("guard_a", {"sector": "A"}, _BROAD, five_tuple=_GUARDED),
        _row("guard_b", {"sector": "B"}, _BROAD, five_tuple=_GUARDED),
        *(
            dataclasses.replace(row, measurement_qualifier="RealValue")
            for row in (
                _row(
                    "guard_real_a_k1", {"channel": "K1", "sector": "A"}, _BROAD, five_tuple=_GUARDED
                ),
                _row("guard_real_a", {"sector": "A"}, second, five_tuple=_GUARDED),
            )
        ),
    )


def _guard_shape(qualifier: str | None) -> ShapeDraft:
    """A shape on `_GUARDED` under `qualifier`, as `derive_shapes` builds it, with the slot
    keys of every row of its five-tuple, whatever dataset publishes it."""
    rows = [row for row in _guard_rows(_BESIDE) if row.measurement_qualifier == qualifier]
    props = sorted({prop for row in rows for prop in row.constraint_properties})
    return ShapeDraft(
        shape_id=f"guard_{qualifier}",
        label="guard amount",
        pop_type_dcid=_GUARDED[0],
        meas_prop_dcid=_GUARDED[1],
        stat_type_dcid="measuredValue",
        meas_qual_dcid=qualifier,
        meas_denom_dcid=None,
        slot_keys=tuple(SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in props),
        slot_taxonomy_index_sourced=True,
        sv_arc_facts_complete=False,
    )


@pytest.mark.parametrize(
    ("second", "stamped"),
    [(_BESIDE, (_BROAD,)), (_DETAIL, (_BROAD, _DETAIL))],
    ids=["opt-in-second-table", "second-table-not-opt-in"],
)
def test_an_alternate_is_judged_by_the_rows_of_the_table_that_serves_it(
    monkeypatch, second, stamped
):
    """An unnamed question answers the real-terms alternate from `_BROAD`. The opt-in
    `_BESIDE` is no candidate, and `_DETAIL` ranks below `_BROAD` in the election. Each
    `_BROAD` row narrows to a channel the elected reading does not name, so the guard
    refuses the alternate. Judged by every candidate table's rows, the second table's
    sector-only row would pass and admit it."""
    _use_index(monkeypatch, _index(rows=_ROWS + _guard_rows(second)))
    elected, alternate = _dataset_candidates([_guard_shape(None), _guard_shape("RealValue")], None)
    assert alternate.datasets == stamped

    reason = inadmissible_reason(query_pins("guard amount"), elected, alternate)

    assert reason is not None
    assert reason.startswith("specialisation:channel")


_FLOW_KEYS = frozenset({"origin", "destination"})


def _real_terms_alternate(constraints: Mapping[str, str]) -> ShapeDraft:
    """A real-terms alternate on `_SHARED`'s population and measure whose one recalled
    member holds `constraints`."""
    row = dataclasses.replace(
        _row("alt", dict(constraints), _DETAIL), measurement_qualifier="RealValue"
    )
    return ShapeDraft(
        shape_id=f"alt_{sorted(constraints.items())}",
        label="amount, real terms",
        pop_type_dcid=_SHARED[0],
        meas_prop_dcid=_SHARED[1],
        stat_type_dcid="measuredValue",
        meas_qual_dcid="RealValue",
        meas_denom_dcid=None,
        slot_keys=tuple(
            SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in sorted(constraints)
        ),
        sv_arc_facts={"alt": _arcs(row)},
        sv_arc_facts_complete=False,
    )


@pytest.mark.parametrize("evidence", ["mentioned", "sole_option"])
@pytest.mark.parametrize(
    ("constraints", "refusal"),
    [
        ({"basis": "H", "channel": "K1", "flow": "F3", "sector": "D"}, None),
        ({"sector": "D"}, None),
        ({"channel": "K1"}, None),
        ({"sector": "A"}, "concept:sector"),
        ({"basis": "G", "flow": "F1", "sector": "A"}, "concept:basis"),
    ],
    ids=["answer-reading", "answer-sector", "answer-channel", "broad-sector", "broad-reading"],
)
def test_the_guard_judges_alternates_against_the_family_the_answer_serves(
    index, constraints, refusal, evidence
):
    """Binding `channel` K1 elects `_DETAIL` over the first-ranked `_BROAD`, so the answer
    serves `detail_d_k1`. The guard elects under the same bindings: its elected reading is
    that family, an alternate holding the answer's values passes, and one holding values
    only `_BROAD`'s rows hold differs in concept, whether the model or the engine bound
    the channel."""
    outcome = _first_resolve(_SHARED, None, "donor-recipient", {"channel": "K1"})
    assert outcome.shape is not None
    assert outcome.elected == _DETAIL
    bindings = [binding.model_copy(update={"evidence": evidence}) for binding in outcome.bindings]
    judge = AlternateJudge(
        query_pins("amount", bindings=bindings), outcome.shape, role_keys=lambda _s: _FLOW_KEYS
    )

    reason = judge.verdict(_real_terms_alternate(constraints)).reason

    assert {member.dcid for member in judge.served} == set(outcome.members)
    if refusal is None:
        assert reason is None
    else:
        assert reason is not None
        assert reason.startswith(refusal)


# ---------------------------------------------------------------------------
# Canonical form: one identity per elected family, whatever pins the input binds
# ---------------------------------------------------------------------------

_TWO_PINS = ("TestFlow", "twoPinAmount")
_PAIRED = "provenance/PairedLedger"

# `_PAIRED` publishes `_TWO_PINS` and declares `basis` G and `flow` F3. Binding either keeps
# `_TOTALS`' one row out of the election, and leaving both open lets `_TOTALS`, ranked above
# it, serve unless the identity names `_PAIRED`.
_TWO_PIN_ROWS = (
    _row("two_totals", {"basis": "H", "flow": "F9", "sector": "A"}, _TOTALS, five_tuple=_TWO_PINS),
    _row(
        "two_paired_a", {"basis": "G", "flow": "F3", "sector": "A"}, _PAIRED, five_tuple=_TWO_PINS
    ),
    _row(
        "two_paired_b", {"basis": "G", "flow": "F3", "sector": "B"}, _PAIRED, five_tuple=_TWO_PINS
    ),
)
_TWO_PIN_RANKS = {**_RANKS, _PAIRED: 6}


def _two_pin_document(source: str | None = None, **bound: str) -> dict:
    """An identity on `_TWO_PINS` declaring `basis`, `flow` and `sector`, each bound to its
    value in `bound` or unbound, with the provenance facet bound to `source` when given."""
    document: dict = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _TWO_PINS[0],
            "measured_property": _TWO_PINS[1],
            "stat_type": "measuredValue",
        },
        "slots": [
            {
                "property": prop,
                "binding": {"state": "value", "ref": bound[prop]}
                if prop in bound
                else {"state": "unbound"},
            }
            for prop in ("basis", "flow", "sector")
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    if source is not None:
        document["facets"] = {"provenance": {"state": "value", "ref": source}}
    return document


@pytest.mark.parametrize(
    "bound",
    [{"basis": "G", "flow": "F3"}, {"basis": "G"}, {"flow": "F3"}],
    ids=["both-pins", "basis-only", "flow-only"],
)
def test_every_binding_of_the_pins_that_elects_one_family_reaches_one_spec_id(monkeypatch, bound):
    """Each identity names `_PAIRED` and elects its two rows, so its canonical form binds
    both invariants. The Datasets doors (`canonical_identity`), the Search resubmit and
    options doors (`canonicalise_identity`) and a first answer's bindings
    (`canonical_pin_bindings`) all reach that one identity, whichever invariants the input
    binds."""
    source = _PAIRED
    rows = _ROWS + _TWO_PIN_ROWS
    index = _index(rows=rows, ranks=_TWO_PIN_RANKS)
    _use_index(monkeypatch, index)
    document = _two_pin_document(source, **bound)
    expected_id, expected = mint_identity(_two_pin_document(source, basis="G", flow="F3"))
    family = family_for_five_tuple(index, identity_family_query(index, document))

    canonical = canonical_identity(index, document)
    resubmitted = canonicalise_identity(mint_identity(document)[1])
    shape = _derived_shape(_TWO_PINS, index_rows=rows)
    drafted = [
        SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])
        for prop, value in bound.items()
    ] + [source_binding(source)]
    answered = canonical_pin_bindings(shape, drafted, ())

    assert family.provenance == _PAIRED
    assert [row.dcid for row in family.rows] == ["two_paired_a", "two_paired_b"]
    assert mint_identity(canonical)[0] == expected_id
    assert resubmitted == expected
    assert _values(answered) == {"basis": "G", "flow": "F3"}
    assert family_for_five_tuple(index, identity_query(expected)) == family


_PARENT_PIN = ("TestFlow", "parentAmount")
_PARENT = "provenance/ParentLedger"

# `_PARENT` publishes `_PARENT_PIN` alone, and its rows all hold sector A, which has a child
# A1, so it declares a code with descendants as an invariant.
_PARENT_PIN_ROWS = (
    _row("parent_g", {"basis": "G", "sector": "A"}, _PARENT, five_tuple=_PARENT_PIN),
    _row("parent_h", {"basis": "H", "sector": "A"}, _PARENT, five_tuple=_PARENT_PIN),
)


@pytest.mark.parametrize(
    "sector",
    [{"state": "unbound"}, {"state": "value", "ref": "A"}, {"state": "exact", "ref": "A"}],
    ids=["unbound", "value", "exact"],
)
def test_an_invariant_with_descendants_binds_as_a_value_binding_spells_it(sector):
    """`_PARENT` declares sector A, a code with a child, so a `value` binding widens to A and A1
    while an `exact` binding does not. All three forms of an identity naming `_PARENT` elect
    the same family. The unbound and `exact` ones canonicalise to the `value` binding Search
    spells, which is canonical as posted."""
    index = _index(
        rows=_ROWS + _PARENT_PIN_ROWS,
        ranks={**_RANKS, _PARENT: 6},
        codes={**_CODES, "sector": (*_CODES["sector"], "A1")},
        parents={"A1": "A"},
    )
    document = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _PARENT_PIN[0],
            "measured_property": _PARENT_PIN[1],
            "stat_type": "measuredValue",
        },
        "slots": [
            {"property": "basis", "binding": {"state": "unbound"}},
            {"property": "sector", "binding": sector},
        ],
        "entities": {},
        "requested_window": {"kind": "all"},
        "facets": {"provenance": {"state": "value", "ref": _PARENT}},
    }
    value_bound = {
        **document,
        "slots": [
            document["slots"][0],
            {**document["slots"][1], "binding": {"state": "value", "ref": "A"}},
        ],
    }

    family = family_for_five_tuple(index, identity_family_query(index, document))
    canonical = canonical_identity(index, document)

    assert [row.dcid for row in family.rows] == ["parent_g", "parent_h"]
    if sector["state"] == "value":
        assert canonical is document
    assert canonical == value_bound
    assert mint_identity(canonical)[0] == mint_identity(value_bound)[0]


def _canonical_query(index: CodeIndex, query: FamilyQuery) -> FamilyQuery:
    """`query` with the slots `invariant_bindings` names bound to their values."""
    bound = invariant_bindings(index, query)
    return dataclasses.replace(
        query,
        slots={**query.slots, **{prop: ("value", value) for prop, value in bound.items()}},
    )


def test_the_canonical_form_depends_only_on_the_elected_family():
    """Over every five-tuple of the fixture index, every binding of its properties, the named
    datasets and three role-key sets, the canonical query elects the query's family and is
    its own canonical form, and every variant that binds another subset of the elected
    dataset's invariants and elects the same family has the same canonical form."""
    index = _index(rows=_ROWS + _TWO_PIN_ROWS, ranks=_TWO_PIN_RANKS)
    variants = 0
    for population_type, measured_property in sorted(
        {(row.population_type, row.measured_property) for row in index.statvars}
    ):
        five_tuple = FiveTuple(
            population_type=population_type,
            measured_property=measured_property,
            stat_type="measuredValue",
        )
        rows = [
            row
            for row in index.statvars
            if (row.population_type, row.measured_property) == (population_type, measured_property)
        ]
        values = {
            prop: sorted(
                {
                    row.constraint_properties[prop]
                    for row in rows
                    if prop in row.constraint_properties
                }
            )
            for prop in sorted({prop for row in rows for prop in row.constraint_properties})
        }
        sources = (None, *index.datasets_by_five_tuple.get(five_tuple, ()))
        for choice, source, role_keys in itertools.product(
            itertools.product(*([None, *found] for found in values.values())),
            sources,
            (frozenset(), frozenset({"origin"}), frozenset({"origin", "destination"})),
        ):
            query = FamilyQuery(
                five_tuple,
                {
                    prop: ("unbound",) if value is None else ("value", value)
                    for prop, value in zip(values, choice, strict=True)
                },
                source,
                role_keys,
            )
            family = family_for_five_tuple(index, query)
            canonical = _canonical_query(index, query)
            assert family_for_five_tuple(index, canonical) == family
            assert invariant_bindings(index, canonical) == {}
            if query.source is None:
                assert canonical == query
                continue
            if family.provenance is None:
                continue
            pins = index.datasets[family.provenance].invariants
            toggled = [
                prop for prop in pins if query.slots[prop] in (("unbound",), ("value", pins[prop]))
            ]
            for flips in itertools.product((False, True), repeat=len(toggled)):
                variant = dataclasses.replace(
                    query,
                    slots={
                        **query.slots,
                        **{
                            prop: ("value", pins[prop]) if bind else ("unbound",)
                            for prop, bind in zip(toggled, flips, strict=True)
                        },
                    },
                )
                if family_for_five_tuple(index, variant) == family:
                    assert _canonical_query(index, variant) == canonical
                    variants += 1
    assert variants > 1000


# ---------------------------------------------------------------------------
# Role-class rebinding: the wire slots and the identity read one set of bindings
# ---------------------------------------------------------------------------

_MIXED = ("TestFlow", "mixedAmount")

# `_BROAD` publishes subject-only rows and pins `basis` G, and `_DETAIL` publishes a
# donor-and-recipient row beside them under `basis` H. Every row carries `channel`.
_MIXED_ROWS = tuple(
    _row(dcid, {**constraints, "channel": "K1"}, dataset, roles=roles, five_tuple=_MIXED)
    for dcid, constraints, dataset, roles in (
        ("mixed_subject_a", {"basis": "G", "sector": "A"}, _BROAD, ()),
        ("mixed_subject_b", {"basis": "G", "sector": "B"}, _BROAD, ()),
        ("mixed_flow_a", {"basis": "H", "sector": "A"}, _DETAIL, _DONOR_RECIPIENT),
    )
)


def _mixed_shape(*, decided: bool) -> ShapeDraft:
    """The recalled shape on `_MIXED`, holding every row as a member and declaring the union
    of their roles. Undecided, the members and the slot keys leave out `channel`, which
    every index row carries, so the index does not decide the shape and the role-class
    election runs among its members (`elected_role_class_shape`)."""
    props = ("basis", "channel", "sector") if decided else ("basis", "sector")
    members = {}
    for row in _MIXED_ROWS:
        constraints = {
            prop: value for prop, value in row.constraint_properties.items() if prop in props
        }
        arcs = _arcs(dataclasses.replace(row, constraint_properties=constraints))
        if row.observation_properties:
            arcs["observationProperties"] = {
                "nodes": [{"dcid": role} for role in row.observation_properties]
            }
        members[row.dcid] = arcs
    return ShapeDraft(
        shape_id=f"mixed_{decided}",
        label="mixed amount",
        pop_type_dcid=_MIXED[0],
        meas_prop_dcid=_MIXED[1],
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=tuple(SlotKeyDraft(axis="how", property_dcid=prop, label=prop) for prop in props),
        declared_roles=_DONOR_RECIPIENT,
        optional_roles=frozenset({"financingRecipient"}),
        sv_arc_facts=members,
        sv_arc_facts_complete=False,
    )


@pytest.mark.parametrize("decided", [True, False], ids=["index-decides", "recall-decides"])
def test_the_wire_slots_and_the_identity_agree_when_the_role_class_election_rebinds(
    monkeypatch, decided
):
    """A question naming no place on a five-tuple mixing subject-only and flow rows. The
    repair canonicalises the bindings under the role keys of the recalled shape's places, and the
    identity is canonical under the role keys of the places bound to the shape the
    role-class election returns. Where the index decides the shape, the election keeps it,
    so the keys are one, and the question names no dataset, so `basis` stays bound. Where it
    does not, the election narrows the role-bearing shape to its subject-only members and
    the places bind again, and the canonical identity binds nothing more, since the index
    decides no family for slot keys it does not cover. Either way the slots the answer states
    are the identity's."""
    _use_index(monkeypatch, _index(rows=_ROWS + _MIXED_ROWS))
    shape = _mixed_shape(decided=decided)
    state = _repair_state(shape, _PLACES["none"], {"basis": "G", "sector": "A"})
    bindings = _repair(monkeypatch, state)

    bound = bind_place_roles(state, shape)
    assert isinstance(bound, PlaceRoles)
    role_query = shape_query(shape, bindings, entity_role_keys(shape, bound.resolved_entities))
    elected = elected_role_class_shape(shape, role_query).value
    rebound = bound_after_election(state, shape, bound, elected)
    assert isinstance(rebound, PlaceRoles)
    assert is_role_bearing(elected) is decided
    assert (rebound is bound) is decided
    assert rebound.resolved_entities == {}

    slots = [
        build_slot(
            key,
            draft,
            [GraphRef(dcid=value, label=value) for value in draft.value_dcids]
            if draft is not None
            else [],
            property_ref=GraphRef(dcid=key.property_dcid, label=key.property_dcid)
            if key.property_dcid is not None
            else None,
        )
        for key, draft in pair_slot_drafts(elected.slot_keys, bindings)
    ]
    _, identity = build_spec_identity(
        identity_shape_for(
            pop_type_dcid=elected.pop_type_dcid,
            meas_prop_dcid=elected.meas_prop_dcid,
            stat_type_dcid=elected.stat_type_dcid,
            meas_qual_dcid=None,
            meas_denom_dcid=None,
        ),
        slots,
        entities=[],
    )

    wire = {
        cast(GraphRef, slot.key.property).dcid: cast(GraphRef, slot.binding.value.ref).dcid
        for slot in slots
        if slot.binding.kind == "value"
    }
    assert wire == _bound_slots(identity)
    assert wire == {"basis": "G", "sector": "A"}


def test_a_constant_no_serving_dataset_declares_stays_in_the_bind_menu_and_the_identity(index):
    """`_BROAD` and `_DETAIL` both hold `basis` G on every row of `_PINNED`, but `_DETAIL`
    does not declare it, since its other rows hold H. The bind menu offers `basis`, and a
    question that binds it keeps it bound in the identity Search mints, so a later export
    that adds `_DETAIL` rows of basis H to the five-tuple still serves basis G alone."""
    (shape,) = _dataset_candidates([_derived_shape(_PINNED)], _mention(None))

    prompt = _build_universal_bind_prompt("pinned amount", [shape])
    outcome = _first_resolve(_PINNED, None, "donor-recipient", {"basis": "G", "flow": "F3"})

    assert "property_dcid=basis" in prompt
    assert outcome.withheld == {}
    assert outcome.elected == _DETAIL
    assert outcome.identity is not None
    assert _bound_slots(outcome.identity) == {"basis": "G", "flow": "F3"}


# ---------------------------------------------------------------------------
# Resubmit: an edit to the entities re-elects the dataset and its pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("five_tuple", "model", "base_roles", "posted_roles", "elected"),
    [
        (_SHARED, {"sector": "A"}, "donor", "donor-recipient", _BROAD),
        (_SHARED, {"sector": "A"}, "donor-recipient", "donor", _DONORS),
        # `_DONORS` pins `flow` F1 and `_BROAD`'s sector B row holds F2.
        (_SHARED, {"sector": "B"}, "donor", "donor-recipient", _BROAD),
        (_PINNED, {"flow": "F3"}, "donor-recipient", "donor-recipient", _DETAIL),
        (_PINNED, {"flow": "F3"}, "donor", "donor-recipient", _DETAIL),
    ],
    ids=[
        "recipient-added",
        "recipient-dropped",
        "recipient-added-other-flow",
        "unchanged-bound-pin",
        "recipient-added-bound-pin",
    ],
)
def test_a_resubmit_that_changes_the_roles_reaches_the_family_a_fresh_question_reaches(
    index, five_tuple, model, base_roles, posted_roles, elected
):
    """The base answer names no dataset, so its identity leaves the pins unbound. A resubmit
    that adds or drops the recipient posts that identity with the new entities, and its
    canonical form is the identity a first resolve of the same question mints, so every door
    serves the dataset those roles elect. A code the question binds that only one dataset reports
    (`flow` F3, `_DETAIL`'s alone) stays bound, since unbinding it would elect another
    dataset, with no edit or with the recipient added."""
    base = _first_resolve(five_tuple, None, base_roles, model)
    fresh = _first_resolve(five_tuple, None, posted_roles, model)
    assert base.identity is not None
    assert fresh.identity is not None
    document = base.identity.model_dump(mode="json", exclude_none=True)
    document["entities"] = fresh.identity.model_dump(mode="json", exclude_none=True)["entities"]
    _, posted = mint_identity(document)

    resubmitted = canonicalise_identity(posted)

    assert fresh.elected == elected
    assert resubmitted == fresh.identity
    _assert_doors_agree(dataclasses.replace(fresh, identity=resubmitted), index)
