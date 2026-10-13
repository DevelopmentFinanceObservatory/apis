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

The fixture index stands in for the aid tables. Datasets whose rows of a five-tuple carry one
property set form one family there (`CodeIndex.family_of`), and the election runs among a
family's datasets. On the shared five-tuple `_BROAD` (rank 0) has donor-and-recipient rows;
`_DETAIL` (rank 1) shares one row with it, alone carries the `channel` codes K1 and K2, and
pins `flow`; `_DONORS` (rank 2) has donor-only rows. The opt-in `_BESIDE` publishes beside
them, and all four carry `basis`, `channel`, `flow` and `sector`, so they are one family. The
opt-in `_ALONE` alone publishes a second five-tuple. `_BROAD` alone publishes a third. On a
fourth, `_BROAD`'s donor-and-recipient rows and `_DONORS`' donor-only rows differ on the
measure axis `netting`. On a fifth, `_TOTALS` publishes donor-only rows without `sector`
beside `_BROAD`'s rows by sector and channel, so the two are separate families. The
perturbation tests show which cells the rank order, opt-in and the exact-role-set preference
decide. The repair-phase tests drive `_phase_repair_bindings` from places the question
names. The drill-down tests read the answer's slot menus (`offers_slot`) and each option's
coverage under the dataset its election serves (`option_families`) on the fifth five-tuple,
for each of its families.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import itertools
import random
import sys
from collections.abc import Collection, Mapping, Sequence
from types import SimpleNamespace
from typing import cast

import pytest

from code_index import members as members_module
from code_index import shared as shared_module
from code_index.members import (
    FamilyQuery,
    canonical_identity,
    elected_pins,
    eligible_datasets,
    family_for_five_tuple,
    identity_family_query,
    invariant_bindings,
    option_families,
)
from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import (
    CodeRow,
    CodeSystemDeclaration,
    DatasetDeclaration,
    DeclaredDefault,
    IndexStamp,
    MeasureAxis,
    MeasureIndex,
    StatVarRow,
)
from datasets_api.adapter import resolve_definition_members
from datasets_api.codes_materialise import RoleSignatureIncompatible
from dfo_platform.roles import RoleAxis
from entity_resolution import Admission, EntityResolver
from materialise.facets import ElectedDataset
from search.engine.bind import (
    SlotBindingDraft,
    _build_universal_bind_prompt,
    _UniversalBindOutput,
)
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.conjoin import _unbound_terms
from search.engine.interpret import Recall
from search.engine.options.population import options_population
from search.engine.qualifier_match import QualifierMatch
from search.engine.quantity import quantity_requirement
from search.engine.region import option_coverage as option_coverage_module
from search.engine.region import phase_election as phase_election_module
from search.engine.region import phase_rank_bind as phase_rank_bind_module
from search.engine.region import related_read as related_read_module
from search.engine.region.alternate_guard import AlternateJudge, inadmissible_reason, query_pins
from search.engine.region.candidates import walk_alternate
from search.engine.region.grounding import _ground_answer
from search.engine.region.option_coverage import (
    RoleServedFilter,
    _maybe_enrich_slot_options,
    option_axis_facts,
)
from search.engine.region.phase_election import _phase_repair_bindings, _phase_vet_election
from search.engine.region.phase_rank_bind import (
    _dataset_candidates,
    _out_of_scope_records,
    _phase_prune_and_rank,
    _resolve_query_qualifiers,
    _undeclared_qualifiers,
    _unheld_bound_codes,
)
from search.engine.region.place_context import empty_place_context
from search.engine.region.place_roles import (
    PlaceAssignment,
    bind_place_roles,
    bound_after_election,
    lacks_query_axis,
)
from search.engine.region.related_read import start_elected_related_read
from search.engine.region.result import RegionResult
from search.engine.region.sentence_events import _build_shape_event
from search.engine.region.state import ResolveState, shape_vocabulary
from search.engine.retrieval.bound_values import (
    PartNotServed,
    RoleKeys,
    SeriesRows,
    bind_shared_values,
    bound_source,
    bound_values_from_bindings,
    canonical_pin_bindings,
    canonicalise_identity,
    elected_role_class_shape,
    entity_role_keys,
    identity_dataset,
    identity_query,
    part_datasets,
    part_source,
    refuse_source_outside_family,
    shape_datasets,
    shape_invariants,
    shape_member_facts,
    shape_query,
    source_binding,
    stated_members,
)
from search.engine.retrieval.member_facts import IndexMemberFacts, MemberFacts
from search.engine.retrieval.read_plan import PairPlan
from search.engine.retrieval.retrieve import Materialised, NoDataDraft
from search.engine.retrieval.role_resolve import _admit_population, _AdmittedPopulation
from search.engine.role_text import DoorRole
from search.engine.shape import (
    ShapeDraft,
    SlotKeyDraft,
    identity_shape_for,
    is_role_bearing,
    role_names_by_axis,
)
from search.engine.source_scope import SourceReference, SourceScope
from search.engine.spec_build import build_slot, pair_slot_drafts, with_served_provenance
from search.engine.spec_id import build_spec_identity, mint_identity
from search.models import (
    BaseIdentity,
    CoverageExact,
    DatasetResubmitInput,
    DefinitePart,
    Entity,
    GraphRef,
    IdentityBindingValue,
    ResolveRequest,
    ResubmitIdentityEntry,
    SlotKey,
    SlotOption,
    SlotOptions,
    SlotValue,
    SourceMention,
    TimeWindow,
)
from search.models.entity import EntityRoleDirectional
from search.models.spec import Spec
from series_coverage.schema import FacetMetadata, RoleNames
from statvar_members import FiveTuple, MemberEnumerator, member_digest
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._member_bounds import mint_bounds
from tests.engine._builders import place_sets, settled_place_roles
from tests.engine._fake_coverage import empty_coverage_snapshot, patch_coverage
from tests.engine._harness import offline_resolve
from tests.engine._index_slots import read_text
from tests.engine._source_scopes import part_scope
from tests.fixtures import FakeGraph
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
    # Each dataset of the shared five-tuple carries `channel` on a row, K3 outside
    # `_DETAIL`, so all four carry one property set and form one family.
    # `_BROAD` holds `basis` G on every row, so it declares `basis` as an invariant.
    _row("broad_a", {"basis": "G", "flow": "F1", "sector": "A"}, _BROAD),
    _row("broad_b", {"basis": "G", "flow": "F2", "sector": "B"}, _BROAD),
    _row("both_c", {"basis": "G", "channel": "K3", "flow": "F3", "sector": "C"}, _BROAD, _DETAIL),
    # `_DETAIL` holds `flow` F3 on every row, so it declares `flow`; `channel` is its own.
    _row("detail_d_k1", {"basis": "H", "channel": "K1", "flow": "F3", "sector": "D"}, _DETAIL),
    _row("detail_d_k2", {"basis": "H", "channel": "K2", "flow": "F3", "sector": "D"}, _DETAIL),
    # `_DONORS` declares the donor alone and holds `basis` G and `flow` F1 on every row.
    _row("donors_a", {"basis": "G", "flow": "F1", "sector": "A"}, _DONORS, roles=_DONOR),
    _row(
        "donors_b",
        {"basis": "G", "channel": "K3", "flow": "F1", "sector": "B"},
        _DONORS,
        roles=_DONOR,
    ),
    # The opt-in `_BESIDE` alone carries sector E.
    _row("beside_a", {"basis": "G", "flow": "F9", "sector": "A"}, _BESIDE),
    _row("beside_e", {"basis": "G", "channel": "K3", "flow": "F9", "sector": "E"}, _BESIDE),
    _row("alone_a", {"basis": "G", "sector": "A"}, _ALONE, five_tuple=_OPT_IN_ONLY),
    _row("alone_b", {"basis": "G", "sector": "B"}, _ALONE, five_tuple=_OPT_IN_ONLY),
    _row("sole_a", {"basis": "G", "sector": "A"}, _BROAD, five_tuple=_SOLE),
    _row("sole_b", {"basis": "G", "sector": "B"}, _BROAD, five_tuple=_SOLE),
    # `_BROAD` reports `netting` gross, the measure vote's preferred value, and `_DONORS`
    # reports it net on its one row of the five-tuple. Both carry `flow`, so they are one
    # family.
    _row(
        "gross_a",
        {"basis": "G", "flow": "F1", "netting": "Gross", "sector": "A"},
        _BROAD,
        five_tuple=_MEASURED,
    ),
    _row(
        "gross_b",
        {"basis": "G", "flow": "F2", "netting": "Gross", "sector": "B"},
        _BROAD,
        five_tuple=_MEASURED,
    ),
    _row(
        "net_a",
        {"basis": "G", "flow": "F1", "netting": "Net", "sector": "A"},
        _DONORS,
        roles=_DONOR,
        five_tuple=_MEASURED,
    ),
    # `_TOTALS` publishes donor-only rows and carries no `sector`; `_BROAD` publishes the
    # same five-tuple by sector and channel for donor and recipient, so the two are separate
    # families, and both record the donor-only channel K3 row.
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
    disjoint: bool = True,
    defaults: Mapping[str, DeclaredDefault] | None = None,
) -> CodeIndex:
    """The fixture index over `rows`, declaring `codes`. A code system with a code in
    `parents` is hierarchical, each such code a child of its parent, and its parent rows
    are apart from their children's when `disjoint` (`disjoint_from_children`), else
    their totals. A code system in `defaults` declares that default. Each dataset declares
    as invariants the values every one of its rows holds (`_invariants`)."""
    parents = parents or {}
    defaults = defaults or {}
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
                disjoint_from_children=disjoint,
                summable=True,
                absent_reading="not_declared",
                plain_name=prop,
                default=defaults.get(prop),
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
                description="A test table.",
                opt_in=provenance in opt_in,
                absent_observation="zero",
                account="provider",
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
    five_tuple: tuple[str, str],
    *,
    index_rows: tuple[StatVarRow, ...] = _ROWS,
    family_of: str | None = None,
) -> ShapeDraft:
    """A shape as `derive_shapes` builds it on `five_tuple` for one index family: the slot
    keys and taxonomy of every row of that family among `index_rows`, whatever dataset of
    the family publishes it, and a source slot. The family is the one `family_of` belongs
    to, else the five-tuple's first (`CodeIndex.families_by_five_tuple`)."""
    population_type, measured_property = five_tuple
    families = _index(rows=index_rows).families_by_five_tuple[
        FiveTuple(
            population_type=population_type,
            measured_property=measured_property,
            stat_type="measuredValue",
        )
    ]
    family = next((found for found in families if family_of in found.datasets), families[0])
    rows = [row for row in index_rows if row in family.rows]
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
            ref=GraphRef(ref=dcid, label=dcid),
            role=EntityRoleDirectional(role=GraphRef(ref=role, label=role), axis=_AXES[role]),
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
    family_of: str | None = None,
    index_rows: tuple[StatVarRow, ...] = _ROWS,
) -> Outcome:
    """Run the first resolve of one part on `five_tuple`'s shape of the family `family_of`
    belongs to among `index_rows` (`_derived_shape`), naming `source` or none, about the
    entities of `roles`, where the bind model binds `model`."""
    candidates = _dataset_candidates(
        [_derived_shape(five_tuple, index_rows=index_rows, family_of=family_of)], part_scope(source)
    )
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
    # Vetting names the dataset the elected shape names (`_phase_vet_election`).
    if shape.named_dataset is not None:
        drafted.append(source_binding(shape.named_dataset))
    return _minted(shape, drafted, roles, window=window)


def _minted(
    shape: ShapeDraft,
    drafted: list[SlotBindingDraft],
    roles: str,
    *,
    window: TimeWindow | None = None,
    open_axes: Collection[RoleAxis] = (),
    open_roles: Collection[RoleAxis] | None = None,
    served: Sequence[str] = (),
) -> Outcome:
    """Mint the identity of `shape` under the vetted `drafted` bindings, about the entities
    of `roles`, as the first resolve does after vetting. The part asks `open_axes` of every
    member, and the identity names `open_roles`, by default the axes asked, and binds its
    source to `served`, as a part with a source mention does. The members are read under
    the shape's query, which equals the identity's own when it binds no source."""
    bindings, _shared = bind_shared_values(shape, drafted)
    asked = entity_role_keys(_ENTITIES[roles], open_axes=open_axes)
    bindings = canonical_pin_bindings(shape, bindings, asked)
    role_keys = (
        asked if open_roles is None else entity_role_keys(_ENTITIES[roles], open_axes=open_roles)
    )
    query = shape_query(shape, bindings, role_keys.bound, open_role_keys=role_keys.open)
    facts, index_sourced = shape_member_facts(shape, query)
    assert index_sourced
    members = {dcid: fact.read_dataset() for dcid, fact in facts.items()}
    elected = {dataset.provenance for dataset in members.values() if dataset is not None}
    assert len(elected) <= 1

    slots = [
        build_slot(
            key,
            draft,
            [GraphRef(ref=value, label=value) for value in draft.value_dcids]
            if draft is not None
            else [],
            property_ref=GraphRef(ref=key.property_dcid, label=key.property_dcid)
            if key.property_dcid is not None
            else None,
        )
        for key, draft in pair_slot_drafts(shape.slot_keys, bindings)
    ]
    if served:
        slots = with_served_provenance(slots, served)
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
        open_roles=sorted(role_keys.open),
    )
    if not served:
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


class _IndexRoles:
    """Serves the declared-role refresh of an index-sourced population from `rows`."""

    def __init__(self, rows: Sequence[StatVarRow]) -> None:
        self._roles = {row.dcid: row.observation_properties for row in rows}

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        return {dcid: self._roles[dcid] for dcid in dcids}


def _admitted(
    outcome: Outcome, roles: str, rows: Sequence[StatVarRow]
) -> NoDataDraft | _AdmittedPopulation:
    """What Search's role path admits for the part `outcome` mints, about the entities of
    `roles` and asking the open roles its identity names
    (`role_resolve._admit_population`)."""
    assert outcome.shape is not None and outcome.identity is not None
    entities = _ENTITIES[roles]
    query = identity_query(outcome.identity)
    return _admit_population(
        outcome.shape,
        query=query,
        bound_values=bound_values_from_bindings(list(outcome.bindings)),
        names=role_names_by_axis(outcome.shape),
        subject_dcids=entities.get("financingRecipient", []),
        origin_dcids=entities.get("financingProvider", []),
        open_role_keys=sorted(query.open_role_keys),
        graph=cast(EngineGraphClient, _IndexRoles(rows)),
        deadline=None,
    )


def _mint(identity: BaseIdentity, index: CodeIndex) -> dict[str, ElectedDataset | None]:
    graph = FakeMintGraphReader(arcs={})
    resolved = resolve_definition_members(
        identity.model_dump(mode="json", exclude_none=True),
        spec_id="spec_test",
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=index,
        bounds=mint_bounds(),
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
    named_by_measure: str | None = None
    """The opt-in dataset the elected shape names for a part that names none."""

    @property
    def names(self) -> str | None:
        """The dataset the identity names."""
        return self.source or self.named_by_measure


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
        datasets=(_ALONE,),
        withheld=_BROAD_PINS,
        by_roles={
            "donor": Expected(_ALONE, {"alone_a": (_ALONE,)}, _BROAD_PINS),
            "donor-recipient": Expected(_ALONE, {"alone_a": (_ALONE,)}, _BROAD_PINS),
        },
        named_by_measure=_ALONE,
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
            self.scenario.five_tuple,
            self.scenario.source,
            self.roles,
            self.model,
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
    if cell.scenario.names is None:
        # A part that names no dataset keeps the pins the model bound, and binds no other.
        assert {prop: bound.get(prop) for prop in expected.pins} == {
            prop: cell.model.get(prop) for prop in expected.pins
        }
    else:
        # The identity names the dataset, so it binds every invariant of it.
        assert {prop: bound.get(prop) for prop in expected.pins} == expected.pins
    facets = outcome.identity.model_dump(mode="json", exclude_none=True).get("facets", {})
    # The identity binds the dataset only when the part or the elected shape names it.
    assert facets.get("provenance", {}).get("ref") == cell.scenario.names


@pytest.mark.parametrize("cell", _CELLS.values(), ids=_CELLS.keys())
def test_the_first_resolve_and_every_identity_door_elect_one_dataset(index, cell: Cell):
    """An unnamed part never answers from an opt-in dataset beside a default one. On a
    five-tuple only one opt-in dataset publishes, the shape names that dataset, so the part
    mints the identity a part naming it mints. A donor-only question reads the donor-only
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
    """A part that names its dataset, or elects a shape that names one, mints one identity
    whether or not the model binds the dataset's invariants. One that names none keeps what the model bound, so the two mint
    two identities that serve the same members."""
    key, roles = scenario_roles
    with_pins = _CELLS["-".join((*key, roles, "none"))].resolve()
    without = _CELLS["-".join((*key, roles, "pins"))].resolve()

    assert with_pins.members == without.members
    assert (with_pins.spec_id == without.spec_id) is (_SCENARIOS[key].names is not None)


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
    table holds no `channel` code, so the term is unavailable on it and the part refuses
    `qualifier_not_bound` once bound. The part naming `_DETAIL` can bind it."""
    (shape,) = _dataset_candidates([_derived_shape(_SHARED)], part_scope(named))
    measure_index = MeasureIndex(axes=(), values=(), partition_labels=())
    vocabulary = shape_vocabulary(shape)
    reading = read_text("K1 aid", index, measure_index, vocabularies=[vocabulary])
    spans = reading.everywhere.spans
    assert [row.code_system for span in spans for row in span.rows] == ["channel"]

    undeclared = _undeclared_qualifiers(
        "K1 aid", spans, [shape], {vocabulary: reading.by_vocabulary[vocabulary].spans}
    )

    assert [record.text for record in undeclared] == (["K1"] if refused else [])


def test_a_part_excluding_a_dataset_answers_from_the_other_publishers(index):
    """A part excluding the first-ranked publisher of `_SHARED` keeps the shape, stamped with
    the other publishers that are not opt-in. A part excluding every one of them keeps
    nothing of the shape, since an opt-in publisher answers only a part that names it."""
    (unnamed,) = _dataset_candidates([_derived_shape(_SHARED)], None)
    first, *others = unnamed.datasets
    assert others

    (shape,) = _dataset_candidates(
        [_derived_shape(_SHARED)], part_scope(excluded=frozenset({first}))
    )

    assert shape.datasets == tuple(others)
    assert (
        _dataset_candidates(
            [_derived_shape(_SHARED)], part_scope(excluded=frozenset(unnamed.datasets))
        )
        == []
    )


# ---------------------------------------------------------------------------
# Source words
# ---------------------------------------------------------------------------

# `_BROAD` and the opt-in `_BESIDE` alone publish a five-tuple of their own, as one family.
_SLOTTED = ("TestFlow", "slotAmount")
_SLOTTED_ROWS = (
    *_ROWS,
    _row("slotted_broad", {"basis": "G", "flow": "F1", "sector": "A"}, _BROAD, five_tuple=_SLOTTED),
    _row(
        "slotted_beside",
        {"basis": "G", "flow": "F9", "sector": "A"},
        _BESIDE,
        five_tuple=_SLOTTED,
    ),
)

# A source a part's words exclude, spelled as the table with this prefix.
_EXCLUDED = "excluded:"


def _excluded(provenance: str) -> str:
    return f"{_EXCLUDED}{provenance}"


def _worded_scope(*sources: str) -> SourceScope | None:
    """The scope of a part whose source words, as a raw-text query's extraction carries
    them, name each table of `sources` and exclude each `_excluded` one: the named tables,
    else every declared one, less the excluded."""
    excluded = {
        source.removeprefix(_EXCLUDED) for source in sources if source.startswith(_EXCLUDED)
    }
    named = [source for source in sources if not source.startswith(_EXCLUDED)]
    if not sources:
        return None
    allowed = frozenset(named) if named else frozenset(get_index().datasets)
    mentions = [
        *(SourceMention(text=table, ref=table) for table in named),
        *(SourceMention(text=table, ref=table, polarity="exclude") for table in sorted(excluded)),
    ]
    return SourceScope(
        allowed=allowed - excluded,
        named=bool(named),
        mentions=tuple(mentions),
        references=tuple(
            SourceReference(ref=m.ref, label=m.ref, kind="provenance")
            for m in mentions
            if m.ref is not None
        ),
        expansions=tuple(frozenset({m.ref}) for m in mentions if m.ref is not None),
    )


def _worded_resolve(
    five_tuple: tuple[str, str],
    source: str,
    roles: str,
    model: Mapping[str, str],
    *,
    index_rows: tuple[StatVarRow, ...] = _ROWS,
    admitted: tuple[QualifierMatch, ...] = (),
) -> Outcome | RegionResult:
    """Run the first resolve of a part on `five_tuple` whose words name the table `source`
    or exclude the `_excluded` one (`_worded_scope`), where the bind model binds `model` and
    the query text matches the codes `admitted`: candidacy under the scope
    (`_dataset_candidates`), whose refusal names the scope's sources
    (`_out_of_scope_records`), then vetting (`_phase_vet_election`). The minted outcome, or
    the refusal."""
    scope = _worded_scope(source)
    assert scope is not None
    derived = dataclasses.replace(
        _derived_shape(five_tuple, index_rows=index_rows),
        # The shape declares the flow roles, so vetting reads the role keys the identity binds.
        declared_roles=_DONOR_RECIPIENT,
        optional_roles=frozenset({"financingRecipient"}),
    )
    candidates = _dataset_candidates([derived], scope)
    if not candidates:
        state = _repair_state(derived, _PLACES[roles], model)
        return state.no_data(
            "qualifier_not_bound", unbound_sources=_out_of_scope_records([derived], scope)
        )
    (shape,) = candidates
    state = _repair_state(shape, _PLACES[roles], model)
    state.capped_shapes = [shape]
    state.scope = scope
    state.query_qualifier_matches = {shape_vocabulary(shape): admitted}
    state.admitted_qualifiers = admitted
    result = asyncio.run(_phase_vet_election(state))
    if result is not None:
        return result
    return _minted(state.shape_draft, list(state.bind_result.bindings), roles)


def _resubmit_door(monkeypatch: pytest.MonkeyPatch, outcome: Outcome) -> Spec:
    """The spec the dataset_resubmit door rebuilds from `outcome`'s identity, posted
    unchanged, where each member of the outcome holds a series from Germany to Kenya under
    the dataset that serves it."""
    assert outcome.identity is not None
    assert outcome.spec_id is not None
    patch_coverage(
        monkeypatch,
        [
            (dcid, served.provenance, _DCIDS["Germany"], _DCIDS["Kenya"], [2020, 2021])
            for dcid, served in outcome.members.items()
            if served is not None
        ],
        role_names=dict.fromkeys(
            outcome.members,
            RoleNames(donor="financingProvider", recipient="financingRecipient"),
        ),
        facet_metadata={
            served.provenance: FacetMetadata(provenance_id=served.provenance, unit="USD")
            for served in outcome.members.values()
            if served is not None
        },
    )
    nodes: dict[str, dict] = {
        place: {"label": place, "type": "Country"} for place in _DCIDS.values()
    }
    for row in _ROWS:
        nodes[row.dcid] = {
            "label": row.dcid,
            "type": "StatisticalVariable",
            "arcs": {
                **_arcs(row),
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "observationProperties": {
                    "nodes": [{"dcid": prop} for prop in row.observation_properties]
                },
            },
        }
    entry = ResubmitIdentityEntry(
        variable_text="aid",
        identity=outcome.identity,
    )
    response = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
        graph=FakeGraph(nodes=nodes, obs={}),
        materialise_graph=FakeGraphReader(
            arcs={
                row.dcid: {
                    "populationType": (row.population_type,),
                    "measuredProperty": (row.measured_property,),
                    "statType": (row.stat_type,),
                    "constraintProperties": tuple(row.constraint_properties),
                    **{prop: (value,) for prop, value in row.constraint_properties.items()},
                }
                for row in _ROWS
            }
        ),
    )
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part.spec


@pytest.mark.parametrize(
    ("source", "roles", "sector", "named"),
    [
        (_DETAIL, "donor-recipient", "C", _DETAIL),
        (_BESIDE, "donor-recipient", "A", _BESIDE),
        (_excluded(_BROAD), "donor-recipient", "C", _DETAIL),
        (_excluded(_DONORS), "donor", "A", _BROAD),
    ],
    ids=["named", "opt-in-named", "excluded-rank", "excluded-roles"],
)
def test_source_words_name_the_table_the_identity_binds(
    index, monkeypatch, source, roles, sector, named
):
    """Words naming a table, the opt-in `_BESIDE` included, make the part a part naming it.
    Words excluding a table name the one the election serves
    among the others: `_DETAIL` for the shared sector C row once `_BROAD` is out, and the
    rank's `_BROAD` for a donor-only question once the donor-only `_DONORS` is out. The
    identity names the table, so it is the identity a part naming it mints, the
    dataset_resubmit door rebuilds it under the same spec_id, and every door agrees."""
    outcome = _worded_resolve(_SHARED, source, roles, {"sector": sector})

    assert isinstance(outcome, Outcome)
    assert outcome.elected == named
    assert outcome.datasets == (named,)
    assert outcome.identity is not None
    facets = outcome.identity.model_dump(mode="json", exclude_none=True)["facets"]
    assert facets["provenance"]["ref"] == named
    assert outcome.spec_id == _first_resolve(_SHARED, named, roles, {"sector": sector}).spec_id
    resubmitted = _resubmit_door(monkeypatch, outcome)
    assert resubmitted.spec_id == outcome.spec_id
    assert identity_dataset(resubmitted.definition.identity) == named
    assert resubmitted.resolved.shape.member_digest == member_digest(outcome.members)
    _assert_doors_agree(outcome, index)


@pytest.mark.parametrize(
    ("source", "roles", "sector"),
    [
        (_excluded(_DETAIL), "donor-recipient", "A"),
        (_excluded(_BESIDE), "donor-recipient", "A"),
        (_excluded(_DONORS), "donor-recipient", "A"),
    ],
    ids=["excluded-not-serving", "excluded-opt-in", "excluded-other-roles"],
)
def test_an_exclusion_that_changes_no_election_leaves_the_part_unnamed(
    index, monkeypatch, source, roles, sector
):
    """`_BROAD` serves the shared sector A row to a donor-and-recipient question, so
    excluding `_DETAIL`, the opt-in `_BESIDE` or the donor-only `_DONORS` leaves the
    election where it was. The part names no table and mints the plain question's identity,
    which the dataset_resubmit door rebuilds under the same spec_id."""
    outcome = _worded_resolve(_SHARED, source, roles, {"sector": sector})
    plain = _first_resolve(_SHARED, None, roles, {"sector": sector})

    assert isinstance(outcome, Outcome)
    assert outcome.elected == plain.elected == _BROAD
    assert outcome.identity is not None
    assert identity_query(outcome.identity).source is None
    assert outcome.spec_id == plain.spec_id
    resubmitted = _resubmit_door(monkeypatch, outcome)
    assert resubmitted.spec_id == plain.spec_id
    assert resubmitted.resolved.shape.member_digest == member_digest(outcome.members)
    _assert_doors_agree(outcome, index)


@pytest.mark.parametrize("excluded", [True, False], ids=["excluded", "not-excluded"])
def test_an_alternate_under_an_exclusion_mints_the_identity_a_fresh_question_mints(index, excluded):
    """Bind elects `_PINNED` and nominates `_SHARED` beside it, and the part's words exclude
    `_BROAD`. With no bindings, `_DETAIL` would serve `_SHARED` to a donor-and-recipient
    question once `_BROAD` is out, but the alternate's representative `donors_a` binds
    sector A, which no `_DETAIL` row holds, so `_DONORS` serves it. The walked alternate
    names `_DONORS` and mints the identity a fresh question electing `_SHARED` under the
    same exclusion and bindings mints. Without the exclusion it names no table, as the
    plain question does."""
    roled = [
        dataclasses.replace(
            _derived_shape(five_tuple),
            declared_roles=_DONOR_RECIPIENT,
            optional_roles=frozenset({"financingRecipient"}),
        )
        for five_tuple in (_PINNED, _SHARED)
    ]
    scope = _worded_scope(_excluded(_BROAD)) if excluded else None
    pinned, shared = _dataset_candidates(roled, scope)
    shared = dataclasses.replace(
        shared,
        sv_arc_facts={
            row.dcid: _arcs(row)
            for row in _ROWS
            if (row.population_type, row.measured_property) == _SHARED
        },
    )
    state = _repair_state(pinned, _PLACES["donor-recipient"], {})
    state.capped_shapes = [pinned, shared]
    state.bind_result.also_plausible = [1]
    state.scope = scope
    assert asyncio.run(_phase_vet_election(state)) is None
    alternate = state.capped_shapes[1]
    roles = bind_place_roles(state, alternate)
    assert isinstance(roles, PlaceAssignment)

    walked = walk_alternate(
        alternate,
        roles.role_keys,
        "donors_a",
        graph=cast(EngineGraphClient, None),
        scope=state.scope,
    )

    assert walked is not None
    model = {"basis": "G", "flow": "F1", "sector": "A"}
    outcome = _minted(walked.shape, list(walked.bindings), "donor-recipient")
    fresh = (
        _worded_resolve(_SHARED, _excluded(_BROAD), "donor-recipient", model)
        if excluded
        else _first_resolve(_SHARED, None, "donor-recipient", model)
    )
    assert isinstance(fresh, Outcome)
    assert walked.shape.named_dataset == (_DONORS if excluded else None)
    assert outcome.spec_id == fresh.spec_id


@pytest.mark.parametrize(
    ("source", "refused"),
    [(_excluded(_BROAD), [[_BROAD], [_BESIDE]]), (_BESIDE, None)],
    ids=["excluded-none-left", "opt-in-named"],
)
def test_words_that_leave_no_table_refuse_naming_the_excluded_and_opt_in_tables(
    monkeypatch, source, refused
):
    """On a family `_BROAD` and the opt-in `_BESIDE` publish, excluding `_BROAD`
    leaves no table an unnamed part may answer from, so the part refuses
    `qualifier_not_bound` naming `_BROAD`, and `_BESIDE`, which a part answers from only by
    naming it. Naming `_BESIDE` answers from it."""
    _use_index(monkeypatch, _index(rows=_SLOTTED_ROWS))

    outcome = _worded_resolve(
        _SLOTTED, source, "donor-recipient", {"sector": "A"}, index_rows=_SLOTTED_ROWS
    )

    if refused is None:
        assert isinstance(outcome, Outcome)
        assert outcome.elected == _BESIDE
        assert outcome.members == {
            "slotted_beside": ElectedDataset(provenance=_BESIDE, member_provenances=(_BESIDE,))
        }
    else:
        assert isinstance(outcome, RegionResult)
        assert outcome.no_data_reason == "qualifier_not_bound"
        assert [
            [ref.ref for ref in record.named]
            for record in (*outcome.unbound_qualifiers, *outcome.unbound_sources)
        ] == refused


@pytest.mark.parametrize(
    ("source", "sector", "forced", "refused"),
    [
        (_DETAIL, "A", False, True),
        (_BESIDE, "B", False, True),
        (_DETAIL, "A", True, True),
        (_DETAIL, "D", False, False),
        (_DETAIL, "D", True, False),
    ],
    ids=[
        "named-without-the-code",
        "opt-in-named-without-the-code",
        "named-without-the-forced-code",
        "named-with-the-code",
        "named-with-the-forced-code",
    ],
)
def test_words_naming_a_table_refuse_a_bound_code_the_table_lacks(
    index, source, sector, forced, refused
):
    """A part whose words name `_DETAIL`, or the opt-in `_BESIDE`, is offered that table's
    sectors alone, so a model binding sector A or B, which no row of the named table
    holds, refuses `qualifier_not_bound` naming the sector, as a part naming the table by
    its ref refuses the term once bound. A sector the query text names, which the engine
    forces into the bindings after the model's (`_force_qualifier_bindings`), refuses
    alike. A sector the named table holds binds."""
    match = QualifierMatch(
        code_system="sector", code_dcid=sector, label=sector, source="code", matched_text=sector
    )
    outcome = _worded_resolve(
        _SHARED,
        source,
        "donor-recipient",
        {} if forced else {"sector": sector},
        admitted=(match,) if forced else (),
    )

    if refused:
        assert isinstance(outcome, RegionResult)
        assert outcome.no_data_reason == "qualifier_not_bound"
        assert [[ref.ref for ref in record.named] for record in outcome.unbound_qualifiers] == [
            [sector]
        ]
    else:
        assert isinstance(outcome, Outcome)
        assert outcome.elected == source


@pytest.mark.parametrize(
    ("source", "sector", "elected"),
    [
        (_excluded(_BROAD), "A", _DONORS),
        (_excluded(_BROAD), "C", _DETAIL),
        (_excluded(_DETAIL), "D", None),
    ],
    ids=["excluded-broad-sector-a", "excluded-broad-sector-c", "excluded-detail-sector-d"],
)
def test_an_exclusion_reads_a_forced_code_as_it_reads_a_drafted_one(index, source, sector, elected):
    """A sector the query text names reaches the bindings as a forced match
    (`_force_qualifier_bindings`), and the exclusion reads it as it reads the same sector
    the model drafts. Once `_BROAD` is out, sector A, which `_DETAIL` lacks, names
    `_DONORS`, and the shared sector C names `_DETAIL`. Once `_DETAIL` is out, no remaining
    table holds sector D, so the part refuses `qualifier_not_bound` naming the sector and
    never answers from the excluded table."""
    match = QualifierMatch(
        code_system="sector", code_dcid=sector, label=sector, source="code", matched_text=sector
    )

    drafted = _worded_resolve(_SHARED, source, "donor-recipient", {"sector": sector})
    forced = _worded_resolve(_SHARED, source, "donor-recipient", {}, admitted=(match,))

    for outcome in (drafted, forced):
        if elected is None:
            assert isinstance(outcome, RegionResult)
            assert outcome.no_data_reason == "qualifier_not_bound"
            assert [[ref.ref for ref in record.named] for record in outcome.unbound_qualifiers] == [
                [sector]
            ]
        else:
            assert isinstance(outcome, Outcome)
            assert outcome.elected == elected
            assert outcome.shape is not None
            assert outcome.shape.named_dataset == elected
    if elected is not None:
        assert isinstance(drafted, Outcome)
        assert isinstance(forced, Outcome)
        assert forced.spec_id == drafted.spec_id


@pytest.mark.parametrize(
    ("excluded", "unheld"),
    [(frozenset(), []), (frozenset({_DETAIL}), [["D"]])],
    ids=["nothing-excluded", "excluded-detail"],
)
def test_the_code_check_reads_the_part_s_tables_less_the_excluded_ones(index, excluded, unheld):
    """Only `_DETAIL` holds sector D on `_SHARED`. The unnamed part's shape is stamped with
    every default publisher, so the check holds sector D, unless the part excludes
    `_DETAIL` and candidacy stamps the shape without it."""
    (shape,) = _dataset_candidates([_derived_shape(_SHARED)], part_scope(excluded=excluded))
    sector_d = SlotBindingDraft(axis="how", property_dcid="sector", kind="value", value_dcids=["D"])

    refused = _unheld_bound_codes([sector_d], shape)

    assert [[ref.ref for ref in record.named] for record in refused] == unheld


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
def test_a_named_part_refuses_a_code_only_another_table_holds_once_bound(
    index, monkeypatch, named, variable, refused
):
    """`_DETAIL` and `_BROAD` both carry `sector` on `_SHARED`, but no `_DETAIL` row holds
    sector A or B and no `_BROAD` row holds sector D. A part naming one table and a sector
    only the other holds goes on to bind and records the sector as unbound with evidence
    "unavailable". Once its bindings are final the part refuses `qualifier_not_bound`
    naming the sector, because no row of its scope's tables holds the forced code
    (`_unheld_bound_codes`). A part naming the table that holds the sector records
    nothing unavailable and binds it."""
    state = _repair_state(_derived_shape(_SHARED), {}, {})
    state.variable = variable
    state.shapes_found = [_derived_shape(_SHARED)]
    state.scope = part_scope(named)
    monkeypatch.setattr(phase_rank_bind_module, "_record_shape_step", _no_record)
    monkeypatch.setattr(
        phase_election_module,
        "get_measure_index",
        lambda: MeasureIndex(axes=(), values=(), partition_labels=()),
    )
    sector = variable.split()[-2]

    assert asyncio.run(_phase_prune_and_rank(state)) is None
    unavailable = [
        [ref.ref for ref in record.named]
        for record in _resolve_query_qualifiers(
            dataclasses.replace(state, shape_draft=state.capped_shapes[0])
        )
        if record.evidence == "unavailable"
    ]
    vetted = asyncio.run(_phase_vet_election(state))

    assert unavailable == ([[sector]] if refused else [])
    if refused:
        assert vetted is not None
        assert vetted.no_data_reason == "qualifier_not_bound"
        assert [[ref.ref for ref in record.named] for record in vetted.unbound_qualifiers] == [
            [sector]
        ]
    else:
        assert vetted is None
        assert asyncio.run(_phase_repair_bindings(state)) is None
        assert [b.value_dcids for b in state.bindings if b.property_dcid == "sector"] == [[sector]]


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
    and the shape only `_ALONE` publishes would name no dataset."""
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


def _flow_shape(
    five_tuple: tuple[str, str],
    *,
    roles: tuple[str, ...],
    index_rows: tuple[StatVarRow, ...] = _ROWS,
    source: str | None = None,
    scope: SourceScope | None = None,
) -> ShapeDraft:
    """The candidate shape of a part under `scope`, else naming `source` or none, on
    `five_tuple` among `index_rows`, declaring `roles`, of which the recipient is
    optional."""
    (shape,) = _dataset_candidates(
        [_derived_shape(five_tuple, index_rows=index_rows)],
        scope if scope is not None else part_scope(source),
    )
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
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        rcl=Recall(
            candidates=[],
            resolutions={},
        ),
        place_sets=place_sets({name: _DCIDS[name] for name in places}),
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
    bound = bind_place_roles(state, shape)
    if isinstance(role_keys, str):
        assert bound == role_keys
        role_keys = set()
    else:
        assert isinstance(bound, PlaceAssignment)
        assert bound.role_keys.bound == role_keys

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
    assignment = bind_place_roles(state, shape)
    assert isinstance(assignment, PlaceAssignment)
    roles = settled_place_roles(shape, assignment)
    query = shape_query(shape, bindings, assignment.role_keys.bound)
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

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        return {}


def _menus(outcome: Outcome) -> dict[str, set[str]]:
    """The values of the answer's slot menus for `outcome`'s identity, by property."""
    assert outcome.shape is not None
    grounded = _ground_answer(
        outcome.shape,
        list(outcome.bindings),
        sorted(outcome.members),
        {_DONOR_PLACE: "origin"},
        cast(EngineGraphClient, _Labels()),
        open_axes=frozenset(),
    )
    slot_options = grounded[7]
    return {
        cast(GraphRef, menu.key.property).ref: {
            cast(GraphRef, option.value.ref).ref for option in menu.options
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
                key=SlotKey(axis="how", property=GraphRef(ref=prop, label=prop), label=prop),
                options=[
                    SlotOption(
                        value=SlotValue(ref=GraphRef(ref=code, label=code), value_kind="enum_value")
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
        open_axes=frozenset(),
    )
    assert warning is None
    return {
        (cast(GraphRef, menu.key.property).ref, cast(GraphRef, option.value.ref).ref): (
            option.coverage.observation_count
        )
        for menu in enriched
        for option in menu.options
        if isinstance(option.coverage, CoverageExact)
    }


def test_a_donor_total_and_its_breakdowns_are_two_families(index):
    """`_TOTALS` carries no sector on `_DRILLED`, so its rows form a family apart from
    `_BROAD`'s breakdowns by sector and channel (`CodeIndex.family_of`), and derive builds a
    shape of each. The donor total's shape elects `_TOTALS` and offers the channel menu
    alone; the breakdowns' shape elects `_BROAD` and offers channel and sector. `shared_k3`,
    which both tables record, is a member of each family."""
    total = _first_resolve(_DRILLED, None, "donor", {}, family_of=_TOTALS)

    assert total.elected == _TOTALS
    assert set(total.members) == {"total", "shared_k3"}
    assert _menus(total) == {"channel": set(_CODES["channel"])}

    breakdowns = _first_resolve(_DRILLED, None, "donor", {}, family_of=_BROAD)

    assert breakdowns.elected == _BROAD
    assert "shared_k3" in breakdowns.members
    assert _menus(breakdowns) == {
        "channel": set(_CODES["channel"]),
        "sector": set(_CODES["sector"]),
    }


def test_a_named_donor_total_offers_no_menu_its_table_does_not_carry(index):
    """Named, `_TOTALS` serves its own family, whose rows carry channel alone. A part naming
    `_TOTALS` keeps no shape of `_BROAD`'s family, which `_TOTALS` does not publish."""
    outcome = _first_resolve(_DRILLED, _TOTALS, "donor", {}, family_of=_TOTALS)

    assert outcome.elected == _TOTALS
    assert _menus(outcome) == {"channel": set(_CODES["channel"])}
    assert _first_resolve(_DRILLED, _TOTALS, "donor", {}, family_of=_BROAD).datasets is None


class _EchoLabels:
    """A graph that labels every node with its dcid."""

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        return {dcid: dcid for dcid in dcids}


@pytest.mark.parametrize(
    ("source", "family_of", "menus"),
    [
        (None, _BROAD, {"channel", "sector"}),
        (None, _TOTALS, {"channel"}),
        (_TOTALS, _TOTALS, {"channel"}),
    ],
    ids=["unnamed-breakdowns", "unnamed-total", "named-total"],
)
def test_the_shape_event_offers_the_menus_of_the_datasets_the_part_answers_from(
    index, source, family_of, menus
):
    """Before bind, a part offers a menu when the datasets of its shape's family it may
    answer from carry the slot's property."""
    (shape,) = _dataset_candidates(
        [_derived_shape(_DRILLED, family_of=family_of)], part_scope(source)
    )

    event = asyncio.run(
        _build_shape_event(shape, variable_index=0, graph=cast(EngineGraphClient, _EchoLabels()))
    )

    assert {cast(GraphRef, menu.key.property).ref for menu in event.slot_options} == menus


# A `_TOTALS` donor row of channel K1, beside `_BROAD`'s `drill_a`.
_TOTALS_K1 = _row("totals_k1", {"channel": "K1"}, _TOTALS, roles=_DONOR, five_tuple=_DRILLED)


@pytest.mark.parametrize(
    "extra",
    [(), (_TOTALS_K1,)],
    ids=["broad-serves-k1", "totals-records-k1-in-its-own-family"],
)
def test_each_option_counts_the_facets_of_the_table_its_election_serves(monkeypatch, extra):
    """Each option of the breakdowns' answer counts only the points of the table that serves
    it once bound, `_BROAD`. Channel K3, which both tables record, counts `_BROAD`'s points.
    A `_TOTALS` row of K1 belongs to the totals' own family, so the K1 option still elects
    `_BROAD` and counts `drill_a` alone."""
    rows = (*_ROWS, *extra)
    _use_index(monkeypatch, _index(rows=rows))
    outcome = _first_resolve(_DRILLED, None, "donor", {}, family_of=_BROAD)

    assert _option_points(
        outcome, rows=rows, points={**_DRILLED_POINTS, "totals_k1": {_TOTALS: 1}}
    ) == {
        ("channel", "K1"): 2,
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
# publishes `collide_broad`, a row of sector A that places the donor under another role
# beside a recipient. Neither role set holds the other, so `_BROAD`'s election of sector A
# ties: a pick of flow F1 elects `_BROAD` and a pick of sector A elects `_DETAIL`, whose
# one role class places the donor exactly.
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
    _row(
        "collide_broad",
        {"sector": "A"},
        _BROAD,
        roles=("creditor", "financingRecipient"),
        five_tuple=_COLLIDE,
    ),
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
    open_axes: frozenset[RoleAxis] = frozenset()

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
# sector by channel, and a second table of the same family, which carries channel too,
# reports sector A alone.
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
                _row(
                    "guard_real_b_k2", {"channel": "K2", "sector": "B"}, second, five_tuple=_GUARDED
                ),
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

    reason = inadmissible_reason(
        query_pins("guard amount", quantity=quantity_requirement(["guard amount"])),
        elected,
        alternate,
    )

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
        query_pins("amount", quantity=quantity_requirement(["amount"]), bindings=bindings),
        outcome.shape,
        role_keys=lambda _s: RoleKeys(bound=_FLOW_KEYS),
    )

    reason = judge.verdict(_real_terms_alternate(constraints)).reason

    assert {member.dcid for member in judge.served} == set(outcome.members)
    if refusal is None:
        assert reason is None
    else:
        assert reason is not None
        assert reason.startswith(refusal)


@pytest.mark.parametrize(
    ("disjoint", "refusal", "named"),
    [(True, None, {"A", "A2"}), (False, "concept:sector", {"A"})],
)
def test_the_guard_widens_an_elected_code_as_its_binding_reads(
    monkeypatch, disjoint, refusal, named
):
    """Sector A2 is a child of A, which `broad_a` holds. Where parent rows are apart from
    their children's, A's reading spans A2 and an alternate on A2 is that reading. Where a
    parent row is its children's total, A names A alone, and A2 is another concept. A
    query naming A pins the same values."""
    index = _index(
        codes={**_CODES, "sector": (*_CODES["sector"], "A2")},
        parents={"A2": "A"},
        disjoint=disjoint,
    )
    _use_index(monkeypatch, index)
    outcome = _first_resolve(_SHARED, None, "donor-recipient", {"sector": "A"})
    assert outcome.shape is not None
    judge = AlternateJudge(
        query_pins("amount", quantity=quantity_requirement(["amount"]), bindings=outcome.bindings),
        outcome.shape,
        role_keys=lambda _s: RoleKeys(bound=_FLOW_KEYS),
    )

    reason = judge.verdict(_real_terms_alternate({"sector": "A2"})).reason

    assert {member.dcid for member in judge.served} == {"broad_a"}
    match = QualifierMatch(
        code_system="sector", code_dcid="A", label="A", source="label", matched_text="A"
    )
    assert query_pins(
        "amount", quantity=quantity_requirement(["amount"]), qualifier_matches=[match]
    ).constraint_matches == {"sector": frozenset(named)}
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
    answered = canonical_pin_bindings(shape, drafted, RoleKeys())

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
    """Over every index family of the fixture index, every binding of its properties, the
    datasets publishing it and three role-key sets, the canonical query elects the query's
    family and is its own canonical form, and every variant that binds another subset of
    the elected dataset's invariants and elects the same family has the same canonical
    form."""
    index = _index(rows=_ROWS + _TWO_PIN_ROWS, ranks=_TWO_PIN_RANKS)
    variants = 0
    for five_tuple, index_family in sorted(
        (
            (five_tuple, found)
            for five_tuple, families in index.families_by_five_tuple.items()
            for found in families
        ),
        key=lambda item: (item[0].measured_property or "", item[1].datasets),
    ):
        rows = index_family.rows
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
        sources = (None, *index_family.datasets)
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
    assert isinstance(bound, PlaceAssignment)
    role_query = shape_query(shape, bindings, bound.role_keys.bound)
    elected = elected_role_class_shape(shape, role_query).value
    rebound = bound_after_election(state, shape, bound, elected)
    assert isinstance(rebound, PlaceAssignment)
    assert is_role_bearing(elected) is decided
    assert (rebound is bound) is decided
    assert rebound.sets == {}

    slots = [
        build_slot(
            key,
            draft,
            [GraphRef(ref=value, label=value) for value in draft.value_dcids]
            if draft is not None
            else [],
            property_ref=GraphRef(ref=key.property_dcid, label=key.property_dcid)
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
        open_roles=(),
    )

    wire = {
        cast(GraphRef, slot.key.property).ref: cast(GraphRef, slot.binding.value.ref).ref
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
    (shape,) = _dataset_candidates([_derived_shape(_PINNED)], None)

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


# ---------------------------------------------------------------------------
# Role axes: each dataset of a family declares its own
# ---------------------------------------------------------------------------

_PARTIAL = "provenance/PartialLedger"
_FULL = "provenance/FullLedger"

# `_PARTIAL` ranks first and publishes a donor-only total beside one donor-and-recipient
# flow, as DAC1's provider totals hold two flows to the West Bank and Gaza Strip. `_FULL`
# publishes every row for donor and recipient. Both carry `sector`, so they are one family.
_PARTIAL_ROWS = (
    _row("partial_a", {"sector": "A"}, _PARTIAL, roles=_DONOR),
    _row("partial_b", {"sector": "B"}, _PARTIAL),
)
_FULL_ROWS = (
    _row("full_a", {"sector": "A"}, _FULL),
    _row("full_b", {"sector": "B"}, _FULL),
)


# Rows no dataset publishes: an index export whose five-tuple has no coverage row.
_UNPUBLISHED_ROWS = (
    _row("unpublished_a", {"sector": "A"}),
    _row("unpublished_b", {"sector": "B"}),
)

_THIRD = "provenance/ThirdLedger"
_THIRD_ROWS = (_row("third_a", {"sector": "A"}, _THIRD),)


def _axis_index(
    monkeypatch: pytest.MonkeyPatch,
    rows: tuple[StatVarRow, ...],
    *,
    opt_in: frozenset[str] = frozenset(),
) -> CodeIndex:
    index = _index(rows=rows, ranks={_PARTIAL: 0, _FULL: 1, _THIRD: 2}, opt_in=opt_in)
    _use_index(monkeypatch, index)
    return index


@pytest.mark.parametrize(
    ("rows", "named", "lacks"),
    [
        (_PARTIAL_ROWS + _FULL_ROWS, None, False),
        (_PARTIAL_ROWS + _FULL_ROWS, _FULL, False),
        (_PARTIAL_ROWS + _FULL_ROWS, _PARTIAL, True),
        (_PARTIAL_ROWS, None, True),
        (_UNPUBLISHED_ROWS, None, False),
        (_PARTIAL_ROWS + _UNPUBLISHED_ROWS, None, True),
    ],
    ids=[
        "beside-a-full-publisher",
        "naming-the-full-publisher",
        "naming-the-partial-publisher",
        "partial-publisher-alone",
        "no-publisher",
        "partial-publisher-beside-unpublished-rows",
    ],
)
def test_a_shape_declares_the_recipient_when_a_table_the_part_reads_holds_it_on_every_row(
    monkeypatch, rows, named, lacks
):
    """ "aid to Kenya" needs the recipient. `_FULL` holds it on every row, so the unnamed
    part keeps the shape though `_PARTIAL`'s total lacks it. Alone, `_PARTIAL` holds it on
    one row only, and the shape lacks it, as DAC1's provider totals do. A part naming
    `_PARTIAL` reads that table alone (`_dataset_candidates`), so its shape lacks the
    recipient beside `_FULL`, as the election of its identity finds. A family no dataset
    publishes declares what its rows hold. Beside `_PARTIAL` those rows serve no part, so
    the shape lacks the recipient as `_PARTIAL` alone does."""
    _axis_index(monkeypatch, rows)
    shape = _flow_shape(_SHARED, roles=_DONOR_RECIPIENT, index_rows=rows, source=named)
    state = _repair_state(shape, {"Kenya": "destination"}, {})

    assert lacks_query_axis(state, shape) is lacks


# `_PARTIAL` and `_FULL` carry `channel` on one row each and `_THIRD` carries `sector`
# alone, so the first two form one family and `_THIRD` another. A shape of the first
# family that keeps the `sector` slot alone selects `_THIRD`'s family by its keys.
_SPLIT_ROWS = (
    _row("partial_a", {"sector": "A"}, _PARTIAL),
    _row("partial_k", {"sector": "A", "channel": "K1"}, _PARTIAL),
    _row("full_a", {"sector": "A"}, _FULL),
    _row("full_b", {"sector": "B"}, _FULL),
    _row("full_k", {"sector": "B", "channel": "K1"}, _FULL),
    _row("third_a", {"sector": "A"}, _THIRD),
    _row("third_b", {"sector": "B"}, _THIRD),
)
# The same families, where `_PARTIAL` holds donor totals and `_FULL` each recipient.
_SPLIT_TOTAL_ROWS = (
    _row("partial_a", {"sector": "A"}, _PARTIAL, roles=_DONOR),
    _row("partial_k", {"sector": "A", "channel": "K1"}, _PARTIAL, roles=_DONOR),
    _row("full_a", {"sector": "A"}, _FULL),
    _row("full_k", {"sector": "A", "channel": "K1"}, _FULL),
    _row("third_a", {"sector": "A"}, _THIRD),
)
_THIRD_TOTAL = _row("third_total", {"sector": "A"}, _THIRD, roles=_DONOR)
_PARTIAL_TOTAL = _row("partial_total", {"sector": "A"}, _PARTIAL, roles=_DONOR)
# One family, since every row carries `channel`. `_PARTIAL` holds sector A and channel K2
# on separate rows, and `_FULL` alone holds them on one.
_JOINT_ROWS = (
    _row("partial_a_k1", {"sector": "A", "channel": "K1"}, _PARTIAL),
    _row("partial_b_k2", {"sector": "B", "channel": "K2"}, _PARTIAL),
    _row("full_a_k2", {"sector": "A", "channel": "K2"}, _FULL),
)
# The same, where `_PARTIAL` holds sector A on a row with no channel.
_JOINT_BARE_ROWS = (
    _row("partial_a", {"sector": "A"}, _PARTIAL),
    _row("partial_b_k2", {"sector": "B", "channel": "K2"}, _PARTIAL),
    _row("full_a_k2", {"sector": "A", "channel": "K2"}, _FULL),
)
# `_PARTIAL` holds sector A only beside a channel, and `_THIRD`, a family of its own,
# holds it alone. A shape of `_PARTIAL`'s family that keeps the `sector` slot alone
# selects `_THIRD`'s family by its keys.
_OTHER_FAMILY_ROWS = (
    _row("partial_b", {"sector": "B"}, _PARTIAL),
    _row("partial_a_k2", {"sector": "A", "channel": "K2"}, _PARTIAL),
    _row("third_a", {"sector": "A"}, _THIRD),
    _row("third_b", {"sector": "B"}, _THIRD),
)


@dataclasses.dataclass(frozen=True)
class NamedPart:
    """One part on `_SHARED` among `rows`, and the table its identity names and serves."""

    rows: tuple[StatVarRow, ...]
    sector: str
    bound: frozenset[str]
    names: str | None
    members: frozenset[str]
    """The members every door serves. Empty for a part no table of its own serves, which
    refuses (`PartNotServed`)."""
    open: frozenset[str] = frozenset()
    excluded: frozenset[str] = frozenset()
    recorded_in: str | None = None
    """The row whose family the shape is built from, keeping the `sector` slot alone. A
    shape with none is derived from the five-tuple's first family."""
    channel: str | None = None
    """The channel the part binds beside its sector, if any."""
    opt_in: frozenset[str] = frozenset()

    @property
    def codes(self) -> dict[str, str]:
        return {"sector": self.sector} | ({"channel": self.channel} if self.channel else {})


_ROLE_PLACES = {"origin": "country/DEU", "destination": "country/KEN"}

_NAMED_PARTS = {
    "one-family": NamedPart(
        rows=_PARTIAL_ROWS + _FULL_ROWS + _THIRD_ROWS,
        sector="A",
        bound=frozenset({"destination"}),
        names=None,
        members=frozenset({"full_a"}),
    ),
    "excluding-the-elected-table": NamedPart(
        rows=_PARTIAL_ROWS + _FULL_ROWS + _THIRD_ROWS,
        sector="A",
        bound=frozenset({"destination"}),
        excluded=frozenset({_FULL}),
        names=_THIRD,
        members=frozenset({"third_a"}),
    ),
    "no-table-holds-the-code": NamedPart(
        rows=_PARTIAL_ROWS + _FULL_ROWS + _THIRD_ROWS,
        sector="C",
        bound=frozenset({"destination"}),
        excluded=frozenset({_FULL}),
        names=None,
        members=frozenset(),
    ),
    "excluded-table-alone-holds-the-codes-together": NamedPart(
        rows=_JOINT_ROWS,
        sector="A",
        channel="K2",
        bound=frozenset({"destination"}),
        excluded=frozenset({_FULL}),
        names=None,
        members=frozenset(),
    ),
    "excluded-table-alone-holds-the-codes-together-beside-a-bare-row": NamedPart(
        rows=_JOINT_BARE_ROWS,
        sector="A",
        channel="K2",
        bound=frozenset({"destination"}),
        excluded=frozenset({_FULL}),
        names=None,
        members=frozenset(),
    ),
    "keys-select-a-default-family-that-alone-holds-the-code": NamedPart(
        rows=_OTHER_FAMILY_ROWS,
        sector="A",
        bound=frozenset({"destination"}),
        recorded_in="partial_b",
        names=None,
        members=frozenset(),
    ),
    "keys-select-an-opt-in-family-that-alone-holds-the-code": NamedPart(
        rows=_OTHER_FAMILY_ROWS,
        sector="A",
        bound=frozenset({"destination"}),
        recorded_in="partial_b",
        opt_in=frozenset({_THIRD}),
        names=None,
        members=frozenset(),
    ),
    "keys-select-another-family": NamedPart(
        rows=_SPLIT_ROWS,
        sector="B",
        bound=frozenset({"destination"}),
        recorded_in="partial_a",
        names=_FULL,
        members=frozenset({"full_b"}),
    ),
    "open-role-keys-select-another-family": NamedPart(
        rows=_SPLIT_TOTAL_ROWS,
        sector="A",
        bound=frozenset({"origin"}),
        open=frozenset({"destination"}),
        recorded_in="partial_a",
        names=_FULL,
        members=frozenset({"full_a"}),
    ),
    "open-role-excluding-the-table-that-lists-it": NamedPart(
        rows=(*_FULL_ROWS, _THIRD_TOTAL),
        sector="A",
        bound=frozenset({"origin"}),
        open=frozenset({"destination"}),
        excluded=frozenset({_FULL}),
        names=_THIRD,
        members=frozenset({"third_total"}),
    ),
    "open-role-no-table-lists": NamedPart(
        rows=(_PARTIAL_TOTAL, _THIRD_TOTAL),
        sector="A",
        bound=frozenset({"origin"}),
        open=frozenset({"destination"}),
        names=None,
        members=frozenset({"partial_total"}),
    ),
}


def _named_part_shape(part: NamedPart) -> ShapeDraft:
    if part.recorded_in is None:
        return _derived_shape(_SHARED, index_rows=part.rows)
    recorded = next(row for row in part.rows if row.dcid == part.recorded_in)
    shape = _derived_shape(_SHARED, index_rows=part.rows, family_of=recorded.provenances[0])
    return dataclasses.replace(
        shape,
        slot_keys=tuple(key for key in shape.slot_keys if key.property_dcid != "channel"),
        sv_arc_facts={recorded.dcid: _arcs(recorded)},
    )


def _rows_held(
    rows: tuple[StatVarRow, ...],
    tables: tuple[str, ...],
    keys: frozenset[str],
    codes: Mapping[str, str],
) -> set[str]:
    """The rows of `rows` that one of `tables` publishes, that declare only `keys`, and
    that hold every code of `codes`: read from the rows themselves, with no election."""
    return {
        row.dcid
        for row in rows
        if set(row.provenances) & set(tables)
        and row.constraint_properties.keys() <= keys
        and all(row.constraint_properties.get(prop) == code for prop, code in codes.items())
    }


@pytest.mark.parametrize("part", _NAMED_PARTS.values(), ids=_NAMED_PARTS.keys())
def test_the_table_search_serves_is_the_table_the_mint_elects_for_the_minted_identity(
    monkeypatch, part: NamedPart
):
    """Search reads the part's members under the role keys it asks, from the shape as
    `part_source` names it. The identity it mints states the bound keys, the open keys
    those members serve and the named table, and the mint resolves it alone. Both read one
    table of the part's own (`part_datasets`) and the same members, whichever table the
    part excludes and whichever family its slot keys select. A part none of whose own
    tables holds a row with its codes together has no such identity and is not served
    (`PartNotServed`), though a table it excludes, or a table of the family its slot keys
    select, holds one."""
    index = _axis_index(monkeypatch, part.rows, opt_in=part.opt_in)
    shape = _named_part_shape(part)
    own = part_datasets(shape, part_scope(excluded=part.excluded), index)
    bindings = [
        SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[code])
        for prop, code in part.codes.items()
    ]
    keys = frozenset(key.property_dcid for key in shape.slot_keys if key.property_dcid)
    held = _rows_held(part.rows, own, keys, part.codes)
    assert bool(held) is bool(part.members)

    if not part.members:
        with pytest.raises(PartNotServed):
            part_source(
                shape,
                bindings,
                lambda: RoleKeys(bound=part.bound, open=part.open),
                scope=part_scope(excluded=part.excluded),
            )
        return
    named = part_source(
        shape,
        bindings,
        lambda: RoleKeys(bound=part.bound, open=part.open),
        scope=part_scope(excluded=part.excluded),
    )

    assert named == part.names
    if named is not None:
        bindings.append(source_binding(named))
    query = shape_query(shape, bindings, part.bound, open_role_keys=part.open)
    served, index_sourced = shape_member_facts(shape, query)
    assert index_sourced
    assert served.keys() == part.members
    tables = {dataset.provenance for fact in served.values() if (dataset := fact.read_dataset())}
    assert tables <= set(own)
    assert len(tables) == 1

    family = members_module.index_family(index, query)
    assert family is not None
    document: dict = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _SHARED[0],
            "measured_property": _SHARED[1],
            "stat_type": "measuredValue",
        },
        "slots": [
            {"property": prop, "binding": {"state": "value", "ref": code}}
            for prop, code in sorted(part.codes.items())
        ],
        "entities": {key: [_ROLE_PLACES[key]] for key in sorted(part.bound)},
        "open_roles": sorted(family.role_keys - part.bound),
        "requested_window": {"kind": "all"},
    }
    if named is not None:
        document["facets"] = {"provenance": {"state": "value", "ref": named}}
    minted = _mint(mint_identity(document)[1], index)

    assert minted.keys() == part.members
    assert {dataset.provenance for dataset in minted.values() if dataset} == tables


def _unread_role_keys() -> RoleKeys:
    raise AssertionError("the role keys decide nothing among fewer than two tables")


def test_a_shape_the_index_does_not_carry_names_no_table(monkeypatch):
    """`_PARTIAL` and `_FULL` publish two families of one five-tuple, and a series keyed
    on `channel` alone covers no row of either, so the index does not carry its shape
    though its key selects `_FULL`'s family. No table serves the part, and its identity
    names none, the elected shape's and a walked alternate's alike."""
    rows = (
        _row("partial_a", {"sector": "A"}, _PARTIAL),
        _row("full_a", {"channel": "K1", "sector": "A"}, _FULL),
    )
    index = _axis_index(monkeypatch, rows)
    unindexed = _row("unindexed", {"channel": "K1"})
    shape = dataclasses.replace(
        _derived_shape(_SHARED, index_rows=rows, family_of=_FULL),
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="channel", label="channel"),),
        slot_taxonomy={"how:channel": ["K1"]},
        slot_taxonomy_index_sourced=False,
        sv_arc_facts={unindexed.dcid: _arcs(unindexed)},
    )
    assert shape_datasets(shape, index) == ()
    assert eligible_datasets(index, shape_query(shape, [], ())) == (_FULL,)

    assert part_source(shape, [], _unread_role_keys) is None


def test_the_role_keys_are_not_read_where_one_table_alone_could_serve(monkeypatch):
    """One table publishes the shape and its slot keys select that family, so the part's
    identity names none whatever places the part binds, and the place binding is not
    read."""
    _axis_index(monkeypatch, _FULL_ROWS)
    shape = _flow_shape(_SHARED, roles=_DONOR_RECIPIENT, index_rows=_FULL_ROWS)

    assert part_source(shape, [], _unread_role_keys) is None


@pytest.mark.parametrize(
    ("model", "members"),
    [({}, {"full_a", "full_b"}), ({"sector": "B"}, {"full_b"})],
    ids=["open", "sector-with-a-partial-flow"],
)
def test_a_recipient_question_never_reads_a_publisher_holding_the_recipient_on_some_rows(
    monkeypatch, model, members
):
    """`_PARTIAL` ranks first and its sector B flow holds the recipient, but its total does
    not, so it does not declare the recipient and `_FULL` serves at every door."""
    index = _axis_index(monkeypatch, _PARTIAL_ROWS + _FULL_ROWS)

    outcome = _first_resolve(
        _SHARED, None, "donor-recipient", model, index_rows=_PARTIAL_ROWS + _FULL_ROWS
    )

    assert outcome.elected == _FULL
    assert set(outcome.members) == members
    _assert_doors_agree(outcome, index)


# ---------------------------------------------------------------------------
# A part no table of its own serves
# ---------------------------------------------------------------------------

# One family of `_SHARED`, since every row carries `channel`. `_BROAD` holds sector A and
# channel K2 on separate rows, and `_DETAIL` alone holds them on one.
_JOINT_LEDGER_ROWS = (
    _row("broad_a_k1", {"sector": "A", "channel": "K1"}, _BROAD),
    _row("broad_b_k2", {"sector": "B", "channel": "K2"}, _BROAD),
    _row("detail_a_k2", {"sector": "A", "channel": "K2"}, _DETAIL),
)


def _part_state(
    shape: ShapeDraft,
    roles: str,
    model: Mapping[str, str],
    *,
    open_axes: Collection[RoleAxis] = (),
    scope: SourceScope | None = None,
) -> ResolveState:
    """The state vetting reads for a part electing `shape`, its one candidate, under the
    source scope `scope`, about the entities of `roles` and asking `open_axes` of every
    member, where the bind model binds `model`."""
    state = _repair_state(shape, _PLACES[roles], model)
    state.open_axes = frozenset(open_axes)
    state.capped_shapes = [shape]
    state.scope = scope
    return state


def _settle(monkeypatch: pytest.MonkeyPatch, state: ResolveState) -> RegionResult | None:
    """Run vetting and binding repair on `state`, under no measure axis: the refusal of
    either, or `None` once `state.shape_draft` and `state.bindings` are final."""
    monkeypatch.setattr(
        phase_election_module,
        "get_measure_index",
        lambda: MeasureIndex(axes=(), values=(), partition_labels=()),
    )
    monkeypatch.setattr(phase_election_module, "_resolve_query_qualifiers", lambda _state: [])
    refused = asyncio.run(_phase_vet_election(state))
    if refused is not None:
        return refused
    return asyncio.run(_phase_repair_bindings(state))


def _refused_names(result: RegionResult | None) -> list[list[str]]:
    assert result is not None
    assert result.no_data_reason == "qualifier_not_bound"
    return [
        [ref.ref for ref in record.named]
        for record in (*result.unbound_qualifiers, *result.unbound_sources)
    ]


@pytest.mark.parametrize(
    ("rows", "source", "slots", "model", "refused", "elected"),
    [
        (_JOINT_LEDGER_ROWS, None, (_excluded(_DETAIL),), {"sector": "A", "channel": "K2"},
         [[_DETAIL]], None),
        (_JOINT_LEDGER_ROWS, None, (), {"sector": "A", "channel": "K2"}, None, _DETAIL),
        (_JOINT_LEDGER_ROWS, None, (_excluded(_DETAIL),), {"sector": "A", "channel": "K1"},
         None, _BROAD),
        (_ROWS, None, (_DETAIL,), {"sector": "C", "channel": "K1"}, [[_DETAIL]], None),
        (_ROWS, _DETAIL, (), {"sector": "C", "channel": "K1"}, [[_DETAIL]], None),
        (_ROWS, _DETAIL, (), {"sector": "D", "channel": "K1"}, None, _DETAIL),
    ],
    ids=[
        "excluded-table-alone-holds-the-codes-together",
        "plain-question-reads-that-table",
        "excluded-table-beside-one-that-holds-them",
        "worded-named-table-lacks-the-codes-together",
        "seam-named-table-lacks-the-codes-together",
        "seam-named-table-holds-them",
    ],
)  # fmt: skip
def test_a_part_none_of_whose_tables_holds_its_codes_together_refuses_naming_the_table(
    monkeypatch, rows, source, slots, model, refused, elected
):
    """Each code the part binds is held by a row of a table it may answer from, so the
    code check passes (`_unheld_bound_codes`), but no such table holds them on one row.
    The part refuses `qualifier_not_bound` once its bindings are final: naming the table it
    excludes, where that table alone holds the row, and the table it names, by its ref or
    by its words, where that table lacks it. It answers from no other table. A part
    one of whose tables holds the row answers from it."""
    _use_index(monkeypatch, _index(rows=rows))
    scope = part_scope(source) if source is not None else _worded_scope(*slots)
    shape = _flow_shape(_SHARED, roles=_DONOR_RECIPIENT, index_rows=rows, scope=scope)
    state = _part_state(shape, "donor-recipient", model, scope=scope)

    result = _settle(monkeypatch, state)

    if refused is not None:
        assert _refused_names(result) == refused
        return
    assert result is None
    outcome = _minted(state.shape_draft, state.bindings, "donor-recipient")
    assert outcome.elected == elected
    assert outcome.members


def test_a_part_naming_two_tables_neither_serves_refuses_naming_only_those_tables(monkeypatch):
    """A part names `_BROAD` and `_DETAIL` by their words. `_BROAD` holds sector B and
    channel K1 on separate rows and `_DETAIL` neither, while the opt-in `_BESIDE` holds
    them on one. The part refuses `qualifier_not_bound` naming the two tables it named and
    no table it left out."""
    rows = (*_JOINT_LEDGER_ROWS, _row("beside_b_k1", {"sector": "B", "channel": "K1"}, _BESIDE))
    _use_index(monkeypatch, _index(rows=rows))
    scope = _worded_scope(_BROAD, _DETAIL)
    shape = _flow_shape(_SHARED, roles=_DONOR_RECIPIENT, index_rows=rows, scope=scope)
    state = _part_state(shape, "donor-recipient", {"sector": "B", "channel": "K1"}, scope=scope)

    result = _settle(monkeypatch, state)

    assert result is not None
    assert result.no_data_reason == "qualifier_not_bound"
    assert result.unbound_sources == ()
    _terms, sources, _naming = _unbound_terms(dataclasses.replace(result, source_scope=scope))
    assert [[ref.ref for ref in record.named] for record in sources or ()] == [
        [_BROAD],
        [_DETAIL],
    ]


def test_a_plain_part_with_one_table_is_left_alone_whatever_rows_the_table_holds(monkeypatch):
    """`_BROAD` alone publishes the family and holds sector A and channel K2 on separate
    rows. The part names and excludes no table and its keys select no other, so no other
    table can serve it: settling refuses nothing, names no table, and leaves the shape and
    the model's bindings as they were. Every door then resolves no member for it."""
    rows = _JOINT_LEDGER_ROWS[:2]
    index = _index(rows=rows)
    _use_index(monkeypatch, index)
    shape = _flow_shape(_SHARED, roles=_DONOR_RECIPIENT, index_rows=rows)
    model = {"sector": "A", "channel": "K2"}
    state = _part_state(shape, "donor-recipient", model)
    assert eligible_datasets(index, shape_query(shape, [], ())) == shape.datasets == (_BROAD,)

    assert _settle(monkeypatch, state) is None

    assert state.shape_draft is shape
    assert state.settled_dataset() is None
    assert bound_source(state.bindings) is None
    assert _values(state.bindings) == model
    assert _minted(shape, state.bindings, "donor-recipient").members == {}


def test_a_part_whose_keys_select_another_family_refuses_naming_its_codes(monkeypatch):
    """`_PARTIAL`'s shape keeping the `sector` slot alone holds sector A on no row, and its
    key selects the family of `_THIRD`, which holds it. The part names and excludes no
    table, so it refuses `qualifier_not_bound` naming the sector, and `_THIRD` never
    answers for it."""
    part = _NAMED_PARTS["keys-select-a-default-family-that-alone-holds-the-code"]
    index = _axis_index(monkeypatch, part.rows)
    (shape,) = _dataset_candidates(
        [dataclasses.replace(_named_part_shape(part), declared_roles=_DONOR_RECIPIENT)], None
    )
    query = shape_query(shape, [], ())
    assert eligible_datasets(index, query) == (_THIRD,)

    result = _settle(monkeypatch, _part_state(shape, "donor-recipient", {"sector": "A"}))

    assert _refused_names(result) == [["A"]]


def test_an_excluded_table_does_not_serve_after_a_slot_default(monkeypatch):
    """`sector` declares the default D, which only `_DETAIL` holds. A part excluding
    `_DETAIL` carries the shape of the tables that remain and the codes their rows hold, so
    the default finds no D to bind, and the family the final bindings elect is not
    `_DETAIL`'s."""
    rows = (
        _row("a", {"basis": "G", "flow": "F1", "sector": "A"}, _BROAD),
        _row("b", {"basis": "G", "flow": "F2", "sector": "B"}, _BROAD),
        _row("d", {"basis": "H", "flow": "F3", "sector": "D"}, _DETAIL),
        _row("e", {"basis": "H", "flow": "F3", "sector": "E"}, _DETAIL),
    )
    default = DeclaredDefault(code="D", when_bound_under=(), when_open=True, note="Default D")
    index = _index(rows=rows, defaults={"sector": default})
    _use_index(monkeypatch, index)
    scope = _worded_scope(_excluded(_DETAIL))
    shape = _flow_shape(_SHARED, roles=_DONOR_RECIPIENT, index_rows=rows, scope=scope)
    state = _part_state(shape, "donor-recipient", {}, scope=scope)
    state.bind_result.bindings = [
        *(
            SlotBindingDraft(axis="how", property_dcid=key.property_dcid, kind="unbound")
            for key in shape.slot_keys
            if key.property_dcid
        ),
        *state.bind_result.bindings,
    ]

    assert asyncio.run(_phase_vet_election(state)) is None
    assert state.scope is not None
    assert _DETAIL not in state.scope.allowed
    assert state.shape_draft.datasets == (_BROAD,)
    bindings = _repair(monkeypatch, state)

    elected = members_module.index_family(
        index, shape_query(state.shape_draft, bindings, {"origin", "destination"})
    )
    assert elected is not None
    assert elected.provenance == _BROAD


@pytest.mark.parametrize(
    ("sources", "sector", "elected"),
    [
        ((_DETAIL, _BROAD), "C", _BROAD),
        ((_excluded(_BROAD), _excluded(_DETAIL)), "A", _DONORS),
        ((_DETAIL, _excluded(_BROAD)), "C", _DETAIL),
    ],
    ids=["two-named", "two-excluded", "named-beside-another-excluded"],
)
def test_several_source_words_compose_into_one_scope(index, monkeypatch, sources, sector, elected):
    """Words naming two tables allow both, and the election serves the first that holds
    the row; words excluding two tables leave the rest; words naming one table and
    excluding another allow the named one. Words naming and excluding one table leave
    nothing, and compile refuses them before any phase runs
    (`source_scope.compile_sources`)."""
    scope = _worded_scope(*sources)
    shape = _flow_shape(_SHARED, roles=_DONOR_RECIPIENT, scope=scope)
    state = _part_state(shape, "donor-recipient", {"sector": sector}, scope=scope)

    result = _settle(monkeypatch, state)

    assert result is None
    assert _minted(state.shape_draft, state.bindings, "donor-recipient").elected == elected


def test_an_exclusion_pin_elects_the_family_and_members_of_the_shape_query(index, monkeypatch):
    """Excluding a table the plain question does not elect leaves the shape naming no
    dataset, so its `shape_query` names none, while the identity binds the table that
    served and its `identity_query` names that table. The two queries differ, and elect
    the same family and members."""
    scope = _worded_scope(_excluded(_DETAIL))
    shape = _flow_shape(_SHARED, roles=_DONOR_RECIPIENT, scope=scope)
    state = _part_state(shape, "donor-recipient", {"sector": "C"}, scope=scope)
    assert _settle(monkeypatch, state) is None

    pinned = _minted(state.shape_draft, state.bindings, "donor-recipient", served=(_BROAD,))

    role_keys = entity_role_keys(_ENTITIES["donor-recipient"], open_axes=())
    queries = (
        shape_query(state.shape_draft, pinned.bindings, role_keys.bound),
        identity_query(cast(BaseIdentity, pinned.identity)),
    )
    assert [query.source for query in queries] == [None, _BROAD]
    by_shape, by_identity = (members_module.index_family(index, query) for query in queries)
    assert by_shape is not None
    assert by_shape == by_identity
    assert by_shape.provenance == _BROAD
    shape_members, identity_members = (
        {
            dcid: fact.read_dataset()
            for dcid, fact in shape_member_facts(state.shape_draft, q)[0].items()
        }
        for q in queries
    )
    assert shape_members == identity_members == pinned.members


# The property test's worlds: two or three default tables and, in some, an opt-in one,
# publishing rows of `_SHARED` by sector, by channel, by both or by neither.
_WORLD_TABLES = (_BROAD, _DETAIL, _DONORS)
_OPEN_AXES: tuple[tuple[RoleAxis, ...], ...] = (
    (),
    ("origin",),
    ("destination",),
    ("origin", "destination"),
)
# Role sets that place one flow axis, two of them beside a role on neither axis.
_ONE_AXIS_ROLES = (
    _DONOR,
    ("financingRecipient",),
    ("financingProvider", "financingChannel"),
    ("financingRecipient", "financingChannel"),
)
_WORLD_CODES = {"sector": ("A", "B"), "channel": ("K1", "K2")}
_WORLD_FIVE_TUPLE = FiveTuple(
    population_type=_SHARED[0], measured_property=_SHARED[1], stat_type="measuredValue"
)


def _world_rows(rng: random.Random) -> tuple[StatVarRow, ...]:
    tables = rng.sample(_WORLD_TABLES, rng.choice((2, 3)))
    if rng.random() < 0.35:
        tables.append(_BESIDE)
    rows: dict[tuple, StatVarRow] = {}
    for number in range(rng.randint(3, 7)):
        constraints = {
            prop: rng.choice(codes)
            for prop, codes in _WORLD_CODES.items()
            if rng.random() < (0.85 if prop == "sector" else 0.4)
        }
        published = rng.sample(tables, 2 if rng.random() < 0.15 else 1)
        roles = rng.choice((_DONOR, _DONOR_RECIPIENT, _DONOR_RECIPIENT))
        rows[tuple(sorted(constraints.items())), roles, published[0]] = _row(
            f"r{number}", constraints, *published, roles=roles
        )
    return tuple(rows.values())


def _one_axis_world_rows(rng: random.Random) -> tuple[StatVarRow, ...]:
    """A world whose three tables each publish rows by sector under one role set of
    `_ONE_AXIS_ROLES`, so no table ties alone and the tables read different open axes."""
    rows = []
    for table, roles in zip(_WORLD_TABLES, rng.sample(_ONE_AXIS_ROLES, 3), strict=True):
        for sector in rng.sample(_WORLD_CODES["sector"], rng.choice((1, 2))):
            rows.append(_row(f"r{len(rows)}", {"sector": sector}, table, roles=roles))
    return tuple(rows)


def _world_shapes(
    index: CodeIndex, rows: tuple[StatVarRow, ...]
) -> list[tuple[ShapeDraft, tuple[str, ...]]]:
    """Each family of the world as a shape with every slot key of the family, and with one
    key left out where a row of the family declares the others alone, beside the tables
    publishing the family. A shape whose keys the index does not decide is left out, and
    so is one the index reads as another family's (`shape_datasets`), as it does a shape
    recorded in a row that tables of two families publish."""
    shapes: list[tuple[ShapeDraft, tuple[str, ...]]] = []
    for family in index.families_by_five_tuple[_WORLD_FIVE_TUPLE]:
        full = dataclasses.replace(
            _derived_shape(_SHARED, index_rows=rows, family_of=family.datasets[0]),
            declared_roles=_DONOR_RECIPIENT,
            optional_roles=frozenset({"financingRecipient"}),
        )
        variants = [full]
        for left_out in sorted(family.properties):
            kept = family.properties - {left_out}
            recorded = next(
                (row for row in family.rows if row.constraint_properties.keys() <= kept), None
            )
            if recorded is not None:
                variants.append(
                    dataclasses.replace(
                        full,
                        slot_keys=tuple(
                            key for key in full.slot_keys if key.property_dcid != left_out
                        ),
                        sv_arc_facts={recorded.dcid: _arcs(recorded)},
                    )
                )
        shapes.extend(
            (shape, family.datasets)
            for shape in variants
            if members_module.index_decides(index, shape_query(shape, [], ()))
            and shape_datasets(shape, index) == family.datasets
        )
    return shapes


def test_no_door_serves_a_part_from_a_table_it_excludes_or_does_not_own(monkeypatch):
    """Over seeded random worlds, each part is one shape, a choice of table (none, one
    named by its ref, one named by its words, one excluded by its words), codes for some of its
    slots, and places. The rows the part may be served are read from the world itself: the
    rows of the shape's own tables, the named one alone or the default ones less the
    excluded, that declare only the shape's keys and hold every code (`_rows_held`).

    Where there are none, the part refuses `qualifier_not_bound` and names what it cannot
    bind, or `variable_not_resolved` when it names and excludes no table and binds no
    code. A part that names and excludes no table, has at most one table of its own and
    whose keys select no other table is the exception: no other table can serve it, so it
    is not refused, and no door resolves a member for it. A part naming the one opt-in
    table that alone publishes its shape mints that part's identity and is treated alike.
    Where there are some, the first answer's members are among them, and the mint,
    the resubmitted identity and the stated members read the identity back to the same
    table and members.

    Each part also asks no open axis, one, or both, less the axes its places bind. Search's
    role path (`_admit_population`) reads the part under the axes asked, the identity
    names the open roles it records, and the mint serves exactly the members it admits,
    from the same table. Two worlds in seven hold three tables that each place one flow
    axis (`_one_axis_world_rows`), where an admitted member can declare a role on an axis
    the election set aside and the identity omits that axis. Where the rows declare role
    sets the election cannot order, Search's role path and the mint both refuse
    `role_signature_incompatible`, and no member count is stated. Where no row places
    every bound role, Search refuses `role_incompatible`."""
    rng = random.Random(1165)
    answered = refused = tied = unplaced = set_aside = 0
    for world in range(280):
        rows = _one_axis_world_rows(rng) if world % 7 >= 5 else _world_rows(rng)
        with monkeypatch.context() as patched:
            index = _index(rows=rows)
            _use_index(patched, index)
            published = index.datasets_by_five_tuple[_WORLD_FIVE_TUPLE]
            for _part in range(12):
                shapes = _world_shapes(index, rows)
                if not shapes:
                    break
                base, family_tables = rng.choice(shapes)
                kind = rng.choice(("plain", "seam", "words", "excluded"))
                table = rng.choice(published)
                roles = rng.choice(("none", "donor", "donor-recipient"))
                open_axes = tuple(
                    axis for axis in rng.choice(_OPEN_AXES) if axis not in _PLACES[roles].values()
                )
                keys = frozenset(key.property_dcid for key in base.slot_keys if key.property_dcid)
                model = {
                    prop: rng.choice(_WORLD_CODES[prop])
                    for prop in sorted(keys)
                    if rng.random() < 0.6
                }
                scope = {
                    "seam": part_scope(table),
                    "words": _worded_scope(table),
                    "excluded": _worded_scope(_excluded(table)),
                }.get(kind)
                candidates = _dataset_candidates([base], scope)
                if not candidates:
                    continue
                (shape,) = candidates
                default = tuple(t for t in family_tables if t not in _OPT_IN)
                if kind in ("seam", "words"):
                    own = (table,) if table in family_tables else ()
                else:
                    own = tuple(
                        t for t in default or family_tables if kind != "excluded" or t != table
                    )
                held = _rows_held(rows, own, keys, model)
                state = _part_state(shape, roles, model, open_axes=open_axes, scope=scope)

                result = _settle(patched, state)

                case = (rows, kind, table, roles, open_axes, model, sorted(keys))
                unnamed = default or family_tables
                selected = eligible_datasets(index, shape_query(base, [], ()))
                alone = (
                    len(unnamed) <= 1
                    and set(selected) <= set(unnamed)
                    and (
                        kind == "plain"
                        or (kind in ("seam", "words") and table in _OPT_IN and own == unnamed)
                    )
                )
                if not held and alone:
                    assert result is None, case
                    assert not _minted(
                        state.shape_draft, state.bindings, roles, open_axes=open_axes
                    ).members, case
                    continue
                if not held:
                    assert result is not None, case
                    if kind == "plain" and not model:
                        assert result.no_data_reason == "variable_not_resolved", case
                    else:
                        assert result.no_data_reason == "qualifier_not_bound", case
                        assert result.unbound_qualifiers or result.unbound_sources, case
                    refused += 1
                    continue
                assert result is None, case
                read = _minted(state.shape_draft, state.bindings, roles, open_axes=open_axes)
                assert read.members, case
                assert read.members.keys() <= held, case
                assert read.elected in own, case
                admitted = _admitted(read, roles, rows)
                if isinstance(admitted, NoDataDraft):
                    assert read.identity is not None
                    if admitted.reason == "role_signature_incompatible":
                        with pytest.raises(RoleSignatureIncompatible):
                            _mint(read.identity, index)
                        assert stated_members(read.identity, None) is None, case
                        tied += 1
                    else:
                        assert admitted.reason == "role_incompatible", case
                        if not open_axes:
                            # No row places every bound role, so the mint keeps every row
                            # for its reproducibility check, or refuses their role sets.
                            with contextlib.suppress(RoleSignatureIncompatible):
                                assert _mint(read.identity, index) == read.members, case
                        unplaced += 1
                    continue
                outcome = _minted(
                    state.shape_draft,
                    state.bindings,
                    roles,
                    open_axes=open_axes,
                    open_roles=admitted.open_axes,
                )
                assert outcome.identity is not None
                assert outcome.elected == read.elected, case
                declared = {axis for axis in admitted.names if axis in open_axes}
                set_aside += admitted.open_axes != declared
                minted = _mint(outcome.identity, index)
                assert minted == outcome.members, case
                assert minted.keys() == set(admitted.surviving_svs), case
                resubmitted = canonicalise_identity(outcome.identity)
                assert identity_dataset(resubmitted) == outcome.elected, case
                stated = stated_members(outcome.identity, None)
                assert stated is not None, case
                assert stated.digest == member_digest(outcome.members), case
                answered += 1
    assert answered > 800
    assert refused > 400
    assert tied
    assert unplaced
    assert set_aside
