"""Search reads a DFO or base candidate's schema from the catalogue's arc tables: deriving a
shape from our Topic and the variables the tables hold sends the graph nothing, and a
variable outside them is read live."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import gc
import json
import logging
import threading
import weakref
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from code_index.tokens import tokenize
from entity_resolution import Admission, EntityResolver
from graph_release.context import ReleaseNotBound, bound_release
from graph_release.service import Release, ReleaseService
from materialise.place_scope import DefaultPlaceScope, DefaultPresence
from origin_catalogue import BASE, UN_SYSTEM, Catalogue
from origin_catalogue.arc_tables import ArcTablesBuilder
from origin_catalogue.artifact import load_catalogue
from origin_catalogue.catalogue import CatalogueSnapshot
from origin_catalogue.families import FamilyKey, FamilySummary, family_tables, summarise_table
from origin_catalogue.frozen_arcs import served_arcs
from origin_catalogue.graph import _table_arcs
from origin_catalogue.un_mirror import EMPTY_UN_MIRROR, UnArc, UnVariable
from scripts.stage_origin_catalogue_fixture import stage_fixture
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.errors import FamilyNotResolvableError, GraphReadFailure
from search.engine.quantity import QuantityRequirement
from search.engine.quantity_read import QuantityOutput
from search.engine.region import phase_place_materialise, pipeline
from search.engine.region import state as region_state
from search.engine.region.alternate_guard import offered_alternates, query_pins
from search.engine.region.arcs import standard_bindings_from_arcs
from search.engine.region.candidates import (
    _Alternate,
    _materialise_alt_shapes,
    _prewarm_places,
    alternate_representative,
)
from search.engine.region.codes import REDUNDANT_BINDING_DROPPED
from search.engine.region.option_coverage import RoleServedFilter, _pick_population
from search.engine.region.phase_place_materialise import (
    _cover_unbound,
    _group_stat_vars,
    _lacks_presence,
    _read_taker_stat_vars,
    _sibling_shapes_of,
    merge_presence,
    record_unevaluated,
)
from search.engine.region.phase_rank_bind import (
    _family_constraints,
    _named_value_bindings,
    _selected_members,
    _value_label_spans,
    value_requirements,
)
from search.engine.region.phase_shared import _elected_flagged_match
from search.engine.region.pipeline import resolve_variable
from search.engine.region.place_context import PlaceContext, place_context
from search.engine.region.place_roles import PlaceAssignment
from search.engine.region.sentence_events import _member_count
from search.engine.region.state import (
    AlternateUnevaluatedReason,
    EngineOptions,
    ResolveState,
    _elected_stat_vars,
)
from search.engine.retrieval import bound_values
from search.engine.retrieval.bound_values import (
    MembershipAuthority,
    MembershipResolution,
    RoleClassWalks,
    RoleKeys,
    _held_members,
    elected_role_class_shape,
    five_tuple_member_facts,
    held_family_exact,
    held_family_memo,
    shape_member_facts,
    shape_query,
)
from search.engine.retrieval.derive import derive_shapes, member_group_class
from search.engine.retrieval.facts import (
    read_constraints,
    read_five_tuple,
    read_name,
)
from search.engine.retrieval.representative import _shape_disagreement_props
from search.engine.retrieval.retrieve import NoDataDraft
from search.engine.retrieval.standard import (
    _admit_standard_members,
    _AdmittedStandardMembers,
    default_scope_members,
)
from search.engine.shape import ShapeDraft
from search.models import GraphRef
from statvar_members import MultiValuedConstraint
from tests.engine._binders import elected_reply
from tests.engine._builders import subject_place_roles
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.graph_release.recorded import recorded_dfo_arcs
from tests.graph_release.support import TEST_REVISION, build_view_release
from tests.origin_catalogue.fake_catalogue import (
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)
from tests.origin_catalogue.support import JobTables, build_arc_tables, build_labelled

_OURS_HOST = "ours.example.org"
_UN_HOST = "un.example.org"

_WDI = "dc/base/WorldDevelopmentIndicators"
_TOPIC = "dc/topic/GDP"
_NOMINAL = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_REAL = "Amount_EconomicActivity_GrossDomesticProduction_RealValue"
_UNLISTED = "Amount_EconomicActivity_GrossNationalIncome_Nominal"


def _variable(name: str, gdp_type: str) -> dict:
    """A GDP variable's `->*` answer: amount of economic activity, gross domestic
    production, of `gdp_type`."""

    def target(dcid: str, *types: str) -> dict:
        return {"nodes": [{"dcid": dcid, "name": dcid, "types": list(types)}]}

    return {
        "typeOf": target("StatisticalVariable", "Class"),
        "name": {"nodes": [{"value": name}]},
        "populationType": target("EconomicActivity", "Class"),
        "measuredProperty": target("amount", "Property"),
        "statType": target("measuredValue", "Property"),
        "constraintProperties": {
            "nodes": [
                {"dcid": "activitySource", "name": "activitySource", "types": ["Property"]},
                {"dcid": "gdpType", "name": "gdpType", "types": ["Property"]},
            ]
        },
        "activitySource": target("GrossDomesticProduction", "EconomicActivitySourceEnum"),
        "gdpType": target(gdp_type, "GDPTypeEnum"),
    }


def _catalogue():
    """Lists both GDP variables as base, with their arcs, and a Topic reaching them."""
    return restrictive_catalogue(
        provenances=[provenance_row(_WDI, BASE)],
        base_variables={_NOMINAL: ([_WDI], []), _REAL: ([_WDI], [])},
        topics={_TOPIC: (_NOMINAL, _REAL)},
        arcs=build_arc_tables(
            dfo={},
            base={
                _NOMINAL: _variable("Nominal GDP", "Nominal"),
                _REAL: _variable("Real GDP", "RealValue"),
            },
            schema={},
        ),
    )


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Every HTTP request the process sends, each answered as an empty graph read."""
    requests: list[httpx.Request] = []

    def send(self: httpx.Client, request: httpx.Request, **_kwargs: object) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": {}}, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


@pytest.fixture
def client() -> LiveGraphClient:
    return LiveGraphClient(base=f"https://{_OURS_HOST}", un_base_url=f"https://{_UN_HOST}/core/api")


def test_deriving_our_topic_and_held_variables_sends_the_graph_nothing(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    (shape,) = derive_shapes(
        confirmed_svs=[_TOPIC, _NOMINAL],
        graph=client,
        sv_scores={_TOPIC: 0.9, _NOMINAL: 0.8},
        sv_types={_TOPIC: "Topic", _NOMINAL: "StatisticalVariable"},
    )

    assert set(shape.sv_arc_facts or {}) == {_NOMINAL, _REAL}
    assert "gdpType" in {slot.property_dcid for slot in shape.slot_keys}
    assert shape.pop_type_name == "EconomicActivity"
    assert sent == []


def test_a_variable_outside_the_arc_tables_is_read_live(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    labels = client.node_labels_batch([_NOMINAL, _UNLISTED])

    assert labels == {_NOMINAL: "Nominal GDP"}
    assert [request.url.host for request in sent] == [_OURS_HOST]
    assert _UNLISTED in sent[0].content.decode()
    assert _NOMINAL not in sent[0].content.decode()


def test_every_recalled_hit_is_grouped_and_only_unmirrored_hits_are_read_live(
    monkeypatch: pytest.MonkeyPatch,
    client: LiveGraphClient,
    bind_catalogue,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Forty-five hits the arc tables hold and two they do not all reach the shape. The
    held ones send the graph nothing, and the two others go in one `->*` read."""
    held = {f"{_NOMINAL}_{i}": _variable(f"GDP {i}", f"Type{i}") for i in range(45)}
    unmirrored = [f"{_UNLISTED}_{i}" for i in range(2)]
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WDI, BASE)],
            base_variables={variable: ([_WDI], []) for variable in held},
            arcs=build_arc_tables(dfo={}, base=held, schema={}),
        )
    )
    posted: list[dict] = []

    def send(self: httpx.Client, request: httpx.Request, **_kwargs: object) -> httpx.Response:
        payload = json.loads(request.content)
        posted.append(payload)
        data = {
            dcid: {"arcs": _variable(dcid, "Type0")}
            for dcid in payload.get("nodes", ())
            if payload.get("property") == "->*" and dcid in unmirrored
        }
        return httpx.Response(200, json={"data": data}, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    hits = [*held, *unmirrored]

    with caplog.at_level(logging.INFO, logger="search.engine.retrieval.derive"):
        shapes = derive_shapes(
            confirmed_svs=hits,
            graph=client,
            sv_scores={dcid: 0.9 - rank * 0.001 for rank, dcid in enumerate(hits)},
        )

    assert {sv for shape in shapes for sv in shape.sv_arc_facts or {}} == set(hits)
    assert posted == [{"nodes": unmirrored, "property": "->*"}]
    (fields,) = [vars(r) for r in caplog.records if "live_read" in vars(r)]
    assert (fields["table_answered"], fields["live_read"]) == (45, 2)


_TOTAL = "Amount_Debt_Outstanding"
_BONDS = "Amount_Debt_Outstanding_Bondholders"
_IMF_GG = "Amount_Debt_Outstanding_IMF_GeneralGovernment"


def _debt(name: str, **constraints: str) -> dict:
    """A debt-stock variable's `->*` answer, asserting `constraints`."""

    def target(dcid: str, *types: str) -> dict:
        return {"nodes": [{"dcid": dcid, "name": dcid, "types": list(types)}]}

    arcs = {
        "typeOf": target("StatisticalVariable", "Class"),
        "name": {"nodes": [{"value": name}]},
        "populationType": target("Debt", "Class"),
        "measuredProperty": target("amountOutstanding", "Property"),
        "statType": target("measuredValue", "Property"),
        **{prop: target(value, "Enum") for prop, value in constraints.items()},
    }
    if constraints:
        arcs["constraintProperties"] = {
            "nodes": [{"dcid": prop, "name": prop, "types": ["Property"]} for prop in constraints]
        }
    return arcs


_DEBT = {
    _TOTAL: _debt("Outstanding external debt"),
    _BONDS: _debt("Outstanding external debt to bondholders", creditorType="Bondholders"),
    _IMF_GG: _debt(
        "Outstanding external debt to the IMF, general government",
        creditorType="IMF",
        debtorType="GeneralGovernment",
    ),
}


def test_a_held_family_gives_the_shape_its_slots_values_and_total_label(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    """Recall finds the bondholder series alone. The shape keeps it as its member, reads
    its slot keys and taxonomy from every member of the family the tables hold, the
    recalled value first, and takes its label from the family total, which asserts no
    constraint."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WDI, BASE)],
            base_variables={variable: ([_WDI], []) for variable in _DEBT},
            arcs=build_arc_tables(dfo={}, base=_DEBT, schema={}),
        )
    )

    (shape,) = derive_shapes(confirmed_svs=[_BONDS], graph=client, sv_scores={_BONDS: 0.8})

    assert list(shape.sv_arc_facts or {}) == [_BONDS]
    assert [slot.property_dcid for slot in shape.slot_keys][:2] == ["creditorType", "debtorType"]
    assert shape.slot_taxonomy == {
        "how:creditorType": ["Bondholders", "IMF"],
        "how:debtorType": ["GeneralGovernment"],
    }
    assert shape.label == "Outstanding external debt"
    assert sent == []


def test_a_family_the_tables_do_not_hold_keeps_its_recalled_members() -> None:
    """With no arc tables, the shape reads its slot keys, taxonomy and label from the
    recalled member alone, though the graph holds the whole family."""
    graph = FakeGraph(
        nodes={dcid: {"label": dcid, "arcs": arcs} for dcid, arcs in _DEBT.items()},
        obs={},
        detect={},
        resolve={},
    )

    (shape,) = derive_shapes(confirmed_svs=[_BONDS], graph=graph, sv_scores={_BONDS: 0.8})

    assert shape.slot_taxonomy == {"how:creditorType": ["Bondholders"]}
    assert shape.label == "Outstanding external debt to bondholders"


_CODED = "Amount_Debt_Outstanding_Coded"


def _coded(code: str) -> dict:
    """A debt-stock variable asserting `code` as its literal `sourceIndicatorCode`."""
    return {
        **_debt(f"Debt {code}"),
        "constraintProperties": {
            "nodes": [
                {
                    "dcid": "sourceIndicatorCode",
                    "name": "sourceIndicatorCode",
                    "types": ["Property"],
                }
            ]
        },
        "sourceIndicatorCode": {"nodes": [{"value": code}]},
    }


def test_an_import_added_constraint_gives_a_held_family_no_slot(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    """Every served read leaves out the `sourceIndicatorCode` target of
    `constraintProperties`, so the family summary counts no such constraint: a family whose
    members differ only in their code has no slot, and its total is its first member."""
    other = f"{_CODED}_Other"
    variables = {_CODED: _coded("DT.DOD.DECT.CD"), other: _coded("DT.DOD.DLXF.CD")}
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WDI, BASE)],
            base_variables={variable: ([_WDI], []) for variable in variables},
            arcs=build_arc_tables(dfo={}, base=variables, schema={}),
        )
    )

    (shape,) = derive_shapes(confirmed_svs=[other], graph=client, sv_scores={other: 0.8})

    assert "sourceIndicatorCode" not in {slot.property_dcid for slot in shape.slot_keys}
    assert shape.slot_taxonomy == {}
    assert shape.label == "Debt DT.DOD.DECT.CD"
    assert sent == []


def test_a_shape_reads_its_family_summary_without_reading_the_members_it_did_not_recall(
    client: LiveGraphClient, bind_catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The family's slot values and total come from the catalogue's family row: derive asks
    the tables for the recalled member's arcs alone."""
    catalogue = restrictive_catalogue(
        provenances=[provenance_row(_WDI, BASE)],
        base_variables={variable: ([_WDI], []) for variable in _DEBT},
        arcs=build_arc_tables(dfo={}, base=_DEBT, schema={}),
    )
    bind_catalogue(catalogue)
    asked: list[str] = []
    held_arcs = type(catalogue).held_arcs

    def spy(self: object, dcid: str) -> object:
        asked.append(dcid)
        return held_arcs(self, dcid)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(type(catalogue), "held_arcs", spy)

    (shape,) = derive_shapes(confirmed_svs=[_BONDS], graph=client, sv_scores={_BONDS: 0.8})

    assert shape.label == "Outstanding external debt"
    assert _BONDS in asked
    assert not {_TOTAL, _IMF_GG} & set(asked)


def test_the_family_tables_summarise_each_family_as_derive_reads_its_members() -> None:
    """The catalogue job reads each variable's five-tuple, role class, constraints and name
    from its arc rows. Over variables that exercise each reading, its summaries equal those
    built from the same variables' arcs as the catalogue serves them (`frozen_arcs.served_arcs`) with
    derive's own readers."""
    provisional = {"nodes": [{"dcid": "dangling", "types": ["ProvisionalNode"]}]}
    variables = {
        **_DEBT,
        "Debt_Qualified": {**_debt("Qualified"), "measurementQualifier": provisional},
        "Debt_Dangling": {**_debt("Dangling"), "measurementDenominator": provisional},
        "Debt_Unvalued": {**_debt("Unvalued"), "measurementDenominator": {"nodes": [{}]}},
        "Debt_ByCreditor": {
            **_debt("By creditor", creditorType="IMF"),
            "observationProperties": {"nodes": [{"dcid": "financingProvider"}]},
        },
        "Debt_Degenerate": {
            **_debt("Degenerate"),
            "observationProperties": {"nodes": [{"dcid": "observationAbout"}]},
        },
        "Debt_Literal": {**_debt("Literal"), "statType": {"nodes": [{"value": "literalStat"}]}},
        _CODED: _coded("DT.DOD.DECT.CD"),
        _NOMINAL: _variable("Nominal GDP", "Nominal"),
        _REAL: _variable("Real GDP", "RealValue"),
    }
    tables = build_arc_tables(dfo={}, base=variables, schema={"dangling": None})

    by_family: dict[FamilyKey, list[dict]] = {}
    for dcid in sorted(tables.base):
        arcs = served_arcs(tables.base[dcid])
        role_bearing, _ = member_group_class(dcid, read_five_tuple(arcs), arcs)
        key = FamilyKey(*read_five_tuple(arcs), role_bearing)
        by_family.setdefault(key, []).append({"dcid": dcid, "arcs": arcs})
    expected = {}
    for key, members in by_family.items():
        values: dict[str, dict[str, None]] = {}
        for member in members:
            for prop, value in read_constraints(member["arcs"]).items():
                values.setdefault(prop, {})[value] = None
        total = next(
            (
                m
                for m in members
                if not read_constraints(m["arcs"]) and read_name(m["arcs"], m["dcid"])
            ),
            None,
        )
        expected[key] = FamilySummary(
            members=tuple(m["dcid"] for m in members),
            values={prop: tuple(found) for prop, found in values.items()},
            shared={
                prop: next(iter(found))
                for prop, found in values.items()
                if len(found) == 1 and all(prop in read_constraints(m["arcs"]) for m in members)
            },
            total=None if total is None else total["dcid"],
            total_name=None if total is None else read_name(total["arcs"], total["dcid"]),
            un_only=False,
        )

    assert family_tables(dfo={}, base=tables.base, un=EMPTY_UN_MIRROR).arcs == expected
    assert len(expected) == 6


_FLOWS = FamilyKey("FinancialFlow", "amount", "measuredValue", "Nominal", None, False, True)
"""The recorded slice's largest DFO family, which `_recorded_catalogue` gives base and UN
members too."""
_FLOW_TOTAL = "Amount_FinancialFlow_Base_Nominal"
_FLOW_DEBT = "AAA_Amount_FinancialFlow_Debt_Nominal"
_UN_FLOW = "undata/WB/Amount_FinancialFlow_Nominal"
_FLOW_TOPIC = "dc/topic/Flows"


def _flow(name: str, **constraints: str) -> dict:
    """A base member of `_FLOWS`: its five-tuple, a recipient role, `name`, and
    `constraints`."""

    def node(dcid: str) -> dict:
        return {"nodes": [{"dcid": dcid}]}

    return {
        "typeOf": node("StatisticalVariable"),
        "populationType": node("FinancialFlow"),
        "measuredProperty": node("amount"),
        "statType": node("measuredValue"),
        "measurementQualifier": node("Nominal"),
        "observationProperties": node("financingRecipient"),
        "name": {"nodes": [{"value": name}]},
        **(
            {"constraintProperties": {"nodes": [{"dcid": p} for p in constraints]}}
            if constraints
            else {}
        ),
        **{prop: node(value) for prop, value in constraints.items()},
    }


def _recorded_catalogue(*, with_dfo: bool) -> CatalogueSnapshot:
    """A catalogue over the recorded slice's DFO variables, read as the catalogue job reads
    them, and two base and one UN member of `_FLOWS`: a total and a member sorting before
    every DFO one. `with_dfo` false leaves the DFO variables out of its tables, as a
    catalogue of another import would hold other ones."""
    builder = ArcTablesBuilder()
    read = recorded_dfo_arcs(builder)
    denominators = {
        arc.dcid
        for rows in builder.table("dfo").values()
        for arc in rows
        if arc.property == "measurementDenominator"
        and arc.dcid is not None
        and arc.dcid not in read.variables
    }
    if not with_dfo:
        builder = ArcTablesBuilder()
    # The schema nodes the catalogue job holds for the DFO denominators, Count_Person apart.
    for denominator in sorted(denominators - {"Count_Person"}):
        builder.add("schema", denominator, None)
    builder.add("base", _FLOW_TOTAL, _flow("Financial flows"))
    builder.add("base", _FLOW_DEBT, _flow("Debt flows", financialFlowType="Debt"))
    builder.add("base", "Count_Person", {"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}})
    un_flow = UnVariable(
        series="FinancialFlow",
        arcs=tuple(
            UnArc(property=prop, dcid=target.get("dcid"), value=target.get("value"))
            for prop, arcs in _flow("UN financial flows").items()
            for target in arcs["nodes"]
        ),
    )
    return restrictive_catalogue(
        dfo_variables=read.variables,
        topics={_FLOW_TOPIC: [_FLOW_DEBT, *sorted(read.variables)[:2]]},
        un_mirror=un_mirror({_UN_FLOW: un_flow}),
        arcs=build_labelled(builder),
    )


def _origin_release(root: Path, *, revision: str = TEST_REVISION) -> Release:
    """The recorded slice's release, opened with its views and its origin view."""
    uri = (root / "releases").as_uri()
    build_view_release(uri, dc_serving_revision=revision)
    return ReleaseService(uri=uri, local_dir=root / "local", recheck_seconds=0).snapshot()


def test_a_release_s_dfo_half_serves_the_catalogue_s_families_and_arcs(tmp_path: Path) -> None:
    """The DFO variables' arcs and family members come from the bound release, and every held
    family and table read equals the one the catalogue job's own tables give for the same
    import (`JobTables`), members in order, whether or not the catalogue holds the job's DFO
    table."""
    full, stripped = _recorded_catalogue(with_dfo=True), _recorded_catalogue(with_dfo=False)
    release = _origin_release(tmp_path)
    keys = [*full.families.arcs, *full.families.un]
    dcids = [*full.arcs.dfo, *full.arcs.base, *full.arcs.schema, _UN_FLOW, _FLOW_TOPIC, _UNLISTED]

    def reads(catalogue: Catalogue) -> tuple[dict, dict]:
        return (
            {key: catalogue.held_family(key) for key in keys},
            {dcid: _table_arcs(catalogue, dcid) for dcid in dcids},
        )

    served = reads(cast("Catalogue", JobTables(full)))
    with bound_release(release):
        assert reads(full) == served
        assert reads(stripped) == served

    flows = served[0][_FLOWS]
    assert flows is not None
    dfo = sorted(dcid for dcid in full.arcs.dfo if dcid in flows.members)
    assert flows.members == (*dfo, _FLOW_DEBT, _FLOW_TOTAL, _UN_FLOW)
    assert len(dfo) > 1
    assert flows.total == _FLOW_TOTAL


def test_a_base_variable_the_release_owns_appears_once_with_the_release_s_values(
    tmp_path: Path,
) -> None:
    """A variable the catalogue's base table holds and the bound release owns as DFO leaves
    every base family its base arcs place it in: its own family lists it once with the
    release's values, and a family it alone made goes."""
    release = _origin_release(tmp_path)
    released = release.origin.families[_FLOWS]
    owned, elsewhere = released.members[:2]
    real = FamilyKey("FinancialFlow", "amount", "measuredValue", "RealValue", None, False, True)
    builder = ArcTablesBuilder()
    builder.add("base", _FLOW_TOTAL, _flow("Financial flows"))
    builder.add("base", owned, _flow("A stale copy", financialFlowType="Stale"))
    moved = _flow("A stale copy elsewhere")
    moved["measurementQualifier"] = {"nodes": [{"dcid": "RealValue"}]}
    builder.add("base", elsewhere, moved)
    catalogue = restrictive_catalogue(arcs=build_labelled(builder))
    assert catalogue.families.arcs[real].members == (elsewhere,)

    with bound_release(release):
        flows = catalogue.held_family(_FLOWS)
        gone = catalogue.held_family(real)

    assert flows is not None
    assert flows.members == (*released.members, _FLOW_TOTAL)
    assert flows.values == released.values
    assert gone is None


@pytest.mark.unbound_release
def test_a_read_refuses_with_no_release_bound() -> None:
    catalogue = _recorded_catalogue(with_dfo=False)

    with pytest.raises(ReleaseNotBound):
        catalogue.held_family(_FLOWS)
    with pytest.raises(ReleaseNotBound):
        _table_arcs(catalogue, _FLOW_TOTAL)


def test_a_held_decode_keys_on_the_bound_release(tmp_path: Path, bind_catalogue) -> None:
    """A request bound to a release whose family holds other DFO members decodes that family
    afresh while a request bound to the first still holds its decode."""
    bind_catalogue(_recorded_catalogue(with_dfo=False))
    first = _origin_release(tmp_path / "a", revision="rev-a")
    origin = first.origin
    dropped = origin.families[_FLOWS].members[0]
    arcs = {dcid: rows for dcid, rows in origin.arcs.items() if dcid != dropped}
    second = dataclasses.replace(
        _origin_release(tmp_path / "b", revision="rev-b"),
        origin=dataclasses.replace(origin, arcs=arcs, families=summarise_table(arcs)),
    )
    assert first.release_id != second.release_id

    with bound_release(first), held_family_memo():
        held = _held_members(_FLOWS)
    # `held` keeps the first request's decode alive, so `_held_decodes` still holds it.
    with bound_release(second), held_family_memo():
        swapped = list(_held_members(_FLOWS).arcs)

    assert dropped in held.arcs
    assert dropped not in swapped
    assert swapped == [dcid for dcid in held.arcs if dcid != dropped]


def _held_debt_shape(
    client: LiveGraphClient, bind_catalogue, variables: dict[str, dict] = _DEBT
) -> ShapeDraft:
    """The shape recall derives from the bondholder series alone, standing for the debt
    family the tables hold, `variables`."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WDI, BASE)],
            base_variables={variable: ([_WDI], []) for variable in variables},
            arcs=build_arc_tables(dfo={}, base=variables, schema={}),
        )
    )
    (shape,) = derive_shapes(confirmed_svs=[_BONDS], graph=client, sv_scores={_BONDS: 0.8})
    return shape


def _imf() -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid="creditorType", kind="value", value_dcids=["IMF"]
    )


def _flagged_member_constraint_properties(shape: ShapeDraft) -> list[list[str] | None]:
    """The constraint properties each member of the flagged identity a quantity refusal
    builds for `shape` states (`phase_shared._elected_flagged_match`), naming the member
    materialise elected, which recall did not find, under a request's memo. The state holds
    what the refusal reads; the rest are placeholders."""
    graph = FakeGraph(
        nodes={
            dcid: {"label": read_name(arcs, dcid), "arcs": arcs} for dcid, arcs in _DEBT.items()
        },
        obs={},
        detect={},
        resolve={},
    )
    state = ResolveState(
        variable="external debt",
        entities=[],
        date_request=None,
        detect_query="external debt",
        role_query="external debt",
        pac=True,
        graph=graph,
        resolver=cast("EntityResolver", None),
        admission=cast("Admission", None),
        llm=cast("SupportsLLM", None),
        sdmx=cast("EngineSdmxClient", None),
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
        shape_draft=shape,
        coverage=empty_coverage_snapshot(),
        # The pass judges coverage before it builds the refusal (`judge_coverage`).
        coverage_judged=True,
    )
    with held_family_memo():
        spec = asyncio.run(_elected_flagged_match(state, members=[_IMF_GG]))
    return [member.constraint_properties for member in spec.resolved.members]


def _sibling(shape: ShapeDraft, elected: list[SlotBindingDraft]) -> ShapeDraft:
    """`shape` as a re-election sibling of an election binding `elected`, which holds the
    members it stands for that hold those values."""
    (sibling,) = _sibling_shapes_of(
        cast(
            "ResolveState",
            SimpleNamespace(bindings=elected, capped_shapes=[shape], reelection_siblings=[0]),
        )
    ).values()
    return sibling


def _unbound_sibling(shape: ShapeDraft) -> ShapeDraft:
    """`shape` as a re-election sibling of an election binding nothing, which holds every
    member the shape stands for."""
    return _sibling(shape, [])


def _sibling_walk_seeds(shape: ShapeDraft) -> list[tuple[str, ...]]:
    """The candidates the mint's walk of `shape` as a re-election sibling of an election
    binding nothing is seeded with: the sibling holds every member the shape stands for."""
    seeds: list[tuple[str, ...]] = []

    class _Walks:
        def family(self, query: object, candidates: tuple[str, ...], *, deadline: object):
            seeds.append(tuple(candidates))
            return MembershipResolution(None, MembershipAuthority.FAILED)

    sibling = _unbound_sibling(shape)
    elected_role_class_shape(
        sibling,
        shape_query(sibling, [], ()),
        walks=cast("RoleClassWalks", _Walks()),
        arcs_graph=cast("EngineGraphClient", object()),
    )
    return seeds


_OWED_TO_THE_IMF = "debt owed to the IMF"


def _value_state(shape: ShapeDraft) -> ResolveState:
    """The fields `value_requirements` reads, for `_OWED_TO_THE_IMF` with `shape` both on
    the menu and elected."""
    return cast(
        "ResolveState",
        SimpleNamespace(
            variable=_OWED_TO_THE_IMF,
            capped_shapes=[shape],
            shape_draft=shape,
            variable_spans_by_vocabulary={},
            variable_spans_in_every_system=[],
            entity_texts=(),
            entity_roles={},
        ),
    )


@pytest.mark.parametrize(
    ("reader", "expected"),
    [
        pytest.param(
            lambda shape: sorted(five_tuple_member_facts(shape)[0]),
            sorted(_DEBT),
            id="pre-bind-members",
        ),
        pytest.param(
            lambda shape: [
                read_name(arcs, dcid) for dcid, arcs in _selected_members(shape, [_imf()]).items()
            ],
            ["Outstanding external debt to the IMF, general government"],
            id="requirement-judging",
        ),
        pytest.param(
            lambda shape: _elected_stat_vars(
                NoDataDraft(reason="no_observations", sv_dcids=[_IMF_GG]), shape
            ),
            [
                GraphRef(
                    ref=_IMF_GG, label="Outstanding external debt to the IMF, general government"
                )
            ],
            id="no-observations-member-labels",
        ),
        pytest.param(
            lambda shape: asyncio.run(_member_count(shape)),
            len(_DEBT),
            id="sentence-member-count",
        ),
        pytest.param(
            lambda shape: alternate_representative(
                shape, elected_bindings=[_imf()], elected_has_data=True
            ),
            _IMF_GG,
            id="alternate-representative",
        ),
        pytest.param(
            _shape_disagreement_props,
            {"creditorType"},
            id="disagreement-axes",
        ),
        pytest.param(
            _flagged_member_constraint_properties,
            [["creditorType", "debtorType"]],
            id="flagged-identity-constraints",
        ),
        pytest.param(
            lambda shape: _value_label_spans([shape], tokenize(_OWED_TO_THE_IMF)),
            [(17, 20)],
            id="value-label-spans",
        ),
        pytest.param(
            lambda shape: [
                requirement.term.text for requirement in value_requirements(_value_state(shape), ())
            ],
            ["IMF"],
            id="value-requirements",
        ),
        pytest.param(
            lambda shape: list(
                _sibling_shapes_of(
                    cast(
                        "ResolveState",
                        SimpleNamespace(
                            bindings=[_imf()], capped_shapes=[shape], reelection_siblings=[0]
                        ),
                    )
                )[0].sv_arc_facts
                or {}
            ),
            [_IMF_GG],
            id="reelection-sibling-members",
        ),
        pytest.param(_sibling_walk_seeds, [(_BONDS,)], id="sibling-walk-seeds-recalled"),
        pytest.param(
            lambda shape: list(
                offered_alternates(
                    query_pins(_OWED_TO_THE_IMF, bindings=[_imf()], quantity=QuantityRequirement()),
                    shape,
                    {0: shape},
                )[0].sv_arc_facts
                or {}
            ),
            [_IMF_GG, _TOTAL],
            id="guard-offered-members",
        ),
        pytest.param(
            lambda shape: [
                (binding.property_dcid, binding.value_dcids)
                for binding in standard_bindings_from_arcs(shape=shape, rep_sv_dcid=_IMF_GG)
            ],
            [("creditorType", ["IMF"]), ("debtorType", ["GeneralGovernment"])],
            id="representative-bindings",
        ),
        pytest.param(
            lambda shape: _named_value_bindings(shape, "external debt owed to the IMF"),
            [_imf()],
            id="named-value-bindings",
        ),
    ],
)
def test_every_reader_of_a_held_shape_reads_its_family(
    client: LiveGraphClient, bind_catalogue, reader, expected
) -> None:
    """Recall finds the bondholder series alone, and every reader of the shape's members
    reads the family the tables hold, the population its taxonomy offers values from, where
    the recalled member alone would answer none of these."""
    assert reader(_held_debt_shape(client, bind_catalogue)) == expected


_IMF = "Amount_Debt_Outstanding_IMF"


@pytest.mark.parametrize(
    ("elected", "bindings", "expected"),
    [
        # Every member holds the values of an election binding nothing; the sibling's
        # answer reads the family total alone.
        pytest.param([], [], [_TOTAL], id="unbound"),
        # The sibling holds the two IMF members, and binds what its representative, the
        # general-government member, asserts. The member owed to the IMF by every debtor
        # lacks the debtor binding, so the resolve elects the general-government member.
        pytest.param(
            [_imf()],
            [
                _imf(),
                SlotBindingDraft(
                    axis="how",
                    property_dcid="debtorType",
                    kind="value",
                    value_dcids=["GeneralGovernment"],
                ),
            ],
            [_IMF_GG],
            id="representative-narrower-than-a-member",
        ),
    ],
)
def test_a_narrowed_siblings_presence_read_covers_the_family_its_resolve_elects(
    client: LiveGraphClient,
    bind_catalogue,
    elected: list[SlotBindingDraft],
    bindings: list[SlotBindingDraft],
    expected: list[str],
) -> None:
    """A re-election sibling narrowed from a held family takes a default scope over the
    members its resolve reads at their own scope (`_group_stat_vars`), which cover the
    family `standard_resolve` elects (`_admit_standard_members`), and not every member it
    holds. Its facet prewarm reads each member at its own scope over that presence alone
    (`_prewarm_places`)."""
    shape = _held_debt_shape(
        client,
        bind_catalogue,
        {**_DEBT, _IMF: _debt("Outstanding external debt to the IMF", creditorType="IMF")},
    )
    sibling = _sibling(shape, elected)
    state = cast("ResolveState", SimpleNamespace(capped_shapes=[], taker_stat_vars={}))
    roles = cast("PlaceAssignment", SimpleNamespace(role_keys=RoleKeys()))

    stat_vars = _group_stat_vars(state, [], [(sibling, bindings, roles)])
    admitted = _admit_standard_members(sibling, bindings, ())

    presence = _presence_of(stat_vars)
    members, _ = shape_member_facts(sibling, shape_query(sibling, bindings, ()))
    prewarmed = _prewarm_places(
        sibling,
        bindings,
        RoleKeys(),
        list(members),
        cast("PlaceContext", SimpleNamespace(presence_for=lambda read: read)),
        presence,
        (),
    )

    assert isinstance(admitted, _AdmittedStandardMembers)
    assert set(admitted.family.order) <= set(stat_vars)
    assert stat_vars == expected
    assert len(stat_vars) < len(sibling.sv_arc_facts or {})
    assert list(prewarmed) == expected


def _presence_of(stat_vars: Iterable[str]) -> DefaultPresence:
    """A presence read covering `stat_vars`, each scoped to Kenya."""
    return DefaultPresence(
        countries=("country/KEN",),
        scopes={
            sv: DefaultPlaceScope(places=("country/KEN",), every_place=False) for sv in stat_vars
        },
    )


_DEBTOR_GG = SlotBindingDraft(
    axis="how", property_dcid="debtorType", kind="value", value_dcids=["GeneralGovernment"]
)


def _debt_by_creditor_and_debtor(client: LiveGraphClient, bind_catalogue) -> ShapeDraft:
    """The held debt family of bondholder debt owed by private debtors, IMF debt owed by
    general government and IMF debt owed by private debtors, recalled through the
    bondholder member. Bondholders with general government match no member together."""
    return _held_debt_shape(
        client,
        bind_catalogue,
        {
            _BONDS: _debt("Debt to bondholders", creditorType="Bondholders", debtorType="Private"),
            _IMF_GG: _DEBT[_IMF_GG],
            "Debt_IMF_Private": _debt("Debt to the IMF", creditorType="IMF", debtorType="Private"),
        },
    )


def _alternate_bound_past_its_recall(client: LiveGraphClient, bind_catalogue) -> _Alternate:
    """An alternate over a held debt family recalled through its bondholder member, binding
    the IMF and public debtors, which match no member together. Unbinding either leaves a
    member past its recall to answer: the IMF's general-government debt, or bilateral
    public debt."""
    shape = _held_debt_shape(
        client,
        bind_catalogue,
        {
            _BONDS: _debt("Debt to bondholders", creditorType="Bondholders", debtorType="Private"),
            _IMF_GG: _DEBT[_IMF_GG],
            "Debt_Bilateral_Public": _debt(
                "Bilateral public debt", creditorType="Bilateral", debtorType="Public"
            ),
        },
    )
    public = SlotBindingDraft(
        axis="how", property_dcid="debtorType", kind="value", value_dcids=["Public"]
    )
    return _Alternate(shape, [_imf(), public])


def test_unbinding_reads_the_presence_the_relaxed_resolve_needs(
    client: LiveGraphClient, bind_catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The elected bindings, bondholders and general government, match no member of the
    held debt family together, so the presence read under them covers the recalled
    bondholder member alone. Unbinding the creditor leaves the general-government member
    to answer, and the presence is topped up with its scope before the shape is resolved
    again (`_cover_unbound`)."""
    shape = _debt_by_creditor_and_debtor(client, bind_catalogue)
    debtor = _DEBTOR_GG
    creditor = _creditor("Bondholders")
    roles = cast("PlaceAssignment", SimpleNamespace(role_keys=RoleKeys()))
    read = _group_stat_vars(
        cast("ResolveState", SimpleNamespace(capped_shapes=[], taker_stat_vars={})),
        [],
        [(shape, [creditor, debtor], roles)],
    )
    asked: list[list[str]] = []

    def read_default_presence(*, stat_vars, known, **_kwargs: object) -> DefaultPresence:
        asked.append(list(stat_vars))
        return _presence_of([*known.scopes, *stat_vars])

    monkeypatch.setattr(region_state, "read_default_presence", read_default_presence)
    state = SimpleNamespace(
        default_presence=_presence_of(read),
        place=SimpleNamespace(
            default_scope=object(),
            roles=SimpleNamespace(resolved_entities={}, open_axes=frozenset()),
        ),
        shape_draft=shape,
        bindings=[debtor],
        graph=None,
        sdmx=None,
        deadline=None,
    )

    asyncio.run(_cover_unbound(cast("ResolveState", state)))

    assert read == [_BONDS]
    assert asked == [[_IMF_GG]]
    assert set(default_scope_members(shape, [debtor], ())) <= set(state.default_presence.scopes)


_UN_SERIES = "undata/wb/SERIES-DEBT"
_UN_BONDS = "undata/wb/DEBT.CREDITOR--BONDS"
_UN_DEBT = {
    "undata/wb/DEBT": un_variable(_UN_SERIES, name="Outstanding external debt"),
    _UN_BONDS: un_variable(
        _UN_SERIES,
        name="Outstanding external debt to bondholders",
        constraints={"creditorType": ("Creditor type", "Bondholders", "Bondholders")},
    ),
    "undata/wb/DEBT.CREDITOR--IMF.DEBTOR--GG": un_variable(
        _UN_SERIES,
        name="Outstanding external debt to the IMF, general government",
        constraints={
            "creditorType": ("Creditor type", "IMF", "IMF"),
            "debtorType": ("Debtor type", "GeneralGovernment", "General government"),
        },
    ),
}


def _held_un_debt_shape(client: LiveGraphClient, bind_catalogue) -> ShapeDraft:
    """The shape recall derives from the UN bondholder series alone, standing for the UN
    debt series the tables hold."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row("undata/p/WB", UN_SYSTEM, segment="wb")],
            un_mirror=un_mirror(_UN_DEBT),
        )
    )
    (shape,) = derive_shapes(confirmed_svs=[_UN_BONDS], graph=client, sv_scores={_UN_BONDS: 0.8})
    return shape


def test_a_un_series_whose_other_role_class_holds_a_base_member_is_not_settled_by_the_tables(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    """A walk reads both role classes of the UN debt series' five-tuple. With a base
    variable in its role-bearing class, which the tables hold only as a floor of the base
    graph, the tables cannot elect what the walk would, so with no walk the members stay
    unsettled and state nothing."""

    def target(dcid: str, *types: str) -> dict:
        return {"nodes": [{"dcid": dcid, "name": dcid, "types": list(types)}]}

    by_creditor = {
        "typeOf": target("StatisticalVariable", "Class"),
        "name": {"nodes": [{"value": "Debt by creditor"}]},
        "populationType": target(_UN_SERIES, "Class"),
        "measuredProperty": target("value", "Property"),
        "statType": target("measuredValue", "Property"),
        "observationProperties": {"nodes": [{"dcid": "financingProvider"}]},
    }
    bind_catalogue(
        restrictive_catalogue(
            provenances=[
                provenance_row("undata/p/WB", UN_SYSTEM, segment="wb"),
                provenance_row(_WDI, BASE),
            ],
            base_variables={"Debt_ByCreditor": ([_WDI], [])},
            arcs=build_arc_tables(dfo={}, base={"Debt_ByCreditor": by_creditor}, schema={}),
            un_mirror=un_mirror(_UN_DEBT),
        )
    )
    (shape,) = derive_shapes(confirmed_svs=[_UN_BONDS], graph=client, sv_scores={_UN_BONDS: 0.8})
    assert shape.held_family is not None
    assert held_family_exact(shape)

    membership = elected_role_class_shape(shape, shape_query(shape, [_imf()], ()))

    assert (membership.authority, membership.walked) == (MembershipAuthority.SAMPLED, None)


@pytest.mark.parametrize(
    ("held", "exact"),
    [
        pytest.param(_held_un_debt_shape, True, id="un"),
        pytest.param(_held_debt_shape, False, id="base"),
    ],
)
def test_a_held_family_is_exact_only_where_the_tables_hold_every_member(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue, held, exact: bool
) -> None:
    """The UN tables hold every member a walk of a UN series can elect, so the readers that
    need a whole population read the held series as one. The arc tables hold the base
    variables a resolver index or topic names, a floor of the base graph's family, so
    those readers read it as recall's sample: the answer is not exact, no source lists
    every member's constraints, an option pick refuses to stand for other bindings, and
    with no walk to settle it the membership is unsettled, where the UN series' is the
    catalogue's to state.
    The members the readers hold are the same either way."""
    shape = held(client, bind_catalogue)
    served_filter = cast(
        "RoleServedFilter", SimpleNamespace(probe_entities={}, open_axes=frozenset())
    )

    admitted = _admit_standard_members(shape, [_imf()], ())
    constraints = _family_constraints(shape)
    pick = _pick_population(shape, [_imf()], served_filter, index_only=True)
    membership = elected_role_class_shape(shape, shape_query(shape, [_imf()], ()))

    assert held_family_exact(shape) is exact
    assert membership.authority.settled is exact
    assert isinstance(admitted, _AdmittedStandardMembers)
    assert admitted.exact is exact
    if exact:
        assert constraints is not None
        assert sorted(sorted(c.items()) for c in constraints) == [
            [],
            [("creditorType", "Bondholders")],
            [("creditorType", "IMF"), ("debtorType", "GeneralGovernment")],
        ]
        assert pick is not None
        assert len(pick.members) == 1
    else:
        assert constraints is None
        assert pick is None


_IMF = "Amount_Debt_Outstanding_IMF"


def test_a_binding_a_held_member_realises_is_kept(bind_catalogue) -> None:
    """Recall finds the IMF series, which states no debtor. The bind binds the one debtor
    value the family the tables hold offers. The IMF general-government member realises
    both bindings, so the repair keeps the debtor binding and the answer reads that member;
    judged over the recalled member alone, the repair would drop it. The binding is
    mentioned: as a ``sole_option`` claim it would leave the slot unbound, since the total
    and the bondholders member state no debtor (`_validate_sole_option_bindings`)."""
    debt = {**_DEBT, _IMF: _debt("Outstanding external debt to the IMF", creditorType="IMF")}
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WDI, BASE)],
            base_variables={variable: ([_WDI], []) for variable in debt},
            arcs=build_arc_tables(dfo={}, base=debt, schema={}),
        )
    )
    query = "external debt owed to the IMF by Kenya"
    facet = {"earliestDate": "2015", "latestDate": "2020", "obsCount": 5}
    graph = FakeGraph(
        nodes={
            "country/KEN": {"label": "Kenya", "type": "Country"},
            **{dcid: {"label": dcid, "arcs": arcs} for dcid, arcs in debt.items()},
        },
        obs={f"{dcid}|country/KEN": [facet] for dcid in (_IMF, _IMF_GG)},
        detect={query: {"svs": [_IMF], "cosine_scores": [0.9]}},
        resolve={"Kenya": "country/KEN"},
    )
    bindings = [
        _imf(),
        SlotBindingDraft(
            axis="how",
            property_dcid="debtorType",
            kind="value",
            value_dcids=["GeneralGovernment"],
            evidence="mentioned",
        ),
    ]

    class _BindLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema is QuantityOutput:
                return NO_QUANTITY
            return elected_reply(0, bindings=bindings), None

    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_BindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )

    (answer,) = result.specs
    assert [sv.ref.ref for sv in answer.resolved.members] == [_IMF_GG]
    assert REDUNDANT_BINDING_DROPPED not in {warning.code for warning in result.warnings}


_EVAL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "origin_catalogue_eval"


def test_concurrent_requests_share_one_decode_of_a_held_family_and_free_it(
    client: LiveGraphClient, bind_catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two requests read the debt family at once: the second waits for the first's decode
    and holds the same one, so each member's arcs decode once. Once both requests end, no
    request holds the decode and it is freed."""
    key = _held_debt_shape(client, bind_catalogue).held_family
    assert key is not None
    decoding, release = threading.Event(), threading.Event()
    decoded: list[str] = []
    mirrored_arcs = bound_values.mirrored_arcs

    def spy(dcid: str, *, catalogue: object) -> dict | None:
        decoded.append(dcid)
        if threading.current_thread().name == "first":
            decoding.set()
            assert release.wait(timeout=10)
        return mirrored_arcs(dcid, catalogue=catalogue)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(bound_values, "mirrored_arcs", spy)
    # An earlier test's decode of this family can linger in uncollected garbage.
    monkeypatch.setattr(bound_values, "_held_decodes", weakref.WeakValueDictionary())
    held: dict[str, weakref.ref[object]] = {}
    both_hold, end = threading.Barrier(3, timeout=10), threading.Event()

    def request() -> None:
        with held_family_memo():
            held[threading.current_thread().name] = weakref.ref(_held_members(key))
            both_hold.wait()
            assert end.wait(timeout=10)

    first, second = (
        threading.Thread(target=contextvars.copy_context().run, args=(request,), name=name)
        for name in ("first", "second")
    )
    first.start()
    assert decoding.wait(timeout=10)
    second.start()
    second.join(timeout=0.2)
    # The second request waits for the first request's decode rather than decoding its own.
    assert second.is_alive()
    release.set()
    both_hold.wait()
    shared = held["first"]() is held["second"]() is not None
    end.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert shared
    assert sorted(decoded) == sorted(_DEBT)
    assert held["first"]() is None
    assert not bound_values._held_decodes


def _creditor(value: str) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid="creditorType", kind="value", value_dcids=[value]
    )


@pytest.mark.parametrize(
    ("binding", "fault", "outcome", "elections"),
    [
        pytest.param(_creditor("IMF"), None, ("authoritative", 1), 1, id="settled"),
        # The walk finds no member, which settles nothing.
        pytest.param(_creditor("Nobody"), None, ("failed", None), 1, id="no-member"),
        pytest.param(_creditor("IMF"), "member-cap", ("failed", None), 1, id="past-member-cap"),
        pytest.param(
            _creditor("IMF"),
            MultiValuedConstraint(stat_var=_UN_BONDS, property="creditorType", values=("a", "b")),
            "FamilyNotResolvableError",
            1,
            id="member-at-fault",
        ),
        # A failure the tables and query do not decide is elected again.
        pytest.param(_creditor("IMF"), OSError("decode"), "OSError", 3, id="other-failure"),
    ],
)
def test_a_query_of_a_family_the_tables_hold_whole_is_elected_once(
    sent: list[httpx.Request],
    client: LiveGraphClient,
    bind_catalogue,
    monkeypatch: pytest.MonkeyPatch,
    binding: SlotBindingDraft,
    fault: object,
    outcome: object,
    elections: int,
) -> None:
    """The UN tables hold the debt series whole, so its members are elected from them, once
    for a query however often a request reads it and however many requests read it while
    one holds the family's decode, whether the election settles the members, settles
    nothing or refuses the query."""
    shape = _held_un_debt_shape(client, bind_catalogue)
    query = shape_query(shape, [binding], ())
    elected: list[object] = []
    group_of_arcs = bound_values.group_of_arcs

    def spy(*args: Any, **kwargs: Any) -> object:
        elected.append(args[0])
        if isinstance(fault, Exception):
            raise fault
        return group_of_arcs(*args, **kwargs)

    monkeypatch.setattr(bound_values, "group_of_arcs", spy)
    monkeypatch.setattr(bound_values, "_held_decodes", weakref.WeakValueDictionary())
    if fault == "member-cap":
        monkeypatch.setattr(
            bound_values, "shared_enumerator", lambda: SimpleNamespace(member_cap=0)
        )

    def read() -> object:
        try:
            resolution = elected_role_class_shape(shape, query)
        except Exception as exc:
            return type(exc).__name__
        stated = resolution.walked.stated.served_count if resolution.walked else None
        return (resolution.authority.value, stated)

    # Another request: the bound catalogue with no memo, so it reads the decode the first
    # request holds.
    other_request = contextvars.copy_context()
    with held_family_memo():
        first, again = read(), read()
        with ThreadPoolExecutor(max_workers=1) as pool:
            other = pool.submit(other_request.run, read).result()

    assert first == again == other == outcome
    assert len(elected) == elections


def test_a_refused_query_keeps_no_state_of_the_requests_that_read_it(
    sent: list[httpx.Request],
    client: LiveGraphClient,
    bind_catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tables refuse the query for a member at fault, and the family's decode keeps that
    refusal for every later reader while a request holds it. Each request that read it can
    still be freed once it ends: the refusal holds none of its frames."""
    shape = _held_un_debt_shape(client, bind_catalogue)
    query = shape_query(shape, [_imf()], ())

    def refuse(*_args: Any, **_kwargs: Any) -> object:
        raise MultiValuedConstraint(stat_var=_UN_BONDS, property="creditorType", values=("a",))

    monkeypatch.setattr(bound_values, "group_of_arcs", refuse)
    monkeypatch.setattr(bound_values, "_held_decodes", weakref.WeakValueDictionary())

    class _RequestState:
        pass

    def request() -> weakref.ref[_RequestState]:
        state = _RequestState()
        with pytest.raises(FamilyNotResolvableError):
            elected_role_class_shape(shape, query)
        return weakref.ref(state)

    # This request holds the family's decode, and with it the kept refusal, throughout.
    with held_family_memo():
        requests = [contextvars.copy_context().run(request) for _ in range(2)]
        gc.collect()
        freed = [state() is None for state in requests]

    assert freed == [True, True]


@pytest.fixture(scope="module")
def eval_catalogue(tmp_path_factory: pytest.TempPathFactory) -> CatalogueSnapshot:
    """The committed dev catalogue build."""
    return load_catalogue(stage_fixture(_EVAL_FIXTURE_DIR, tmp_path_factory.mktemp("eval")))


def test_a_request_reads_each_held_family_once_however_it_alternates(
    eval_catalogue: CatalogueSnapshot, bind_catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One request reads the two largest subject-only families of the recorded dev build,
    8,248 BLS members each, three times in turn, and decodes each member's arcs once. Reads
    the dev build itself, about half a second of decoding, because a smaller family would
    fit a memo too small for these."""
    bind_catalogue(eval_catalogue)
    first, second = sorted(
        (key for key in eval_catalogue.families.arcs if not key.role_bearing),
        key=lambda key: -len(eval_catalogue.families.arcs[key].members),
    )[:2]
    sizes = [len(eval_catalogue.families.arcs[key].members) for key in (first, second)]
    decoded: list[str] = []
    mirrored_arcs = bound_values.mirrored_arcs

    def spy(dcid: str, *, catalogue: object) -> dict | None:
        decoded.append(dcid)
        return mirrored_arcs(dcid, catalogue=catalogue)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(bound_values, "mirrored_arcs", spy)

    with held_family_memo():
        read = [len(_held_members(key).arcs) for key in (first, second) * 3]

    assert sizes == [8_248, 8_248]
    assert read == sizes * 3
    assert len(decoded) == sum(sizes)


@pytest.mark.parametrize(
    ("failure", "pass_failure", "reason"),
    [
        pytest.param(TimeoutError("deadline"), None, "timed_out", id="timed-out"),
        pytest.param(
            GraphReadFailure("unavailable", upstream_status=503),
            None,
            "upstream_error",
            id="upstream-error",
        ),
        # After the pass's optional presence read failed, no other is tried.
        pytest.param(None, "no_capacity", "no_capacity", id="after-a-failed-read"),
    ],
)
def test_an_alternate_whose_presence_read_after_unbinding_fails_is_unevaluated(
    sent: list[httpx.Request],
    client: LiveGraphClient,
    bind_catalogue,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
    pass_failure: str | None,
    reason: str,
) -> None:
    """An alternate binds values that match no member together, so they are unbound and it
    is read again (`_alternate_bound_past_its_recall`). The presence read its new bindings
    need does not complete, so the alternate is left unevaluated for that reason, where it
    would read as found empty, and the request records it (`record_unevaluated`): the pass
    states `alternates_unevaluated` and is not cached."""
    alternate = _alternate_bound_past_its_recall(client, bind_catalogue)
    shape = alternate.shape
    known = _presence_of([_BONDS])

    def read_default_presence(**_kwargs: object) -> DefaultPresence:
        assert failure is not None, "no presence read is tried after the pass's failed"
        raise failure

    monkeypatch.setattr(region_state, "read_default_presence", read_default_presence)

    read = _materialise_alt_shapes(
        [alternate],
        places=[place_context(shape, subject_place_roles(), stat_vars=[_BONDS], presence=known)],
        presence=known,
        presence_failure=cast("AlternateUnevaluatedReason | None", pass_failure),
        named_place_dcids=[],
        graph=client,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )
    state = SimpleNamespace(
        capped_shapes=[shape],
        alternates_presence_failure=None,
        unevaluated_alternates={},
    )
    record_unevaluated(
        cast("ResolveState", state),
        {index: why for index, (unread, why) in enumerate(read.unevaluated) if unread is alternate},
    )

    assert (read.candidates, read.unevaluated) == ([], ((alternate, reason),))
    assert state.unevaluated_alternates == {0: reason}
    stated = pipeline._stating_unevaluated(
        pipeline.RegionResult(
            variable_text="external debt",
            status="no_data",
            specs=(),
            no_data_reason="no_observations",
            warnings=(),
            timing_by_step={},
        ),
        cast("ResolveState", state),
    )
    assert stated.alternates_left_out


def test_after_one_alternate_s_top_up_fails_no_other_is_tried(
    sent: list[httpx.Request],
    client: LiveGraphClient,
    bind_catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two alternates each need a presence read after unbinding
    (`_alternate_bound_past_its_recall`). The first read times out, so the call tries no
    other, as a pass tries no other optional presence read after one failed, and both
    alternates are unevaluated for the timeout."""
    first = _alternate_bound_past_its_recall(client, bind_catalogue)
    second = dataclasses.replace(first, bindings=list(first.bindings))
    known = _presence_of([_BONDS])
    reads: list[object] = []

    def read_default_presence(**_kwargs: object) -> DefaultPresence:
        reads.append(_kwargs["stat_vars"])
        raise TimeoutError("deadline")

    monkeypatch.setattr(region_state, "read_default_presence", read_default_presence)
    place = place_context(first.shape, subject_place_roles(), stat_vars=[_BONDS], presence=known)

    read = _materialise_alt_shapes(
        [first, second],
        places=[place, place],
        presence=known,
        named_place_dcids=[],
        graph=client,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    assert len(reads) == 1
    assert read.unevaluated == ((first, "timed_out"), (second, "timed_out"))


def test_a_top_up_on_a_countries_fallback_leaves_the_pass_s_scope_unsettled(
    sent: list[httpx.Request],
    client: LiveGraphClient,
    bind_catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alternate's presence read after unbinding gives the member it now answers through
    a countries fallback for this request only (`_alternate_bound_past_its_recall`). The
    read comes back with the alternates' result and joins the request's presence
    (`merge_presence`), so the pass's scope is unsettled and the answer stays out of the
    cache."""
    alternate = _alternate_bound_past_its_recall(client, bind_catalogue)
    shape = alternate.shape
    known = _presence_of([_BONDS])

    def read_default_presence(*, stat_vars, known, **_kwargs: object) -> DefaultPresence:
        return dataclasses.replace(
            _presence_of([*known.scopes, *stat_vars]), unsettled=frozenset(stat_vars)
        )

    monkeypatch.setattr(region_state, "read_default_presence", read_default_presence)

    read = _materialise_alt_shapes(
        [alternate],
        places=[place_context(shape, subject_place_roles(), stat_vars=[_BONDS], presence=known)],
        presence=known,
        named_place_dcids=[],
        graph=client,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )
    state = SimpleNamespace(default_presence=known)
    merge_presence(cast("ResolveState", state), read.presence)

    assert read.presence is not None and read.presence.unsettled
    assert _BONDS not in read.presence.unsettled
    assert state.default_presence.unsettled


def test_each_default_scope_taker_is_elected_once_off_the_event_loop(
    client: LiveGraphClient, bind_catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sibling narrowed from the held debt family takes a default scope. Its served and
    covered StatVars are elected once, in a worker thread, however many of the readers on
    the event loop read them: the presence read's StatVars, its lack check and the place
    context's StatVars (`_read_taker_stat_vars`)."""
    sibling = _unbound_sibling(_held_debt_shape(client, bind_catalogue))
    taker = (sibling, [], cast("PlaceAssignment", SimpleNamespace(sets={}, role_keys=RoleKeys())))
    elections: list[bool] = []
    elect = phase_place_materialise.taker_stat_vars

    def spy(*args: Any) -> object:
        elections.append(threading.current_thread() is threading.main_thread())
        return elect(*args)

    monkeypatch.setattr(phase_place_materialise, "taker_stat_vars", spy)
    state = cast(
        "ResolveState",
        SimpleNamespace(capped_shapes=[], taker_stat_vars={}, default_presence=None),
    )

    asyncio.run(_read_taker_stat_vars(state, [taker]))
    asyncio.run(_read_taker_stat_vars(state, [taker]))
    covered = _group_stat_vars(state, [], [taker])
    lacks = _lacks_presence(state, [], [taker])

    assert elections == [False]
    assert covered == [_TOTAL]
    assert lacks
