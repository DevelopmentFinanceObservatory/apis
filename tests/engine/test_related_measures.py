"""Related measures: the other members of the smallest StatVarGroup a definite answer's one
StatVar belongs to, with data at its places (`related`, `ResolvedSpec.related_measures`).

`related_measures_pakistan.json` holds dc/g/SDG_17.13.1 (Macroeconomic Dashboard), its 20
members and their facets at Pakistan as the dfo-dev graph serves them, the group "inflation
in Pakistan" elects from. Synthetic graphs pin the group rule, the member filters and the
minted identities; offline resolves pin the wiring into a definite answer, its fallback to
null and its serial depth.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import Any

import pytest

from code_index.query import CodeIndex
from datasets_api.adapter import resolve_definition_members
from dc_sdmx import SdmxClient
from search.engine import related
from search.engine.clients.graph import SubjectsPage
from search.engine.concurrency import scoped_concurrency
from search.engine.core import resolve_async
from search.engine.errors import EngineInputError, GraphInfraError
from search.engine.region import RegionResult, related_read, resolve_spec_resubmit
from search.engine.related import NO_RELATED, Related, read_related, related_measures
from search.engine.retrieval.facts import read_five_tuple
from search.engine.shape import identity_shape_for
from search.engine.spec_id import mint_identity
from search.models import (
    BaseIdentity,
    GraphRef,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    RelatedMeasure,
    RequestedWindow,
    ResubmitIdentityPayload,
    Spec,
    TimeWindow,
)
from statvar_members import MemberEnumerator
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api.test_resolve_definition_members_index import _STAMP
from tests.engine._base_identity import make_base, subject_entity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._virtual_clock import VirtualRequests
from tests.engine.test_facet_unit_gate_multi_statvar import _NODES as _UNIT_GATE_NODES
from tests.engine.test_facet_unit_gate_multi_statvar import _SV_A as _UNIT_GATE_A
from tests.engine.test_facet_unit_gate_multi_statvar import _SV_B as _UNIT_GATE_B
from tests.engine.test_facet_unit_gate_multi_statvar import _SV_C as _UNIT_GATE_ALT
from tests.engine.test_facet_unit_gate_multi_statvar import (
    _obs_facet,
    _SetPlusAltBindLLM,
)
from tests.engine.test_facet_unit_gate_multi_statvar import _resolution as _unit_gate_resolution
from tests.engine.test_first_answer_place_scenarios import (
    _EXECUTORS,
    _FREE_CALLS,
    _POOL_MODULES,
)
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM, _load_json
from tests.materialise.conftest import FakeGraphReader

_FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "related_measures_pakistan.json").read_text()
)
_INFLATION = "sdg/FP_CPI_TOTL_ZG"
_DASHBOARD = "dc/g/SDG_17.13.1"
_PAK, _KEN, _GHA = "country/PAK", "country/KEN", "country/GHA"
# Every reading of a 23-year Pakistan series overlaps inflation's 2000 to 2022 in full.
_FIRST_FIVE = [
    "sdg/BN_CAB_XOKA_GD_ZS",
    "sdg/BN_KLT_PTXL_CD",
    "sdg/BX_KLT_DINV_WD_GD_ZS",
    "sdg/DT_DOD_DECT_GN_ZS",
    "sdg/FI_RES_TOTL_MO",
]


_TWINNED = "sdg/BN_KLT_PTXL_CD"
# The deadline of the tests whose reads keep their threads and sleep up to 0.4 s: past every
# sleep, with room for a worker's garbage collection pause, which runs to about 2 s.
_THREADED_DEADLINE_S = 5.0


def _pakistan_graph(*, twins: bool = False) -> FakeGraph:
    nodes, obs = _pakistan_nodes(twins=twins)
    return FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})


def _pakistan_nodes(*, twins: bool = False) -> tuple[dict, dict]:
    """The dfo-dev graph's nodes and facets around inflation in Pakistan. `twins` adds to the
    dashboard a copy of `_TWINNED` and one of inflation, each sharing its original's
    five-tuple and constraints and so its identity, with its original's Pakistan facets."""
    nodes = {**_load_json("graph_nodes.json"), **_FIXTURE["nodes"]}
    obs = {**_load_json("graph_obs.json"), **_FIXTURE["obs"]}
    if twins:
        for original in (_TWINNED, _INFLATION):
            twin = f"{original}_TWIN"
            arcs = {**nodes[original]["arcs"], "name": {"nodes": [{"value": f"{twin} name"}]}}
            nodes[twin] = {**nodes[original], "arcs": arcs}
            obs[f"{twin}|{_PAK}"] = obs[f"{original}|{_PAK}"]
    return nodes, obs


def _read(
    graph: FakeGraph,
    *,
    stat_var: str = _INFLATION,
    arcs: Mapping | None = None,
    places: tuple[str, ...] = (_PAK,),
    window: RequestedWindow | None = None,
) -> Related:
    return read_related(
        stat_var=stat_var,
        arcs=arcs,
        places=places,
        window=window,
        graph=graph,
        deadline=time.monotonic() + 30,
    )


def _inflation_arcs() -> dict:
    return _FIXTURE["nodes"][_INFLATION]["arcs"]


def test_inflation_in_pakistan_offers_five_macroeconomic_dashboard_members():
    found = _read(_pakistan_graph(), arcs=_inflation_arcs())

    assert found.group == GraphRef(dcid=_DASHBOARD, label="17.13.1: Macroeconomic Dashboard")
    assert [m.dcids for m in found.members] == [(dcid,) for dcid in _FIRST_FIVE]
    assert found.members[0].label == "Current account balance as a proportion of GDP"


def test_members_order_by_shared_years_then_dcid_and_a_range_keeps_those_with_a_year_in_it(
    monkeypatch,
):
    """Bank capital (2013 to 2022) shares 10 years with inflation, the two cash and tax
    series (2000 only) 1; central government debt holds no Pakistan data."""
    monkeypatch.setattr(related, "RELATED_MEASURES_MAX", 20)
    graph = _pakistan_graph()

    every = [m.dcids[0] for m in _read(graph, arcs=_inflation_arcs()).members]
    in_range = [
        m.dcids[0]
        for m in _read(
            graph, arcs=_inflation_arcs(), window=TimeWindow(start_year=2015, end_year=2020)
        ).members
    ]

    assert len(every) == 18
    assert every[-3:] == ["sdg/FB_BNK_CAPA_ZS", "sdg/GC_BAL_CASH_GD_ZS", "sdg/GC_TAX_TOTL_GD_ZS"]
    assert {_INFLATION, "sdg/DP_DOD_DLD2_CR_CG_Z1"}.isdisjoint(every)
    assert in_range == every[:-2]


def test_a_stat_var_without_its_arcs_reads_its_groups():
    graph = _pakistan_graph()

    assert _read(graph) == _read(graph, arcs=_inflation_arcs())


# ---------------------------------------------------------------------------
# The group rule and the member filters, on synthetic graphs.
# ---------------------------------------------------------------------------

_SV = "Elected_Var"


def _var(
    *groups: str, measured: str = "count", constraints: Mapping[str, Mapping] | None = None
) -> dict:
    """A StatVar of Person in `groups`. Members sharing `measured` and `constraints` share
    one identity, so each member of a synthetic group names its own `measured`."""
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": measured}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "memberOf": {"nodes": [{"dcid": group} for group in groups]},
    }
    for prop, target in (constraints or {}).items():
        arcs.setdefault("constraintProperties", {"nodes": []})["nodes"].append({"dcid": prop})
        arcs[prop] = {"nodes": [target]}
    return {"label": None, "type": "StatisticalVariable", "arcs": arcs}


def _group_graph(sizes: Mapping[str, int], *, elected_in: tuple[str, ...]) -> FakeGraph:
    """`_SV` in `elected_in`, and each group of `sizes` holding that many members, `_SV`
    among them where it belongs, every member with data at Kenya."""
    nodes = {_SV: _var(*elected_in)}
    for group, size in sizes.items():
        nodes[group] = _group(group)
        others = size - (group in elected_in)
        for i in range(others):
            nodes[f"{group}_member_{i:02d}"] = _var(group, measured=f"{group}_{i:02d}")
    obs = {f"{dcid}|{_KEN}": [_facet("2010", "2020")] for dcid in nodes}
    return FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})


def _group(dcid: str) -> dict:
    return {
        "label": f"Group {dcid}",
        "type": "StatVarGroup",
        "arcs": {"name": {"nodes": [{"value": f"Group {dcid}"}]}},
    }


def _facet(earliest: str, latest: str) -> dict:
    return {"earliestDate": earliest, "latestDate": latest, "obsCount": 1}


@pytest.mark.parametrize(
    ("sizes", "elected_in", "chosen"),
    [
        ({"g/a": 5, "g/b": 3}, ("g/a", "g/b"), "g/b"),
        ({"g/b": 3, "g/a": 3}, ("g/a", "g/b"), "g/a"),
        ({"g/a": 41, "g/b": 40}, ("g/a", "g/b"), "g/b"),
        ({"g/a": 1, "g/b": 4}, ("g/a", "g/b"), "g/b"),
        ({"g/a": 41}, ("g/a",), None),
        ({"g/a": 1}, ("g/a",), None),
    ],
    ids=["smallest", "tie_by_dcid", "over_40_ignored", "alone_ignored", "only_over_40", "alone"],
)
def test_the_smallest_group_of_2_to_40_members_is_read(sizes, elected_in, chosen):
    found = _read(
        _group_graph(sizes, elected_in=elected_in),
        stat_var=_SV,
        arcs=_var(*elected_in)["arcs"],
        places=(_KEN,),
    )

    if chosen is None:
        assert found == NO_RELATED
        return
    assert found.group == GraphRef(dcid=chosen, label=f"Group {chosen}")
    assert [m.dcids for m in found.members] == [
        (f"{chosen}_member_{i:02d}",) for i in range(min(5, sizes[chosen] - 1))
    ]


class _Logged(FakeGraph):
    """Logs every graph method called by name."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log: list[str] = []

    def node_arcs_many(self, *args, **kwargs):
        self.log.append("node_arcs_many")
        return super().node_arcs_many(*args, **kwargs)

    def node_subjects_many(self, *args, **kwargs) -> dict[str, SubjectsPage]:
        self.log.append("node_subjects_many")
        return super().node_subjects_many(*args, **kwargs)

    def node_arcs_batch(self, *args, **kwargs):
        self.log.append("node_arcs_batch")
        return super().node_arcs_batch(*args, **kwargs)

    def observation_facets_batch(self, *args, **kwargs):
        self.log.append("observation_facets_batch")
        return super().observation_facets_batch(*args, **kwargs)


def test_a_stat_var_in_no_group_reads_nothing():
    graph = _Logged(nodes={_SV: _var()}, obs={}, detect={}, resolve={})

    assert _read(graph, stat_var=_SV, arcs=_var()["arcs"], places=(_KEN,)) == NO_RELATED
    assert graph.log == []


def test_a_member_needs_data_at_every_place_and_resolved_constraints():
    resolved = {"gender": {"dcid": "Female"}}
    unresolved = {"gender": {"types": ["ProvisionalNode"]}}
    nodes = {
        _SV: _var("g/a"),
        "g/a": _group("g/a"),
        "Both_Places": _var("g/a", constraints=resolved),
        "One_Place": _var("g/a", measured="age"),
        "Unresolved": _var("g/a", constraints=unresolved),
    }
    both = [_facet("2010", "2020")]
    obs = {
        f"{_SV}|{_KEN}": both,
        f"{_SV}|{_GHA}": both,
        f"Both_Places|{_KEN}": both,
        f"Both_Places|{_GHA}": both,
        f"One_Place|{_KEN}": both,
        f"Unresolved|{_KEN}": both,
        f"Unresolved|{_GHA}": both,
    }
    graph = FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})

    found = _read(graph, stat_var=_SV, arcs=nodes[_SV]["arcs"], places=(_KEN, _GHA))

    assert [(m.dcids, dict(m.constraints)) for m in found.members] == [
        (("Both_Places",), {"gender": "Female"})
    ]


@pytest.mark.parametrize(
    ("window", "kept"),
    [
        (TimeWindow(start_year=2000, end_year=2000), False),
        (TimeWindow(start_year=2020, end_year=2020), False),
        (TimeWindow(start_year=1990, end_year=2020), True),
        (None, True),
    ],
    ids=["neither_place", "one_place", "both_places", "every_year"],
)
def test_a_range_keeps_a_member_only_when_every_place_holds_a_year_in_it(
    window: TimeWindow | None, kept: bool
):
    """Kenya holds the member in 1990 and Ghana in 2020, so their combined span covers 2000
    though neither place holds it then; the card offers it at both places."""
    nodes = {_SV: _var("g/a"), "g/a": _group("g/a"), "Split": _var("g/a", measured="age")}
    obs = {
        f"{_SV}|{_KEN}": [_facet("1990", "2020")],
        f"{_SV}|{_GHA}": [_facet("1990", "2020")],
        f"Split|{_KEN}": [_facet("1990", "1990")],
        f"Split|{_GHA}": [_facet("2020", "2020")],
    }
    graph = FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})

    found = _read(graph, stat_var=_SV, arcs=nodes[_SV]["arcs"], places=(_KEN, _GHA), window=window)

    assert [m.dcids for m in found.members] == ([("Split",)] if kept else [])


def test_a_group_with_no_other_member_kept_states_none():
    """The group qualifies, but its other member holds no facet at Kenya, so no measure is
    offered and no group is named."""
    nodes = {_SV: _var("g/a"), "g/a": _group("g/a"), "No_Data": _var("g/a", measured="age")}
    obs = {f"{_SV}|{_KEN}": [_facet("2010", "2020")]}
    graph = FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})

    assert _read(graph, stat_var=_SV, arcs=nodes[_SV]["arcs"], places=(_KEN,)) == NO_RELATED


# ---------------------------------------------------------------------------
# Each member's identity.
# ---------------------------------------------------------------------------


def test_a_member_identity_is_its_own_shape_and_constraints_with_the_answer_places_and_years():
    nodes = {"Both_Places": _var("g/a", constraints={"gender": {"dcid": "Female"}})}
    arcs = nodes["Both_Places"]["arcs"]
    member = related.RelatedMember(
        dcids=("Both_Places",),
        label="Female count",
        five_tuple=read_five_tuple(arcs),
        constraints={"gender": "Female"},
    )
    window = TimeWindow(start_year=2015, end_year=2020)
    _spec_id, elected = make_base(
        identity_shape_for(
            pop_type_dcid="EconomicActivity",
            meas_prop_dcid="amount",
            stat_type_dcid="measuredValue",
            meas_qual_dcid="Nominal",
            meas_denom_dcid=None,
        ),
        entities=[subject_entity(_KEN)],
        requested_window=window,
        explicit_facets={
            "unit": IdentityFacetValueRequest(
                request=IdentityUnitRequest(currency="EUR", prices="current")
            )
        },
    )

    [measure] = related_measures(
        Related(group=GraphRef(dcid="g/a", label="Group a"), members=(member,)), elected
    )

    identity = measure.definition.identity
    assert measure.label == "Female count"
    assert measure.resolved_stat_var_dcids == ["Both_Places"]
    assert identity.model_dump(include={"shape"}, exclude_none=True)["shape"] == {
        "kind": "statvar_schema",
        "population_type": "Person",
        "measured_property": "count",
        "stat_type": "measuredValue",
    }
    assert [(s.property, s.binding.model_dump()) for s in identity.slots] == [
        ("gender", {"state": "value", "ref": "Female"})
    ]
    assert identity.facets is None
    assert identity.entities == elected.entities
    assert identity.requested_window == elected.requested_window
    assert measure.definition.spec_id == mint_identity(identity.model_dump(exclude_none=True))[0]


@pytest.mark.parametrize("twins", [False, True], ids=["dev_graph", "twin_identities"])
def test_every_inflation_in_pakistan_member_resubmits_to_a_definite_answer_electing_it(
    monkeypatch, twins: bool
):
    """Each of the 18 Macroeconomic Dashboard measures with Pakistan data, posted back with
    its `resolved_stat_var_dcids`, answers definitely from the StatVars it names. Members
    sharing an identity make one measure naming them all, and one sharing inflation's makes
    none, since a mint of an identity reads every member that holds it."""
    monkeypatch.setattr(related, "RELATED_MEASURES_MAX", 20)
    nodes, obs = _pakistan_nodes(twins=twins)
    graph = FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})
    measures = _pakistan_measures(graph)
    assert len(measures) == 18
    named = [dcid for m in measures for dcid in m.resolved_stat_var_dcids]
    assert f"{_INFLATION}_TWIN" not in named
    twinned = next(m for m in measures if _TWINNED in m.resolved_stat_var_dcids)
    if twins:
        assert twinned.resolved_stat_var_dcids == [_TWINNED, f"{_TWINNED}_TWIN"]
        assert twinned.label.endswith(f" / {_TWINNED}_TWIN name")
    else:
        assert twinned.resolved_stat_var_dcids == [_TWINNED]

    mint_graph = _mint_graph(nodes)
    for measure in measures:
        region = _resubmit(measure, graph)

        assert region.status == "definite", (measure.label, region.no_data_reason)
        [spec] = region.specs
        assert spec.spec_id == measure.definition.spec_id
        assert spec.resolved.stat_vars.member_count == len(measure.resolved_stat_var_dcids)
        assert {sv.ref for sv in spec.resolved.stat_vars.items} <= set(
            measure.resolved_stat_var_dcids
        )
        # A Datasets mint of the definition, seeded with the measure's StatVars, reads them.
        family = resolve_definition_members(
            measure.definition.identity.model_dump(exclude_none=True),
            spec_id=measure.definition.spec_id,
            graph=mint_graph,
            enumerator=MemberEnumerator(graph=mint_graph),
            code_index=CodeIndex(stamp=_STAMP, declarations=(), codes=(), statvars=()),
            seed=measure.resolved_stat_var_dcids,
        ).family
        assert family == set(measure.resolved_stat_var_dcids), measure.label
        # An answer reading several StatVars offers no related measures (`related_read`).
        if len(measure.resolved_stat_var_dcids) > 1:
            assert spec.resolved.related_measures is None
            continue
        # The answer lists its own siblings, never itself.
        assert spec.resolved.related_group == GraphRef(
            dcid=_DASHBOARD, label="17.13.1: Macroeconomic Dashboard"
        )
        listed = {
            dcid for m in spec.resolved.related_measures or () for dcid in m.resolved_stat_var_dcids
        }
        assert listed
        assert listed.isdisjoint(measure.resolved_stat_var_dcids)


def test_a_resubmitted_sibling_lists_its_own_siblings():
    measure = next(
        m for m in _pakistan_measures() if m.resolved_stat_var_dcids == ["sdg/BN_CAB_XOKA_GD_ZS"]
    )

    [spec] = _resubmit(measure, _pakistan_graph()).specs

    assert [m.resolved_stat_var_dcids for m in spec.resolved.related_measures or ()] == [
        ["sdg/BN_KLT_PTXL_CD"],
        ["sdg/BX_KLT_DINV_WD_GD_ZS"],
        ["sdg/DT_DOD_DECT_GN_ZS"],
        ["sdg/FI_RES_TOTL_MO"],
        ["sdg/FM_LBL_BMNY_IR_ZS"],
    ]


_CAB = "sdg/BN_CAB_XOKA_GD_ZS"


@pytest.mark.related_read_threads
@pytest.mark.parametrize(
    ("places", "facets", "outcome", "reads"),
    [
        ([_PAK], None, "definite", [_CAB]),
        ([_KEN], None, "no_observations", [_CAB]),
        ([], None, "no_observations", []),
        (
            [_PAK],
            {"measurementMethod": {"state": "value", "ref": "NoSuchMethod"}},
            "facet_not_available",
            [_CAB],
        ),
    ],
    ids=["definite", "no_data_at_place", "no_place", "refusal"],
)
def test_a_resubmit_waits_for_its_related_read_however_it_answers(
    monkeypatch,
    search_settings,
    places: list[str],
    facets: dict | None,
    outcome: str,
    reads: list[str],
):
    """The resubmit door waits for the related-measures read it started before it answers,
    so no read outlives the response. The read sleeps 0.2 s to outlast grounding, so only that
    wait lets it finish before an answer that ends before finalisation returns. An answer
    with no place starts no read (`related_read._qualifies`), so `no_matching_slice`, which
    grounding answers only with no place bound, never has one to wait for."""
    search_settings(related_measures_timeout_s=_THREADED_DEADLINE_S)
    finished: list[str] = []

    def _slow_read(**kwargs: Any) -> Related:
        time.sleep(0.2)
        finished.append(kwargs["stat_var"])
        return NO_RELATED

    monkeypatch.setattr(related_read, "read_related", _slow_read)
    measure = next(m for m in _pakistan_measures() if m.resolved_stat_var_dcids == [_CAB])
    posted = measure.definition.identity.model_dump(exclude_none=True)
    posted["entities"] = {"observationAbout": places} if places else {}
    if facets is not None:
        posted["facets"] = facets
    identity = BaseIdentity.model_validate(posted)

    try:
        region = resolve_spec_resubmit(
            inp=ResubmitIdentityPayload(
                identity=identity,
                base_spec_id=mint_identity(identity.model_dump(exclude_none=True))[0],
                base_identity=identity,
                resolved_stat_var_dcids=[_CAB],
            ),
            graph=_pakistan_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        answered = region.no_data_reason or region.status
    except EngineInputError as refusal:
        answered = refusal.code

    assert answered == outcome
    assert finished == reads


def test_a_resubmit_logs_a_failed_related_read_it_answers_without(monkeypatch, caplog):
    """A defect in the read of a resubmit that answers before finalisation, which never
    takes it, is logged at error with its traceback, as one the answer takes is."""

    def _broken_read(**_kwargs: Any) -> Related:
        raise ValueError("a malformed page")

    monkeypatch.setattr(related_read, "read_related", _broken_read)
    measure = next(m for m in _pakistan_measures() if m.resolved_stat_var_dcids == [_CAB])
    posted = measure.definition.identity.model_dump(exclude_none=True)
    posted["entities"] = {"observationAbout": [_KEN]}
    identity = BaseIdentity.model_validate(posted)

    with caplog.at_level(logging.WARNING, logger=related_read.__name__):
        region = resolve_spec_resubmit(
            inp=ResubmitIdentityPayload(
                identity=identity,
                base_spec_id=mint_identity(identity.model_dump(exclude_none=True))[0],
                base_identity=identity,
                resolved_stat_var_dcids=[_CAB],
            ),
            graph=_pakistan_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

    assert region.no_data_reason == "no_observations"
    [record] = [r for r in caplog.records if r.name == related_read.__name__]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None


def _pakistan_measures(graph: FakeGraph | None = None) -> list[RelatedMeasure]:
    """Every Macroeconomic Dashboard member with Pakistan data, as a measure related to
    inflation in Pakistan, up to `RELATED_MEASURES_MAX`."""
    _spec_id, elected = make_base(
        identity_shape_for(
            pop_type_dcid="SDG_FP_CPI_TOTL_ZG",
            meas_prop_dcid="value",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
        ),
        entities=[subject_entity(_PAK, "Pakistan")],
    )
    return related_measures(_read(graph or _pakistan_graph(), arcs=_inflation_arcs()), elected)


def _mint_graph(nodes: Mapping[str, dict]) -> FakeMintGraphReader:
    """`nodes` as the Datasets member walk reads them."""
    return FakeMintGraphReader(
        arcs={
            dcid: {
                prop: tuple(n["dcid"] for n in arc.get("nodes", ()) if "dcid" in n)
                for prop, arc in node["arcs"].items()
            }
            for dcid, node in nodes.items()
            if isinstance(node, dict) and node.get("arcs")
        }
    )


def _resubmit(measure: RelatedMeasure, graph: FakeGraph) -> RegionResult:
    """`measure` posted back with its seed, as a client switching to it does."""
    definition = measure.definition
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=definition.identity,
            base_spec_id=definition.spec_id,
            base_identity=definition.identity,
            resolved_stat_var_dcids=measure.resolved_stat_var_dcids,
        ),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )


# ---------------------------------------------------------------------------
# Wiring into a definite answer.
# ---------------------------------------------------------------------------


def test_a_definite_answer_carries_its_related_group_and_measures():
    """Kenya's population reads Count_Person, whose smallest fixture group is Demographics."""
    [part] = offline_resolve(make_request("population of Kenya")).parts

    resolved = part.spec.resolved
    assert resolved.related_group == GraphRef(dcid="dc/g/Demographics", label="dc/g/Demographics")
    assert [(m.label, m.resolved_stat_var_dcids) for m in resolved.related_measures or ()] == [
        ("Rate of Population Growth", ["GrowthRate_Count_Person"]),
        ("Life Expectancy", ["LifeExpectancy_Person"]),
        ("Population Density", ["Count_Person_PerArea"]),
    ]


class _FailingSubjects(FakeGraph):
    def node_subjects_many(self, *args, **kwargs) -> dict[str, SubjectsPage]:
        raise GraphInfraError("the reverse read failed")


class _SlowSubjects(FakeGraph):
    def node_subjects_many(self, *args, **kwargs) -> dict[str, SubjectsPage]:
        time.sleep(0.3)
        return super().node_subjects_many(*args, **kwargs)


class _BrokenSubjects(FakeGraph):
    def node_subjects_many(self, *args, **kwargs) -> dict[str, SubjectsPage]:
        raise ValueError("a malformed page")


@pytest.mark.related_read_threads
@pytest.mark.parametrize(
    ("graph_type", "level"),
    [
        (_FailingSubjects, logging.WARNING),
        (_SlowSubjects, logging.WARNING),
        (_BrokenSubjects, logging.ERROR),
    ],
    ids=["fails", "late", "raises"],
)
def test_a_failed_or_late_read_leaves_the_answer_definite_with_null_related_measures(
    graph_type, level, search_settings, caplog
):
    """A graph failure or a timeout logs a warning; any other exception in the read is a
    defect, logged at error with its traceback, and still leaves the answer standing."""
    search_settings(related_measures_timeout_s=0.05)

    with caplog.at_level(logging.WARNING, logger=related_read.__name__):
        [part] = offline_resolve(make_request("population of Kenya"), graph=graph_type()).parts

    assert part.status == "definite"
    assert part.spec.resolved.related_group is None
    assert part.spec.resolved.related_measures is None
    [record] = [r for r in caplog.records if r.name == related_read.__name__]
    assert record.levelno == level
    assert (record.exc_info is not None) is (level == logging.ERROR)


class _StalledSubjects(FakeGraph):
    def node_subjects_many(self, *args, deadline: float, **kwargs) -> dict[str, SubjectsPage]:
        time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)
        return super().node_subjects_many(*args, deadline=deadline, **kwargs)


def _stalled_first_answer() -> Spec:
    [part] = offline_resolve(make_request("population of Kenya"), graph=_StalledSubjects()).parts
    return part.spec


def _stalled_resubmit() -> Spec:
    nodes, obs = _pakistan_nodes()
    measure = next(m for m in _pakistan_measures() if m.resolved_stat_var_dcids == [_CAB])
    graph = _StalledSubjects(nodes=nodes, obs=obs, detect={}, resolve={})
    [spec] = _resubmit(measure, graph).specs
    return spec


@pytest.mark.parametrize(
    ("answer", "group"),
    [(_stalled_first_answer, "dc/g/Demographics"), (_stalled_resubmit, _DASHBOARD)],
    ids=["first", "resubmit"],
)
def test_an_offline_read_stalled_past_its_deadline_is_still_stated(
    answer, group: str, search_settings
):
    """These tests run each related read on the thread that starts it
    (`conftest._related_reads_inline`), so an answer states what the doubles serve however
    long the machine stalls. On its own thread a read stalled past its deadline, as a garbage
    collection pausing a loaded worker stalls it, leaves the answer without it."""
    search_settings(related_measures_timeout_s=0.05)

    spec = answer()

    assert spec.resolved.related_group is not None
    assert spec.resolved.related_group.dcid == group


def _first_answer() -> Coroutine[Any, Any, object]:
    return resolve_async(
        make_request("population of Kenya"), graph=FakeGraph(), llm=FakeLLM(), sdmx=FakeSdmxClient()
    )


def _sibling_resubmit() -> Coroutine[Any, Any, object]:
    """A resubmit of the current account balance, a Macroeconomic Dashboard member, in
    Pakistan, run off the event loop as the resubmit door runs it."""
    measure = next(
        m for m in _pakistan_measures() if m.resolved_stat_var_dcids == ["sdg/BN_CAB_XOKA_GD_ZS"]
    )
    return asyncio.to_thread(_resubmit, measure, _pakistan_graph())


_ALT_GROUP = "g/unit_gate_alt"
_ALT_SIBLING = "Test_UnitGate_AltSibling"


def _alternate_survivor() -> Coroutine[Any, Any, object]:
    """A bound USD unit drops the elected SET group, whose SourceB member holds only EUR, and
    the alternate answers alone over a streamed preview
    (`test_candidate_with_partial_coverage_is_dropped_leaving_the_full_alternate`). The
    alternate belongs to a two-member group whose other member has data at Kenya."""
    usd = [_obs_facet(unit="CurrencyUSD", value=20.0)]
    alternate = _UNIT_GATE_NODES[_UNIT_GATE_ALT]
    nodes = {
        **_UNIT_GATE_NODES,
        _UNIT_GATE_ALT: {
            **alternate,
            "arcs": {**alternate["arcs"], "memberOf": {"nodes": [{"dcid": _ALT_GROUP}]}},
        },
        _ALT_GROUP: _group(_ALT_GROUP),
        _ALT_SIBLING: _var(_ALT_GROUP, measured="age"),
    }
    resolution = _unit_gate_resolution(
        obs={
            f"{_UNIT_GATE_A}|{_KEN}": [_obs_facet(unit="CurrencyUSD", value=10.0)],
            f"{_UNIT_GATE_B}|{_KEN}": [_obs_facet(unit="CurrencyEUR", value=5.0)],
            f"{_UNIT_GATE_ALT}|{_KEN}": usd,
            f"{_ALT_SIBLING}|{_KEN}": usd,
        },
        unit_text="US dollars",
        llm=_SetPlusAltBindLLM(),
        also_plausible_svs=[_UNIT_GATE_ALT],
        stream_observations=True,
        nodes=nodes,
    )

    async def scoped() -> object:
        # A streamed answer launches its observations task into the request's scope.
        async with scoped_concurrency():
            return await resolution

    return scoped()


@pytest.mark.related_read_threads
@pytest.mark.parametrize(
    "door",
    [_first_answer, _sibling_resubmit, _alternate_survivor],
    ids=["first", "resubmit", "alternate_survivor"],
)
def test_the_related_reads_add_no_serial_step(monkeypatch, search_settings, door):
    """An answer takes as many serial steps with its related reads as without them on the
    first door, the resubmit door and this candidates answer whose survivor is an alternate.
    The guarantee in general is weaker. The reads start once the elected StatVar is known and
    run concurrently with the answer's other reads, so they delay it by at most their
    deadline. On the alternate-survivor path they add no step only because the options and
    preview take at least two serial reads."""
    search_settings(related_measures_timeout_s=_THREADED_DEADLINE_S)

    def depth() -> tuple[int, list[str]]:
        requests = VirtualRequests()
        with monkeypatch.context() as patched:
            requests.install(
                patched,
                doubles=(FakeGraph, FakeSdmxClient, FakeGraphReader, SdmxClient),
                pools=(*_POOL_MODULES, related, related_read),
                executors=_EXECUTORS,
                free=_FREE_CALLS,
            )
            patched.setattr("search.engine.extract.date", _PinnedDate)
            requests.run(door())
        return requests.serial_depth(), [name for name, *_ in requests.spans]

    with_reads, names = depth()
    monkeypatch.setattr(related_read, "read_related", lambda **_kwargs: NO_RELATED)
    without_reads, _ = depth()

    assert "node_subjects_many" in names
    assert with_reads == without_reads


_GDP_NOMINAL = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"


def _population_and_gdp() -> Coroutine[Any, Any, object]:
    """Brazil's population answers definitely and takes its read. Its GDP answers with
    candidates, which take none of the read the elected `_GDP_NOMINAL` started."""
    return resolve_async(
        make_request("population and GDP in Brazil"),
        graph=FakeGraph(),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
    )


@pytest.mark.related_read_threads
@pytest.mark.parametrize(
    ("door", "read_s"),
    [(_first_answer, {"Count_Person": 0.2}), (_population_and_gdp, {_GDP_NOMINAL: 0.4})],
    ids=["taken", "untaken"],
)
def test_a_pass_waits_for_every_related_read_it_started(
    monkeypatch, search_settings, door, read_s: dict[str, float]
):
    """However a pass answers, every related-measures read it started has ended when the
    answer returns, including one no answer takes. The read named in `read_s` sleeps past
    the rest of the answer, others 0.05 s, and the check runs before `asyncio.run` shuts
    the loop's executor down, which would wait for them too."""
    search_settings(related_measures_timeout_s=_THREADED_DEADLINE_S)
    started: list[str] = []
    finished: list[str] = []

    def _slow_read(**kwargs: Any) -> Related:
        started.append(kwargs["stat_var"])
        time.sleep(read_s.get(kwargs["stat_var"], 0.05))
        finished.append(kwargs["stat_var"])
        return NO_RELATED

    monkeypatch.setattr(related_read, "read_related", _slow_read)

    async def answer() -> tuple[list[str], list[str]]:
        await door()
        return sorted(started), sorted(finished)

    started_by_return, finished_by_return = asyncio.run(answer())

    assert set(read_s) <= set(started_by_return)
    assert finished_by_return == started_by_return


@pytest.mark.related_read_threads
def test_a_cancelled_pass_ends_without_waiting_for_its_related_read(monkeypatch):
    """A pass being cancelled, as on a client disconnect, cancels its related-measures reads
    and ends at once. Waiting would hold the cancellation for the read's remaining time.
    The read sleeps 0.8 s, so a pass that waited would end well past the 0.3 s bound."""
    started = threading.Event()

    def _slow_read(**_kwargs: Any) -> Related:
        started.set()
        time.sleep(0.8)
        return NO_RELATED

    monkeypatch.setattr(related_read, "read_related", _slow_read)

    async def cancelled_after_the_read_starts() -> float:
        answer = asyncio.create_task(_first_answer())
        assert await asyncio.to_thread(started.wait, 5)
        cancelled_at = time.monotonic()
        answer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await answer
        return time.monotonic() - cancelled_at

    assert asyncio.run(cancelled_after_the_read_starts()) < 0.3


class _PinnedDate:
    """`datetime.date` with today pinned, as `offline_resolve` pins extraction's clock."""

    @staticmethod
    def today():
        return PINNED_DATE

    def __new__(cls, *args, **kwargs):
        from datetime import date

        return date(*args, **kwargs)
