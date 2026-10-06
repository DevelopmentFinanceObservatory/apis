"""Tests entity-option serialization and discovery on standard and role-bearing queries.

Open directional axes carry role counts and, when requested, member lists (#664)."""

from __future__ import annotations

import asyncio
import json

from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.models import (
    EntityOptions,
    GraphRef,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    RoleAxisFact,
    SlotOption,
    SlotValue,
)
from series_coverage.schema import RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    door_roles_for_test_query,
    materialise_double_from_obs,
    offline_resolve,
)
from tests.fixtures import FakeGraph, make_resolver

# Materialise requires a provenance id on every facet.
_BRAZIL_FERTILITY_OBS = {
    "FertilityRate_Person_Female|country/BRA": [
        {
            "earliestDate": "2000",
            "latestDate": "2022",
            "obsCount": 20,
            "provenanceId": "dc/base/WorldBank",
        }
    ],
}


def test_entity_options_absent_when_no_donor():
    """A standard query with real observations has no directional entity options."""
    req = ResolveRequest(
        input=RawTextInput(query="fertility rate in Brazil"),
        response_options=ResolveOptions(include_observations=True, include_option_coverage=True),
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_BRAZIL_FERTILITY_OBS)
    r = offline_resolve(
        req,
        sdmx=FakeSdmxClient(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    ).parts[0]
    assert r.status == "definite"
    spec = r.spec
    assert spec.resolved.observations is not None
    assert spec.resolved.entity_options == []


def test_entity_options_event_serialize_sse_roundtrip():
    from pydantic import TypeAdapter

    from search.engine.events import EntityOptionsEvent, Event, serialize_sse

    ev = EntityOptionsEvent(
        spec_id="abc123",
        entity_ref="country/FRA",
        role="origin",
        options=[
            SlotOption(
                value=SlotValue(
                    ref=GraphRef(dcid="country/FRA", label="France"), value_kind="entity"
                )
            )
        ],
        truncated=False,
    )
    frame = serialize_sse(ev)
    lines = frame.strip("\n").split("\n")
    assert lines[0] == "event: entity_options"
    assert lines[1].startswith("data: ")
    payload = json.loads(lines[1][len("data: ") :])
    assert payload["kind"] == "entity_options"
    assert payload["entity_ref"] == "country/FRA"

    # Validates through the discriminated Event union.
    adapter = TypeAdapter(Event)
    parsed = adapter.validate_json(ev.model_dump_json())
    assert isinstance(parsed, EntityOptionsEvent)
    assert parsed == ev


def test_entity_options_round_trips_with_entity_ref_absent():
    """An open-axis entry omits ``entity_ref`` on the wire and round-trips (#880)."""
    eo = EntityOptions(
        entity_ref=None, role="origin", options=[], truncated=True, source="observed"
    )
    dumped = eo.model_dump(mode="json", exclude_none=True)
    assert "entity_ref" not in dumped
    assert EntityOptions.model_validate(dumped) == eo


# ---------------------------------------------------------------------------
# Role-bearing: an open directional axis populates entity_options
# ---------------------------------------------------------------------------

_SV_DCID = "TESTFIXTURE/EntityOptions/Flow"
_KEN = "country/KEN"
_NOR = "country/NOR"
_USA = "country/USA"


def _facet(*, obs_count: int = 3, facet_id: str = "f1") -> Facet:
    return Facet(
        earliest_date="2015",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"],
        values=[0.0],
        facet_id=facet_id,
    )


def _declaring_arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
    }


class _FixedBindLLM:
    """Elects shape 0 without bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _run(
    *,
    entities: list[str],
    resolve: dict[str, str],
    role_query: str,
    sdmx: FakeSdmxClient,
    include_option_coverage: bool = False,
    coverage: CoverageSnapshot | None = None,
):
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
            _SV_DCID: {"label": "Test flow", "arcs": _declaring_arcs()},
        },
        detect={role_query: {"svs": [_SV_DCID], "cosine_scores": [0.9]}},
        resolve=resolve,
        obs={},
    )
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=entities,
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, entities),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=coverage if coverage is not None else empty_coverage_snapshot(),
            options=EngineOptions(include_option_coverage=include_option_coverage),
        )
    )


def _providers_observations() -> dict[str, dict[tuple[tuple[str, str], ...], list[Facet]]]:
    return {
        _SV_DCID: {
            (("creditor", _NOR), ("debtor", _KEN)): [_facet()],
            (("creditor", _USA), ("debtor", _KEN)): [_facet()],
        }
    }


def _providers_sdmx() -> FakeSdmxClient:
    return FakeSdmxClient(observations=_providers_observations())


_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")


def _providers_coverage_rows() -> list[tuple[str, str, str, str, list[int]]]:
    """Returns two donor coverage rows at the bound destination, Kenya."""
    return [
        (_SV_DCID, "f1", _NOR, _KEN, [2018, 2019, 2020]),
        (_SV_DCID, "f2", _USA, _KEN, [2019, 2020]),
    ]


def test_bound_destination_open_origin_reports_the_count_without_the_list(monkeypatch):
    """#664/#880's core case: 'aid to Kenya' names only the recipient.

    Two providers (Norway, USA) reach the client as a role count and an open-origin option.
    The member list is empty until ``include_option_coverage`` is enabled."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _providers_coverage_rows(), role_names={_SV_DCID: _ROLE_NAMES}
    )
    result = _run(
        entities=["Kenya"],
        resolve={"Kenya": _KEN, "Norway": _NOR, "USA": _USA},
        role_query="aid to Kenya",
        sdmx=_providers_sdmx(),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    spec = result.specs[0]
    assert spec.resolved.role_axis_facts == [
        RoleAxisFact(role="origin", member_count=2, members_available=True)
    ]
    assert spec.resolved.entity_options == [
        EntityOptions(entity_ref=None, role="origin", options=[], truncated=True, source="observed")
    ]


def test_bound_destination_open_origin_lists_members_under_include_option_coverage(monkeypatch):
    """With option coverage enabled, the open origin lists labelled members."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _providers_coverage_rows(), role_names={_SV_DCID: _ROLE_NAMES}
    )
    result = _run(
        entities=["Kenya"],
        resolve={"Kenya": _KEN, "Norway": _NOR, "USA": _USA},
        role_query="aid to Kenya",
        sdmx=_providers_sdmx(),
        include_option_coverage=True,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    entity_options = result.specs[0].resolved.entity_options
    assert len(entity_options) == 1
    eo = entity_options[0]
    assert eo.entity_ref is None
    assert eo.role == "origin"
    assert eo.truncated is False
    assert {o.value.ref.dcid: o.value.ref.label for o in eo.options if o.value.ref} == {
        _NOR: "Norway",
        _USA: "USA",
    }


def test_both_axes_open_entity_options_carries_both_roles(monkeypatch):
    """With no named entity, both discovered axes expose role counts and options."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _providers_coverage_rows(), role_names={_SV_DCID: _ROLE_NAMES}
    )
    sdmx = FakeSdmxClient(observations=_providers_observations())

    result = _run(
        entities=[],
        resolve={"Kenya": _KEN, "Norway": _NOR, "USA": _USA},
        role_query="aid disbursed",
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    spec = result.specs[0]
    facts = {f.role: f for f in spec.resolved.role_axis_facts or []}
    assert facts == {
        "destination": RoleAxisFact(role="destination", member_count=1, members_available=True),
        "origin": RoleAxisFact(role="origin", member_count=2, members_available=True),
    }
    options_by_role = {eo.role: eo for eo in spec.resolved.entity_options}
    assert set(options_by_role) == {"origin", "destination"}
    assert all(eo.entity_ref is None for eo in options_by_role.values())


_ALT_ROLE_SV = "TESTFIXTURE/EntityOptions/AltFlow"


def _alt_role_arcs() -> dict:
    """Builds a sibling flow shape with no observations."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amountAlt"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
    }


class _CollapseBindLLM:
    """Elects shape 0 and includes the no-data sibling as plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None
        ), None


def test_candidates_single_survivor_collapse_populates_entity_options(monkeypatch):
    """A single surviving candidate publishes entity options and option timing."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _providers_coverage_rows(), role_names={_SV_DCID: _ROLE_NAMES}
    )
    graph = FakeGraph(
        nodes={
            "country/KEN": {"label": "Kenya", "type": "Country"},
            "country/NOR": {"label": "Norway", "type": "Country"},
            "country/USA": {"label": "USA", "type": "Country"},
            _SV_DCID: {"label": "Test flow", "arcs": _declaring_arcs()},
            _ALT_ROLE_SV: {"label": "Alt test flow", "arcs": _alt_role_arcs()},
        },
        detect={
            "aid to Kenya": {
                "svs": [_SV_DCID, _ALT_ROLE_SV],
                "cosine_scores": [0.9, 0.85],
            }
        },
        resolve={"Kenya": _KEN, "Norway": _NOR, "USA": _USA},
        obs={},
    )
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Kenya"],
            date_request=None,
            detect_query="aid to Kenya",
            role_query="aid to Kenya",
            entity_roles=door_roles_for_test_query("aid to Kenya", ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_CollapseBindLLM(),
            sdmx=_providers_sdmx(),
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            options=EngineOptions(include_option_coverage=True),
        )
    )
    assert result.status == "definite", result
    spec = result.specs[0]
    assert spec.resolved.members[0].ref.dcid == _SV_DCID
    assert spec.resolved.role_axis_facts == [
        RoleAxisFact(role="origin", member_count=2, members_available=True)
    ]
    assert spec.resolved.entity_options != []
    eo = spec.resolved.entity_options[0]
    assert eo.role == "origin"
    assert {o.value.ref.dcid for o in eo.options if o.value.ref} == {_NOR, _USA}
    assert "options" in result.timing_by_step
