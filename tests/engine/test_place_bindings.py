"""Place-axis binding follows each entity's door occurrence.

A role-bearing shape binds an axis-less place to its one required role and
refuses it when the shape requires two. A subject-only shape binds a
directional place as its subject and reports the dropped direction. Fixtures
use a neutral namespace because the deployment corpus lacks these shapes.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.departures import warning_key
from search.engine.quantity_read import QuantityOutput
from search.engine.region import DIRECTION_DISCARDED
from search.engine.region.pipeline import resolve_variable
from search.heading.departure_fact import DepartureFact
from search.models import (
    DatasetResubmitInput,
    ParsedEntity,
    ParsedPart,
    ResolveRequest,
    ResubmitParsedEntry,
)
from series_coverage.schema import ABSENT_ENTITY, RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_dataset_resubmit import _identity_entry_from_spec
from tests.engine.test_role_bearing_pipeline_e2e import _declaring_arcs
from tests.fixtures import FakeGraph, make_resolver

_SV_DCID = "TESTFIXTURE/PlaceBindings/Flow"
_KEN = "country/KEN"
_NOR = "country/NOR"

# _declaring_arcs' own observationProperties, "creditor"/"debtor", mapped to
# this deployment's fixed origin/destination axis labels -- the pair the
# coverage rows below key their RoleNames under.
_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")


def _facet(*, obs_count: int = 3, facet_id: str = "f1") -> Facet:
    """A summary-only fixture Facet: one dated point plus a facet_id, the two
    things FakeSdmxClient's derived role_facet_index needs to see this as a
    LATEST-confirmable candidate at all."""
    return Facet(
        earliest_date="2015",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"],
        values=[0.0],
        facet_id=facet_id,
    )


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _graph(
    *, resolve: dict[str, str], sv_arcs: dict, detect_query: str, obs: dict | None = None
) -> FakeGraph:
    return FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
            _SV_DCID: {"label": "Test flow", "arcs": sv_arcs},
        },
        detect={detect_query: {"svs": [_SV_DCID], "cosine_scores": [0.9]}},
        resolve=resolve,
        obs=obs if obs is not None else {},
    )


def _sdmx(observations: dict) -> FakeSdmxClient:
    return FakeSdmxClient(observations=observations)


def _run(
    *,
    entities: list[str],
    resolve: dict[str, str],
    sv_arcs: dict,
    role_query: str,
    sdmx: FakeSdmxClient,
    obs: dict | None = None,
    coverage: CoverageSnapshot | None = None,
):
    graph = _graph(resolve=resolve, sv_arcs=sv_arcs, detect_query=role_query, obs=obs)
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
        )
    )


# ---------------------------------------------------------------------------
# (a) A solitary place with no door occurrence refuses.
# ---------------------------------------------------------------------------


def test_solitary_axis_less_place_refuses_role_not_resolved():
    """An "aid Kenya" query has no directional occurrence, so it refuses before probing."""
    sdmx = _sdmx({_SV_DCID: {(("debtor", _KEN),): [_facet()]}})
    result = _run(
        entities=["Kenya"],
        resolve={"Kenya": _KEN},
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        role_query="aid Kenya",
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert result.status == "no_data", result
    assert result.no_data_reason == "role_not_resolved"
    assert result.specs == ()
    assert result.flagged_match is None
    assert sdmx.series_calls == []


def test_place_with_a_door_occurrence_still_binds_on_that_axis(monkeypatch):
    """Adding "to" binds Kenya to the destination axis."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _NOR, _KEN, [2018, 2019, 2020])],
        role_names={_SV_DCID: _ROLE_NAMES},
    )
    result = _run(
        entities=["Kenya"],
        resolve={"Kenya": _KEN},
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        role_query="aid to Kenya",
        sdmx=_sdmx({}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    assert result.specs[0].definition.identity.entities == {"destination": [_KEN]}


def test_two_occurrences_of_one_name_on_opposite_axes_bind_both(monkeypatch):
    """Two occurrences bind Kenya to both ends of the flow."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _KEN, _KEN, [2018, 2019, 2020])],
        role_names={_SV_DCID: _ROLE_NAMES},
    )
    result = _run(
        entities=["Kenya"],
        resolve={"Kenya": _KEN},
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        role_query="aid from Kenya to Kenya",
        sdmx=_sdmx({}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    assert result.specs[0].definition.identity.entities == {
        "origin": [_KEN],
        "destination": [_KEN],
    }


# ---------------------------------------------------------------------------
# (b) The subject idiom warns when it drops a direction.
# ---------------------------------------------------------------------------


def _subject_arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


def test_lone_origin_on_a_subject_only_shape_warns_and_answers_as_subject():
    """A subject-only shape binds Kenya as subject and warns that "from" was dropped."""
    result = _run(
        entities=["Kenya"],
        resolve={"Kenya": _KEN},
        sv_arcs=_subject_arcs(),
        role_query="migration from Kenya",
        sdmx=_sdmx({}),
        obs={f"{_SV_DCID}|{_KEN}": [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3}]},
        coverage=empty_coverage_snapshot(),
    )
    assert result.status == "definite", result
    assert result.specs[0].definition.identity.entities == {"observationAbout": [_KEN]}
    warnings = [w for w in result.warnings if w.code == DIRECTION_DISCARDED]
    assert len(warnings) == 1
    assert warnings[0].severity == "warn"
    assert "Kenya" in warnings[0].message
    assert "provider" in warnings[0].message
    assert result.departure_facts[warning_key(warnings[0])] == DepartureFact(
        code="direction_discarded", binding_ref="where:", asked="Kenya", over="origin"
    )
    assert warnings[0].message == (
        "Kenya was named as the provider; this shape reports the place as its "
        "own subject, so the direction was dropped."
    )


def test_no_direction_at_all_on_a_subject_only_shape_carries_no_warning():
    """A subject query without direction emits no dropped-direction warning."""
    result = _run(
        entities=["Kenya"],
        resolve={"Kenya": _KEN},
        sv_arcs=_subject_arcs(),
        role_query="population of Kenya",
        sdmx=_sdmx({}),
        obs={f"{_SV_DCID}|{_KEN}": [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3}]},
        coverage=empty_coverage_snapshot(),
    )
    assert result.status == "definite", result
    assert result.specs[0].definition.identity.entities == {"observationAbout": [_KEN]}
    assert not any(w.code == DIRECTION_DISCARDED for w in result.warnings)


def test_parsed_door_role_text_also_warns_when_the_shape_drops_it():
    """Parsed `role_text="to"` is dropped on a subject-only shape, with a warning.

    The part remains definite and binds Kenya as its subject."""
    entry = ResubmitParsedEntry(
        part=ParsedPart(
            variable_text="inward remittances",
            entities=[ParsedEntity(text="Kenya", role_text="to")],
        )
    )
    graph = _graph(
        resolve={"Kenya": _KEN},
        sv_arcs=_subject_arcs(),
        detect_query="inward remittances",
        obs={f"{_SV_DCID}|{_KEN}": [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3}]},
    )
    response = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
        graph=graph,
        llm=_FixedBindLLM(),
    )
    part = response.parts[0]
    assert part.status == "definite", part
    assert part.spec.definition.identity.entities == {"observationAbout": ["country/KEN"]}
    warnings = [w for w in part.warnings if w.code == DIRECTION_DISCARDED]
    assert len(warnings) == 1
    assert "Kenya" in warnings[0].message
    assert "recipient" in warnings[0].message


# ---------------------------------------------------------------------------
# (c) The raw_text -> dataset_resubmit round trip keeps the status.
# ---------------------------------------------------------------------------


def test_directional_binding_replays_to_the_same_status(monkeypatch):
    """A dataset-resubmit replay preserves the raw-text query's resolved direction."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _NOR, _KEN, [2018, 2019, 2020])],
        role_names={_SV_DCID: _ROLE_NAMES},
    )
    sv_arcs = _declaring_arcs(observation_properties=["creditor", "debtor"])
    role_query = "aid to Kenya"
    fresh_graph = _graph(resolve={"Kenya": _KEN}, sv_arcs=sv_arcs, detect_query=role_query)
    fresh = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Kenya"],
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, ["Kenya"]),
            pac=True,
            graph=fresh_graph,
            resolver=make_resolver(fresh_graph),
            llm=_FixedBindLLM(),
            sdmx=_sdmx({}),
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )
    assert fresh.status == "definite", fresh

    entry = _identity_entry_from_spec(fresh.specs[0], "aid")
    replay_graph = _graph(resolve={"Kenya": _KEN}, sv_arcs=sv_arcs, detect_query=role_query)
    replay = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
        graph=replay_graph,
        sdmx=_sdmx({}),
    )
    assert replay.status == fresh.status == "definite"
    assert replay.parts[0].spec.definition.identity.entities == {"destination": [_KEN]}


# ---------------------------------------------------------------------------
# (d) The parsed door on a role-bearing shape refuses an axis-less place.
# ---------------------------------------------------------------------------


def test_parsed_door_roleless_place_on_a_role_bearing_shape_refuses_role_not_resolved():
    """A parsed place without `role_text` refuses before observation probing."""
    entry = ResubmitParsedEntry(
        part=ParsedPart(variable_text="aid", entities=[ParsedEntity(text="Kenya")])
    )
    graph = _graph(
        resolve={"Kenya": _KEN},
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        detect_query="aid",
    )
    sdmx = _sdmx({_SV_DCID: {(("debtor", _KEN),): [_facet()]}})
    response = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
        graph=graph,
        llm=_FixedBindLLM(),
        sdmx=sdmx,
    )
    part = response.parts[0]
    assert part.status == "no_data", part
    assert part.no_data.reason == "role_not_resolved"


# ---------------------------------------------------------------------------
# (e) An undirected place binds to a shape's one required role.
# ---------------------------------------------------------------------------

_UGA = "country/UGA"
_ONE_ROLE_NAMES = RoleNames(donor=None, recipient="debtor")


def test_undirected_place_binds_to_the_one_required_role(monkeypatch):
    """A debtor-only shape binds the undirected Kenya in "debt Kenya" as the debtor."""
    index = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", ABSENT_ENTITY, _KEN, [2018, 2019, 2020])],
        role_names={_SV_DCID: _ONE_ROLE_NAMES},
    )
    result = _run(
        entities=["Kenya"],
        resolve={"Kenya": _KEN},
        sv_arcs=_declaring_arcs(observation_properties=["debtor"]),
        role_query="debt Kenya",
        sdmx=_sdmx({}),
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert result.status == "definite", result
    assert result.specs[0].definition.identity.entities == {"destination": [_KEN]}


def test_two_undirected_places_on_a_one_role_shape_refuse_role_not_resolved():
    """ "debt Kenya Uganda" names two places for one role, so it refuses."""
    sdmx = _sdmx({})
    result = _run(
        entities=["Kenya", "Uganda"],
        resolve={"Kenya": _KEN, "Uganda": _UGA},
        sv_arcs=_declaring_arcs(observation_properties=["debtor"]),
        role_query="debt Kenya Uganda",
        sdmx=sdmx,
    )
    assert result.status == "no_data", result
    assert result.no_data_reason == "role_not_resolved"
    assert sdmx.series_calls == []


def test_undirected_place_refuses_when_a_directed_place_holds_the_required_axis():
    """In "debt Kenya to Uganda" Uganda already holds the debtor axis, so Kenya refuses."""
    sdmx = _sdmx({})
    result = _run(
        entities=["Kenya", "Uganda"],
        resolve={"Kenya": _KEN, "Uganda": _UGA},
        sv_arcs=_declaring_arcs(observation_properties=["debtor"]),
        role_query="debt Kenya to Uganda",
        sdmx=sdmx,
    )
    assert result.status == "no_data", result
    assert result.no_data_reason == "role_not_resolved"
    assert sdmx.series_calls == []


def test_undirected_place_beside_a_directed_one_on_a_two_role_shape_refuses():
    """A creditor-debtor shape requires both roles, so undirected Kenya stays ambiguous."""
    sdmx = _sdmx({})
    result = _run(
        entities=["Kenya", "Norway"],
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        role_query="aid Kenya from Norway",
        sdmx=sdmx,
    )
    assert result.status == "no_data", result
    assert result.no_data_reason == "role_not_resolved"
    assert sdmx.series_calls == []
