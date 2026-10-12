"""Pins the response cache's part memo: what keys a resubmitted identity, what it refuses to
store, and that a hit answers as the fresh resolution it replaces on both transports."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging

import pytest

from code_index.shared import get_index, get_measure_index
from graph_generation import FakeGeneration
from origin_catalogue import bound_catalogue
from search.engine.core import resolve_events
from search.engine.errors import GraphInfraError
from search.engine.events import ObservationsEvent, ResponseEvent
from search.engine.region import resolve_spec_resubmit
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.response_cache_flow import PartMemoScope
from search.engine.response_cache_keys import part_memo_key
from search.engine.response_cache_memo import PartMemo, region_bytes
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.models import (
    BaseIdentity,
    DatasetResubmitInput,
    EntityGroup,
    ResolveOptions,
    ResolveRequest,
    ResolveResponse,
    ResubmitIdentityEntry,
    ResubmitIdentityPayload,
    Warning,
)
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._base_identity import make_identity, subject_entity
from tests.engine._builders import value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    materialise_double_from_obs,
    member_walk_reader,
    offline_resolve,
)
from tests.engine.test_dataset_resubmit import _gender_entry, _gender_fold_graph
from tests.engine.test_membership_authority import (
    _PLACES,
    _UNSETTLED_WALKS,
    _debt_coverage,
)
from tests.engine.test_membership_authority import _run as _first_answer
from tests.engine.test_resubmit_entity_groups import _CONFIRMED, _NODE, _TRIANGLE, _nodes
from tests.engine.test_resubmit_entity_groups import _OBS as _TRIANGLE_OBS
from tests.engine.test_resubmit_entity_groups import _identity as _triangle_identity
from tests.engine.test_role_class_shapes import _DEBT_NODES, _OBS, _TOTAL, _walk_reader
from tests.fixtures import FakeGraph, FakeLLM, _load_json
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

# ---------------------------------------------------------------------------
# The key: every input that changes the answer changes it
# ---------------------------------------------------------------------------

_SHAPE = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "fertilityRate",
    "stat_type": "measuredValue",
}
_FERTILITY_SV = "FertilityRate_Person_Female"
_FEMALE = value_slot("how", "gender", "gender", "Female", "Female")


def _identity(slots: list) -> BaseIdentity:
    return make_identity(_SHAPE, slots, entities=[subject_entity("country/KEN")])


def _payload(
    identity: BaseIdentity | None = None,
    entity_groups: dict[str, EntityGroup] | None = None,
) -> ResubmitIdentityPayload:
    return ResubmitIdentityPayload(
        identity=identity or _identity([_FEMALE]), entity_groups=entity_groups
    )


_FLAGS = {
    "include_option_coverage": False,
    "stream_slot_options": False,
    "include_observations": False,
    "stream_observations": False,
}
_STAMPS: dict[str, str | None] = {
    "build_commit": "build-a",
    "graph_generation": "gen-a",
    "coverage_stamp": "coverage-a",
    "conversion_catalogue": "catalogue-a",
    "origin_catalogue": "build-a",
    "today": "2026-10-06",
}


def _key(
    payload: ResubmitIdentityPayload | None = None,
    *,
    flag: str | None = None,
    un_system_enabled: bool = True,
    **stamps: str | None,
) -> str:
    """The key of `payload` under the baseline flags and stamps, with `flag` turned on and
    `stamps` replaced."""
    scope = {**_STAMPS, **stamps}
    return part_memo_key(
        payload=payload if payload is not None else _payload(),
        engine_flags={**_FLAGS, **({flag: True} if flag is not None else {})},
        build_commit=scope["build_commit"] or "",
        graph_generation=scope["graph_generation"] or "",
        coverage_stamp=scope["coverage_stamp"] or "",
        conversion_catalogue=scope["conversion_catalogue"],
        origin_catalogue=scope["origin_catalogue"] or "",
        un_system_enabled=un_system_enabled,
        today=scope["today"] or "",
    )


_GROUPS = {"observationAbout": EntityGroup(dcid=None, label="East Africa")}


@pytest.mark.parametrize(
    "perturbed",
    [
        pytest.param(lambda: _key(_payload(entity_groups=_GROUPS)), id="entity_groups"),
        pytest.param(lambda: _key(_payload(identity=_identity([]))), id="identity_slots"),
        *(pytest.param(lambda flag=flag: _key(flag=flag), id=flag) for flag in _FLAGS),
        *(
            pytest.param(lambda stamp=stamp: _key(**{stamp: "changed"}), id=stamp)
            for stamp in _STAMPS
        ),
        pytest.param(lambda: _key(conversion_catalogue=None), id="no_conversion_catalogue"),
        pytest.param(lambda: _key(un_system_enabled=False), id="un_system_enabled"),
    ],
)
def test_each_key_input_alone_changes_the_key(perturbed):
    assert perturbed() != _key()


def test_equal_inputs_built_independently_share_a_key():
    assert _key(_payload()) == _key(_payload())


@pytest.mark.parametrize(
    ("live", "same"),
    [
        pytest.param(FakeCatalogue(), True, id="same-catalogue"),
        pytest.param(FakeCatalogue(build_id="build-b"), False, id="new-build"),
        pytest.param(FakeCatalogue(un_system_enabled=False), False, id="un-system-off"),
    ],
)
def test_the_scope_keys_on_the_origin_catalogue_it_was_built_under(live, same: bool):
    def scope_key() -> str:
        scope = PartMemoScope(
            PartMemo(), build_commit="b", today="2026-10-06", coverage=empty_coverage_snapshot()
        )
        return scope.key(_payload(), EngineOptions())

    with bound_catalogue(FakeCatalogue()):
        before = scope_key()
    with bound_catalogue(live):
        assert (scope_key() == before) is same


# ---------------------------------------------------------------------------
# A region that swallowed a failed read is never stored
# ---------------------------------------------------------------------------


class _GroupReadFails(FakeGraph):
    def node_subjects_many(self, *args, **kwargs):
        raise GraphInfraError("members read failed")


def _edited_debt_resubmit(
    member_graph, *, group_read_fails: bool = False
) -> tuple[ResubmitIdentityPayload, RegionResult]:
    """Edit a recipient-only debt total to the public guarantee, so the mint's walk elects
    the members. With `group_read_fails` the identity also names a destination group whose
    members read fails, which drops the group and answers with no warning."""
    base = _first_answer([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0]
    document = base.definition.identity.model_dump(mode="json", exclude_none=True)
    edited = BaseIdentity.model_validate(
        {
            **document,
            "slots": [
                *document["slots"],
                {"property": "guarantee", "binding": {"state": "value", "ref": "Public"}},
            ],
        }
    )
    payload = ResubmitIdentityPayload(
        identity=edited,
        entity_groups=(
            {"destination": EntityGroup(dcid="africa", label="Africa")}
            if group_read_fails
            else None
        ),
    )
    graph = (_GroupReadFails if group_read_fails else FakeGraph)(
        nodes={**_PLACES, **_DEBT_NODES},
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in _DEBT_NODES},
    )
    region = resolve_spec_resubmit(
        inp=payload,
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=_debt_coverage(),
        options=EngineOptions(materialise_graph=member_graph),
    )
    return payload, region


@pytest.mark.parametrize(
    ("group_read_fails", "stored"),
    [(False, 1), (True, 0)],
    ids=["settled", "group-read-failed"],
)
def test_a_region_that_swallowed_a_failed_read_is_not_stored(group_read_fails, stored):
    payload, region = _edited_debt_resubmit(
        _walk_reader(_DEBT_NODES), group_read_fails=group_read_fails
    )
    memo = PartMemo()
    scope = PartMemoScope(
        memo, build_commit="b", today="2026-10-06", coverage=empty_coverage_snapshot()
    )

    scope.store(scope.key(payload, EngineOptions()), region)

    assert region.warnings == ()
    assert region.degraded is group_read_fails
    assert memo.stats()["entries"] == stored


@pytest.mark.parametrize(
    ("walk", "error"), [("timeout", TimeoutError), ("graph_failure", GraphInfraError)]
)
def test_a_walk_that_fails_raises_and_leaves_no_region_to_store(walk, error):
    """A resubmit refuses a walk past its deadline (504) or a failed walk read (503), so no
    region reaches the memo."""
    with pytest.raises(error):
        _edited_debt_resubmit(_UNSETTLED_WALKS[walk]())


# ---------------------------------------------------------------------------
# A hit answers as the fresh resolution: folded, buffered and streamed
# ---------------------------------------------------------------------------


def _without_timing(response: ResolveResponse) -> dict:
    """The response's wire form less what describes one run: its trace, timing and usage."""
    document = response.model_dump(mode="json")
    for key in ("trace_id", "timing_ms", "llm_usage"):
        document["diagnostics"].pop(key, None)
    for part in document.get("parts", []):
        part.pop("timing_ms", None)
    return document


def test_a_hit_folded_with_a_fresh_sibling_equals_an_all_fresh_run(caplog):
    """The female part is stored by a request that resolves it alone. A request posting it
    beside a male part hits it, resolves the male part fresh, and folds the two into one
    Spec exactly as a request that resolves both fresh does. Its log line counts the hit."""
    graph = _gender_fold_graph()
    sdmx = FakeSdmxClient(observations={}, availability={})
    female = _gender_entry("Female", "Count_Person_Female", window=None)
    male = _gender_entry("Male", "Count_Person_Male", window=None)
    both = ResolveRequest(input=DatasetResubmitInput(entries=[female, male]))

    walk = member_walk_reader(graph)
    fresh = offline_resolve(both, graph=graph, sdmx=sdmx, materialise_graph=walk)
    memo = PartMemo()
    offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[female])),
        graph=graph,
        sdmx=sdmx,
        materialise_graph=walk,
        part_memo=memo,
    )
    # Another test can leave the logger at INFO, so the two earlier resolves log too.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="search.engine.core"):
        warm = offline_resolve(both, graph=graph, sdmx=sdmx, materialise_graph=walk, part_memo=memo)

    assert any(w.code == "binding_folded" for w in warm.parts[1].warnings)
    assert _without_timing(warm) == _without_timing(fresh)
    assert set(warm.parts[0].timing_ms.by_step or {}) == {"response_cache_part_memo"}
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("resolve ")]
    logged = json.loads(line.removeprefix("resolve "))
    assert (logged["memo_hits"], logged["memo_misses"]) == (1, 1)


def test_a_memo_hit_overlays_comparison_without_mutating_the_stored_region():
    graph = _gender_fold_graph()
    sdmx = FakeSdmxClient(observations={}, availability={})
    memo = PartMemo()
    plain = _gender_entry("Female", "Count_Person_Female", window=None)
    comparison = plain.model_copy(update={"separate": True})

    def resolve(entry: ResubmitIdentityEntry):
        return offline_resolve(
            ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
            graph=graph,
            sdmx=sdmx,
            materialise_graph=member_walk_reader(graph),
            part_memo=memo,
        )

    assert resolve(plain).parts[0].separate is False
    assert resolve(comparison).parts[0].separate is True
    assert resolve(plain).parts[0].separate is False
    assert (memo.stats()["hits"], memo.stats()["entries"]) == (2, 1)


@pytest.mark.parametrize("grouped_first", [True, False], ids=["group_then_none", "none_then_group"])
def test_one_identity_posted_with_and_without_a_group_gets_an_answer_for_each(grouped_first):
    """The same identity posted with a graph-confirmed group and with none, through one live
    memo: each post is a miss, the grouped one serves the group on `resolved.entity_groups`
    and the other serves none."""
    spec_id, identity = _triangle_identity(_TRIANGLE)
    graph = FakeGraph(nodes=_nodes(_TRIANGLE), obs=_TRIANGLE_OBS)
    memo = PartMemo()
    posted = {"observationAbout": EntityGroup(dcid=_NODE, label="posted")}
    order = [posted, None] if grouped_first else [None, posted]

    served = []
    for entity_groups in order:
        entry = ResubmitIdentityEntry(
            variable_text="population", identity=identity, entity_groups=entity_groups
        )
        response = offline_resolve(
            ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            part_memo=memo,
        )
        (part,) = response.parts
        assert part.spec.spec_id == spec_id
        served.append(part.spec.resolved.entity_groups)

    assert served == ([_CONFIRMED, None] if grouped_first else [None, _CONFIRMED])
    assert (memo.stats()["hits"], memo.stats()["entries"]) == (0, 2)


def _fertility_request(options: ResolveOptions) -> ResolveRequest:
    payload = _payload()
    return ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                ResubmitIdentityEntry(
                    variable_text="fertility",
                    identity=payload.identity,
                )
            ]
        ),
        response_options=options,
    )


class _WalkAndReadGraph:
    """The mint's member walk over `walk` and every other read over `reads`."""

    def __init__(self, walk: FakeMintGraphReader, reads: object) -> None:
        self._walk, self._reads = walk, reads

    def node_subjects(self, *args, **kwargs):
        return self._walk.node_subjects(*args, **kwargs)

    def node_arcs_many(self, *args, **kwargs):
        return self._walk.node_arcs_many(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._reads, name)


def _fertility_materialise():
    """A materialise graph that walks the fixture graph's members and serves the fertility
    series at Kenya, and its SDMX double."""
    key = f"{_FERTILITY_SV}|country/KEN"
    reads, sdmx = materialise_double_from_obs({key: _load_json("graph_obs.json")[key]})
    return _WalkAndReadGraph(member_walk_reader(), reads), sdmx


def test_cold_and_warm_buffered_responses_are_equal_except_timing():
    materialise_graph, materialise_sdmx = _fertility_materialise()
    request = _fertility_request(
        ResolveOptions(include_observations=True, include_option_coverage=True)
    )
    memo = PartMemo()

    def _resolve() -> ResolveResponse:
        return offline_resolve(
            request,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            part_memo=memo,
        )

    cold = _resolve()
    warm = _resolve()

    assert cold.status == "definite"
    assert cold.parts[0].spec.resolved.observations is not None
    assert memo.stats()["hits"] == 1
    assert _without_timing(warm) == _without_timing(cold)


def test_a_streamed_hit_still_emits_its_trailing_observations_event():
    """The stored region holds no closure, so the hit binds this request's own and its
    observations read streams after the response as on the fresh run."""
    materialise_graph, materialise_sdmx = _fertility_materialise()
    request = _fertility_request(ResolveOptions(include_observations=True))
    memo = PartMemo()
    generation = FakeGeneration(token="gen-a")

    async def _stream() -> list:
        return [
            event
            async for event in resolve_events(
                request,
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(),
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                stream_slot_options=True,
                part_memo=memo,
                generation=generation,
            )
        ]

    cold = asyncio.run(_stream())
    warm = asyncio.run(_stream())

    assert memo.stats()["hits"] == 1
    assert [event.kind for event in warm] == [event.kind for event in cold]
    cold_observations = [e for e in cold if isinstance(e, ObservationsEvent)]
    warm_observations = [e for e in warm if isinstance(e, ObservationsEvent)]
    assert cold_observations
    assert warm_observations == cold_observations
    (cold_response,) = [e.response for e in cold if isinstance(e, ResponseEvent)]
    (warm_response,) = [e.response for e in warm if isinstance(e, ResponseEvent)]
    assert _without_timing(warm_response) == _without_timing(cold_response)


# ---------------------------------------------------------------------------
# The memo is bounded by bytes
# ---------------------------------------------------------------------------


def _region_of(message: str) -> RegionResult:
    """A no-data region weighing what its one warning serialises to."""
    return RegionResult(
        variable_text="x",
        status="no_data",
        specs=(),
        no_data_reason="no_observations",
        warnings=(Warning(code="part_restated", severity="info", message=message),),
        timing_by_step={},
    )


def test_a_region_over_the_entry_cap_is_served_and_not_kept():
    small, large = _region_of("a" * 10), _region_of("a" * 500)
    memo = PartMemo(max_bytes=10_000, max_entry_bytes=region_bytes(small))

    memo.put("gen-a", "small", small)
    memo.put("gen-a", "large", large)

    assert memo.get("gen-a", "small") is small
    assert memo.get("gen-a", "large") is None
    stats = memo.stats()
    assert (stats["entries"], stats["oversized"], stats["bytes"]) == (1, 1, region_bytes(small))


def _holding_members(count: int) -> RegionResult:
    """A no-data region whose wire part is one short warning, holding `count` served members
    and their observation entities, as a large role-bearing family's region does."""
    return dataclasses.replace(
        _region_of("a"),
        served_members=tuple(
            ArcsMemberFacts(arcs={"constraintProperties": [f"prop{i}"], "name": [f"sv {i}"]})
            for i in range(count)
        ),
        observation_entities={"recipient": [f"country/{i:03d}" for i in range(count)]},
    )


def test_a_region_is_weighed_by_everything_it_holds_beside_its_wire_part():
    """Two regions with the same wire part weigh apart by the members they hold, so the
    memo refuses the one whose members pass the per-entry cap."""
    few, many = _holding_members(1), _holding_members(2_000)
    memo = PartMemo(max_bytes=1 << 30, max_entry_bytes=10 * region_bytes(few))

    memo.put("gen-a", "few", few)
    memo.put("gen-a", "many", many)

    assert region_bytes(many) > 100 * region_bytes(few)
    assert memo.get("gen-a", "few") is few
    assert memo.get("gen-a", "many") is None
    assert memo.stats()["oversized"] == 1


@pytest.mark.parametrize(
    "shared",
    [
        pytest.param(get_index, id="code_index"),
        pytest.param(get_measure_index, id="measure_index"),
        pytest.param(lambda: empty_coverage_snapshot().index, id="coverage_index"),
    ],
)
def test_a_region_reaching_a_process_wide_index_is_weighed_by_its_own_heap(shared):
    """A role-bearing region reaches the coverage index through its served filter, and the
    process holds each index whether or not the memo keeps the region, so the weight
    leaves it out."""

    def holding(obj: object) -> RegionResult:
        return dataclasses.replace(_region_of("a"), observation_entities={"x": [obj]})

    assert region_bytes(holding(shared())) == region_bytes(holding(None))


def test_weighing_stops_once_past_the_limit():
    many = _holding_members(2_000)
    limit = region_bytes(_holding_members(10))

    assert limit < region_bytes(many, limit=limit) < region_bytes(many) // 10


def test_the_total_bound_evicts_the_least_recently_used_region_by_bytes():
    regions = {name: _region_of(name * 100) for name in "abc"}
    size = region_bytes(regions["a"])
    memo = PartMemo(max_bytes=2 * size, max_entry_bytes=size)

    memo.put("gen-a", "a", regions["a"])
    memo.put("gen-a", "b", regions["b"])
    assert memo.get("gen-a", "a") is regions["a"]
    memo.put("gen-a", "c", regions["c"])

    assert memo.get("gen-a", "b") is None
    assert memo.get("gen-a", "a") is regions["a"]
    assert memo.get("gen-a", "c") is regions["c"]
    assert memo.stats()["bytes"] == 2 * size


def test_a_zero_byte_bound_turns_the_memo_off():
    memo = PartMemo(max_bytes=0)
    scope = PartMemoScope(
        memo, build_commit="b", today="2026-10-06", coverage=empty_coverage_snapshot()
    )
    assert not scope.active
