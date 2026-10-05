"""An index member recorded under two datasets is read from its elected one on every door.

The answer-invariants world's `overlap` rows hold a broad `Ids` source and a narrower `Alt`
one for one flow variable. `code_index.members.index_family` is replaced where Search calls
it, so the family elects `Alt` for a row recorded under both and `shape_member_facts` runs
as written, on the first answer's recall and on a resubmit's posted member alike. Ranking
alone would serve `Ids`.
"""

from __future__ import annotations

import dataclasses

import pytest

from code_index.members import DatasetFamily
from code_index.shared import get_index
from code_index.types import DatasetDeclaration, StatVarRow
from materialise import MaterialisationRequest, materialise_preview
from materialise.errors import MaterialisationError
from materialise.facets import ElectedDataset, FacetBinding
from materialise.rate_tables import RateTableCache
from materialise.window import RequestedWindow
from search.engine import spec_build
from search.engine.region.option_coverage import _maybe_build_facet_options
from search.engine.retrieval import bound_values
from search.models import IdentityBindingValue, Spec
from tests.engine import _answer_invariants_world as world_module
from tests.engine._answer_invariants_world import ALAND, ALT, BLAND, IDS, SOURCE_LABELS, World

_ROLE_ROUTES = ("named_pair", "named_bulk", "discovered_pair", "discovered_bulk")


def _served(spec: Spec) -> set[str | None]:
    observations = spec.resolved.observations
    assert observations is not None
    return {series.facet["provenance"].ref for series in observations.series}


_STALE = "provenance/StaleElected"
"""A dataset the family elects whose rows the world holds no facet of, while the code index
records the member under it alone: an index that drifted from coverage."""


def _electing(
    monkeypatch, search_settings, route: str, *, provenance: str, recorded: tuple[str, ...]
) -> World:
    """The route's `overlap` world, its one variable an index row under `recorded` whose
    family elects `provenance`."""
    world_module.carry_logical_clock(monkeypatch)
    world_module.log_unit_decisions(monkeypatch)
    world_module.log_phases(monkeypatch)
    world_module.emit_bind_stage_sentence(monkeypatch)
    search_settings(obs_preview_max_pairs=1 if route.endswith("bulk") else 120)
    world = world_module.world(route, "overlap", "definite")
    (stat_var,) = world.stat_vars
    row = StatVarRow(
        dcid=stat_var,
        population_type="EconomicActivity",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=world.declared_roles,
        constraint_properties={},
        provenances=tuple(sorted(recorded)),
    )
    family = DatasetFamily(provenance=provenance, rows=(row,))
    monkeypatch.setattr(bound_values, "index_family", lambda _index, _query: family)

    real = get_index()
    absent = next(iter(real.datasets.values())).absent_observation
    labels = {**SOURCE_LABELS, _STALE: "Stale"}

    declared = {
        **real.datasets,
        **{
            declared: DatasetDeclaration(
                provenance=declared,
                label=labels[declared],
                synonyms=(),
                opt_in=False,
                absent_observation=absent,
            )
            for declared in {ALT, IDS, provenance}
        },
    }

    class _DeclaringBoth:
        def __init__(self) -> None:
            self.datasets = declared

        def __getattr__(self, name: str):
            return getattr(real, name)

    declaring = _DeclaringBoth()
    monkeypatch.setattr(spec_build, "get_index", lambda: declaring)
    return world


@pytest.fixture
def two_dataset_world(monkeypatch, search_settings, request) -> World:
    """The route's world, its one variable an index row under `Alt` and `Ids` whose family
    elects `Alt`."""
    return _electing(
        monkeypatch, search_settings, request.param, provenance=ALT, recorded=(ALT, IDS)
    )


def _mint_request(
    world: World, *, provenance: str = ALT, recorded: tuple[str, ...] = (ALT, IDS)
) -> MaterialisationRequest:
    (stat_var,) = world.stat_vars
    return MaterialisationRequest(
        stat_var=f"dcid:{stat_var}",
        role_bindings=(
            {}
            if world.discovered
            else {
                "debtor" if "debtor" in world.declared_roles else "observationAbout": [ALAND, BLAND]
            }
        ),
        declared_roles=world.declared_roles,
        facet_bindings={},
        window=RequestedWindow(kind="all"),
        dataset=ElectedDataset(provenance=provenance, member_provenances=tuple(sorted(recorded))),
    )


def _mint(world: World, request: MaterialisationRequest):
    return materialise_preview(
        request,
        sdmx=world.materialise_sdmx,
        graph=world.materialise_graph,
        limit=10_000,
        coverage=world.index,
        rate_cache=RateTableCache(),
    )


@pytest.mark.parametrize("two_dataset_world", _ROLE_ROUTES, indirect=True)
def test_every_door_serves_the_elected_dataset_the_mint_serves(two_dataset_world: World) -> None:
    """The first answer, its unchanged resubmit and a mint of the identity serve `Alt`, and
    the two answers state the same sentence. A resubmit asking years `Alt` holds no point in
    answers `no_observations` and does not fall back to `Ids`. A resubmit pinning `Ids` is
    refused `facet_not_available`, as the mint refuses that pin."""
    world = two_dataset_world
    with world_module.door_loop():
        first = world.first([ALAND, BLAND], None, preview=True, unit_text=None)
        (spec,) = first.specs
        again = world.resubmit(spec, preview=True)
        windowed = world.resubmit(
            spec, preview=True, window={"kind": "range", "start_year": 2000, "end_year": 2018}
        )
        pinned = world.resubmit(
            spec, preview=True, facets={"provenance": IdentityBindingValue(ref=IDS)}
        )

    assert spec.resolved.facets["provenance"].ref == ALT
    assert _served(spec) == {ALT}
    assert [t.text for t in spec.resolved.sentence_plan or [] if t.axis == "source"] == [
        "Alternative"
    ]

    (resubmitted,) = again.specs
    assert resubmitted.resolved.facets["provenance"].ref == ALT
    assert _served(resubmitted) == {ALT}
    assert resubmitted.resolved.rendered_sentence == spec.resolved.rendered_sentence

    assert windowed.result is not None
    assert (windowed.result.status, windowed.result.no_data_reason) == (
        "no_data",
        "no_observations",
    )

    assert pinned.result is None
    assert pinned.refusal == "facet_not_available"

    request = _mint_request(world)
    assert _mint(world, request).selections["provenance"].ref == ALT
    with pytest.raises(MaterialisationError):
        _mint(
            world,
            dataclasses.replace(
                request, facet_bindings={"provenance": FacetBinding(state="bound", ref=IDS)}
            ),
        )


@pytest.mark.parametrize("two_dataset_world", _ROLE_ROUTES, indirect=True)
def test_every_offered_source_is_one_the_mint_serves(two_dataset_world: World) -> None:
    """The first answer and its resubmit offer `Alt` as the only source, and a mint pinning
    each offered source serves it. `Ids`, which the mint refuses, is not offered."""
    world = two_dataset_world
    with world_module.door_loop():
        first = world.first([ALAND, BLAND], None, preview=True, unit_text=None)
        (spec,) = first.specs
        again = world.resubmit(spec, preview=True)

    request = _mint_request(world)
    for answer in (first, again):
        assert answer.result is not None and answer.result.facet_probe is not None
        options, warning = _maybe_build_facet_options(
            answer.result.facet_probe,
            graph=world.graph(),
            coverage=world.index,
            include_option_coverage=True,
            deadline=None,
        )
        assert warning is None
        offered = [
            option.value.ref.dcid
            for facet in options
            if facet.axis == "provenance"
            for option in facet.options
            if option.value.ref is not None
        ]
        assert offered == [ALT]
        for provenance in offered:
            pinned = dataclasses.replace(
                request,
                facet_bindings={"provenance": FacetBinding(state="bound", ref=provenance)},
            )
            assert _mint(world, pinned).selections["provenance"].ref == provenance


@pytest.mark.parametrize("route", ("standard", *_ROLE_ROUTES))
def test_a_member_only_unrecorded_datasets_publish_is_refused_as_the_mint_refuses_it(
    monkeypatch, search_settings, route: str
) -> None:
    """The family elects a dataset the world holds no facet of, and the code index records
    the member under that one alone, so `Alt` and `Ids` publish its observations unrecorded.
    The mint raises `DatasetDisagreementError`, and so does the first answer. Search refuses
    it with the platform's catch-all, the status, code and retry advice the mint's door
    gives it."""
    from datasets_api.doors.mint_errors import _classify_dataset_disagreement
    from dfo_platform import refusal
    from materialise.errors import DatasetDisagreementError
    from search.engine.app import classify_search_exception

    world = _electing(monkeypatch, search_settings, route, provenance=_STALE, recorded=(_STALE,))
    with world_module.door_loop():
        first = world.first([ALAND, BLAND], None, preview=True, unit_text=None)

    assert first.result is None and first.refusal is None
    assert (first.error or "").startswith("DatasetDisagreementError")
    with pytest.raises(DatasetDisagreementError) as minted:
        _mint(world, _mint_request(world, provenance=_STALE, recorded=(_STALE,)))
    searched = classify_search_exception(minted.value) or refusal.classify_default(minted.value)
    door = _classify_dataset_disagreement(minted.value)
    assert (searched.status, searched.code, searched.retryable) == (
        door.status,
        door.code,
        door.retryable,
    )


@pytest.mark.parametrize("route", ("standard", "named_pair", "named_bulk"))
def test_a_window_holding_only_unrecorded_points_is_no_disagreement(
    monkeypatch, search_settings, route: str
) -> None:
    """The family elects `Ids` and the code index records the member under `Ids` alone, so
    `Alt` is unrecorded. In 2022 to 2023 Aland holds `Alt` points only, but `Ids` publishes
    the member in other years, so the mint, whose candidates span every year, serves `Ids`.
    The answer raises no disagreement either, and as `Ids` holds no point in those years it
    answers `no_observations`."""
    from materialise.window import RequestedWindow
    from search.engine.extract import DateRequest
    from search.models import TimeWindow

    world = _electing(monkeypatch, search_settings, route, provenance=IDS, recorded=(IDS,))
    window = DateRequest(window=TimeWindow(start_year=2022, end_year=2023), latest=False)
    with world_module.door_loop():
        first = world.first([ALAND, BLAND], window, preview=True, unit_text=None)

    assert first.error is None and first.refusal is None
    assert first.result is not None
    assert (first.result.status, first.result.no_data_reason) == ("no_data", "no_observations")
    minted = _mint(
        world,
        dataclasses.replace(
            _mint_request(world, provenance=IDS, recorded=(IDS,)),
            window=RequestedWindow(kind="range", start_year=2022, end_year=2023),
        ),
    )
    assert minted.selections["provenance"].ref == IDS
