"""First answers across place scopes, pinned against a stored snapshot.

Each scenario resolves one query part through `resolve_variable` on the offline graph and SDMX
doubles. The scenarios cover first elections with and without a named place, flagged
alternates and mixed menus of subject-only and role-bearing shapes, zero-observation re-elections
that adopt one sibling or offer several,
index-sourced shapes that take no default scope, places over the bound a Spec binds, regions
and directed places.

`test_first_answer_matches_its_snapshot` projects each answer: its status and reason, every
offered spec (`specs`, `flagged_match`, `alternate_match`, `nearest_real`) with its identity
entities and resolved view, the warnings in order, the entities each observation read was
built over, and the graph and SDMX requests in a canonical order, since the walks of a query's
alternates and siblings run beside the elected shape's reads, and the sentence the one-part
response assembled from the answer states, with the part's departures. It compares the
projection with the one stored under the scenario's id in
`tests/fixtures/first_answer_place_scenarios.json`, and checks every sentence that response
carries against the sentence invariants (`_sentence_violations`).

`test_first_answer_keeps_its_serial_depth` runs each scenario on a virtual clock
(`_virtual_clock.VirtualRequests`), where every request takes one tick and no wall clock is
read. A call the live graph client answers without a request (`_FREE_CALLS`) takes none. Its serial depth is the longest chain of requests each starting once the previous one
has ended, so a request moved off a concurrent path or added in series changes it where a
count would not. The depth must
equal the scenario's baseline depth, stored as `baseline_serial_depth` and measured on the
engine that reads each StatVar's default scope from SDMX availability, offers a subject-only
alternate from its recalled members without the mint's member walk, and probes an alternate
at the places the elected shape resolved. Both engines settle the elected shape's members
through the mint's member walk (`RoleClassWalks`) over `_materialise_graph`, so its reads
count on both sides. A scenario may be deeper only by the steps `_SERIAL_STEPS_ADDED` lists
for it, and only with an answer that differs from the baseline's, stored as `baseline_answer`:
its status and a hash of the series it serves (`_answer_record`), or by the steps
`_SERIAL_STEPS_ADDED_FOR_A_SHAPE_THE_BASELINE_DROPS` lists for a shape the baseline leaves
out of the election. A scenario `_SERIAL_STEPS_SAVED` lists is shallower by those steps,
with or without a changed answer. A scenario in no table takes the baseline's depth, so a
read added in series fails it until a table names the step.

With `SEARCH_RECORD_SCENARIOS=1` the snapshot test stores what it observes instead of
comparing. With `SEARCH_RECORD_BASELINE_DEPTH=1`, on a checkout of the baseline engine, the
depth test stores each scenario's baseline depth and answer record. Recording refuses to run
under xdist, so one process writes the file.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import inspect
import json
import logging
import os
import threading
from collections.abc import Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from materialise import place_scope
from search.engine.bind import _UniversalBindOutput
from search.engine.clients import materialise_bridge
from search.engine.conjoin import assemble_region
from search.engine.envelope import now_ms
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import (
    candidate_grounding,
    phase_answer,
    phase_place_materialise,
    resubmit_request,
)
from search.engine.region.observations import fetch as observation_fetch
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.retrieval import bound_values
from search.models import Resolution, Spec
from series_coverage.schema import RoleNames
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._facet_option_builds import wrap_facet_option_builds
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.engine._request_log import _UNRECORDED_METHODS, _plain, _RequestLog
from tests.engine._virtual_clock import VirtualRequests
from tests.fixtures import FakeGraph, make_resolver

_RECORD_ENV = "SEARCH_RECORD_SCENARIOS"
_RECORD_BASELINE_ENV = "SEARCH_RECORD_BASELINE_DEPTH"
_SNAPSHOT = Path(__file__).parent.parent / "fixtures" / "first_answer_place_scenarios.json"
_RECORDED_CLASSES: tuple[type, ...] = (FakeGraph, FakeSdmxClient, FakeMintGraphReader)
# The modules whose thread pools fan requests out inside one executor function.
_POOL_MODULES = (observation_fetch, materialise_bridge, candidate_grounding, resubmit_request)
# The pools a module builds at import, on which requests run beside the resolve.
_EXECUTORS = (
    (phase_place_materialise, "_ALTERNATES_POOL"),
    (phase_place_materialise, "_WALK_POOL"),
)
# The doubles' methods the live graph client answers without a request
# (`dc_graph.GraphClient.default_facet_policy`, `generation_stamp`, `observation_read_groups`,
# `source_key`, `un_availability`), which take no tick of the virtual clock.
_FREE_CALLS = frozenset(
    {
        "default_facet_policy",
        "generation_stamp",
        "observation_read_groups",
        "source_key",
        "un_availability",
    }
)

_KEN, _GHA, _TZA, _NOR, _USA = (
    "country/KEN",
    "country/GHA",
    "country/TZA",
    "country/NOR",
    "country/USA",
)
_AFRICA = "africa"
_EARTH = "Earth"
_SOURCE = "dc/base/ScenarioFixture"

# A subject-only family: the elected count, two qualifier siblings, and an unrelated reading
# the bind step can flag. The unrelated reading's label holds every word of the query, so the
# question does not pick the elected count over it, and a nomination of it is a rival
# (status rule 2(b)).
_MORT = "Count_MortalityEventScenario"
_MORT_EST = "Count_MortalityEventEstimatedScenario"
_MORT_PROJ = "Count_MortalityEventProjectedScenario"
_FAR = "Count_UnrelatedTopicScenario"
# An estimated per-capita reading of the unrelated population. No scenario recalls it. It is
# part of the graph every scenario's member walk reads.
_FAR_PC_EST = "Count_UnrelatedTopicPerCapitaEstimatedScenario"
# A reading of another population whose label repeats the query's words and scores close to
# the elected count.
_LIVESTOCK = "Count_LivestockMortalityScenario"

# Gross national income: two role-bearing IDS series (one by creditor) and three subject-only
# readings. All five are qualifier variants of one another. The IDS series read at a price
# basis off the declared default one, so a question naming none leaves the price basis open
# beside them, and each other reading the bind step names is a rival (status rule 2(a)).
_IDS_QUALIFIER = "CurrentPricesScenario"
_IDS = "Amount_EconomicActivity_GrossNationalIncome_NominalScenario"
_IDS_BC = "Amount_EconomicActivity_GrossNationalIncome_Nominal_ByCreditorScenario"
_REAL = "Amount_EconomicActivity_GrossNationalIncome_RealValueScenario"
_PPP = "Amount_EconomicActivity_GrossNationalIncome_PPPScenario"
_ATLAS = "Amount_EconomicActivity_GrossNationalIncome_AtlasScenario"
_YEARS = [2020, 2021, 2022]
_ROLE_NAMES = {
    _IDS: RoleNames(donor=None, recipient="financingRecipient"),
    _IDS_BC: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    _REAL: RoleNames(donor=None, recipient="financingRecipient"),
}


def _mortality_arcs(label: str, *, pop_type: str, meas_qual: str | None = None) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if meas_qual:
        arcs["measurementQualifier"] = {"nodes": [{"dcid": meas_qual}]}
    return arcs


def _constrained(arcs: dict, prop: str, value: str) -> dict:
    """`arcs` with one more constraint property asserted at `value`."""
    return {
        **arcs,
        "constraintProperties": {"nodes": [{"dcid": prop}]},
        prop: {"nodes": [{"dcid": value}]},
    }


def _income_arcs(qualifier: str, constraints: dict[str, str], roles: tuple[str, ...] = ()) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": qualifier}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }
    if roles:
        arcs["observationProperties"] = {"nodes": [{"dcid": role} for role in roles]}
    return arcs


def _sv(label: str, arcs: dict) -> dict:
    return {"label": label, "type": "StatisticalVariable", "arcs": arcs}


_NODES = {
    _EARTH: {"label": "Earth", "type": "Place"},
    _KEN: {"label": "Kenya", "type": "Country"},
    _GHA: {"label": "Ghana", "type": "Country"},
    _TZA: {"label": "Tanzania", "type": "Country"},
    _NOR: {"label": "Norway", "type": "Country"},
    _USA: {"label": "United States", "type": "Country"},
    _AFRICA: {"label": "Africa", "type": "Continent", "arcs": {}},
    _MORT: _sv(
        "Mortality event count",
        _mortality_arcs("Mortality event count", pop_type="MortalityEvent"),
    ),
    _MORT_EST: _sv(
        "Estimated mortality event count",
        _mortality_arcs(
            "Estimated mortality event count", pop_type="MortalityEvent", meas_qual="Estimated"
        ),
    ),
    _MORT_PROJ: _sv(
        "Projected mortality event count",
        _mortality_arcs(
            "Projected mortality event count", pop_type="MortalityEvent", meas_qual="Projected"
        ),
    ),
    _FAR: _sv(
        "Mortality count of an unrelated topic",
        _mortality_arcs("Mortality count of an unrelated topic", pop_type="UnrelatedThing"),
    ),
    _FAR_PC_EST: _sv(
        "Estimated mortality count of an unrelated topic per capita",
        {
            **_mortality_arcs(
                "Estimated mortality count of an unrelated topic per capita",
                pop_type="UnrelatedThing",
                meas_qual="Estimated",
            ),
            "measurementDenominator": {"nodes": [{"dcid": "Count_Person"}]},
        },
    ),
    _LIVESTOCK: _sv(
        "Mortality count of livestock",
        _mortality_arcs("Mortality count of livestock", pop_type="LivestockMortality"),
    ),
    _IDS: _sv(
        "Gross national income (current US$)",
        _income_arcs(
            _IDS_QUALIFIER,
            {
                "economicActivityType": "GrossNationalIncome",
            },
            ("financingRecipient",),
        ),
    ),
    _IDS_BC: _sv(
        "Gross national income (current US$), by creditor",
        _income_arcs(
            _IDS_QUALIFIER,
            {
                "economicActivityType": "GrossNationalIncome",
            },
            ("financingProvider", "financingRecipient"),
        ),
    ),
    _REAL: _sv(
        "Gross national income (constant US$)",
        _income_arcs(
            "RealValue",
            {
                "economicActivityType": "GrossNationalIncome",
            },
            ("financingRecipient",),
        ),
    ),
    _PPP: _sv(
        "Gross national income based on purchasing power parity",
        _income_arcs("PurchasingPowerParity", {"activitySource": "GrossNationalIncome"}),
    ),
    _ATLAS: _sv(
        "Gross national income, Atlas method",
        _income_arcs("AtlasMethod", {"activitySource": "GrossNationalIncome"}),
    ),
}
_RESOLVE = {
    "Kenya": _KEN,
    "Ghana": _GHA,
    "Tanzania": _TZA,
    "Norway": _NOR,
    "United States": _USA,
}
_PLACES = {"Africa": {"dcid": _AFRICA, "type_of": "Continent"}}
_CHILDREN = {
    _EARTH: {"Country": [_KEN, _GHA, _TZA, _NOR, _USA]},
    _AFRICA: {"Country": [_KEN, _GHA, _TZA]},
}


@pytest.fixture(autouse=True)
def _regions(declare_regions):
    """Search reads Africa's members from the membership policy."""
    declare_regions({_AFRICA: _CHILDREN[_AFRICA]["Country"]})


@pytest.fixture(autouse=True)
def _no_wall_clock_budgets(monkeypatch):
    """A snapshot pins the reads a scenario makes, so no wall-clock budget may cut them.

    The role-class walk and the every-place availability wait each stop reading or fall
    back when their budget runs out, which a stalled worker can reach.
    """
    monkeypatch.setattr(bound_values, "_ROLE_CLASS_WALK_BUDGET_S", 1e6)
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 1e6)


def _facets(*, count: int = 5, start: str = "2015", end: str = "2022") -> list[dict]:
    return [{"earliestDate": start, "latestDate": end, "obsCount": count, "provenanceId": _SOURCE}]


def _at(stat_var: str, *places: str, count: int = 5) -> dict[str, list[dict]]:
    """The facet fixture rows giving `stat_var` data at each of `places`."""
    return {f"{stat_var}|{place}": _facets(count=count) for place in places}


@dataclass(frozen=True)
class _Scenario:
    """One query part and the graph it resolves against."""

    query: str
    entities: tuple[str, ...]
    svs: tuple[str, ...]
    scores: tuple[float, ...]
    obs: Mapping[str, list[dict]] = field(default_factory=dict)
    chosen: int = 0
    also_plausible: tuple[int, ...] = ()
    # Role coverage rows: (variable, facet_id, donor, recipient, years).
    coverage_rows: tuple[tuple[str, str, str, str, list[int]], ...] = ()
    # Read each place's direction from the query's `from` and `to`.
    directed: bool = False
    # The bound on the places a Spec binds, when the scenario lowers it.
    max_child_places: int | None = None
    include_observations: bool = False
    include_option_coverage: bool = False
    stream_slot_options: bool = False
    # StatVars whose shapes the code index is reported to source, so those shapes can take no
    # default scope (`_index_sourced`).
    index_sourced: tuple[str, ...] = ()
    date_request: DateRequest | None = None


class _Bind:
    """Elects `chosen` and flags `also_plausible`."""

    def __init__(self, chosen: int, also_plausible: tuple[int, ...]) -> None:
        self._chosen = chosen
        self._also_plausible = list(also_plausible)

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=self._chosen,
            bindings=[],
            also_plausible=self._also_plausible,
            ask=None,
        ), None


_MORTALITY = "mortality count"
_GNI = "gross national income"
_SIBLINGS = (_MORT, _MORT_EST, _MORT_PROJ)
_SIBLING_SCORES = (0.90, 0.85, 0.82)

_SCENARIOS: dict[str, _Scenario] = {
    # First election, no place named.
    "placeless-definite-countries": _Scenario(
        _MORTALITY, (), (_MORT,), (0.9,), _at(_MORT, _KEN, _GHA)
    ),
    "placeless-definite-one-country": _Scenario(_MORTALITY, (), (_MORT,), (0.9,), _at(_MORT, _TZA)),
    "placeless-definite-earth": _Scenario(_MORTALITY, (), (_MORT,), (0.9,), _at(_MORT, _EARTH)),
    "placeless-definite-earth-and-country": _Scenario(
        _MORTALITY, (), (_MORT,), (0.9,), _at(_MORT, _EARTH, _KEN)
    ),
    "placeless-no-data": _Scenario(_MORTALITY, (), (_MORT,), (0.9,)),
    "placeless-definite-observations": _Scenario(
        _MORTALITY, (), (_MORT,), (0.9,), _at(_MORT, _KEN, _GHA), include_observations=True
    ),
    "placeless-flagged-alternate-elsewhere": _Scenario(
        _MORTALITY,
        (),
        (_MORT, _FAR),
        (0.90, 0.84),
        {**_at(_MORT, _KEN), **_at(_FAR, _TZA)},
        also_plausible=(1,),
    ),
    "placeless-flagged-alternate-elsewhere-option-coverage": _Scenario(
        _MORTALITY,
        (),
        (_MORT, _FAR),
        (0.90, 0.84),
        {**_at(_MORT, _KEN), **_at(_FAR, _TZA)},
        also_plausible=(1,),
        include_option_coverage=True,
    ),
    "placeless-flagged-alternate-empty": _Scenario(
        _MORTALITY, (), (_MORT, _FAR), (0.90, 0.84), _at(_MORT, _KEN), also_plausible=(1,)
    ),
    "placeless-worldwide-elected-country-alternate": _Scenario(
        _MORTALITY,
        (),
        (_MORT, _FAR),
        (0.90, 0.84),
        {**_at(_MORT, _EARTH), **_at(_FAR, _KEN)},
        also_plausible=(1,),
    ),
    "placeless-empty-elected-flagged-alternate-data": _Scenario(
        _MORTALITY, (), (_MORT, _FAR), (0.90, 0.84), _at(_FAR, _KEN), also_plausible=(1,)
    ),
    # The elected count holds data beside a qualifier sibling, no place named. Re-election
    # does not run. The sibling's StatVar rides the elected count's presence read.
    "placeless-definite-beside-sibling": _Scenario(
        _MORTALITY, (), _SIBLINGS[:2], _SIBLING_SCORES[:2], _at(_MORT, _KEN)
    ),
    # No place named, beside a subject-only shape the code index sources, which can take no
    # default scope. The answer does not rest on that shape: it is left out of every presence
    # read and out of the offer, and the request answers. In the first three the baseline
    # gives the same answer: a sibling of an elected count with data, a widened search's
    # reading when the tied reading holds data, and a flagged alternate with no data.
    "placeless-definite-beside-index-sourced-sibling": _Scenario(
        _MORTALITY,
        (),
        _SIBLINGS[:2],
        _SIBLING_SCORES[:2],
        _at(_MORT, _KEN),
        index_sourced=(_MORT_EST,),
    ),
    "placeless-index-sourced-flagged-alternate-empty": _Scenario(
        _MORTALITY,
        (),
        (_MORT, _FAR),
        (0.90, 0.84),
        _at(_MORT, _KEN),
        also_plausible=(1,),
        index_sourced=(_FAR,),
    ),
    # The dropped shape holds data the baseline offers or adopts: a flagged alternate with
    # data elsewhere, and the one sibling of an elected count with none.
    "placeless-index-sourced-flagged-alternate-elsewhere": _Scenario(
        _MORTALITY,
        (),
        (_MORT, _FAR),
        (0.90, 0.84),
        {**_at(_MORT, _KEN), **_at(_FAR, _TZA)},
        also_plausible=(1,),
        index_sourced=(_FAR,),
    ),
    "placeless-reelect-index-sourced-sibling": _Scenario(
        _MORTALITY,
        (),
        _SIBLINGS[:2],
        _SIBLING_SCORES[:2],
        _at(_MORT_EST, _KEN, _GHA),
        index_sourced=(_MORT_EST,),
    ),
    # Zero-observation re-election, no place named.
    "placeless-reelect-adopt": _Scenario(
        _MORTALITY,
        (),
        _SIBLINGS[:2],
        _SIBLING_SCORES[:2],
        _at(_MORT_EST, _KEN, _GHA),
    ),
    "placeless-reelect-adopt-observations": _Scenario(
        _MORTALITY,
        (),
        _SIBLINGS[:2],
        _SIBLING_SCORES[:2],
        _at(_MORT_EST, _KEN, _GHA),
        include_observations=True,
    ),
    "placeless-reelect-adopt-earth": _Scenario(
        _MORTALITY,
        (),
        _SIBLINGS[:2],
        _SIBLING_SCORES[:2],
        _at(_MORT_EST, _EARTH),
    ),
    "placeless-reelect-no-data": _Scenario(_MORTALITY, (), _SIBLINGS[:2], _SIBLING_SCORES[:2]),
    "placeless-reelect-two-siblings-apart": _Scenario(
        _MORTALITY,
        (),
        _SIBLINGS,
        _SIBLING_SCORES,
        {**_at(_MORT_EST, _KEN, _GHA), **_at(_MORT_PROJ, _TZA)},
    ),
    "placeless-reelect-two-siblings-apart-option-coverage": _Scenario(
        _MORTALITY,
        (),
        _SIBLINGS,
        _SIBLING_SCORES,
        {**_at(_MORT_EST, _KEN, _GHA), **_at(_MORT_PROJ, _TZA)},
        include_option_coverage=True,
    ),
    "placeless-reelect-two-siblings-together": _Scenario(
        _MORTALITY,
        (),
        _SIBLINGS,
        _SIBLING_SCORES,
        {**_at(_MORT_EST, _KEN), **_at(_MORT_PROJ, _KEN)},
    ),
    "placeless-reelect-one-of-two-siblings": _Scenario(
        _MORTALITY, (), _SIBLINGS, _SIBLING_SCORES, _at(_MORT_EST, _KEN)
    ),
    "placeless-reelect-earth-sibling-and-country-sibling": _Scenario(
        _MORTALITY,
        (),
        _SIBLINGS,
        _SIBLING_SCORES,
        {**_at(_MORT_EST, _EARTH), **_at(_MORT_PROJ, _KEN)},
    ),
    "placeless-reelect-flagged-alternate-elsewhere": _Scenario(
        _MORTALITY,
        (),
        (_MORT, _MORT_EST, _FAR),
        (0.90, 0.85, 0.84),
        {**_at(_MORT_EST, _KEN, _GHA), **_at(_FAR, _TZA)},
        also_plausible=(2,),
    ),
    "placeless-reelect-flagged-alternate-overlapping": _Scenario(
        _MORTALITY,
        (),
        (_MORT, _MORT_EST, _FAR),
        (0.90, 0.85, 0.84),
        {**_at(_MORT_EST, _KEN, _GHA), **_at(_FAR, _KEN, _TZA)},
        also_plausible=(2,),
    ),
    "placeless-reelect-flagged-alternate-empty": _Scenario(
        _MORTALITY,
        (),
        (_MORT, _MORT_EST, _FAR),
        (0.90, 0.85, 0.84),
        _at(_MORT_EST, _KEN, _GHA),
        also_plausible=(2,),
    ),
    "placeless-reelect-menu-beside-flagged-alternate": _Scenario(
        _MORTALITY,
        (),
        (*_SIBLINGS, _FAR),
        (*_SIBLING_SCORES, 0.80),
        {**_at(_MORT_EST, _KEN), **_at(_MORT_PROJ, _TZA), **_at(_FAR, _GHA)},
        also_plausible=(3,),
    ),
    "placeless-reelect-menu-beside-flagged-alternate-together": _Scenario(
        _MORTALITY,
        (),
        (*_SIBLINGS, _FAR),
        (*_SIBLING_SCORES, 0.80),
        {**_at(_MORT_EST, _KEN), **_at(_MORT_PROJ, _KEN), **_at(_FAR, _KEN)},
        also_plausible=(3,),
    ),
    # No place named, role-bearing and subject-only readings together.
    "placeless-role-bearing-definite": _Scenario(
        _GNI, (), (_IDS,), (0.85,), coverage_rows=((_IDS, "f_ids", "", _USA, _YEARS),)
    ),
    "placeless-role-bearing-elected-subject-alternate": _Scenario(
        _GNI,
        (),
        (_IDS, _PPP),
        (0.85, 0.84),
        _at(_PPP, _GHA),
        also_plausible=(1,),
        coverage_rows=((_IDS, "f_ids", "", _USA, _YEARS),),
    ),
    "placeless-role-bearing-elected-subject-alternate-at-recipient": _Scenario(
        _GNI,
        (),
        (_IDS, _PPP),
        (0.85, 0.84),
        _at(_PPP, _USA),
        also_plausible=(1,),
        coverage_rows=((_IDS, "f_ids", "", _USA, _YEARS),),
    ),
    "placeless-subject-elected-role-bearing-alternate": _Scenario(
        _GNI,
        (),
        (_PPP, _IDS),
        (0.85, 0.84),
        _at(_PPP, _GHA),
        also_plausible=(1,),
        coverage_rows=((_IDS, "f_ids", "", _USA, _YEARS),),
    ),
    "placeless-reelect-subject-and-role-bearing-siblings": _Scenario(
        _GNI,
        (),
        (_ATLAS, _PPP, _IDS),
        (0.85, 0.84, 0.83),
        _at(_PPP, _GHA),
        coverage_rows=((_IDS, "f_ids", "", _USA, _YEARS),),
    ),
    "placeless-reelect-role-bearing-sibling-widens-scope": _Scenario(
        _GNI,
        (),
        (_ATLAS, _PPP, _IDS),
        (0.85, 0.84, 0.83),
        {**_at(_PPP, _GHA), **_at(_IDS, _NOR)},
        coverage_rows=((_IDS, "f_ids", "", _NOR, _YEARS),),
    ),
    "placeless-reelect-role-bearing-sibling-only": _Scenario(
        _GNI,
        (),
        (_ATLAS, _IDS),
        (0.85, 0.84),
        coverage_rows=((_IDS, "f_ids", "", _USA, _YEARS),),
    ),
    "placeless-reelect-subject-sibling-beside-empty-role-bearing": _Scenario(
        _GNI,
        (),
        (_ATLAS, _PPP, _IDS),
        (0.85, 0.84, 0.83),
        _at(_PPP, _GHA, _NOR),
    ),
    # No place named, a role-bearing elected shape beside subject-only qualifier siblings and
    # no flagged alternate. The elected shape discovers its scope, so re-election does not
    # follow it and the siblings take no presence read.
    "placeless-role-bearing-elected-subject-siblings": _Scenario(
        _GNI, (), (_IDS, _PPP, _ATLAS), (0.85, 0.84, 0.83), _at(_PPP, _GHA)
    ),
    "placeless-role-bearing-elected-with-data-subject-siblings": _Scenario(
        _GNI,
        (),
        (_IDS, _PPP, _ATLAS),
        (0.85, 0.84, 0.83),
        _at(_PPP, _GHA),
        coverage_rows=((_IDS, "f_ids", "", _USA, _YEARS),),
    ),
    # One named place.
    "named-definite": _Scenario(
        "mortality count in Kenya", ("Kenya",), (_MORT,), (0.9,), _at(_MORT, _KEN)
    ),
    "named-no-data": _Scenario("mortality count in Kenya", ("Kenya",), (_MORT,), (0.9,)),
    "named-reelect-adopt": _Scenario(
        "mortality count in Kenya",
        ("Kenya",),
        _SIBLINGS[:2],
        _SIBLING_SCORES[:2],
        _at(_MORT_EST, _KEN),
    ),
    "named-reelect-adopt-observations": _Scenario(
        "mortality count in Kenya",
        ("Kenya",),
        _SIBLINGS[:2],
        _SIBLING_SCORES[:2],
        _at(_MORT_EST, _KEN),
        include_observations=True,
    ),
    "named-reelect-menu": _Scenario(
        "mortality count in Kenya",
        ("Kenya",),
        _SIBLINGS,
        _SIBLING_SCORES,
        {**_at(_MORT_EST, _KEN), **_at(_MORT_PROJ, _KEN)},
    ),
    "named-flagged-alternate": _Scenario(
        "mortality count in Kenya",
        ("Kenya",),
        (_MORT, _FAR),
        (0.90, 0.84),
        {**_at(_MORT, _KEN), **_at(_FAR, _KEN)},
        also_plausible=(1,),
    ),
    "named-two-places": _Scenario(
        "mortality count in Kenya and Ghana",
        ("Kenya", "Ghana"),
        (_MORT,),
        (0.9,),
        _at(_MORT, _KEN, _GHA),
    ),
    "named-region": _Scenario(
        "mortality count in Africa", ("Africa",), (_MORT,), (0.9,), _at(_MORT, _KEN, _TZA)
    ),
    "named-region-reelect": _Scenario(
        "mortality count in Africa",
        ("Africa",),
        _SIBLINGS[:2],
        _SIBLING_SCORES[:2],
        _at(_MORT_EST, _GHA),
    ),
    "named-region-and-country": _Scenario(
        "mortality count in Africa and Norway",
        ("Africa", "Norway"),
        (_MORT,),
        (0.9,),
        _at(_MORT, _KEN, _NOR),
    ),
    # Two named places over a bound of one, so the elected shape refuses and reads no place.
    "named-over-bound": _Scenario(
        "mortality count in Kenya and Ghana",
        ("Kenya", "Ghana"),
        (_MORT,),
        (0.9,),
        _at(_MORT, _KEN),
        max_child_places=1,
    ),
    "named-directed-reelect": _Scenario(
        "mortality count from Kenya",
        ("Kenya",),
        _SIBLINGS[:2],
        _SIBLING_SCORES[:2],
        _at(_MORT_EST, _KEN),
        directed=True,
    ),
    # Named places, role-bearing and subject-only readings together.
    "income-role-bearing-elected-subject-alternate": _Scenario(
        "gross national income of Ghana",
        ("Ghana",),
        (_IDS, _PPP),
        (0.75, 0.74),
        _at(_PPP, _GHA),
        also_plausible=(1,),
        coverage_rows=((_IDS, "f_ids", "", _GHA, _YEARS),),
    ),
    "income-subject-elected-role-bearing-alternate": _Scenario(
        "gross national income of Ghana",
        ("Ghana",),
        (_PPP, _IDS),
        (0.75, 0.74),
        _at(_PPP, _GHA),
        also_plausible=(1,),
        coverage_rows=((_IDS, "f_ids", "", _GHA, _YEARS),),
    ),
    "income-reelect-subject-from-role-bearing": _Scenario(
        "gross national income of Ghana",
        ("Ghana",),
        (_IDS, _PPP),
        (0.85, 0.84),
        _at(_PPP, _GHA),
        coverage_rows=((_IDS, "f_ids", "", _NOR, _YEARS),),
    ),
    "income-reelect-directed-two-places": _Scenario(
        "gross national income of Ghana from Norway",
        ("Ghana", "Norway"),
        (_IDS, _IDS_BC, _PPP),
        (0.85, 0.85, 0.84),
        {**_at(_PPP, _GHA, count=21), **_at(_PPP, _NOR, count=35)},
        coverage_rows=(
            (_IDS, "f_ids", "", _USA, _YEARS),
            (_IDS_BC, "f_ids", _USA, _USA, _YEARS),
        ),
        directed=True,
    ),
    "income-reelect-siblings-in-separate-scopes": _Scenario(
        "gross national income of Ghana from Norway",
        ("Ghana", "Norway"),
        (_IDS, _IDS_BC, _PPP, _REAL, _ATLAS),
        (0.85, 0.85, 0.84, 0.83, 0.82),
        {**_at(_PPP, _GHA), **_at(_ATLAS, _GHA)},
        coverage_rows=(
            (_IDS, "f_ids", "", _USA, _YEARS),
            (_IDS_BC, "f_ids", _USA, _USA, _YEARS),
            (_REAL, "f_real", "", _GHA, _YEARS),
        ),
        directed=True,
    ),
    "income-alternate-role-cannot-take-the-place": _Scenario(
        "gross national income from Norway",
        ("Norway",),
        (_PPP, _IDS),
        (0.75, 0.74),
        _at(_PPP, _NOR),
        also_plausible=(1,),
        coverage_rows=((_IDS, "f_ids", "", _GHA, _YEARS),),
        directed=True,
    ),
    "income-role-bearing-elected-subject-alternate-at-both": _Scenario(
        "gross national income of Ghana from Norway",
        ("Ghana", "Norway"),
        (_IDS, _IDS_BC, _PPP),
        (0.85, 0.85, 0.84),
        _at(_PPP, _GHA, _NOR),
        also_plausible=(1,),
        coverage_rows=(
            (_IDS, "f_ids", "", _GHA, _YEARS),
            (_IDS_BC, "f_ids", _NOR, _GHA, _YEARS),
        ),
        directed=True,
    ),
    "income-discovered-recipients-subject-alternate-at-origin": _Scenario(
        "gross national income from Norway",
        ("Norway",),
        (_IDS_BC, _PPP),
        (0.85, 0.84),
        _at(_PPP, _NOR),
        also_plausible=(1,),
        coverage_rows=(
            (_IDS_BC, "f_ids", _NOR, _GHA, _YEARS),
            (_IDS_BC, "f_ids", _NOR, _USA, _YEARS),
        ),
        directed=True,
    ),
    "income-discovered-recipients-subject-alternate-at-recipient": _Scenario(
        "gross national income from Norway",
        ("Norway",),
        (_IDS_BC, _PPP),
        (0.85, 0.84),
        _at(_PPP, _GHA),
        also_plausible=(1,),
        coverage_rows=(
            (_IDS_BC, "f_ids", _NOR, _GHA, _YEARS),
            (_IDS_BC, "f_ids", _NOR, _USA, _YEARS),
        ),
        directed=True,
    ),
}


class _ObservationEntities:
    """Records the entities every observation read is built over, in call order."""

    def __init__(self) -> None:
        self.reads: list[Any] = []
        self._lock = threading.Lock()

    def wrap(self, module: Any) -> Callable[..., Any]:
        real = module._observation_entities
        name = module.__name__.rsplit(".", 1)[-1]

        @functools.wraps(real)
        def recorded(shape, **kwargs):
            entities, warning = real(shape, **kwargs)
            with self._lock:
                self.reads.append([name, _plain(entities)])
            return entities, warning

        return recorded


@contextlib.contextmanager
def _index_sourced(stat_vars: tuple[str, ...]) -> Iterator[None]:
    """Report every shape holding one of `stat_vars` as index-sourced where the engine
    decides which shapes take a default scope. The baseline engine makes no such decision
    and probes these shapes as any other."""
    try:
        from search.engine.region import place_context
    except ImportError:
        place_context = None
    if not stat_vars or place_context is None:
        yield
        return
    real = place_context.shape_member_facts

    def member_facts(shape, bound_values):
        facts, sourced = real(shape, bound_values)
        return facts, sourced or any(sv in (shape.sv_arc_facts or {}) for sv in stat_vars)

    with mock.patch.object(place_context, "shape_member_facts", member_facts):
        yield


def _resolve(
    scenario: _Scenario,
    run: Callable[[Coroutine[Any, Any, RegionResult]], RegionResult] = asyncio.run,
) -> RegionResult:
    with _index_sourced(scenario.index_sourced):
        return _resolve_scenario(scenario, run)


def _materialise_graph(graph: FakeGraph, scenario: _Scenario) -> FakeMintGraphReader:
    """What materialise reads for a scenario: `graph`'s nodes, facets and places. The role
    class election walks the mint's member enumeration through it (`RoleClassWalks`), as the
    request path walks the live graph. Its facets are read off `graph` unrecorded."""
    facets_of = inspect.unwrap(FakeGraph.observation_facets)
    pairs = [tuple(key.split("|", 1)) for key in scenario.obs]
    return FakeMintGraphReader(
        facets={
            (stat_var, place): facets_of(graph, stat_var=stat_var, entity=place)
            for stat_var, place in pairs
        },
        labels={dcid: node["label"] for dcid, node in _NODES.items()},
        arcs={
            dcid: {
                prop: tuple(n["dcid"] for n in arc["nodes"] if "dcid" in n)
                for prop, arc in node.get("arcs", {}).items()
            }
            for dcid, node in _NODES.items()
        },
        child_places={
            (parent, child_type): children
            for parent, by_type in _CHILDREN.items()
            for child_type, children in by_type.items()
        },
    )


def _resolve_scenario(
    scenario: _Scenario,
    run: Callable[[Coroutine[Any, Any, RegionResult]], RegionResult],
) -> RegionResult:
    graph = FakeGraph(
        nodes=_NODES,
        obs=dict(scenario.obs),
        detect={
            scenario.query: {"svs": list(scenario.svs), "cosine_scores": list(scenario.scores)}
        },
        resolve=_RESOLVE,
        places=_PLACES,
        children=_CHILDREN,
    )
    coverage = FakeCoverageService(
        build_coverage_index(list(scenario.coverage_rows), role_names=_ROLE_NAMES)
    )
    entities = list(scenario.entities)
    return run(
        resolve_variable(
            scenario.query,
            entities=entities,
            date_request=scenario.date_request,
            detect_query=scenario.query,
            role_query=scenario.query,
            entity_roles=(
                door_roles_for_test_query(scenario.query, entities) if scenario.directed else None
            ),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_Bind(scenario.chosen, scenario.also_plausible),
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            base_timing={},
            coverage=coverage.snapshot(),
            options=EngineOptions(
                include_observations=scenario.include_observations,
                include_option_coverage=scenario.include_option_coverage,
                stream_slot_options=scenario.stream_slot_options,
                materialise_graph=_materialise_graph(graph, scenario),
            ),
        )
    )


def _with_deadline_marked(arguments: dict[str, Any]) -> dict[str, Any]:
    """A request's arguments with an absolute `deadline` shown as `"set"`: the role-class walk
    bounds its reads by a monotonic time, which differs on every run."""
    if arguments.get("deadline") is None:
        return arguments
    return {**arguments, "deadline": "set"}


def _without_timings(value: Any) -> Any:
    """`value` without the step timings (`ms`) a trace records."""
    if isinstance(value, dict):
        return {key: _without_timings(item) for key, item in value.items() if key != "ms"}
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


def _spec(spec: Spec) -> dict[str, Any]:
    resolved = _without_timings(
        spec.resolved.model_dump(mode="json", exclude={"shape", "coverage_stamp"})
    )
    return {
        "spec_id": spec.spec_id,
        "identity_entities": spec.definition.identity.entities,
        "resolved": resolved,
    }


def _projection(result: RegionResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "no_data_reason": result.no_data_reason,
        "unsupported": None if result.unsupported is None else result.unsupported.code,
        "specs": [_spec(spec) for spec in result.specs],
        "flagged_match": [_spec(spec) for spec in result.flagged_match or ()],
        "alternate_match": None
        if result.alternate_match is None
        else _spec(result.alternate_match),
        "nearest_real": [spec.spec_id for spec in result.nearest_real or ()],
        "elected_stat_vars": [ref.ref for ref in result.elected_stat_vars or ()],
        "no_data_role_axis": result.no_data_role_axis,
        "no_data_bound_role_dcids": {
            role: list(dcids) for role, dcids in result.no_data_bound_role_dcids.items()
        },
        "warnings": [[w.code, w.severity, w.message, w.spec_id] for w in result.warnings],
    }


def _response(result: RegionResult, query: str) -> Resolution:
    """The one-part response the raw-text door assembles from `result`, with sentences."""
    return assemble_region(
        result,
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
        query=query,
    )


def _sentence(response: Resolution) -> dict[str, Any]:
    """The response's composed sentence and its one part's departures."""
    [part] = response.parts
    return {
        "rendered_sentence": response.rendered_sentence,
        "departures": [[d.code, d.binding_ref, d.asked, d.shown, d.text] for d in part.departures],
    }


# Wording the sentence no longer writes: the coverage tail, the unnamed-year caveat, a
# singular role count read as a plural, and the unit stated twice.
_RETIRED_WORDING = (
    "observations across",
    "No year was named",
    "1 donors",
    "1 providers",
    "1 recipients",
    "(current US$; US dollar)",
)


def _sentence_violations(response: Resolution) -> list[str]:
    """Each way a sentence in `response` breaks the sentence invariants: retired wording, a
    plan whose plain join differs from its `rendered_sentence`, a token with no segment or
    of kind "note", a departure token on a spec's own plan, and a role token naming more than
    three entities. Every sentence the response carries is checked: the composed one, each
    part's own, and each offered spec's."""
    violations: list[str] = []
    labels = {dcid: node["label"] for dcid, node in _NODES.items()}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(value, dict):
            return
        plan = value.get("sentence_plan")
        if plan is not None:
            text = value.get("rendered_sentence")
            if "".join(token["text"] for token in plan) != text:
                violations.append(f"{path}: plain join differs from {text!r}")
            violations.extend(f"{path}: retired {w!r}" for w in _RETIRED_WORDING if w in text)
            for token in plan:
                if token.get("segment") not in ("heading", "byline", "departure"):
                    violations.append(f"{path}: token {token['text']!r} has no segment")
                if token["kind"] == "note":
                    violations.append(f"{path}: note token {token['text']!r}")
                if "stat_vars" in value and token.get("segment") == "departure":
                    violations.append(f"{path}: spec plan holds departure {token['text']!r}")
                named = [
                    d for d in token.get("entity_refs", []) if labels.get(d, d) in token["text"]
                ]
                if token.get("role") and len(named) > 3:
                    violations.append(f"{path}: role token lists {len(named)} entities")
        for key, item in value.items():
            visit(item, f"{path}/{key}")

    visit(response.model_dump(mode="json"), "")
    return violations


def _offered(result: RegionResult) -> list[Any]:
    """Every spec the answer offers, in order: its specs, flagged matches, alternate match
    and nearest-real suggestions."""
    return [
        *result.specs,
        *(result.flagged_match or ()),
        *(() if result.alternate_match is None else (result.alternate_match,)),
        *(result.nearest_real or ()),
    ]


def _answer_record(result: RegionResult) -> dict[str, str]:
    """The answer's status and a hash of the series it serves: for each offered spec its
    StatVars, the places it binds and is read over, and the coverage it states."""
    served = []
    for spec in _offered(result):
        resolved = spec.resolved.model_dump(mode="json")
        served.append(
            {
                "stat_vars": [stat_var.ref.ref for stat_var in spec.resolved.members],
                "identity_entities": _plain(spec.definition.identity.entities),
                "entities": resolved["entities"],
                "scope": resolved["scope"],
                "coverage": resolved["coverage"],
                "available_at_place": resolved["available_at_place"],
            }
        )
    digest = hashlib.sha256(json.dumps(served, sort_keys=True).encode()).hexdigest()
    return {"status": result.status, "served_series": digest}


def _first_difference(expected: Any, observed: Any, path: str = "") -> str | None:
    """The path of the first value `observed` holds differently from `expected`, with both
    values, or `None` when they are equal."""
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in [*expected, *(k for k in observed if k not in expected)]:
            if key not in expected or key not in observed:
                return f"{path}/{key}: expected {expected.get(key)!r}, got {observed.get(key)!r}"
            if (
                found := _first_difference(expected[key], observed[key], f"{path}/{key}")
            ) is not None:
                return found
        return None
    if isinstance(expected, list) and isinstance(observed, list):
        for index, (want, got) in enumerate(zip(expected, observed, strict=False)):
            if (found := _first_difference(want, got, f"{path}[{index}]")) is not None:
                return found
        if len(expected) != len(observed):
            return f"{path}: expected {len(expected)} items, got {len(observed)}"
        return None
    return None if expected == observed else f"{path}: expected {expected!r}, got {observed!r}"


def _patch_settings(scenario: _Scenario, search_settings: Callable[..., Any]) -> None:
    if scenario.max_child_places is not None:
        search_settings(max_child_places=scenario.max_child_places)


def _recording(env: str = _RECORD_ENV) -> bool:
    if os.environ.get(env) != "1":
        return False
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail(f"{env}=1 records scenarios only without xdist (-p no:xdist)")
    return True


def _stored(scenario_id: str, key: str, env: str = _RECORD_ENV) -> Any:
    snapshot = json.loads(_SNAPSHOT.read_text()) if _SNAPSHOT.exists() else {}
    if key not in snapshot.get(scenario_id, {}):
        pytest.fail(f"no stored {key} for {scenario_id} in {_SNAPSHOT}; record with {env}=1")
    return snapshot[scenario_id][key]


def _store(scenario_id: str, key: str, value: Any) -> None:
    snapshot = json.loads(_SNAPSHOT.read_text()) if _SNAPSHOT.exists() else {}
    snapshot.setdefault(scenario_id, {})[key] = value
    _SNAPSHOT.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("scenario_id", list(_SCENARIOS))
def test_first_answer_matches_its_snapshot(scenario_id, monkeypatch, search_settings) -> None:
    """The answer, its warnings in order, the entities its observation reads are built over,
    the graph and SDMX requests in canonical order, and the sentence the assembled response
    states with its part's departures equal the stored projection. Every sentence the
    response carries keeps the sentence invariants (`_sentence_violations`)."""
    scenario = _SCENARIOS[scenario_id]
    _patch_settings(scenario, search_settings)
    log = _RequestLog()
    for owner in _RECORDED_CLASSES:
        for name, method in list(vars(owner).items()):
            if (
                not name.startswith("_")
                and name not in _UNRECORDED_METHODS
                and inspect.isfunction(method)
            ):
                monkeypatch.setattr(owner, name, log.wrap(owner, name, method))
    reads = _ObservationEntities()
    for module in (phase_answer, candidate_grounding, phase_place_materialise):
        monkeypatch.setattr(module, "_observation_entities", reads.wrap(module))

    result = _resolve(scenario)
    answer = _projection(result)
    calls = [[name, _with_deadline_marked(arguments)] for name, arguments in log.calls]
    response = _response(result, scenario.query)
    assert _sentence_violations(response) == []
    observed = json.loads(
        json.dumps(
            {
                **answer,
                "sentence": _sentence(response),
                "observation_entities": reads.reads,
                "requests": sorted(calls, key=json.dumps),
            }
        )
    )

    if _recording():
        _store(scenario_id, "answer", observed)
        return
    expected = _stored(scenario_id, "answer")
    difference = _first_difference(expected, observed)
    assert difference is None, difference


# The serial steps a scenario takes beyond its baseline depth, each with the reason its
# group gives. Every scenario here gives an answer the baseline does not.
_SERIAL_STEPS_ADDED: dict[str, int] = {}

# The serial steps a scenario takes beyond its baseline depth while giving the baseline's
# status, each for a shape the baseline leaves out of the election.
_SERIAL_STEPS_ADDED_FOR_A_SHAPE_THE_BASELINE_DROPS: dict[str, int] = {}

# The serial steps a scenario saves against its baseline depth, each with the reason its
# group gives. A saving needs no changed answer.
_SERIAL_STEPS_SAVED: dict[str, int] = {
    # The baseline clamps the probe union of two named places to one place and reads it.
    # Here the two places exceed the bound of one, so the answer refuses and reads no place.
    "named-over-bound": 6,
    # A sibling is walked before it is materialised, so an adopted sibling is materialised
    # once, where the baseline materialises it, walks it, and materialises it again.
    "named-directed-reelect": 5,
    "named-reelect-adopt": 5,
    "named-reelect-adopt-observations": 5,
    # As above, and the default scopes of the elected shape and its siblings come from one
    # presence read beside the elected walk, where the baseline reads availability for each
    # shape's StatVars in turn. A candidate answer also grounds its candidates side by side.
    "placeless-reelect-adopt": 7,
    "placeless-reelect-adopt-earth": 7,
    # Its scope read serves no place, so it reads no labels and saves one more step.
    "placeless-reelect-adopt-observations": 8,
    "placeless-reelect-flagged-alternate-elsewhere": 10,
    "placeless-reelect-flagged-alternate-empty": 8,
    "placeless-reelect-flagged-alternate-overlapping": 10,
    "placeless-reelect-one-of-two-siblings": 7,
    "placeless-reelect-subject-sibling-beside-empty-role-bearing": 7,
    # Re-election offers its siblings beside the elected shape and candidate admission takes
    # each as re-election materialised it, where the baseline materialises each a second
    # time, and the candidates are grounded side by side. Named or not, a sibling's facet
    # reads are made once.
    "named-reelect-menu": 6,
    "placeless-reelect-earth-sibling-and-country-sibling": 9,
    "placeless-reelect-two-siblings-apart": 9,
    "placeless-reelect-two-siblings-apart-option-coverage": 9,
    "placeless-reelect-two-siblings-together": 6,
    # With no place named, the default scope is one listing of Earth's countries beside the
    # elected walk and one presence read, where the baseline reads availability for each
    # StatVar and checks the places it lists, shape by shape. A candidate answer also
    # grounds its candidates side by side.
    "placeless-definite-beside-sibling": 1,
    "placeless-definite-countries": 1,
    "placeless-definite-earth": 1,
    "placeless-definite-earth-and-country": 1,
    # Its scope read serves no place, so it reads no labels and saves one more step.
    "placeless-definite-observations": 2,
    "placeless-definite-one-country": 1,
    "placeless-empty-elected-flagged-alternate-data": 2,
    "placeless-flagged-alternate-elsewhere": 4,
    "placeless-flagged-alternate-elsewhere-option-coverage": 4,
    "placeless-flagged-alternate-empty": 2,
    "placeless-reelect-no-data": 2,
    "placeless-worldwide-elected-country-alternate": 4,
    # A StatVar whose availability lists no place takes the empty scope without the check of
    # listed places the baseline makes after each listing.
    "placeless-no-data": 1,
    # An index-sourced alternate or sibling is dropped before any read made for it.
    "placeless-definite-beside-index-sourced-sibling": 1,
    "placeless-index-sourced-flagged-alternate-empty": 2,
    "placeless-index-sourced-flagged-alternate-elsewhere": 6,
    "placeless-reelect-index-sourced-sibling": 12,
    # An alternate derive reads no data for at the named place has a confirmed absence, so
    # the status rule offers it nothing and admission neither walks nor materialises it.
    "income-discovered-recipients-subject-alternate-at-recipient": 5,
    # Status rule 1 tries no sibling of another role class in the elected shape's place, so
    # no such sibling is walked, materialised or grounded. Where the baseline adopts one or
    # offers several, the part answers no data.
    "income-reelect-directed-two-places": 9,
    "income-reelect-siblings-in-separate-scopes": 13,
    "income-reelect-subject-from-role-bearing": 7,
    "placeless-reelect-role-bearing-sibling-only": 1,
    "placeless-role-bearing-elected-subject-siblings": 5,
    # Several qualifier siblings with data are offered without the elected shape and its
    # flagged alternate, so the alternate is neither materialised nor grounded beside them.
    "placeless-reelect-menu-beside-flagged-alternate": 14,
    "placeless-reelect-menu-beside-flagged-alternate-together": 10,
    # A part naming a set reads its menu's availability at the set's node and members before
    # bind (`availability.read_menu_availability`): one step in series, which leaves four of
    # the five steps an adopted sibling saves above.
    "named-region-reelect": 4,
    # The candidates of an answer are grounded side by side, where the baseline grounds them
    # one after another, and the alternates' walks start beside the elected shape's.
    "income-role-bearing-elected-subject-alternate": 3,
    "named-flagged-alternate": 3,
    # As above, and its role-bearing alternate resolves from coverage, so the facet read that
    # warmed the alternate's cache before admission, one step in series, is not made.
    "income-subject-elected-role-bearing-alternate": 7,
    # An answer that binds several places reads their types in one request and the labels
    # of those types in one more, where the baseline reads each place's type and its label
    # in turn, which saves two steps for a second place. Three of these also save the steps their
    # groups above give: an adopted sibling's (5), re-election's offered siblings' (8) and
    # candidates grounded side by side (5).
    "named-two-places": 2,
    # It also asks role admission about the region's node and member types in one batch,
    # where the baseline asks of the region and then of the country in turn, which saves
    # the country's two-step class walk.
    "named-region-and-country": 4,
    "income-role-bearing-elected-subject-alternate-at-both": 7,
    # With no place named, the elected one-role shape stays in the election, and its subject
    # siblings take fewer reads in series than the baseline's subject-only answer, or join
    # it as candidates grounded side by side.
    "placeless-reelect-role-bearing-sibling-widens-scope": 7,
    "placeless-reelect-subject-and-role-bearing-siblings": 7,
    "placeless-subject-elected-role-bearing-alternate": 1,
}


# The scenarios beside an index-sourced shape whose answer the baseline gives as well: the
# dropped shape holds nothing the baseline's answer rests on.
_ANSWERS_AS_THE_BASELINE_BESIDE_AN_INDEX_SOURCED_SHAPE = (
    "placeless-definite-beside-index-sourced-sibling",
    "placeless-index-sourced-flagged-alternate-empty",
)


def test_every_serial_step_table_entry_names_one_scenario_and_a_step() -> None:
    tables = (
        _SERIAL_STEPS_ADDED,
        _SERIAL_STEPS_ADDED_FOR_A_SHAPE_THE_BASELINE_DROPS,
        _SERIAL_STEPS_SAVED,
    )
    named = [scenario_id for table in tables for scenario_id in table]
    assert set(named) <= set(_SCENARIOS)
    assert len(named) == len(set(named))
    assert all(steps > 0 for table in tables for steps in table.values())


@pytest.mark.parametrize("scenario_id", list(_SCENARIOS))
def test_first_answer_keeps_its_serial_depth(scenario_id, monkeypatch, search_settings) -> None:
    """On a virtual clock where every request takes one tick, the longest chain of requests
    each awaiting the one before equals the scenario's baseline depth. A scenario in
    `_SERIAL_STEPS_ADDED` is deeper by its steps and gives an answer the baseline does not;
    one in `_SERIAL_STEPS_ADDED_FOR_A_SHAPE_THE_BASELINE_DROPS` is deeper by its steps; one
    in `_SERIAL_STEPS_SAVED` is shallower by its steps. The requests a Spec's facet options
    make (`_maybe_build_facet_options`) are left out of the chain when the scenario sets no
    option coverage, since the baseline engine built them only under it; every other
    request counts."""
    scenario = _SCENARIOS[scenario_id]
    _patch_settings(scenario, search_settings)
    requests = VirtualRequests()
    requests.install(
        monkeypatch,
        doubles=_RECORDED_CLASSES,
        pools=_POOL_MODULES,
        executors=_EXECUTORS,
        free=_FREE_CALLS,
    )
    facet_spans: set[int] = set()

    def spanned(build):
        # A build runs inline on the virtual clock's one thread, so the spans recorded
        # while it runs are its own.
        def run(*args, **kwargs):
            first = len(requests.spans)
            try:
                return build(*args, **kwargs)
            finally:
                facet_spans.update(range(first, len(requests.spans)))

        return run

    if not scenario.include_option_coverage:
        wrap_facet_option_builds(monkeypatch, spanned)

    answer = _answer_record(_resolve(scenario, requests.run))
    depth = requests.serial_depth(excluding=facet_spans)

    if _recording(_RECORD_BASELINE_ENV):
        _store(scenario_id, "baseline_serial_depth", depth)
        _store(scenario_id, "baseline_answer", answer)
        return
    baseline = _stored(scenario_id, "baseline_serial_depth", _RECORD_BASELINE_ENV)
    added = _SERIAL_STEPS_ADDED.get(scenario_id, 0)
    dropped = _SERIAL_STEPS_ADDED_FOR_A_SHAPE_THE_BASELINE_DROPS.get(scenario_id, 0)
    saved = _SERIAL_STEPS_SAVED.get(scenario_id, 0)
    assert depth == baseline + added + dropped - saved, (
        baseline,
        added,
        dropped,
        saved,
    )
    if added:
        assert answer != _stored(scenario_id, "baseline_answer", _RECORD_BASELINE_ENV)


@pytest.mark.parametrize(
    ("scenario_id", "offered"),
    [
        ("placeless-role-bearing-definite", [[_IDS]]),
        ("placeless-role-bearing-elected-subject-alternate", [[_IDS], [_PPP]]),
    ],
)
def test_a_placeless_one_role_question_discovers_its_recipients_from_coverage(
    scenario_id, offered, search_settings
) -> None:
    """With no place named, the recipient-only series stays in the election, alone or beside
    a subject-only reading, and answers over the recipients its one-role coverage rows name."""
    scenario = _SCENARIOS[scenario_id]
    _patch_settings(scenario, search_settings)

    result = _resolve(scenario)

    assert [[sv.ref.ref for sv in spec.resolved.members] for spec in result.specs] == offered
    ids_spec = result.specs[0]
    assert ids_spec.resolved.scope is not None
    assert ids_spec.resolved.scope.axis == "destination"
    assert [member.ref for member in ids_spec.resolved.scope.members] == [_USA]
    assert "role_scope_defaulted" in [warning.code for warning in result.warnings]


# A subject-only series with the recipient-only IDS series' five-tuple and constraints. With
# no role bound, the mint's walk elects the subject-only class, so a shape recalled through
# `_IDS` is answered through this series.
_GNI_SUBJECT = "Amount_EconomicActivity_GrossNationalIncome_Nominal_SubjectScenario"
_IDS_AT_USA = ((_IDS, "f_ids", "", _USA, _YEARS),)
_WALKS_TO_SUBJECT: dict[str, tuple[_Scenario, str, list[tuple[str, list[str]]]]] = {
    # The walked shape is elected: its presence is read and it answers over its own scope.
    "elected": (
        _Scenario(_GNI, (), (_IDS,), (0.85,), _at(_GNI_SUBJECT, _KEN), coverage_rows=_IDS_AT_USA),
        "definite",
        [(_GNI_SUBJECT, [_KEN])],
    ),
    # GNI at purchasing power parity shares the walked series' family and holds
    # `activitySource`, which the walked series leaves open and the query does not name, so
    # the guard refuses it and the walked shape answers alone.
    "elected-beside-subject-alternate": (
        _Scenario(
            _GNI,
            (),
            (_IDS, _PPP),
            (0.85, 0.84),
            {**_at(_GNI_SUBJECT, _KEN), **_at(_PPP, _GHA)},
            also_plausible=(1,),
            coverage_rows=_IDS_AT_USA,
        ),
        "definite",
        [(_GNI_SUBJECT, [_KEN])],
    ),
    # The walked shape is a flagged alternate: it is offered over its own scope when it holds
    # data there, and left out when it holds none.
    "alternate": (
        _Scenario(
            _GNI,
            (),
            (_PPP, _IDS),
            (0.85, 0.84),
            {**_at(_PPP, _GHA), **_at(_GNI_SUBJECT, _KEN)},
            also_plausible=(1,),
            coverage_rows=_IDS_AT_USA,
        ),
        "candidates",
        [(_PPP, [_GHA]), (_GNI_SUBJECT, [_KEN])],
    ),
    "alternate-without-data": (
        _Scenario(
            _GNI,
            (),
            (_PPP, _IDS),
            (0.85, 0.84),
            _at(_PPP, _GHA),
            also_plausible=(1,),
            coverage_rows=_IDS_AT_USA,
        ),
        "definite",
        [(_PPP, [_GHA])],
    ),
    # The walked shape is a re-election sibling of an elected shape with no data.
    "sibling": (
        _Scenario(
            _GNI,
            (),
            (_ATLAS, _IDS),
            (0.85, 0.84),
            _at(_GNI_SUBJECT, _KEN),
            coverage_rows=_IDS_AT_USA,
        ),
        "definite",
        [(_GNI_SUBJECT, [_KEN])],
    ),
}


@pytest.mark.parametrize("case", list(_WALKS_TO_SUBJECT))
def test_a_shape_the_walk_makes_subject_only_takes_the_default_scope_of_its_own_stat_vars(
    case, monkeypatch, search_settings
) -> None:
    """No place is named, and the role-class walk replaces the recalled recipient-only shape
    with the subject-only family of its five-tuple. Elected, flagged as an alternate or
    re-elected as a sibling, the walked shape has its presence read and is offered over the
    default scope of its own StatVars, with exact coverage and the scope warning."""
    scenario, status, offered = _WALKS_TO_SUBJECT[case]
    _patch_settings(scenario, search_settings)
    subject = _sv(
        "Gross national income (current US$), subject",
        _income_arcs(
            _IDS_QUALIFIER,
            {
                "economicActivityType": "GrossNationalIncome",
            },
        ),
    )
    monkeypatch.setitem(_NODES, _GNI_SUBJECT, subject)

    result = _resolve(scenario)

    assert result.status == status, result
    assert [
        (
            [sv.ref.ref for sv in spec.resolved.members],
            spec.resolved.scope and [member.ref for member in spec.resolved.scope.members],
        )
        for spec in result.specs
    ] == [([stat_var], places) for stat_var, places in offered]
    assert all(spec.resolved.coverage.kind == "exact" for spec in result.specs)
    assert "place_scope_defaulted" in [warning.code for warning in result.warnings]


def test_a_bound_recipient_keeps_the_recipient_only_shape_beside_a_subject_only_family(
    monkeypatch, search_settings
) -> None:
    """With the recipient named, the walk keeps the recipient-only class, which answers at
    the named place from coverage."""
    scenario = _Scenario(
        "gross national income of Ghana",
        ("Ghana",),
        (_IDS,),
        (0.85,),
        _at(_GNI_SUBJECT, _GHA),
        coverage_rows=((_IDS, "f_ids", "", _GHA, _YEARS),),
    )
    _patch_settings(scenario, search_settings)
    monkeypatch.setitem(
        _NODES,
        _GNI_SUBJECT,
        _sv(
            "Gross national income, subject",
            _income_arcs(
                _IDS_QUALIFIER,
                {
                    "economicActivityType": "GrossNationalIncome",
                },
            ),
        ),
    )

    result = _resolve(scenario)

    assert result.status == "definite", result
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_IDS]


@pytest.mark.parametrize("scenario_id", _ANSWERS_AS_THE_BASELINE_BESIDE_AN_INDEX_SOURCED_SHAPE)
def test_an_index_sourced_shape_the_answer_does_not_rest_on_leaves_the_answer_standing(
    scenario_id, search_settings, caplog
) -> None:
    """A subject-only index-sourced shape can take no default scope. Elected, it refuses the
    request. As a sibling, a widened search's reading or a flagged alternate holding nothing
    the answer rests on, it is dropped with a log line and the request gives the baseline's
    answer."""
    scenario = _SCENARIOS[scenario_id]
    _patch_settings(scenario, search_settings)

    with caplog.at_level(logging.INFO, logger="search.engine.region.phase_place_materialise"):
        answer = _answer_record(_resolve(scenario))

    assert answer == _stored(scenario_id, "baseline_answer", _RECORD_BASELINE_ENV)
    assert any("takes no default scope" in record.getMessage() for record in caplog.records)
