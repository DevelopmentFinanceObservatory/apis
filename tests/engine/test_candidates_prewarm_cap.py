"""Alternate-shape pre-warm caps index populations at `settings().region_max_members`.

A recall pool unions to a few hundred dcids; an index-sourced population can reach thousands.
The test counts the dcids in the pre-warm's facets call.
"""

from __future__ import annotations

import dataclasses

import pytest

from code_index.shared import get_index
from materialise.place_scope import DefaultPresence, scope_of_places
from search.engine.config import settings
from search.engine.place import ConsultedPresence
from search.engine.region.place_context import PlaceContext
from search.engine.shape import ShapeDraft
from tests.engine._builders import (
    bound_place_context,
    materialise_alternates,
    subject_place_roles,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._index_slots import index_slot_keys
from tests.fixtures import FakeGraph

_POP_TYPE = "FinancialObligation"
_MEAS_PROP = "amount"
_STAT_TYPE = "measuredValue"
_MEAS_QUAL = "Nominal"
_MEAS_DENOM = None


class _SpyGraph(FakeGraph):
    """Record each observation-facets request before delegating to `FakeGraph`."""

    def __init__(self) -> None:
        super().__init__(nodes={}, obs={}, detect={}, resolve={})
        self.calls: list[list[str]] = []

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, *, max_bytes=None):
        self.calls.append(list(stat_vars))
        return super().observation_facets_batch(
            stat_vars, entities, needs_dates=needs_dates, max_bytes=max_bytes
        )


def test_the_prewarm_caps_an_index_sourced_populations_union_at_the_region_cap():
    index = get_index()
    target = (_POP_TYPE, _MEAS_PROP, _STAT_TYPE, _MEAS_QUAL, _MEAS_DENOM)
    population = [
        row.dcid
        for row in index.statvars
        if (
            row.population_type or "",
            row.measured_property,
            row.stat_type or "",
            row.measurement_qualifier,
            row.measurement_denominator,
        )
        == target
    ]
    assert len(population) > settings().region_max_members, (
        "expected a real CRS five-tuple group larger than the scope cap"
    )

    # Incomplete arcs let index facts supply the full shape population.
    recalled_sv = population[0]
    shape = ShapeDraft(
        shape_id="test_broad_alt_shape",
        label="test shape",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=_MEAS_QUAL,
        meas_denom_dcid=_MEAS_DENOM,
        slot_keys=index_slot_keys(
            population_type=_POP_TYPE,
            measured_property=_MEAS_PROP,
            stat_type=_STAT_TYPE,
            measurement_qualifier=_MEAS_QUAL,
        ),
        sv_arc_facts={recalled_sv: {}},
        sv_arc_facts_complete=False,
    )
    graph = _SpyGraph()

    results, unsupported = materialise_alternates(
        [shape],
        places=[bound_place_context(subject_place_roles("country/KEN"))],
        elected_bindings=[],
        elected_has_data=False,
        named_place_dcids=["country/KEN"],
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    assert unsupported is None
    # The empty fixture drops the shape after pre-warm. Only the pre-warm request is measured.
    assert results == []
    assert len(graph.calls) >= 1
    assert len(graph.calls[0]) == settings().region_max_members
    assert len(graph.calls[0]) < len(population)


class _EntitySpyGraph(FakeGraph):
    """Record each observation-facets request's StatVars and entities."""

    def __init__(self) -> None:
        super().__init__(nodes={}, obs={}, detect={}, resolve={})
        self.calls: list[tuple[list[str], list[str]]] = []

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, *, max_bytes=None):
        self.calls.append((list(stat_vars), list(entities)))
        return super().observation_facets_batch(
            stat_vars, entities, needs_dates=needs_dates, max_bytes=max_bytes
        )


def _one_member_shape(sv: str) -> ShapeDraft:
    return ShapeDraft(
        shape_id=sv,
        label=sv,
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={sv: {}},
    )


def test_the_prewarm_batches_shapes_by_probe_places_and_every_default_scope_together():
    """Shapes probed at the same bound places share one prewarm. Every default-scoped shape
    joins one prewarm of its members at their own scopes, here one read over their union,
    which costs fewer POSTs than one per scope. Batches follow first-shape order, and each
    default-scoped shape is probed over the scope of its own StatVars."""
    ken, gha, tza = "country/KEN", "country/GHA", "country/TZA"
    roles = subject_place_roles()
    places = [
        bound_place_context(subject_place_roles(ken), ken),
        PlaceContext(
            roles=roles, probe_dcids=(ken, tza), mode="countries_fallback", scope_warning=None
        ),
        bound_place_context(subject_place_roles(ken), ken),
        bound_place_context(subject_place_roles(gha), gha),
        PlaceContext(
            roles=roles, probe_dcids=(gha,), mode="countries_fallback", scope_warning=None
        ),
    ]
    graph = _EntitySpyGraph()
    presence = ConsultedPresence(
        DefaultPresence(
            countries=(ken, gha, tza),
            scopes={
                "sv_1": scope_of_places([ken, tza], every_place=False),
                "sv_4": scope_of_places([gha], every_place=False),
            },
        )
    )

    results, unsupported = materialise_alternates(
        [_one_member_shape(f"sv_{i}") for i in range(len(places))],
        places=places,
        presence=presence,
        elected_bindings=[],
        elected_has_data=False,
        named_place_dcids=[],
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    assert (results, unsupported) == ([], None)
    prewarms = graph.calls[:3]
    assert prewarms == [
        (["sv_0", "sv_2"], [ken]),
        (["sv_1", "sv_4"], [ken, tza, gha]),
        (["sv_3"], [gha]),
    ]
    # Each default-scoped shape is then probed over the scope its own presence gives.
    assert [call for call in graph.calls[3:] if call[0] in (["sv_1"], ["sv_4"])] == [
        (["sv_1"], [ken, tza]),
        (["sv_4"], [gha]),
    ]


def test_the_prewarm_reads_the_pairs_the_probe_reads():
    """A default-scoped shape of two members: one has data at Earth alone, the other in
    Kenya. The prewarm and the probe choose the same read plan over the same scopes, so the
    read the prewarm warms finds every pair the probe asks for."""
    ken, earth = "country/KEN", "Earth"
    shape = dataclasses.replace(
        _one_member_shape("sv_worldwide"), sv_arc_facts={"sv_worldwide": {}, "sv_kenya": {}}
    )
    presence = ConsultedPresence(
        DefaultPresence(
            countries=(ken,),
            scopes={
                "sv_worldwide": scope_of_places([earth], every_place=False),
                "sv_kenya": scope_of_places([ken], every_place=False),
            },
        )
    )
    place = PlaceContext(
        roles=subject_place_roles(),
        probe_dcids=(ken,),
        mode="countries_fallback",
        scope_warning=None,
    )
    graph = _EntitySpyGraph()

    materialise_alternates(
        [shape],
        places=[place],
        presence=presence,
        elected_bindings=[],
        elected_has_data=False,
        named_place_dcids=[],
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    prewarm, probe = graph.calls[:2]
    assert prewarm == (["sv_worldwide", "sv_kenya"], [earth, ken])
    assert probe == prewarm


@pytest.mark.parametrize(
    ("declared_roles", "prewarmed"),
    [
        pytest.param((), True, id="subject-only"),
        pytest.param(("financingProvider", "financingRecipient"), False, id="role-bearing"),
    ],
)
def test_the_prewarm_reads_only_alternates_the_standard_path_resolves(declared_roles, prewarmed):
    """A role-bearing alternate resolves from coverage (`graph_confirm_resolve`) and reads no
    warmed facet, so the prewarm leaves it out; a subject-only alternate is warmed."""
    ken = "country/KEN"
    shape = dataclasses.replace(_one_member_shape("sv_alt"), declared_roles=declared_roles)
    graph = _EntitySpyGraph()

    materialise_alternates(
        [shape],
        places=[bound_place_context(subject_place_roles(ken), ken)],
        elected_bindings=[],
        elected_has_data=False,
        named_place_dcids=[ken],
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    assert ((["sv_alt"], [ken]) in graph.calls) is prewarmed
