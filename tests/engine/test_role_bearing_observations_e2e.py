"""End-to-end tests for role-bearing observation series returned by resolve_variable.
Synthetic fixtures use provider/recipient, the role names supported by the series wire model."""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    build_materialise_double_union_candidates,
    door_roles_for_test_query,
)
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_SV_DCID = "TESTFIXTURE/RoleBearingObservationsE2E/Flow"
_NOR = "country/NOR"
_KEN = "country/KEN"
_SWE = "country/SWE"
_UGA = "country/UGA"

_ROLE_NAMES = RoleNames(donor="provider", recipient="recipient")
"""Role names declared by _declaring_arcs and stored in coverage."""


def _dated_facet(dates: list[str], *, facet_id: str = "f1") -> Facet:
    """Build a dated facet with a facet_id for LATEST confirmation."""
    return Facet(
        earliest_date=dates[0],
        latest_date=dates[-1],
        obs_count=len(dates),
        dates=dates,
        values=[float(i) for i in range(len(dates))],
        facet_id=facet_id,
    )


def _materialise_points(dates: list[str]) -> tuple[tuple[str, float], ...]:
    """Convert dated facet points to the materialisation fixture format."""
    return tuple((d, float(i)) for i, d in enumerate(dates))


def _declaring_arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]},
    }


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _run(
    *,
    entities: list[str],
    resolve: dict[str, str],
    role_query: str,
    sdmx: FakeSdmxClient,
    sv_dcid: str = _SV_DCID,
    sv_score: float = 0.9,
    materialise_graph=None,
    materialise_sdmx=None,
    monkeypatch=None,
    coverage_rows=None,
    coverage_facet_provenance=None,
):
    """Run resolve_variable with optional patched coverage and materialisation doubles.

    Coverage facet provenance must match each ObsFacetFixture or selection filters out its rows."""
    coverage_snapshot = empty_coverage_snapshot()
    if coverage_rows is not None:
        assert monkeypatch is not None, "coverage_rows needs a monkeypatch fixture"
        facet_metadata = {
            facet_id: FacetMetadata(provenance_id=provenance)
            for facet_id, provenance in (coverage_facet_provenance or {}).items()
        }
        _coverage_index = patch_coverage(
            monkeypatch,
            coverage_rows,
            role_names={sv_dcid: _ROLE_NAMES},
            facet_metadata=facet_metadata,
        )
        coverage_snapshot = FakeCoverageService(_coverage_index).snapshot()
    graph = FakeGraph(
        nodes={
            **{name: {"label": name, "type": "Country"} for name in resolve},
            **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
            sv_dcid: {"label": "Test flow", "arcs": _declaring_arcs()},
        },
        detect={role_query: {"svs": [sv_dcid], "cosine_scores": [sv_score]}},
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
            coverage=coverage_snapshot,
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            ),
        )
    )


def _series_entities(series) -> frozenset[tuple[str, str]]:
    return frozenset((e.role, e.ref) for e in series.entities)


class TestDirectionalWordingBothOrders:
    """Equivalent directional wording returns the same Norway-to-Kenya series."""

    def _sdmx(self) -> FakeSdmxClient:
        return FakeSdmxClient(
            observations={
                _SV_DCID: {
                    (("provider", _NOR), ("recipient", _KEN)): [
                        _dated_facet(["2018", "2019", "2020"])
                    ],
                    # Self-flow poison: a role-blind filter that folds the
                    # origin entity into the destination side would match this.
                    (("provider", _NOR), ("recipient", _NOR)): [
                        _dated_facet(["2010", "2011", "2012", "2013", "2014"])
                    ],
                    # Reverse-flow poison: the two named places with their
                    # roles swapped.
                    (("provider", _KEN), ("recipient", _NOR)): [
                        _dated_facet(["2001", "2002", "2003", "2004", "2005", "2006", "2007"])
                    ],
                },
            },
        )

    def _coverage_rows(self):
        """Poison rows with a different donor or recipient fail the bound-role filter."""
        return [
            (_SV_DCID, "f1", _NOR, _KEN, [2018, 2019, 2020]),
            # Self-flow poison: a role-blind filter that folds the origin
            # entity into the destination side would match this.
            (_SV_DCID, "f2", _NOR, _NOR, [2010, 2011, 2012, 2013, 2014]),
            # Reverse-flow poison: the two named places with their roles
            # swapped.
            (_SV_DCID, "f3", _KEN, _NOR, [2001, 2002, 2003, 2004, 2005, 2006, 2007]),
        ]

    def _materialise(self):
        """Only the role-split Norway-to-Kenya pair has a materialisation fixture."""
        return build_materialise_double(
            {
                (_SV_DCID, (("provider", _NOR), ("recipient", _KEN))): [
                    ObsFacetFixture(
                        facet_id="f1",
                        points=_materialise_points(["2018", "2019", "2020"]),
                        provenance_id="dc/base/RoleBearingObsE2E",
                    )
                ],
            }
        )

    def _assert_clean_result(self, result) -> None:
        assert result.status == "definite", result
        observations = result.specs[0].resolved.observations
        assert observations is not None
        assert observations.total_obs == 3, (
            f"expected only the Norway-to-Kenya row's 3 points, got {observations.total_obs}"
        )
        assert len(observations.series) == 1, (
            f"expected exactly one series, got {len(observations.series)}"
        )
        assert _series_entities(observations.series[0]) == {
            ("origin", _NOR),
            ("destination", _KEN),
        }

    def _coverage_facet_provenance(self):
        # Must match _materialise's own ObsFacetFixture.provenance_id for
        # f1 exactly (see _run's own docstring); f2/f3 are poison rows
        # never dispatched, so their provenance is never read.
        return {"f1": "dc/base/RoleBearingObsE2E"}

    def test_from_norway_to_kenya(self, monkeypatch):
        materialise_graph, materialise_sdmx, _requests = self._materialise()
        result = _run(
            entities=["Norway", "Kenya"],
            resolve={"Norway": _NOR, "Kenya": _KEN},
            role_query="aid from Norway to Kenya",
            sdmx=self._sdmx(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            monkeypatch=monkeypatch,
            coverage_rows=self._coverage_rows(),
            coverage_facet_provenance=self._coverage_facet_provenance(),
        )
        self._assert_clean_result(result)

    def test_to_kenya_from_norway(self, monkeypatch):
        materialise_graph, materialise_sdmx, _requests = self._materialise()
        result = _run(
            entities=["Kenya", "Norway"],
            resolve={"Norway": _NOR, "Kenya": _KEN},
            role_query="aid to Kenya from Norway",
            sdmx=self._sdmx(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            monkeypatch=monkeypatch,
            coverage_rows=self._coverage_rows(),
            coverage_facet_provenance=self._coverage_facet_provenance(),
        )
        self._assert_clean_result(result)


class TestNoDirectionalWordingRefuses:
    """A lone named place with no "from"/"to" wording carries no axis at all,
    so the role-bearing shape has nothing to bind it to and the query
    refuses. The poison row where that same place is the origin of a
    different flow is never reached either way."""

    def test_lone_place_with_no_direction_refuses_role_not_resolved(self):
        sdmx = FakeSdmxClient(
            observations={
                _SV_DCID: {
                    (("provider", _NOR), ("recipient", _KEN)): [
                        _dated_facet(["2018", "2019", "2020"])
                    ],
                    # Kenya-as-origin poison: aid that Kenya itself gives,
                    # never asked for by a directionless query.
                    (("provider", _KEN), ("recipient", _NOR)): [
                        _dated_facet(["2001", "2002", "2003", "2004", "2005"])
                    ],
                },
            },
        )
        # The refusal fires before any observation fetch: no per-pair probe
        # and no series read is ever made, so the poison pair's own
        # (provider, recipient) shape above is never requested either.
        materialise_graph, materialise_sdmx, _requests = build_materialise_double(
            {
                (_SV_DCID, (("recipient", _KEN),)): [
                    ObsFacetFixture(
                        facet_id="f1",
                        points=_materialise_points(["2018", "2019", "2020"]),
                        provenance_id="dc/base/RoleBearingObsE2E",
                    )
                ],
            },
            declared_roles={_SV_DCID: ("provider", "recipient")},
        )
        result = _run(
            entities=["Kenya"],
            resolve={"Kenya": _KEN, "Norway": _NOR},
            role_query="aid Kenya",
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        )
        assert result.status == "no_data", result
        assert result.no_data_reason == "role_not_resolved"
        assert result.specs == ()


class TestMultiPlace:
    """Plural origins and destinations keep their own roles in every series."""

    def test_one_origin_two_destinations_stay_on_their_own_axis(self, monkeypatch):
        resolve = {"Norway": _NOR, "Kenya": _KEN, "Uganda": _UGA}
        # The coverage-backed confirmation's own bound-role filter guards
        # against the self-flow/axis-swap poison rows below the same way
        # TestDirectionalWordingBothOrders._coverage_rows' own does.
        coverage_rows = [
            (_SV_DCID, "f1", _NOR, _KEN, [2018, 2019, 2020]),
            (_SV_DCID, "f1", _NOR, _UGA, [2015, 2016, 2017, 2018]),
            (_SV_DCID, "f3", _NOR, _NOR, [2001, 2002, 2003, 2004, 2005]),
            (_SV_DCID, "f4", _KEN, _NOR, [2001, 2002, 2003, 2004, 2005, 2006]),
        ]
        sdmx = FakeSdmxClient(
            observations={
                _SV_DCID: {
                    (("provider", _NOR), ("recipient", _KEN)): [
                        _dated_facet(["2018", "2019", "2020"])
                    ],
                    (("provider", _NOR), ("recipient", _UGA)): [
                        _dated_facet(["2015", "2016", "2017", "2018"])
                    ],
                    # Self-flow and axis-swap poisons: the origin as its own
                    # destination, and a destination place acting as an
                    # origin toward the real origin.
                    (("provider", _NOR), ("recipient", _NOR)): [
                        _dated_facet(["2001", "2002", "2003", "2004", "2005"])
                    ],
                    (("provider", _KEN), ("recipient", _NOR)): [
                        _dated_facet(["2001", "2002", "2003", "2004", "2005", "2006"])
                    ],
                },
            },
        )
        # The observations preview's own per-pair fetch: one pair per
        # (provider NOR, recipient X) destination. The poison pairs above
        # are never enumerated in the first place (see
        # TestDirectionalWordingBothOrders._materialise's own docstring).
        # NOR is shared as "provider" across both pairs, so the
        # union-candidates double is needed (see its own docstring).
        materialise_graph, materialise_sdmx, _requests = build_materialise_double_union_candidates(
            {
                (_SV_DCID, (("provider", _NOR), ("recipient", _KEN))): [
                    ObsFacetFixture(
                        facet_id="f1",
                        points=_materialise_points(["2018", "2019", "2020"]),
                        provenance_id="dc/base/RoleBearingObsE2EKen",
                    )
                ],
                (_SV_DCID, (("provider", _NOR), ("recipient", _UGA))): [
                    ObsFacetFixture(
                        facet_id="f1",
                        points=_materialise_points(["2015", "2016", "2017", "2018"]),
                        provenance_id="dc/base/RoleBearingObsE2EKen",
                    )
                ],
            }
        )
        result = _run(
            entities=["Norway", "Kenya", "Uganda"],
            resolve=resolve,
            role_query="aid from Norway to Kenya and Uganda",
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            monkeypatch=monkeypatch,
            coverage_rows=coverage_rows,
            # Both pairs hold one facet, the group the variable serves at every pair. It must
            # match the materialise fixture's own ObsFacetFixture provenance_id exactly (see
            # _run's own docstring); f3/f4 are poison rows never dispatched.
            coverage_facet_provenance={"f1": "dc/base/RoleBearingObsE2EKen"},
        )
        assert result.status == "definite", result
        observations = result.specs[0].resolved.observations
        assert observations is not None
        assert observations.total_obs == 7, (
            "expected only the Norway->Kenya (3) and Norway->Uganda (4) rows, "
            f"got {observations.total_obs}"
        )
        assert len(observations.series) == 2
        entity_pairs = {_series_entities(s) for s in observations.series}
        assert entity_pairs == {
            frozenset({("origin", _NOR), ("destination", _KEN)}),
            frozenset({("origin", _NOR), ("destination", _UGA)}),
        }, entity_pairs

    def test_two_origins_both_feed_fixed_destination_observations(self, monkeypatch):
        """A fixed destination retains every provider's role tuple.

        The materialise preview sees both complete provider/recipient pairs,
        rather than truncating to the first origin while the emitted identity
        still names both providers.
        """
        resolve = {"Norway": _NOR, "Sweden": _SWE, "Kenya": _KEN}
        # Both rows share facet_id "f1": the materialise fixture below
        # elects facet_id "f1" for BOTH the (NOR, KEN) and (SWE, KEN) pairs
        # (see its own ObsFacetFixture entries), and the delivered rows
        # survive only under the facet_id select_facet actually elects (see
        # _run's own docstring).
        coverage_rows = [
            (_SV_DCID, "f1", _NOR, _KEN, [2018, 2019, 2020]),
            (_SV_DCID, "f1", _SWE, _KEN, [2015, 2016, 2017, 2018]),
        ]
        sdmx = FakeSdmxClient(
            observations={
                _SV_DCID: {
                    (("provider", _NOR), ("recipient", _KEN)): [
                        _dated_facet(["2018", "2019", "2020"])
                    ],
                    (("provider", _SWE), ("recipient", _KEN)): [
                        _dated_facet(["2015", "2016", "2017", "2018"])
                    ],
                },
            },
        )
        # The observation preview enumerates each provider/recipient pair.
        materialise_graph, materialise_sdmx, _requests = build_materialise_double(
            {
                (_SV_DCID, (("provider", _NOR), ("recipient", _KEN))): [
                    ObsFacetFixture(
                        facet_id="f1",
                        points=_materialise_points(["2018", "2019", "2020"]),
                        provenance_id="dc/base/RoleBearingObsE2E",
                    )
                ],
                (_SV_DCID, (("provider", _SWE), ("recipient", _KEN))): [
                    ObsFacetFixture(
                        facet_id="f1",
                        points=_materialise_points(["2015", "2016", "2017", "2018"]),
                        provenance_id="dc/base/RoleBearingObsE2E",
                    )
                ],
            }
        )
        result = _run(
            entities=["Norway", "Sweden", "Kenya"],
            resolve=resolve,
            role_query="aid from Norway and Sweden to Kenya",
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            monkeypatch=monkeypatch,
            coverage_rows=coverage_rows,
            coverage_facet_provenance={"f1": "dc/base/RoleBearingObsE2E"},
        )
        assert result.status == "definite", result
        observations = result.specs[0].resolved.observations
        assert observations is not None
        assert observations.total_obs == 7
        assert {_series_entities(series) for series in observations.series} == {
            frozenset({("origin", _NOR), ("destination", _KEN)}),
            frozenset({("origin", _SWE), ("destination", _KEN)}),
        }


class TestEntityRoleLabelling:
    """Spec.resolved.entities labels an origin place by its directional
    role."""

    def test_origin_and_destination_entities_carry_their_directional_role(self, monkeypatch):
        sdmx = FakeSdmxClient(
            observations={
                _SV_DCID: {
                    (("provider", _NOR), ("recipient", _KEN)): [
                        _dated_facet(["2018", "2019", "2020"])
                    ],
                },
            },
        )
        result = _run(
            entities=["Norway", "Kenya"],
            resolve={"Norway": _NOR, "Kenya": _KEN},
            role_query="aid from Norway to Kenya",
            sdmx=sdmx,
            monkeypatch=monkeypatch,
            coverage_rows=[(_SV_DCID, "f1", _NOR, _KEN, [2018, 2019, 2020])],
        )
        assert result.status == "definite", result
        roles_by_ref = {e.ref: e.role for e in result.specs[0].resolved.entities}
        assert roles_by_ref == {_NOR: "origin", _KEN: "destination"}
