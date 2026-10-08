"""Tests facet-binding availability checks for standard and role-indexed candidates."""

from __future__ import annotations

import pytest

from search.engine.clients.graph import Facet
from search.engine.clients.sdmx import RoleFacetCandidate
from search.engine.facet_refusals import (
    candidate_facet_refusal,
    facet_refusal,
    facet_satisfies_bindings,
    role_index_facets,
)
from search.models import (
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
)
from tests.engine._refusal_detail import assert_names_plainly
from tests.engine._role_candidates import role_candidate
from tests.fixtures import FakeGraph

_SV = "Test/SomeStatVar"
_SV_A = "Test/SomeStatVarA"
_SV_B = "Test/SomeStatVarB"
_ENTITY = "country/USA"


def _graph(facets: list[dict], currency_units: dict | None = None) -> FakeGraph:
    kwargs = {} if currency_units is None else {"currency_units": currency_units}
    return FakeGraph(obs={f"{_SV}|{_ENTITY}": facets}, **kwargs)


@pytest.mark.parametrize(
    ("obs", "stat_var_dcids", "bindings"),
    [
        pytest.param(
            {
                f"{_SV}|{_ENTITY}": [
                    {
                        "earliestDate": "2015",
                        "latestDate": "2020",
                        "obsCount": 3,
                        "provenanceId": "src/A",
                    }
                ]
            },
            [_SV],
            {"provenance": IdentityBindingValue(ref="src/A")},
            id="matched_provenance",
        ),
        pytest.param(
            {
                f"{_SV}|{_ENTITY}": [
                    {
                        "earliestDate": "2015",
                        "latestDate": "2020",
                        "obsCount": 3,
                        "measurementMethod": "GrossDisbursement",
                    }
                ]
            },
            [_SV],
            {"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")},
            id="matched_measurement_method",
        ),
        pytest.param(
            {
                f"{_SV_A}|{_ENTITY}": [
                    {
                        "earliestDate": "2015",
                        "latestDate": "2020",
                        "obsCount": 3,
                        "measurementMethod": "GrossDisbursement",
                    }
                ],
                f"{_SV_B}|{_ENTITY}": [],
            },
            [_SV_A],
            {"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")},
            id="single_stat_var",
        ),
    ],
)
def test_matched_bindings_return_none(obs, stat_var_dcids, bindings):
    assert (
        facet_refusal(FakeGraph(obs=obs), stat_var_dcids, [_ENTITY], bindings, deadline=None)
        is None
    )


def test_no_relevant_binding_returns_none():
    graph = _graph([{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3}])
    assert facet_refusal(graph, [_SV], [_ENTITY], None, deadline=None) is None
    assert facet_refusal(graph, [_SV], [_ENTITY], {}, deadline=None) is None


def test_unmatched_provenance_value_binding_refuses_facet_not_available():
    """A provenance binding matches the facet's source dcid."""
    graph = _graph(
        [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3, "provenanceId": "src/A"}]
    )
    result = facet_refusal(
        graph, [_SV], [_ENTITY], {"provenance": IdentityBindingValue(ref="src/B")}, deadline=None
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert result.pointer == "/input/facets/provenance"
    assert "src/B" in str(result)


def test_provenance_set_binding_names_the_absent_member():
    """A set provenance binding refuses and names each missing source."""
    graph = _graph(
        [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3, "provenanceId": "src/A"}]
    )
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {"provenance": IdentityBindingSet(refs=["src/A", "src/B"])},
        deadline=None,
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert result.pointer == "/input/facets/provenance"
    assert "src/B" in str(result)
    assert "src/A" not in str(result).split("missing")[-1]


def test_mixed_missing_members_across_provenance_and_unit_named_together():
    """One refusal reports missing members across provenance and unit axes."""
    graph = _graph(
        [
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 3,
                "provenanceId": "src/A",
                "unit": "USD",
            }
        ]
    )
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {
            "provenance": IdentityBindingValue(ref="src/B"),
            "unit": IdentityBindingValue(ref="EUR"),
        },
        deadline=None,
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert "provenance" in str(result)
    assert "src/B" in str(result)
    assert "unit" in str(result)
    assert "EUR" in str(result)


def test_unmatched_value_binding_refuses_facet_not_available():
    graph = _graph(
        [
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 3,
                "measurementMethod": "GrossDisbursement",
            }
        ]
    )
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {"measurementMethod": IdentityBindingValue(ref="Commitment")},
        deadline=None,
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert result.status == 422
    assert result.pointer == "/input/facets/measurementMethod"
    assert "Commitment" in str(result)


def test_a_set_binding_missing_many_members_names_five_and_counts_the_rest():
    graph = _graph(
        [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3, "provenanceId": "src/A"}]
    )
    missing = [f"src/Z{letter}" for letter in "abcdefg"]

    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {"provenance": IdentityBindingSet(refs=["src/A", *missing])},
        deadline=None,
    )

    assert result is not None
    assert result.code == "facet_not_available"
    assert_names_plainly(str(result), missing)


def test_all_or_nothing_set_binding_names_every_missing_member():
    graph = _graph(
        [
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 3,
                "measurementMethod": "GrossDisbursement",
            }
        ]
    )
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {
            "measurementMethod": IdentityBindingSet(
                refs=["GrossDisbursement", "Commitment", "GrantEquivalent"]
            )
        },
        deadline=None,
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert "Commitment" in str(result)
    assert "GrantEquivalent" in str(result)
    # The one present member is never named as missing.
    assert "GrossDisbursement" not in str(result).split("missing")[-1]


def test_all_or_nothing_excludes_the_members_own_axis_from_the_other_check():
    """Other bindings filter candidate facets. A binding leaves its own axis unfiltered."""
    graph = _graph(
        [
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 3,
                "measurementMethod": "GrossDisbursement",
                "unit": "USD",
            },
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 3,
                "measurementMethod": "Commitment",
                "unit": "EUR",
            },
        ]
    )
    # The unit is USD, so the Commitment facet fails the other-axis filter.
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {
            "measurementMethod": IdentityBindingSet(refs=["GrossDisbursement", "Commitment"]),
            "unit": IdentityBindingValue(ref="USD"),
        },
        deadline=None,
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert "measurementMethod" in str(result)
    assert "Commitment" in str(result)


def test_member_with_only_out_of_window_observations_is_not_missing():
    """Membership uses unwindowed `obsCount`, even when facet date ranges differ."""
    graph = _graph(
        [
            {
                "earliestDate": "1990",
                "latestDate": "1995",
                "obsCount": 3,
                "measurementMethod": "GrossDisbursement",
            },
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 3,
                "measurementMethod": "Commitment",
            },
        ]
    )
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {"measurementMethod": IdentityBindingSet(refs=["GrossDisbursement", "Commitment"])},
        deadline=None,
    )
    assert result is None


def test_zero_obs_count_facet_does_not_count_as_present():
    graph = _graph(
        [
            {
                "earliestDate": None,
                "latestDate": None,
                "obsCount": 0,
                "measurementMethod": "Commitment",
            }
        ]
    )
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {"measurementMethod": IdentityBindingValue(ref="Commitment")},
        deadline=None,
    )
    assert result is not None
    assert result.code == "facet_not_available"


def test_inexpressible_unit_request_refuses_facet_binding_unsupported_before_any_data_lookup():
    """A currency outside ISO 4217 returns 501 before any graph data lookup."""
    graph = FakeGraph(raise_on_call=True)
    request = IdentityUnitRequest(currency="XYZ", prices="current")
    result = facet_refusal(
        graph, [_SV], [_ENTITY], {"unit": IdentityFacetValueRequest(request=request)}, deadline=None
    )
    assert result is not None
    assert result.code == "facet_binding_unsupported"
    assert result.status == 501
    assert result.pointer == "/input/facets/unit"


def test_matched_unit_request_form_returns_none():
    graph = _graph(
        [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3, "unit": "CurrencyUSD"}],
        currency_units={"CurrencyUSD": "USD"},
    )
    request = IdentityUnitRequest(currency="USD", prices="current")
    result = facet_refusal(
        graph, [_SV], [_ENTITY], {"unit": IdentityFacetValueRequest(request=request)}, deadline=None
    )
    assert result is None


def test_foreign_currency_unit_request_form_refuses_without_a_map():
    """Without `elected_units`, an EUR facet does not satisfy a USD request."""
    graph = _graph(
        [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3, "unit": "CurrencyEUR"}],
        currency_units={"CurrencyEUR": "EUR", "CurrencyUSD": "USD"},
    )
    request = IdentityUnitRequest(currency="USD", prices="current")
    result = facet_refusal(
        graph, [_SV], [_ENTITY], {"unit": IdentityFacetValueRequest(request=request)}, deadline=None
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert result.pointer == "/input/facets/unit"


def test_unit_ref_form_mismatch_still_refuses_facet_not_available():
    """A unit ref binding must match its literal unit ref."""
    graph = _graph(
        [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3, "unit": "CurrencyEUR"}]
    )
    result = facet_refusal(
        graph, [_SV], [_ENTITY], {"unit": IdentityBindingValue(ref="CurrencyUSD")}, deadline=None
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert result.pointer == "/input/facets/unit"
    assert "CurrencyUSD" in str(result)


def test_elected_units_admits_a_facet_the_literal_request_would_refuse():
    """A unit in `elected_units` satisfies the request when it matches the minted facet."""
    graph = _graph(
        [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3, "unit": "CurrencyUSD"}],
        currency_units={"CurrencyUSD": "USD", "CurrencyEUR": "EUR"},
    )
    request = IdentityUnitRequest(currency="EUR", prices="current")
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {"unit": IdentityFacetValueRequest(request=request)},
        elected_units={_SV: frozenset({"CurrencyUSD"})},
        deadline=None,
    )
    assert result is None


def test_elected_units_stat_var_absent_from_the_map_refuses():
    """A StatVar absent from `elected_units` has no admitted unit facets."""
    graph = FakeGraph(
        obs={
            f"{_SV_A}|{_ENTITY}": [
                {"earliestDate": "2015", "latestDate": "2020", "obsCount": 3, "unit": "CurrencyUSD"}
            ],
            f"{_SV_B}|{_ENTITY}": [
                {"earliestDate": "2015", "latestDate": "2020", "obsCount": 3, "unit": "CurrencyUSD"}
            ],
        },
        currency_units={"CurrencyUSD": "USD"},
    )
    request = IdentityUnitRequest(currency="USD", prices="current")
    result = facet_refusal(
        graph,
        [_SV_A, _SV_B],
        [_ENTITY],
        {"unit": IdentityFacetValueRequest(request=request)},
        elected_units={_SV_A: frozenset({"CurrencyUSD"})},
        deadline=None,
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert result.pointer == "/input/facets/unit"


def test_provenance_set_member_absent_still_refuses_with_elected_units_unit_admitted():
    """An admitted unit facet does not hide a missing provenance member."""
    graph = _graph(
        [
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 3,
                "provenanceId": "src/A",
                "unit": "CurrencyUSD",
            }
        ],
        currency_units={"CurrencyUSD": "USD"},
    )
    request = IdentityUnitRequest(currency="USD", prices="current")
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {
            "provenance": IdentityBindingSet(refs=["src/A", "src/B"]),
            "unit": IdentityFacetValueRequest(request=request),
        },
        elected_units={_SV: frozenset({"CurrencyUSD"})},
        deadline=None,
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert result.pointer == "/input/facets/provenance"
    assert "src/B" in str(result)
    assert "src/A" not in str(result).split("missing")[-1]


def test_facet_satisfies_bindings_with_elected_units_counts_only_the_elected_unit():
    """Coverage counts only the elected USD facet, excluding the EUR facet's 5 observations."""
    usd_facet = Facet(earliest_date="2015", latest_date="2020", obs_count=100, unit="CurrencyUSD")
    eur_facet = Facet(earliest_date="2015", latest_date="2020", obs_count=5, unit="CurrencyEUR")
    request = IdentityUnitRequest(currency="USD", prices="current")
    bindings = {"unit": IdentityFacetValueRequest(request=request)}
    elected_units = {_SV: frozenset({"CurrencyUSD"})}
    graph = _graph([])
    counted = [
        f
        for f in (usd_facet, eur_facet)
        if facet_satisfies_bindings(
            graph, f, bindings, sv=_SV, elected_units=elected_units, deadline=None
        )
    ]
    assert sum(f.obs_count for f in counted) == 100


def test_no_candidate_facets_at_all_returns_none():
    """The caller handles an empty facet set as `no_observations`."""
    graph = _graph([])
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {"measurementMethod": IdentityBindingValue(ref="Commitment")},
        deadline=None,
    )
    assert result is None


def test_empty_sv_or_entities_returns_none():
    graph = _graph([{"earliestDate": "2015", "latestDate": "2020", "obsCount": 3}])
    binding = {"measurementMethod": IdentityBindingValue(ref="Commitment")}
    assert facet_refusal(graph, [], [_ENTITY], binding, deadline=None) is None
    assert facet_refusal(graph, [_SV], [], binding, deadline=None) is None


class TestMultiStatVarPerSvSatisfaction:
    """Each StatVar needs its own facet for a binding, or its series may be omitted downstream."""

    def _graph_multi(
        self, facets_a: list[dict], facets_b: list[dict], currency_units: dict | None = None
    ) -> FakeGraph:
        kwargs = {} if currency_units is None else {"currency_units": currency_units}
        return FakeGraph(
            obs={
                f"{_SV_A}|{_ENTITY}": facets_a,
                f"{_SV_B}|{_ENTITY}": facets_b,
            },
            **kwargs,
        )

    def test_binding_satisfied_by_only_one_stat_var_refuses_facet_not_available(self):
        """Refuses when only SV_A carries the bound measurement method."""
        graph = self._graph_multi(
            facets_a=[
                {
                    "earliestDate": "2015",
                    "latestDate": "2020",
                    "obsCount": 3,
                    "measurementMethod": "GrossDisbursement",
                }
            ],
            facets_b=[
                {
                    "earliestDate": "2015",
                    "latestDate": "2020",
                    "obsCount": 3,
                    "measurementMethod": "Commitment",
                }
            ],
        )
        result = facet_refusal(
            graph,
            [_SV_A, _SV_B],
            [_ENTITY],
            {"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")},
            deadline=None,
        )
        assert result is not None
        assert result.code == "facet_not_available"
        assert result.pointer == "/input/facets/measurementMethod"
        assert "GrossDisbursement" in str(result)

    def test_binding_satisfied_by_every_stat_var_returns_none(self):
        """Both StatVars carry the bound method, so no refusal is returned."""
        graph = self._graph_multi(
            facets_a=[
                {
                    "earliestDate": "2015",
                    "latestDate": "2020",
                    "obsCount": 3,
                    "measurementMethod": "GrossDisbursement",
                }
            ],
            facets_b=[
                {
                    "earliestDate": "2015",
                    "latestDate": "2020",
                    "obsCount": 2,
                    "measurementMethod": "GrossDisbursement",
                }
            ],
        )
        result = facet_refusal(
            graph,
            [_SV_A, _SV_B],
            [_ENTITY],
            {"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")},
            deadline=None,
        )
        assert result is None

    def test_unit_request_satisfied_by_only_one_stat_var_refuses(self):
        """The request-form counterpart: SV_B has no USD-satisfying facet at
        all, so the per-stat_var check must refuse even though SV_A alone
        would pass."""
        graph = self._graph_multi(
            facets_a=[
                {
                    "earliestDate": "2015",
                    "latestDate": "2020",
                    "obsCount": 3,
                    "unit": "CurrencyUSD",
                }
            ],
            facets_b=[
                {
                    "earliestDate": "2015",
                    "latestDate": "2020",
                    "obsCount": 3,
                    "unit": "CurrencyEUR",
                }
            ],
            currency_units={"CurrencyUSD": "USD", "CurrencyEUR": "EUR"},
        )
        request = IdentityUnitRequest(currency="USD", prices="current")
        result = facet_refusal(
            graph,
            [_SV_A, _SV_B],
            [_ENTITY],
            {"unit": IdentityFacetValueRequest(request=request)},
            deadline=None,
        )
        assert result is not None
        assert result.code == "facet_not_available"
        assert result.pointer == "/input/facets/unit"


def test_alternatives_lists_values_that_exist_for_the_pointed_axis():
    graph = _graph(
        [
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 5,
                "measurementMethod": "GrossDisbursement",
            },
            {
                "earliestDate": "2015",
                "latestDate": "2020",
                "obsCount": 2,
                "measurementMethod": "Commitment",
            },
        ]
    )
    result = facet_refusal(
        graph,
        [_SV],
        [_ENTITY],
        {"measurementMethod": IdentityBindingValue(ref="GrantEquivalent")},
        deadline=None,
    )
    assert result is not None
    assert result.alternatives is not None
    refs = [alt.ref for alt in result.alternatives]
    assert refs == ["GrossDisbursement", "Commitment"]  # ranked by obsCount, descending


class TestRoleIndexElectedUnits:
    """Role-index refusal helpers apply the same `elected_units` rules as `facet_refusal`."""

    def _role_index(self, unit: str) -> dict[str, list[RoleFacetCandidate]]:
        return {
            _SV: [
                role_candidate(
                    facet_id="facet/1",
                    role_tuples=frozenset(),
                    latest_date="2020",
                    provenance_id="src/A",
                    unit=unit,
                )
            ]
        }

    def test_role_index_facets_without_elected_units_follows_literal_rule(self):
        """Without an elected unit, EUR does not satisfy a USD request."""
        graph = FakeGraph(currency_units={"CurrencyEUR": "EUR", "CurrencyUSD": "USD"})
        request = IdentityUnitRequest(currency="USD", prices="current")
        result = candidate_facet_refusal(
            graph,
            role_index_facets(self._role_index("CurrencyEUR")),
            {"unit": IdentityFacetValueRequest(request=request)},
            deadline=None,
        )
        assert result is not None
        assert result.code == "facet_not_available"
        assert result.pointer == "/input/facets/unit"

    def test_role_index_facets_with_elected_units_admit_the_elected_facet(self):
        """The elected USD facet satisfies the request."""
        graph = FakeGraph(currency_units={"CurrencyUSD": "USD", "CurrencyEUR": "EUR"})
        request = IdentityUnitRequest(currency="EUR", prices="current")
        result = candidate_facet_refusal(
            graph,
            role_index_facets(self._role_index("CurrencyUSD")),
            {"unit": IdentityFacetValueRequest(request=request)},
            elected_units={_SV: frozenset({"CurrencyUSD"})},
            deadline=None,
        )
        assert result is None

    def test_role_index_facets_with_elected_units_refuse_a_facet_outside_the_set(self):
        """A literal EUR match fails when only USD is elected for the StatVar."""
        graph = FakeGraph(currency_units={"CurrencyEUR": "EUR", "CurrencyUSD": "USD"})
        request = IdentityUnitRequest(currency="EUR", prices="current")
        result = candidate_facet_refusal(
            graph,
            role_index_facets(self._role_index("CurrencyEUR")),
            {"unit": IdentityFacetValueRequest(request=request)},
            elected_units={_SV: frozenset({"CurrencyUSD"})},
            deadline=None,
        )
        assert result is not None
        assert result.code == "facet_not_available"
        assert result.pointer == "/input/facets/unit"


class TestRoleIndexProvenancePinElectedUnits:
    """A provenance member counts only on a facet that also satisfies the unit binding.

    The joint failure points at the provenance pin, the axis the shared check orders first.
    """

    def _role_index(self, unit: str) -> dict[str, list[RoleFacetCandidate]]:
        return {
            _SV: [
                role_candidate(
                    facet_id="facet/1",
                    role_tuples=frozenset(),
                    latest_date="2020",
                    provenance_id="src/A",
                    unit=unit,
                )
            ]
        }

    def _facets(self, request: IdentityUnitRequest) -> dict:
        return {
            "provenance": IdentityBindingValue(ref="src/A"),
            "unit": IdentityFacetValueRequest(request=request),
        }

    def test_provenance_pin_refusal_without_elected_units_follows_literal_rule(self):
        """Without an elected unit, EUR fails the USD request and the provenance check."""
        graph = FakeGraph(currency_units={"CurrencyEUR": "EUR", "CurrencyUSD": "USD"})
        request = IdentityUnitRequest(currency="USD", prices="current")
        result = candidate_facet_refusal(
            graph,
            role_index_facets(self._role_index("CurrencyEUR")),
            self._facets(request),
            deadline=None,
        )
        assert result is not None
        assert result.code == "facet_not_available"
        assert result.pointer == "/input/facets/provenance"
        assert "unit" in str(result)

    def test_provenance_pin_refusal_with_elected_units_admits_the_elected_facet(self):
        graph = FakeGraph(currency_units={"CurrencyUSD": "USD", "CurrencyEUR": "EUR"})
        request = IdentityUnitRequest(currency="EUR", prices="current")
        result = candidate_facet_refusal(
            graph,
            role_index_facets(self._role_index("CurrencyUSD")),
            self._facets(request),
            elected_units={_SV: frozenset({"CurrencyUSD"})},
            deadline=None,
        )
        assert result is None

    def test_provenance_pin_refusal_with_elected_units_refuses_a_facet_outside_the_set(self):
        """A unit outside the elected set fails the provenance pin's joint check."""
        graph = FakeGraph(currency_units={"CurrencyEUR": "EUR", "CurrencyUSD": "USD"})
        request = IdentityUnitRequest(currency="EUR", prices="current")
        result = candidate_facet_refusal(
            graph,
            role_index_facets(self._role_index("CurrencyEUR")),
            self._facets(request),
            elected_units={_SV: frozenset({"CurrencyUSD"})},
            deadline=None,
        )
        assert result is not None
        assert result.code == "facet_not_available"
        assert result.pointer == "/input/facets/provenance"
        assert "unit" in str(result)


def _method_role_index(*methods: str | None) -> dict[str, list[RoleFacetCandidate]]:
    return {
        _SV: [
            role_candidate(
                facet_id=f"F{i}",
                role_tuples=frozenset({(("recipient", "country/KEN"),)}),
                latest_date="2024",
                provenance_id="dc/base/TestIDS",
                measurement_method=method,
                unit="USDollar",
            )
            for i, method in enumerate(methods)
        ]
    }


def test_null_method_ref_passes_when_a_facet_declares_no_method():
    result = candidate_facet_refusal(
        FakeGraph(),
        role_index_facets(_method_role_index(None, "Estimated")),
        {"measurementMethod": IdentityFacetValueAbsent(ref=None)},
        deadline=None,
    )
    assert result is None


def test_null_method_ref_refuses_when_every_facet_declares_a_method():
    result = candidate_facet_refusal(
        FakeGraph(),
        role_index_facets(_method_role_index("Estimated")),
        {"measurementMethod": IdentityFacetValueAbsent(ref=None)},
        deadline=None,
    )
    assert result is not None
    assert result.code == "facet_not_available"
    assert result.pointer == "/input/facets/measurementMethod"
    assert "a facet with no measurementMethod" in str(result)
    assert [a.ref for a in result.alternatives or []] == ["Estimated"]


@pytest.mark.parametrize(
    ("axis", "binding"),
    [
        ("measurementMethod", IdentityFacetValueAbsent(ref=None)),
        ("provenance", IdentityBindingValue(ref="dc/base/TestIDS")),
        ("measurementMethod", IdentityBindingSet(refs=["Estimated"])),
    ],
    ids=["null-ref", "ref", "set-member"],
)
def test_member_presence_follows_binding_admits_value(monkeypatch, axis, binding):
    """The per-StatVar member check matches through `binding_admits_value` and no other
    equality: the index carries every bound value, so the refusal comes from the rule."""
    index = _method_role_index(None, "Estimated")
    assert (
        candidate_facet_refusal(
            FakeGraph(), role_index_facets(index), {axis: binding}, deadline=None
        )
        is None
    )

    monkeypatch.setattr(
        "search.engine.facet_refusals.binding_admits_value", lambda binding, value: False
    )

    refusal = candidate_facet_refusal(
        FakeGraph(), role_index_facets(index), {axis: binding}, deadline=None
    )
    assert refusal is not None
    assert refusal.code == "facet_not_available"
    assert refusal.pointer == f"/input/facets/{axis}"
