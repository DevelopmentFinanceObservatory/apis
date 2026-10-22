"""Tests bound-unit conversion decisions on text and promote requests.
Fixtures cover native USD facets, Nominal qualifiers, and conversion recipes.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from unittest.mock import patch

import pytest

from materialise.facets import ElectedDataset, FacetSelection
from materialise.graph import SubjectPage
from materialise.rate_tables import RateTableCache
from materialise.recipes import CONVERSION_NODE_PROPERTIES
from search.engine.errors import EngineInfraError, EngineInputError
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_spec_resubmit, resolve_variable
from search.engine.region.convertibility import decide_part_unit, elected_unit_refs
from search.engine.region.state import EngineOptions
from search.models import (
    IdentityBindingSet,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    ResubmitIdentityPayload,
)
from tests.engine import _answer_invariants_world as world_module
from tests.engine._base_identity import make_identity, subject_entity
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs, walk_options, with_member_arcs
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver


class _SoleShapeLLM:
    """Elects the sole (unconstrained) candidate shape, no bindings needed."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            elected_reply(0, bindings=[], also_plausible=None),
            None,
        )


_ENTITY = "country/KEN"
_SV = "Test_Convert_Expenditure"
_PROVENANCE = "dc/base/ConvertSrc"
_OTHER_PROVENANCE = "dc/base/ConvertSrcOther"
_RECIPE = "ConversionRecipeDAC"
_QUERY = "convertibility test variable in Kenya"
# Every direct decide_part_unit() call in this file declares
# ("observationAbout",), subject-only, so `coverage` is never read on that
# path (materialise.facets.probe_facets' role-bearing branch alone reads
# it) -- an empty index, never consulted, satisfies the required argument.
_SUBJECT_ONLY_COVERAGE = build_coverage_index([], role_names={})

_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "TestConvertActivity",
    "measured_property": "testConvertExpenditure",
    "stat_type": "measuredValue",
}

_SV_ARCS: dict = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "populationType": {"nodes": [{"dcid": "TestConvertActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testConvertExpenditure"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}

_NODES: dict = {
    _ENTITY: {"label": "Kenya", "type": "Country"},
    _SV: {"label": "Convert expenditure", "type": "StatisticalVariable", "arcs": _SV_ARCS},
}

_CURRENCY_ARCS = {
    "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
    "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
    # A withdrawn currency is a CurrencyUnit node that no rate table serves.
    "CurrencyFRF": {"typeOf": ("CurrencyUnit",)},
}


def _member_arcs(node_arcs: dict) -> dict[str, tuple[str, ...]]:
    """Mirrors test_facet_unit_gate_multi_statvar.py's own helper: this
    fixture's `{"prop": {"nodes": [...]}}` node-arc shape, flattened into
    the `{property: (value, ...)}` tuple form FakeGraphReader expects."""
    return {
        prop: tuple(n.get("dcid") or n.get("value") for n in entry["nodes"])
        for prop, entry in node_arcs.items()
    }


def _sv_arcs(*, nominal: bool, qualifier: str | None = None) -> dict[str, tuple[str, ...]]:
    """`qualifier` names a declared `measurementQualifier` other than
    `Nominal` (e.g. `"RealValue"`), for proving a natively satisfied
    request still refuses a declared non-nominal qualifier; ignored when
    `nominal` is `True`. `nominal=False, qualifier=None` leaves the StatVar
    carrying no `measurementQualifier` arc at all."""
    arcs = {**_member_arcs(_SV_ARCS), "populationType": ("FinancialFlow",)}
    if nominal:
        arcs["measurementQualifier"] = ("Nominal",)
    elif qualifier is not None:
        arcs["measurementQualifier"] = (qualifier,)
    return arcs


def _materialise_arcs(
    *,
    nominal: bool,
    provenance_declares_recipe: dict[str, bool],
    qualifier: str | None = None,
) -> dict[str, dict]:
    """`provenance_declares_recipe` maps every provenance id this test's own
    obs facets carry to whether IT declares a conversionRecipe -- lets a
    fixture put the recipe on a non-elected provenance to prove which facet
    the decision actually checks."""
    arcs: dict[str, dict] = {_SV: _sv_arcs(nominal=nominal, qualifier=qualifier), **_CURRENCY_ARCS}
    for provenance_id, declares in provenance_declares_recipe.items():
        arcs[provenance_id] = {"conversionRecipe": (_RECIPE,)} if declares else {}
    return arcs


def _obs_facet(
    *,
    unit: str | None,
    provenance_id: str | None = _PROVENANCE,
    facet_id: str = "f1",
    value: float = 10.0,
    obs_count: int = 1,
    measurement_method: str | None = None,
) -> dict:
    return {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": obs_count,
        "provenanceId": provenance_id,
        "measurementMethod": measurement_method,
        "unit": unit,
        "dates": ["2020"],
        "values": [value],
        "facetId": facet_id,
    }


def _run_text(
    *,
    unit_text: str,
    obs_facets: list[dict],
    nominal: bool,
    provenance_declares_recipe: dict[str, bool],
    qualifier: str | None = None,
    unit_text_best_effort: bool = False,
):
    obs = {f"{_SV}|{_ENTITY}": obs_facets}
    graph = FakeGraph(
        nodes=_NODES,
        detect={_QUERY: {"svs": [_SV], "cosine_scores": [0.95]}},
        resolve={"Kenya": _ENTITY},
        obs=obs,
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=nominal, provenance_declares_recipe=provenance_declares_recipe, qualifier=qualifier
    )
    materialise_graph._subjects = {
        ("testConvertExpenditure", "measuredProperty"): SubjectPage(dcids=(_SV,), truncated=False),
    }
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_SoleShapeLLM(),
            base_timing={},
            unit_text=unit_text,
            unit_text_best_effort=unit_text_best_effort,
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
            ),
        )
    )


def test_native_currency_wins_over_a_lower_ranked_non_recipe_facet():
    """The top-ranked USD facet wins over an earlier-listed EUR facet without a recipe.
    The resolve is definite on both text and promote doors.
    """
    result = _run_text(
        unit_text="USD",
        obs_facets=[
            _obs_facet(unit="CurrencyEUR", facet_id="f_eur", value=3.0, obs_count=1),
            _obs_facet(unit="CurrencyUSD", facet_id="f_usd", value=1.0, obs_count=3),
        ],
        nominal=True,
        provenance_declares_recipe={_PROVENANCE: False},
    )
    assert result.status == "definite", f"expected definite, got {result.status!r}"


def test_reverse_elected_facet_lacks_recipe_refuses_unsupported_with_cause():
    """The top-ranked facet lacks a recipe, although a lower-ranked facet has one.
    A converting EUR request refuses with `recipe_not_declared` and its cause.
    """
    result = _run_text(
        unit_text="EUR",
        obs_facets=[
            _obs_facet(
                unit="CurrencyUSD",
                provenance_id=_PROVENANCE,
                facet_id="f1",
                value=10.0,
                obs_count=3,
            ),
            _obs_facet(
                unit="CurrencyUSD",
                provenance_id=_OTHER_PROVENANCE,
                facet_id="f2",
                value=1.0,
                obs_count=1,
            ),
        ],
        nominal=True,
        provenance_declares_recipe={_PROVENANCE: False, _OTHER_PROVENANCE: True},
    )
    assert result.status == "unsupported", f"expected unsupported, got {result.status!r}"
    assert result.unsupported is not None
    assert result.unsupported.code == "recipe_not_declared"
    assert result.unsupported.cause is not None
    assert result.unsupported.cause.code == "recipe_undeclared"


def test_no_qualifier_arc_resolves_definite_on_a_natively_satisfied_request():
    """A native USD match resolves without a measurementQualifier arc because no conversion runs."""
    result = _run_text(
        unit_text="USD",
        obs_facets=[_obs_facet(unit="CurrencyUSD")],
        nominal=False,
        provenance_declares_recipe={_PROVENANCE: False},
    )
    assert result.status == "definite", f"expected definite, got {result.status!r}"


def test_declared_non_nominal_qualifier_refuses_unsupported_with_cause():
    """A declared `RealValue` qualifier refuses a native USD request with `nominal_undeclared`.
    A missing qualifier arc permits the native request.
    """
    result = _run_text(
        unit_text="USD",
        obs_facets=[_obs_facet(unit="CurrencyUSD")],
        nominal=False,
        qualifier="RealValue",
        provenance_declares_recipe={_PROVENANCE: False},
    )
    assert result.status == "unsupported", f"expected unsupported, got {result.status!r}"
    assert result.unsupported is not None
    assert result.unsupported.code == "recipe_not_declared"
    assert result.unsupported.cause is not None
    assert result.unsupported.cause.code == "nominal_undeclared"


def test_provenance_default_undefined_refuses_facet_not_available():
    """Missing provenance and default policy raises `facet_not_available` before conversion checks.
    The fixture also has no nominal qualifier, confirming the earlier refusal.
    """
    result = _run_text(
        unit_text="USD",
        obs_facets=[_obs_facet(unit="CurrencyUSD", provenance_id=None)],
        nominal=False,
        provenance_declares_recipe={},
    )
    assert result.status == "no_data", f"expected no_data, got {result.status!r}"
    assert result.no_data_reason == "facet_not_available"


def test_no_unit_at_all_refuses_facet_binding_unsupported():
    """A candidate with no unit ref raises `NonCurrencyUnitError`.
    Search maps the refusal to `facet_binding_unsupported`.
    """
    result = _run_text(
        unit_text="USD",
        obs_facets=[_obs_facet(unit=None)],
        nominal=True,
        provenance_declares_recipe={_PROVENANCE: False},
    )
    assert result.status == "unsupported", f"expected unsupported, got {result.status!r}"
    assert result.unsupported is not None
    assert result.unsupported.code == "facet_binding_unsupported"


def test_a_typed_unit_phrase_the_data_cannot_convert_to_answers_without_it():
    """The raw-text door's typed unit phrase is best effort. Where the explicit request above
    refuses, the leg answers unconverted, binds no unit, and departs `measure_not_applied`
    for the phrase. Data in no currency has no unit token, so no one slot clears it."""
    result = _run_text(
        unit_text="in US dollars",
        obs_facets=[_obs_facet(unit=None)],
        nominal=True,
        provenance_declares_recipe={_PROVENANCE: False},
        unit_text_best_effort=True,
    )
    assert result.status == "definite", result.no_data_reason
    assert result.specs[0].definition.identity.facets is None
    assert [(f.code, f.binding_ref, f.asked) for f in result.departure_facts.values()] == [
        ("measure_not_applied", None, "in US dollars")
    ]


def test_missing_materialise_graph_raises_infra_error():
    """A bound unit request with no materialise_graph at all raises
    EngineInfraError, whether or not any facet was found."""
    obs = {f"{_SV}|{_ENTITY}": [_obs_facet(unit="CurrencyUSD")]}
    graph = FakeGraph(
        nodes=_NODES,
        detect={_QUERY: {"svs": [_SV], "cosine_scores": [0.95]}},
        resolve={"Kenya": _ENTITY},
        obs=obs,
    )
    with pytest.raises(EngineInfraError):
        asyncio.run(
            resolve_variable(
                _QUERY,
                entities=["Kenya"],
                date_request=None,
                detect_query=_QUERY,
                role_query=_QUERY,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=_SoleShapeLLM(),
                base_timing={},
                unit_text="USD",
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(include_observations=True),
            )
        )


def test_ref_form_unit_is_unaffected():
    """A ref-form unit makes no conversion decision and resolves definite."""
    obs = {f"{_SV}|{_ENTITY}": [_obs_facet(unit="CurrencyUSD")]}
    graph = FakeGraph(
        nodes=_NODES,
        detect={_QUERY: {"svs": [_SV], "cosine_scores": [0.95]}},
        resolve={"Kenya": _ENTITY},
        obs=obs,
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=False, provenance_declares_recipe={_PROVENANCE: False}
    )
    with patch(
        "search.engine.region.phase_answer.decide_part_unit",
        side_effect=AssertionError("decide_part_unit must not be called with no unit request"),
    ) as spy:
        result = asyncio.run(
            resolve_variable(
                _QUERY,
                entities=["Kenya"],
                date_request=None,
                detect_query=_QUERY,
                role_query=_QUERY,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=_SoleShapeLLM(),
                base_timing={},
                unit_text=None,
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(
                    include_observations=True,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                ),
            )
        )
    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spy.assert_not_called()


# ---------------------------------------------------------------------------
# Discovered-scope role-bearing resolve: the decision still runs
# ---------------------------------------------------------------------------
#
# Reuses test_role_scope_pipeline_e2e.py's own fixture helpers
# (_coverage_seed_for/_facet/_declaring_arcs/_graph_for), which already prove
# discover_role_scope's own coverage-lookup discovery wiring -- this test's
# own concern is only that the convertibility decision reaches the
# discovered entities, not re-proving discovery itself.

_ROLE_SV_DCID = "TESTFIXTURE/Convertibility/RoleFlow"
_ROLE_FRA = "country/FRA"
_ROLE_KEN = "country/KEN"


def test_discovered_scope_role_bearing_resolve_runs_the_decision(monkeypatch):
    """The unit decision receives the destination discovered from an origin-bound query.
    The spy wraps the real function to verify the decision runs.
    """
    from series_coverage.schema import FacetMetadata
    from tests.engine._fake_coverage import patch_coverage
    from tests.engine._harness import (
        ObsFacetFixture,
        build_materialise_double,
        door_roles_for_test_query,
    )
    from tests.engine.test_role_scope_pipeline_e2e import (
        _coverage_seed_for,
        _declaring_arcs,
        _facet,
        _FixedBindLLM,
        _graph_for,
    )

    filters = {"creditor": [_ROLE_FRA]}
    rows, role_names = _coverage_seed_for("debtor", [_ROLE_KEN], [_ROLE_SV_DCID], filters=filters)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        rows,
        role_names=role_names,
        facet_metadata={"f1": FacetMetadata(provenance_id=_PROVENANCE, unit="CurrencyUSD")},
    )
    observations = {_ROLE_SV_DCID: {(("creditor", _ROLE_FRA), ("debtor", _ROLE_KEN)): [_facet()]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})
    sv_arcs = {
        **_declaring_arcs(observation_properties=["creditor", "debtor"]),
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
    }
    graph = _graph_for(
        sv_dcid=_ROLE_SV_DCID,
        sv_arcs=sv_arcs,
        resolve={"France": _ROLE_FRA},
        role_query="aid from France",
        sv_score=0.9,
    )

    materialise_graph, materialise_sdmx, _ = build_materialise_double(
        {
            (
                _ROLE_SV_DCID,
                (("creditor", _ROLE_FRA), ("debtor", _ROLE_KEN)),
            ): [
                ObsFacetFixture(
                    facet_id="f1",
                    points=(("2020", 10.0),),
                    provenance_id=_PROVENANCE,
                    unit="CurrencyUSD",
                )
            ]
        }
    )
    materialise_graph._arcs = {
        _ROLE_SV_DCID: {"measurementQualifier": ("Nominal",)},
        **_CURRENCY_ARCS,
        _PROVENANCE: {},
    }

    from search.engine.region import convertibility as convertibility_module

    real_decide_part_unit = convertibility_module.decide_part_unit
    with patch(
        "search.engine.region.phase_answer.decide_part_unit", side_effect=real_decide_part_unit
    ) as spy:
        result = asyncio.run(
            resolve_variable(
                "aid",
                entities=["France"],
                date_request=None,
                detect_query="aid from France",
                role_query="aid from France",
                entity_roles=door_roles_for_test_query("aid from France", ["France"]),
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=sdmx,
                llm=_FixedBindLLM(),
                base_timing={},
                unit_text="USD",
                coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
                options=EngineOptions(
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    rate_cache=RateTableCache(),
                ),
            )
        )
    spy.assert_called_once()
    _, kwargs = spy.call_args
    assert kwargs["role_bindings"], "the discovered destination entity must reach the decision"
    assert result.status == "definite", f"expected definite, got {result.status!r}"


# ---------------------------------------------------------------------------
# Promote door (spec resubmit)
# ---------------------------------------------------------------------------


def _resubmit(
    *,
    request: IdentityUnitRequest | None,
    obs_facets: list[dict],
    nominal: bool,
    provenance_declares_recipe: dict[str, bool],
    provenance_refs: Sequence[str] | None = None,
    include_materialise: bool = True,
):
    obs = {f"{_SV}|{_ENTITY}": obs_facets}
    # The graph Search reads states the StatVar as materialise does, so the resubmit's member
    # walk elects it for an identity of its own five-tuple.
    sv_arcs = _sv_arcs(nominal=nominal)
    graph = FakeGraph(
        nodes={
            **_NODES,
            _SV: {
                "label": "Convert expenditure",
                "type": "StatisticalVariable",
                "arcs": {
                    prop: {"nodes": [{"dcid": value} for value in values]}
                    for prop, values in sv_arcs.items()
                },
            },
        },
        obs=obs,
    )
    shape = {
        **_SHAPE,
        "population_type": "FinancialFlow",
        **({"measurement_qualifier": "Nominal"} if nominal else {}),
    }
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=nominal, provenance_declares_recipe=provenance_declares_recipe
    )
    materialise_graph._subjects = {
        ("testConvertExpenditure", "measuredProperty"): SubjectPage(dcids=(_SV,), truncated=False),
    }
    entities = [subject_entity(_ENTITY)]
    facets: dict[str, IdentityFacetValueRequest | IdentityBindingSet] = {}
    if request is not None:
        facets["unit"] = IdentityFacetValueRequest(request=request)
    if provenance_refs is not None:
        # Post provenance refs directly on the identity facet.
        facets["provenance"] = IdentityBindingSet(refs=list(provenance_refs))
    base_identity = make_identity(shape, slots=[], entities=entities)
    inp = ResubmitIdentityPayload(
        identity=base_identity.model_copy(update={"facets": facets}) if facets else base_identity,
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            graph,
            EngineOptions(
                include_observations=True,
                materialise_graph=with_member_arcs(
                    materialise_graph if include_materialise else None, graph
                ),
                materialise_sdmx=materialise_sdmx if include_materialise else None,
                # Elected-mode planning requires a RateTableCache, even for a native match.
                rate_cache=RateTableCache() if request is not None else None,
            ),
        ),
    )


def test_resubmit_reverse_elected_facet_raises_422_with_cause():
    """Mirrors the text-door reverse scenario: the elected (top-ranked)
    facet's own provenance declares no recipe; raises
    EngineInputError(recipe_not_declared, 422), cause recipe_undeclared,
    pointer rewritten onto the nested identity."""
    request = IdentityUnitRequest(currency="EUR", prices="current")
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(
            request=request,
            obs_facets=[
                _obs_facet(
                    unit="CurrencyUSD", provenance_id=_PROVENANCE, facet_id="f1", value=10.0
                ),
                _obs_facet(
                    unit="CurrencyUSD", provenance_id=_OTHER_PROVENANCE, facet_id="f2", value=1.0
                ),
            ],
            nominal=True,
            provenance_declares_recipe={_PROVENANCE: False, _OTHER_PROVENANCE: True},
        )
    assert exc_info.value.code == "recipe_not_declared"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/input/identity/facets/unit"
    assert exc_info.value.cause is not None
    assert exc_info.value.cause.code == "recipe_undeclared"


def test_resubmit_native_currency_is_definite():
    """A native `{USD, current}` request on spec resubmit is definite, same
    as the text door -- USD is the true top-ranked (higher obs_count)
    candidate the real preview independently serves too."""
    request = IdentityUnitRequest(currency="USD", prices="current")
    region = _resubmit(
        request=request,
        obs_facets=[
            _obs_facet(unit="CurrencyEUR", facet_id="f_eur", value=5.0, obs_count=1),
            _obs_facet(unit="CurrencyUSD", facet_id="f_usd", value=1.0, obs_count=3),
        ],
        nominal=True,
        provenance_declares_recipe={_PROVENANCE: False},
    )
    assert region.status == "definite", f"expected definite, got {region.status!r}"


def test_resubmit_no_unit_request_makes_no_decision_call():
    """No unit binding at all (a plain resubmit) makes no decision call."""
    with patch(
        "search.engine.region.promote_ground.decide_part_unit",
        side_effect=AssertionError("decide_part_unit must not be called with no unit request"),
    ) as spy:
        region = _resubmit(
            request=None,
            obs_facets=[_obs_facet(unit="CurrencyUSD")],
            nominal=False,
            provenance_declares_recipe={_PROVENANCE: False},
        )
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    spy.assert_not_called()


def test_resubmit_missing_materialise_graph_raises_infra_error():
    """Mirrors the text door's own test_missing_materialise_graph_raises_
    infra_error: a bound unit request with no materialise_graph/
    materialise_sdmx at all raises EngineInfraError on the promote door
    too."""
    request = IdentityUnitRequest(currency="USD", prices="current")
    with pytest.raises(EngineInfraError):
        _resubmit(
            request=request,
            obs_facets=[_obs_facet(unit="CurrencyUSD")],
            nominal=True,
            provenance_declares_recipe={_PROVENANCE: False},
            include_materialise=False,
        )


def test_resubmit_provenance_default_undefined_raises_facet_not_available_422():
    """Promote-door mirror: undeclared-provenance-default raises
    DefaultFacetUndefinedError, mapped to EngineInputError(facet_not_available, 422)."""
    request = IdentityUnitRequest(currency="USD", prices="current")
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(
            request=request,
            obs_facets=[_obs_facet(unit="CurrencyUSD", provenance_id=None)],
            nominal=False,
            provenance_declares_recipe={},
        )
    assert exc_info.value.code == "facet_not_available"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/input/identity/facets/unit"
    assert exc_info.value.cause is None


def test_resubmit_no_unit_at_all_raises_facet_binding_unsupported_422():
    """Promote-door mirror: the sole candidate facet carries no unit dcid,
    and its provenance/measurementMethod both resolve, so
    decide_member_unit's own preamble re-raises NonCurrencyUnitError
    unchanged, mapped to
    EngineInputError(facet_binding_unsupported, 422)."""
    request = IdentityUnitRequest(currency="USD", prices="current")
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(
            request=request,
            obs_facets=[_obs_facet(unit=None)],
            nominal=True,
            provenance_declares_recipe={_PROVENANCE: False},
        )
    assert exc_info.value.code == "facet_binding_unsupported"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/input/identity/facets/unit"


def test_decide_part_unit_native_currency_returns_native_ref():
    """A single-ref binding, natively satisfied (`{USD,current}` on a
    native `CurrencyUSD` facet, no conversion): the returned mapping
    carries one decided combination, whose `"unit"` selection is that
    facet's own ref for the sv; `elected_unit_refs` derives the same
    ref as before."""
    obs = {f"{_SV}|{_ENTITY}": [_obs_facet(unit="CurrencyUSD")]}
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=True, provenance_declares_recipe={_PROVENANCE: False}
    )
    elected = decide_part_unit(
        [_SV],
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        role_bindings={"observationAbout": [_ENTITY]},
        declared_roles=("observationAbout",),
        bound_facets={"provenance": _PROVENANCE},
        years=None,
        graph=materialise_graph,
        sdmx=materialise_sdmx,
        coverage=_SUBJECT_ONLY_COVERAGE,
    )
    assert len(elected[_SV]) == 1
    assert elected[_SV][0]["unit"].ref == "CurrencyUSD"
    assert elected_unit_refs(elected) == {_SV: frozenset({"CurrencyUSD"})}


@pytest.mark.parametrize(
    ("elected", "decided"),
    [(_PROVENANCE, _PROVENANCE), (None, _OTHER_PROVENANCE)],
    ids=["elected_dataset", "no_election"],
)
def test_decide_part_unit_probes_within_the_members_elected_dataset(elected, decided):
    """With no plan probe the decision probes for itself. The other dataset's facet holds
    more points and wins a plain ranking; a member whose family elected `_PROVENANCE`
    decides on that dataset's facet, as a mint of it does."""
    obs = {
        f"{_SV}|{_ENTITY}": [
            _obs_facet(unit="CurrencyUSD", provenance_id=_OTHER_PROVENANCE, facet_id="f-other")
            | {"obsCount": 9},
            _obs_facet(unit="CurrencyUSD", facet_id="f-elected"),
        ]
    }
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=True,
        provenance_declares_recipe={_PROVENANCE: False, _OTHER_PROVENANCE: False},
    )
    datasets = (
        {
            _SV: ElectedDataset(
                provenance=elected, member_provenances=(_PROVENANCE, _OTHER_PROVENANCE)
            )
        }
        if elected is not None
        else {}
    )

    decision = decide_part_unit(
        [_SV],
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        role_bindings={"observationAbout": [_ENTITY]},
        declared_roles=("observationAbout",),
        years=None,
        graph=materialise_graph,
        sdmx=materialise_sdmx,
        coverage=_SUBJECT_ONLY_COVERAGE,
        datasets=datasets,
    )

    (selections,) = decision[_SV]
    assert selections["provenance"].ref == decided


def test_decide_part_unit_converting_facet_returns_its_native_ref():
    """A single-ref binding that converts (`{EUR,current}` on a native
    `CurrencyUSD` facet whose provenance declares a recipe): the decided
    combination's `"unit"` selection still carries the elected facet's own
    native unit ref, not the requested currency -- the same ref a native
    request would elect; `elected_unit_refs` derives the same ref as before."""
    obs = {f"{_SV}|{_ENTITY}": [_obs_facet(unit="CurrencyUSD")]}
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=True, provenance_declares_recipe={_PROVENANCE: True}
    )
    elected = decide_part_unit(
        [_SV],
        unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
        role_bindings={"observationAbout": [_ENTITY]},
        declared_roles=("observationAbout",),
        bound_facets={"provenance": _PROVENANCE},
        years=None,
        graph=materialise_graph,
        sdmx=materialise_sdmx,
        coverage=_SUBJECT_ONLY_COVERAGE,
    )
    assert len(elected[_SV]) == 1
    assert elected[_SV][0]["unit"].ref == "CurrencyUSD"
    assert elected_unit_refs(elected) == {_SV: frozenset({"CurrencyUSD"})}


def test_decide_part_unit_withdrawn_native_currency_passes_the_decision():
    """A native `CurrencyFRF` that no rate table serves still resolves as a `CurrencyUnit`.
    The decision converts it and elects its own native ref. The missing rate
    surfaces only when the tables are applied.
    """
    obs = {f"{_SV}|{_ENTITY}": [_obs_facet(unit="CurrencyFRF")]}
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=True, provenance_declares_recipe={_PROVENANCE: True}
    )
    elected = decide_part_unit(
        [_SV],
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        role_bindings={"observationAbout": [_ENTITY]},
        declared_roles=("observationAbout",),
        bound_facets={"provenance": _PROVENANCE},
        years=None,
        graph=materialise_graph,
        sdmx=materialise_sdmx,
        coverage=_SUBJECT_ONLY_COVERAGE,
    )
    assert elected_unit_refs(elected) == {_SV: frozenset({"CurrencyFRF"})}


def test_decide_part_unit_null_method_ref_elects_the_facet_without_a_method():
    """A null measurementMethod ref is one combination that binds the facet declaring no
    method. The Estimated facet has more observations, so an unbound method would elect it."""
    obs = {
        f"{_SV}|{_ENTITY}": [
            _obs_facet(unit="CurrencyUSD", facet_id="f-recorded"),
            _obs_facet(
                unit="CurrencyUSD",
                facet_id="f-estimated",
                measurement_method="Estimated",
                obs_count=5,
            ),
        ]
    }
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=True, provenance_declares_recipe={_PROVENANCE: False}
    )

    def decide(bound_facets):
        return decide_part_unit(
            [_SV],
            unit_request=IdentityUnitRequest(currency="USD", prices="current"),
            role_bindings={"observationAbout": [_ENTITY]},
            declared_roles=("observationAbout",),
            bound_facets=bound_facets,
            years=None,
            graph=materialise_graph,
            sdmx=materialise_sdmx,
            coverage=_SUBJECT_ONLY_COVERAGE,
        )

    (pinned,) = decide({"measurementMethod": None})[_SV]
    (unpinned,) = decide({})[_SV]

    assert (pinned["measurementMethod"].ref, pinned["measurementMethod"].basis) == (None, "bound")
    assert pinned["unit"].dispatch_facet_id == "f-recorded"
    assert unpinned["unit"].dispatch_facet_id == "f-estimated"


def test_decide_part_unit_set_second_member_refusal_raises_after_first_converts():
    """A second Set member's `RecipeNotDeclaredError` propagates after the first converts."""
    from materialise.recipes import RecipeNotDeclaredError

    obs = {
        f"{_SV}|{_ENTITY}": [
            _obs_facet(unit="CurrencyUSD", provenance_id=_PROVENANCE, facet_id="f1"),
            _obs_facet(unit="CurrencyUSD", provenance_id=_OTHER_PROVENANCE, facet_id="f2"),
        ]
    }
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=True,
        provenance_declares_recipe={_PROVENANCE: True, _OTHER_PROVENANCE: False},
    )
    with pytest.raises(RecipeNotDeclaredError):
        decide_part_unit(
            [_SV],
            unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
            role_bindings={"observationAbout": [_ENTITY]},
            declared_roles=("observationAbout",),
            bound_facets={"provenance": [_PROVENANCE, _OTHER_PROVENANCE]},
            years=None,
            graph=materialise_graph,
            sdmx=materialise_sdmx,
            coverage=_SUBJECT_ONLY_COVERAGE,
        )


def test_resubmit_provenance_set_both_members_lack_recipe_raises_recipe_not_declared():
    """Neither candidate facet declares a recipe for an EUR conversion request.
    The first `RecipeNotDeclaredError` is propagated, not skipped.
    """
    request = IdentityUnitRequest(currency="EUR", prices="current")
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(
            request=request,
            obs_facets=[
                _obs_facet(
                    unit="CurrencyUSD", provenance_id=_PROVENANCE, facet_id="f1", obs_count=3
                ),
                _obs_facet(
                    unit="CurrencyUSD", provenance_id=_OTHER_PROVENANCE, facet_id="f2", obs_count=1
                ),
            ],
            nominal=True,
            provenance_declares_recipe={_PROVENANCE: False, _OTHER_PROVENANCE: False},
            provenance_refs=[_PROVENANCE, _OTHER_PROVENANCE],
        )
    assert exc_info.value.code == "recipe_not_declared"
    assert exc_info.value.status == 422


def test_decide_part_unit_provenance_set_one_member_has_no_candidate_succeeds():
    """Tests `decide_part_unit`'s per-member fallback directly.
    Only `_OTHER_PROVENANCE` has a facet; the empty first result raises `DefaultFacetUndefinedError`.
    The USD match returns the second member's unit ref. A full promote request still fails because
    `facet_refusals` requires a candidate for every posted set member.
    """
    obs = {f"{_SV}|{_ENTITY}": [_obs_facet(unit="CurrencyUSD", provenance_id=_OTHER_PROVENANCE)]}
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=True, provenance_declares_recipe={_OTHER_PROVENANCE: False}
    )
    elected = decide_part_unit(
        [_SV],
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        role_bindings={"observationAbout": [_ENTITY]},
        declared_roles=("observationAbout",),
        bound_facets={"provenance": [_PROVENANCE, _OTHER_PROVENANCE]},
        years=None,
        graph=materialise_graph,
        sdmx=materialise_sdmx,
        coverage=_SUBJECT_ONLY_COVERAGE,
    )
    assert len(elected[_SV]) == 1
    assert elected[_SV][0]["unit"].ref == "CurrencyUSD"
    assert elected_unit_refs(elected) == {_SV: frozenset({"CurrencyUSD"})}


def test_decide_part_unit_two_member_set_returns_each_combinations_own_provenance_ref():
    """Each set member keeps its own provenance ref in `bound_facets` order."""
    obs = {
        f"{_SV}|{_ENTITY}": [
            _obs_facet(unit="CurrencyUSD", provenance_id=_PROVENANCE, facet_id="f1"),
            _obs_facet(unit="CurrencyUSD", provenance_id=_OTHER_PROVENANCE, facet_id="f2"),
        ]
    }
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=True,
        provenance_declares_recipe={_PROVENANCE: False, _OTHER_PROVENANCE: False},
    )
    elected = decide_part_unit(
        [_SV],
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        role_bindings={"observationAbout": [_ENTITY]},
        declared_roles=("observationAbout",),
        bound_facets={"provenance": [_PROVENANCE, _OTHER_PROVENANCE]},
        years=None,
        graph=materialise_graph,
        sdmx=materialise_sdmx,
        coverage=_SUBJECT_ONLY_COVERAGE,
    )
    assert [selections["provenance"].ref for selections in elected[_SV]] == [
        _PROVENANCE,
        _OTHER_PROVENANCE,
    ]


def test_elected_unit_refs_skips_a_selection_with_a_none_unit_ref():
    """A missing unit ref yields an empty set for the StatVar, not a missing key."""
    elected = {
        _SV: (
            {
                "unit": FacetSelection(
                    axis="unit",
                    ref=None,
                    basis="declared_default",
                    default_policy=None,
                    default_policy_version=None,
                    selection_reason="absent_on_selected_facet",
                    dispatch_facet_id="f1",
                )
            },
        )
    }
    assert elected_unit_refs(elected) == {_SV: frozenset()}


def test_resubmit_provenance_set_no_member_has_candidate_raises_facet_not_available():
    """Neither posted provenance has a facet, so the request raises `facet_not_available` (422)."""
    request = IdentityUnitRequest(currency="USD", prices="current")
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(
            request=request,
            obs_facets=[
                _obs_facet(unit="CurrencyUSD", provenance_id="dc/base/ConvertSrcThird"),
            ],
            nominal=True,
            provenance_declares_recipe={},
            provenance_refs=[_PROVENANCE, _OTHER_PROVENANCE],
        )
    assert exc_info.value.code == "facet_not_available"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/input/identity/facets/unit"


# ---------------------------------------------------------------------------
# Direct phase test for an undeclared currency role.
# ---------------------------------------------------------------------------
# `_phase_ground_build` maps an undeclared role axis to `request_invalid`.
# Full requests refuse role-shaped currency earlier as `facet_binding_unsupported` (501).


def test_role_shaped_currency_naming_undeclared_axis_raises_request_invalid():
    from search.engine.region.place_context import place_context
    from search.engine.region.promote_ground import _phase_ground_build
    from search.engine.region.promote_state import _PromoteState
    from search.engine.retrieval.member_facts import ArcsMemberFacts
    from search.engine.retrieval.retrieve import Materialised
    from search.engine.shape import ShapeDraft
    from search.models import CoverageBare, IdentityCurrencyRole
    from tests.engine._builders import subject_place_roles

    obs = {f"{_SV}|{_ENTITY}": [_obs_facet(unit="CurrencyUSD")]}
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = _materialise_arcs(
        nominal=True, provenance_declares_recipe={_PROVENANCE: True}
    )
    # role-shaped currency field in the request.
    request = IdentityUnitRequest(
        currency=IdentityCurrencyRole(role="destination"), prices="current"
    )
    entities = [subject_entity(_ENTITY)]
    base_identity = make_identity(_SHAPE, slots=[], entities=entities)
    inp = ResubmitIdentityPayload(
        identity=base_identity.model_copy(
            update={"facets": {"unit": IdentityFacetValueRequest(request=request)}}
        ),
    )
    shape_draft = ShapeDraft(
        shape_id="test_shape",
        label="Test",
        pop_type_dcid="TestConvertActivity",
        meas_prop_dcid="testConvertExpenditure",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=("observationAbout",),
    )
    state = _PromoteState(
        inp=inp,
        graph=None,
        sdmx=None,
        deadline=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    state.shape_draft = shape_draft
    state.entity_dcid = _ENTITY
    state.place = place_context(
        shape_draft,
        subject_place_roles(_ENTITY),
        stat_vars=(),
        presence=None,
    )
    state.mat_result = Materialised(
        sv_dcids=[_SV],
        has_data=True,
        coverage=CoverageBare(has_data=True),
        member_facts={_SV: ArcsMemberFacts(arcs={})},
    )

    with pytest.raises(EngineInputError) as exc_info:
        _phase_ground_build(state)
    assert exc_info.value.code == "request_invalid"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/input/facets/unit"
    assert exc_info.value.cause is None


def test_decide_part_unit_reads_one_recipe_per_provenance_however_many_members_share_it():
    """Fifty StatVars under one provenance, converting `{EUR, current}` from native
    `CurrencyUSD`. Each member's decision reads its StatVar, the provenance and both
    currencies in one `node_arcs_many` read, so fifty members make fifty reads when each
    reads its own. Through `conversion_graph_reader` the request makes one. The first
    decision's read folds in every StatVar, and every later decision is served from memory.
    """
    stat_vars = [f"{_SV}_{index}" for index in range(50)]
    obs = {f"{sv}|{_ENTITY}": [_obs_facet(unit="CurrencyUSD")] for sv in stat_vars}
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = {
        **{sv: _sv_arcs(nominal=True) for sv in stat_vars},
        **_CURRENCY_ARCS,
        _PROVENANCE: {"conversionRecipe": (_RECIPE,)},
    }

    elected = decide_part_unit(
        stat_vars,
        unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
        role_bindings={"observationAbout": [_ENTITY]},
        declared_roles=("observationAbout",),
        bound_facets={"provenance": _PROVENANCE},
        years=None,
        graph=materialise_graph,
        sdmx=materialise_sdmx,
        coverage=_SUBJECT_ONLY_COVERAGE,
    )

    assert elected_unit_refs(elected) == {sv: frozenset({"CurrencyUSD"}) for sv in stat_vars}
    calls = materialise_graph.node_arcs_many_calls
    assert len([dcids for dcids, _properties in calls if _PROVENANCE in dcids]) == 1
    assert len(calls) == 1
    [(dcids, properties)] = calls
    assert set(dcids) == {*stat_vars, _PROVENANCE, "CurrencyUSD", "CurrencyEUR"}
    assert set(properties) == set(CONVERSION_NODE_PROPERTIES)


def _materialise_graph_reads(unit_text: str | None) -> int:
    """The materialise graph reads of a single-StatVar, single-source Search answer with
    observations, entering through `resolve_variable`, less the rate-table version guard's
    reads, which the recorder names apart."""
    overlap = world_module.world("standard", "overlap", "definite")
    single = dataclasses.replace(
        overlap,
        rows=tuple(
            row
            for row in overlap.rows
            if row.place == world_module.ALAND and row.facet_id == "f-ids"
        ),
    )
    with world_module.door_loop():
        answer = single.first([world_module.ALAND], None, preview=True, unit_text=unit_text)
    assert answer.error is None and answer.refusal is None
    assert answer.result is not None and answer.result.status == "definite"
    return answer.recorder.counts()["materialise_graph.node_arcs_many"]


@pytest.mark.parametrize("unit_text", ["EUR", "USD"])
def test_a_converted_search_answer_reads_the_graph_once_more_than_an_unconverted_one(
    unit_text: str,
):
    """The options step's unit decision, its coverage check, and the observation fetch's
    decision share the request's memo, so the conversion costs one graph read in all."""
    assert _materialise_graph_reads(unit_text) - _materialise_graph_reads(None) == 1
