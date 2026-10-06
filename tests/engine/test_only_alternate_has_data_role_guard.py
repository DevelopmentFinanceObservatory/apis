"""Tests role-aware alternate suppression using bound role dcids from a real engine run.

The tests show a role-correct alternate and withhold the same result with origin and destination
swapped by hand, since no pipeline path inverts roles."""

from __future__ import annotations

import asyncio

from search.engine.assemble import build_no_data_part
from search.engine.bind import _UniversalBindOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import ONLY_ALTERNATE_HAS_DATA
from search.models import ResolvedEntity
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.fixtures import FakeGraph, make_resolver

_SV_DCID = "TESTFIXTURE/OnlyAlternateHasDataRoleGuard/Flow"
_ALT_SV_DCID = "TESTFIXTURE/OnlyAlternateHasDataRoleGuard/FlowAlt"
_NOR = "country/NOR"
_KEN = "country/KEN"
_ROLE_QUERY = "aid from Norway to Kenya"
_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")
"""The role names declared by `_declaring_arcs`."""


def _declaring_arcs(*, measured_property: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": measured_property}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in ("creditor", "debtor")]},
    }


class _ElectNoDataAlsoPlausibleHasDataLLM:
    """Elects the data-empty StatVar and marks the data-bearing one as plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None
        ), None


def _run(monkeypatch):
    graph = FakeGraph(
        nodes={
            _NOR: {"label": "Norway", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            _SV_DCID: {"label": "Test flow", "arcs": _declaring_arcs(measured_property="amount")},
            _ALT_SV_DCID: {
                "label": "Test alternate flow",
                "arcs": _declaring_arcs(measured_property="amountAlt"),
            },
        },
        detect={
            _ROLE_QUERY: {
                "svs": [_SV_DCID, _ALT_SV_DCID],
                "cosine_scores": [0.9, 0.85],
            }
        },
        resolve={"Norway": _NOR, "Kenya": _KEN},
        obs={},
    )
    # The elected StatVar has no coverage; the alternate has coverage for the same bound pair.
    # Both declare the same creditor/debtor roles, so the shared entity map binds them alike.
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_ALT_SV_DCID, "f1", _NOR, _KEN, [2018, 2019, 2020])],
        role_names={_ALT_SV_DCID: _ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=["Norway", "Kenya"],
            date_request=None,
            detect_query=_ROLE_QUERY,
            role_query=_ROLE_QUERY,
            entity_roles=door_roles_for_test_query(_ROLE_QUERY, ["Norway", "Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectNoDataAlsoPlausibleHasDataLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )


def test_only_alternate_has_data_reaches_a_both_bound_shape(monkeypatch):
    """The fixture reaches the guard with both donor and recipient bound."""
    result = _run(monkeypatch)
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.gate == ONLY_ALTERNATE_HAS_DATA
    assert result.alternate_match is not None
    assert result.no_data_role_axis is None, (
        "both roles are bound, so no_data_role_axis (only ever set when a "
        "role-bearing query left an axis fully open) must stay None here -- "
        "this is exactly the shape the review found the old guard could "
        "never confirm safe"
    )


def test_no_data_bound_role_dcids_names_both_axes_the_query_bound(monkeypatch):
    """Bound role dcids are recorded when both flow axes have entities."""
    result = _run(monkeypatch)
    assert result.no_data_bound_role_dcids == {"origin": (_NOR,), "destination": (_KEN,)}


def _build_part(result, alternate_match):
    return build_no_data_part(
        result.no_data_reason,
        variable_text=result.variable_text,
        warnings=list(result.warnings),
        gate=result.gate,
        alternate_match=alternate_match,
        elected_stat_vars=result.elected_stat_vars,
        entity_terms=None,
        role_axis=result.no_data_role_axis,
        bound_role_dcids=result.no_data_bound_role_dcids,
    )


def test_role_correct_alternate_is_shown_end_to_end(monkeypatch):
    """The alternate shares the query's bound donor and recipient, so its sentence is shown."""
    result = _run(monkeypatch)
    assert [
        e.role for e in result.alternate_match.resolved.entities if e.role != "observationAbout"
    ]
    part = _build_part(result, result.alternate_match)
    assert part.rendered_sentence is not None
    assert "Data exists for" in part.rendered_sentence
    assert "Norway" in part.rendered_sentence and "Kenya" in part.rendered_sentence


def test_a_swapped_alternate_is_withheld_against_the_real_bound_role_dcids(monkeypatch):
    """The guard withholds the alternate when it swaps Norway from donor to recipient."""
    result = _run(monkeypatch)
    alt = result.alternate_match
    assert alt is not None
    swapped_role = {"origin": "destination", "destination": "origin"}
    inverted_entities = [
        ResolvedEntity(
            ref=e.ref,
            label=e.label,
            entity_type=e.entity_type,
            role=swapped_role.get(e.role, e.role),
        )
        for e in alt.resolved.entities
    ]
    inverted_alt = alt.model_copy(
        update={"resolved": alt.resolved.model_copy(update={"entities": inverted_entities})}
    )
    part = _build_part(result, inverted_alt)
    assert part.rendered_sentence is not None
    assert "Data exists for" not in part.rendered_sentence
    assert "Norway" not in part.rendered_sentence
