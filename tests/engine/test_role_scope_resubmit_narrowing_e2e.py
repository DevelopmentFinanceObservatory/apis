"""End-to-end tests for role-index narrowing on role-bearing resubmission.
Synthetic fixtures cover facet validation, sources, and scope narrowing."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.errors import EngineInputError
from search.engine.extract import DateRequest
from search.engine.region import resolve_spec_resubmit
from search.engine.region.pipeline import resolve_variable
from search.models import (
    BaseIdentity,
    IdentityBindingSet,
    IdentityBindingValue,
    ResubmitIdentityPayload,
    TimeWindow,
)
from series_coverage.schema import FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._base_identity import base_from_spec
from tests.engine._fake_coverage import (
    CoverageRow,
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

pytestmark = pytest.mark.usefixtures("request_log")

_SV_DCID = "TESTFIXTURE/RoleScopeResubmitNarrowingE2E/Flow"
_KEN = "country/KEN"
_NOR = "country/NOR"


def _facet(
    *,
    facet_id: str,
    unit: str | None = None,
    measurement_method: str | None = None,
    provenance_id: str | None = None,
    latest_date: str = "2020",
) -> Facet:
    return Facet(
        earliest_date="2015",
        latest_date=latest_date,
        obs_count=1,
        dates=[latest_date],
        values=[0.0],
        facet_id=facet_id,
        unit=unit,
        measurement_method=measurement_method,
        provenance_id=provenance_id,
    )


def _declaring_arcs() -> dict:
    """A bilateral creditor/debtor flow shape -- discovery runs on the
    recipient (debtor) axis when no entity is posted, mirroring
    test_role_scope_pipeline_e2e.py's own fixture shape."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in ("creditor", "debtor")]},
    }


_UNBOUND_DONOR = "TESTFIXTURE/RoleScopeResubmitNarrowingE2E/UnboundDonor"
_ROLE_NAMES = {_SV_DCID: RoleNames(donor="creditor", recipient="debtor")}


def _coverage_seed(
    members: list[str],
    *,
    facet_id: str = "f1",
    member_facet_ids: dict[str, str] | None = None,
    years: list[int] | None = None,
) -> list[CoverageRow]:
    """Seed one coverage row per recipient, with the donor axis unbound.

    Override facet IDs for per-member metadata; years default to [2020]."""
    member_facet_ids = member_facet_ids or {}
    row_years = years if years is not None else [2020]
    return [
        (_SV_DCID, member_facet_ids.get(member, facet_id), _UNBOUND_DONOR, member, row_years)
        for member in members
    ]


def _nodes(*, resolve: dict[str, str], label_dcids: list[str]) -> dict:
    return {
        **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
        _SV_DCID: {"label": "Test flow", "arcs": _declaring_arcs()},
        **{dcid: {"label": dcid.rsplit("/", 1)[-1]} for dcid in label_dcids},
    }


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _fresh_discover(
    *,
    resolve: dict[str, str],
    label_dcids: list[str],
    sdmx: FakeSdmxClient,
    date_request: DateRequest | None = None,
    coverage: CoverageSnapshot | None = None,
):
    """A fresh, unfiltered discovered-scope resolve_variable() call -- the
    Spec a real client would have already downloaded and now edits with a
    facets binding, echoed via ResubmitIdentityPayload below."""
    graph = FakeGraph(
        nodes=_nodes(resolve=resolve, label_dcids=label_dcids),
        detect={"aid disbursed": {"svs": [_SV_DCID], "cosine_scores": [0.9]}},
        resolve=resolve,
        obs={},
    )
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=[],
            date_request=date_request,
            detect_query="aid disbursed",
            role_query="aid disbursed",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_steps=[],
            base_timing={},
            coverage=coverage if coverage is not None else empty_coverage_snapshot(),
        )
    )


def _resubmit_with_facets(
    *,
    fresh_spec,
    resolve: dict[str, str],
    label_dcids: list[str],
    sdmx: FakeSdmxClient,
    facets: dict,
    coverage: CoverageSnapshot | None = None,
):
    """Echo `fresh_spec`'s own identity back through resolve_spec_resubmit,
    adding `facets` as a fresh binding this edit posts -- no facet axis is
    bound in the base, so this is a genuine new constraint, never an
    edit_incomplete re-send failure (identity_edit.check_edit_incomplete
    only diffs the base's OWN bound facet axes)."""
    graph = FakeGraph(
        nodes=_nodes(resolve=resolve, label_dcids=label_dcids), detect={}, resolve={}, obs={}
    )
    base_spec_id, base_identity = base_from_spec(fresh_spec)
    echo = ResubmitIdentityPayload(
        identity=BaseIdentity.model_validate(
            {**fresh_spec.definition.identity.model_dump(mode="python"), "facets": facets or None}
        ),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[sv.ref for sv in fresh_spec.resolved.stat_vars.items],
    )
    return resolve_spec_resubmit(
        inp=echo,
        graph=graph,
        sdmx=sdmx,
        coverage=coverage if coverage is not None else empty_coverage_snapshot(),
    )


def _resubmit_with_source_slot(
    *,
    fresh_spec,
    resolve,
    label_dcids,
    sdmx,
    source_dcid,
    facets: dict | None = None,
    coverage: CoverageSnapshot | None = None,
):
    graph = FakeGraph(
        nodes=_nodes(resolve=resolve, label_dcids=label_dcids), detect={}, resolve={}, obs={}
    )
    base_spec_id, base_identity = base_from_spec(fresh_spec)
    identity_facets = {"provenance": IdentityBindingValue(ref=source_dcid)}
    if facets:
        identity_facets.update(facets)
    echo = ResubmitIdentityPayload(
        identity=BaseIdentity.model_validate(
            {
                **fresh_spec.definition.identity.model_dump(mode="python"),
                "facets": identity_facets,
            }
        ),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[sv.ref for sv in fresh_spec.resolved.stat_vars.items],
    )
    return resolve_spec_resubmit(
        inp=echo,
        graph=graph,
        sdmx=sdmx,
        coverage=coverage if coverage is not None else empty_coverage_snapshot(),
    )


def test_unit_binding_narrows_which_sources_are_cited(monkeypatch):
    """Binding USD keeps Kenya's tuple, citing one source and one recipient."""
    rows = _coverage_seed([_KEN, _NOR], member_facet_ids={_KEN: "fA", _NOR: "fB"}, years=[2021])
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        rows,
        role_names=_ROLE_NAMES,
        facet_metadata={
            "fA": FacetMetadata(unit="USD", provenance_id="source/A"),
            "fB": FacetMetadata(unit="EUR", provenance_id="source/B"),
        },
    )
    observations = {
        _SV_DCID: {
            (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [
                _facet(facet_id="fA", unit="USD", provenance_id="source/A", latest_date="2021")
            ],
            (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [
                _facet(facet_id="fB", unit="EUR", provenance_id="source/B", latest_date="2021")
            ],
        }
    }
    resolve = {"Kenya": _KEN, "Norway": _NOR}
    label_dcids = ["source/A", "source/B"]
    fresh = _fresh_discover(
        resolve=resolve,
        label_dcids=label_dcids,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        date_request=DateRequest(window=None, latest=True),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    region = _resubmit_with_facets(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        label_dcids=label_dcids,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        facets={"unit": IdentityBindingValue(ref="USD")},
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert sorted(r.dcid for r in spec.resolved.trace.resolved_sources) == ["source/A"]
    dims = {d.label: d.count for d in spec.resolved.coverage.dimensions}
    assert dims["sources"] == 1
    assert dims["recipients"] == 1
    assert spec.resolved.coverage.window == TimeWindow(start_year=2021, end_year=2021)
    when_slot = next(slot for slot in spec.resolved.slots if slot.key.axis == "when")
    assert when_slot.binding.value.time_window == TimeWindow(start_year=2021, end_year=2021)
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == [_KEN]


def test_measurement_method_binding_narrows_which_sources_are_cited(monkeypatch):
    """Two discovered recipients whose role tuples share no facet_id but
    disagree on measurement_method: KEN/Census/source-A, NOR/Estimate/source-B.
    Binding facets.measurementMethod="dcAggregate/Census" drops NOR's role
    tuple (same _narrow_candidate_by_metadata mechanism as the unit case, at
    the "measurementMethod" axis instead), so resolved_sources cites only
    source/A, the sources dimension recounts to 1, the recipients dimension
    recounts to 1, and resolved.scope.members narrows to KEN alone.
    """
    rows = _coverage_seed([_KEN, _NOR], member_facet_ids={_KEN: "fA", _NOR: "fB"})
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        rows,
        role_names=_ROLE_NAMES,
        facet_metadata={
            "fA": FacetMetadata(measurement_method="dcAggregate/Census", provenance_id="source/A"),
            "fB": FacetMetadata(
                measurement_method="dcAggregate/Estimate", provenance_id="source/B"
            ),
        },
    )
    observations = {
        _SV_DCID: {
            (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [
                _facet(
                    facet_id="fA",
                    measurement_method="dcAggregate/Census",
                    provenance_id="source/A",
                )
            ],
            (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [
                _facet(
                    facet_id="fB",
                    measurement_method="dcAggregate/Estimate",
                    provenance_id="source/B",
                )
            ],
        }
    }
    resolve = {"Kenya": _KEN, "Norway": _NOR}
    label_dcids = ["source/A", "source/B"]
    fresh = _fresh_discover(
        resolve=resolve,
        label_dcids=label_dcids,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    region = _resubmit_with_facets(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        label_dcids=label_dcids,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        facets={"measurementMethod": IdentityBindingValue(ref="dcAggregate/Census")},
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert sorted(r.dcid for r in spec.resolved.trace.resolved_sources) == ["source/A"]
    dims = {d.label: d.count for d in spec.resolved.coverage.dimensions}
    assert dims["sources"] == 1
    assert dims["recipients"] == 1
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == [_KEN]


# test_unit_binding_narrows_a_shared_facet_id_to_its_matching_role_tuples
# deleted here: it proved a SHARED facet_id disagreeing on unit/provenance
# across role tuples, a real shape a bilateral SDMX read could produce.
# CoverageIndex.facet_metadata carries exactly one unit/provenance per
# facet_id, no per-role-tuple grain at all (see retrieval/role_scope.py's
# own docstring), so that shape can no longer arise on a coverage-built role
# index -- distinct metadata now always means distinct facet_id, the
# premise test_unit_binding_narrows_which_sources_are_cited above already
# covers.


# Three tests naming the old per-scope omitted-index flag deleted here (one
# proving a degraded facets/preview carrier that kept has_data, one proving
# a bound-unit scope_too_broad refusal, one proving a bound-source
# scope_too_broad refusal): each drove a raised SdmxCapExceededError through
# sdmx.role_facet_index, the read that used to build discover_role_scope's
# own role_index_by_sv on re-discovery. That read is gone -- role_index_by_sv
# is a local build off confirmed_coverage now (retrieval/role_scope.py's
# docstring), which cannot cap-trip. Materialised no longer carries a
# per-scope omitted-index flag at all, and none of the three outcomes (a
# degraded facets/preview carrier, a bound-unit scope_too_broad refusal, or
# a bound-source scope_too_broad refusal) can be reached from a resubmit's
# own re-discovery any more.


@pytest.mark.parametrize("source_dcid, expected_error", [("source/B", True), ("source/A", False)])
def test_dateless_source_slot_is_validated_against_the_role_index(
    monkeypatch, source_dcid, expected_error
):
    """A source slot must be checked from the role index even without a
    requested window or observations preview, and point any mismatch at the
    slot rather than the facets map."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _coverage_seed([_KEN]),
        role_names=_ROLE_NAMES,
        facet_metadata={"f1": FacetMetadata(provenance_id="source/A")},
    )
    observations = {
        _SV_DCID: {(("debtor", _KEN),): [_facet(facet_id="fA", provenance_id="source/A")]}
    }
    resolve = {"Kenya": _KEN}
    fresh = _fresh_discover(
        resolve=resolve,
        label_dcids=["source/A", "source/B"],
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    if expected_error:
        with pytest.raises(EngineInputError) as exc_info:
            _resubmit_with_source_slot(
                fresh_spec=fresh.specs[0],
                resolve=resolve,
                label_dcids=["source/A", "source/B"],
                sdmx=FakeSdmxClient(observations=observations, availability={}),
                source_dcid=source_dcid,
                coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            )
        assert exc_info.value.code == "facet_not_available"
        assert exc_info.value.pointer == "/input/identity/facets/provenance"
        assert "provenance:" in exc_info.value.detail
    else:
        region = _resubmit_with_source_slot(
            fresh_spec=fresh.specs[0],
            resolve=resolve,
            label_dcids=["source/A", "source/B"],
            sdmx=FakeSdmxClient(observations=observations, availability={}),
            source_dcid=source_dcid,
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
        assert region.status == "definite", region


@pytest.mark.parametrize(
    ("source_dcid", "unit", "expected_error"),
    [("source/A", "USD", True), ("source/A", "EUR", False), ("source/B", "USD", False)],
)
def test_source_slot_and_unit_must_share_one_role_tuple(
    monkeypatch, source_dcid, unit, expected_error
):
    """Source A/EUR and source B/USD are not interchangeable bindings.

    The role index must reject a source-A-plus-USD request even though each
    member appears independently, while retaining the two same-tuple controls.
    """
    rows = _coverage_seed([_KEN, _NOR], member_facet_ids={_KEN: "fA", _NOR: "fB"})
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        rows,
        role_names=_ROLE_NAMES,
        facet_metadata={
            "fA": FacetMetadata(unit="EUR", provenance_id="source/A"),
            "fB": FacetMetadata(unit="USD", provenance_id="source/B"),
        },
    )
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet(facet_id="fA", unit="EUR", provenance_id="source/A")],
            (("debtor", _NOR),): [_facet(facet_id="fB", unit="USD", provenance_id="source/B")],
        }
    }
    resolve = {"Kenya": _KEN, "Norway": _NOR}
    fresh = _fresh_discover(
        resolve=resolve,
        label_dcids=["source/A", "source/B"],
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    kwargs = {
        "fresh_spec": fresh.specs[0],
        "resolve": resolve,
        "label_dcids": ["source/A", "source/B"],
        "sdmx": FakeSdmxClient(observations=observations, availability={}),
        "source_dcid": source_dcid,
        "facets": {"unit": IdentityBindingValue(ref=unit)},
        "coverage": FakeCoverageService(_auto_coverage_index_1).snapshot(),
    }
    if expected_error:
        with pytest.raises(EngineInputError) as exc_info:
            _resubmit_with_source_slot(**kwargs)
        assert exc_info.value.code == "facet_not_available"
        assert exc_info.value.pointer == "/input/identity/facets/provenance"
        assert "provenance:" in exc_info.value.detail
    else:
        region = _resubmit_with_source_slot(**kwargs)
        assert region.status == "definite", region


@pytest.mark.parametrize(
    ("facets", "axis", "missing"),
    [
        ({"unit": IdentityBindingValue(ref="EUR")}, "unit", "EUR"),
        (
            {"measurementMethod": IdentityBindingValue(ref="dcAggregate/Estimate")},
            "measurementMethod",
            "dcAggregate/Estimate",
        ),
        ({"provenance": IdentityBindingValue(ref="source/B")}, "provenance", "source/B"),
    ],
)
def test_unavailable_role_scope_facet_refuses_from_role_index(monkeypatch, facets, axis, missing):
    """A placeless role-scope resubmit has no subject entity for the normal
    facet endpoint. Its discovered role index must therefore refuse a bound
    unit, measurement method, or provenance that none of its tuples carries,
    rather than finalising a definite empty scope.
    """
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _coverage_seed([_KEN]), role_names=_ROLE_NAMES
    )
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [
                _facet(
                    facet_id="fA",
                    unit="USD",
                    measurement_method="dcAggregate/Census",
                    provenance_id="source/A",
                )
            ]
        }
    }
    resolve = {"Kenya": _KEN}
    fresh = _fresh_discover(
        resolve=resolve,
        label_dcids=["source/A"],
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    with pytest.raises(EngineInputError) as exc_info:
        _resubmit_with_facets(
            fresh_spec=fresh.specs[0],
            resolve=resolve,
            label_dcids=["source/A"],
            sdmx=FakeSdmxClient(observations=observations, availability={}),
            facets=facets,
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )

    error = exc_info.value
    assert error.code == "facet_not_available"
    assert error.status == 422
    assert error.pointer == f"/input/identity/facets/{axis}"
    assert missing in error.detail


def test_role_scope_facet_set_requires_every_member_from_every_stat_var(monkeypatch):
    """The role-index path shares facet_refusal's all-set-members contract:
    a present USD does not cover a posted USD/EUR set when the discovered
    tuple index has no EUR candidate.
    """
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _coverage_seed([_KEN]), role_names=_ROLE_NAMES
    )
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [
                _facet(
                    facet_id="fA",
                    unit="USD",
                    provenance_id="source/A",
                )
            ]
        }
    }
    resolve = {"Kenya": _KEN}
    fresh = _fresh_discover(
        resolve=resolve,
        label_dcids=["source/A"],
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    with pytest.raises(EngineInputError) as exc_info:
        _resubmit_with_facets(
            fresh_spec=fresh.specs[0],
            resolve=resolve,
            label_dcids=["source/A"],
            sdmx=FakeSdmxClient(observations=observations, availability={}),
            facets={"unit": IdentityBindingSet(refs=["USD", "EUR"])},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )

    error = exc_info.value
    assert error.code == "facet_not_available"
    assert error.pointer == "/input/identity/facets/unit"
    assert "EUR" in error.detail


@pytest.mark.parametrize(
    "facets",
    [
        {"unit": IdentityBindingValue(ref="USD")},
        {"measurementMethod": IdentityBindingValue(ref="dcAggregate/Census")},
        {"provenance": IdentityBindingValue(ref="source/A")},
    ],
)
def test_available_role_scope_facet_still_resolves_definite(monkeypatch, facets):
    """A matching facet remains a definite placeless role-scope answer."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _coverage_seed([_KEN]),
        role_names=_ROLE_NAMES,
        facet_metadata={
            "f1": FacetMetadata(
                unit="USD", measurement_method="dcAggregate/Census", provenance_id="source/A"
            )
        },
    )
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [
                _facet(
                    facet_id="fA",
                    unit="USD",
                    measurement_method="dcAggregate/Census",
                    provenance_id="source/A",
                )
            ]
        }
    }
    resolve = {"Kenya": _KEN}
    fresh = _fresh_discover(
        resolve=resolve,
        label_dcids=["source/A"],
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    result = _resubmit_with_facets(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        label_dcids=["source/A"],
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        facets=facets,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert result.status == "definite", result.no_data_reason
