"""End-to-end tests for role-bearing scope discovery through resolve and resubmit."""

from __future__ import annotations

import asyncio
import sys

import pytest

from materialise.roles import axis_for_role
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.concurrency import scoped_concurrency
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import materialise_step as materialise_step_module
from search.engine.region import observations as _observations_pkg  # noqa: F401
from search.engine.region import resolve_spec_resubmit
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.models import CoverageBreadth, DateRange, ResubmitIdentityPayload, TimeWindow
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import (
    CoverageRow,
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query, walk_options
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_SV_DCID = "TESTFIXTURE/RoleScopePipelineE2E/Flow"
_ORIGIN_ONLY_SV_DCID = "TESTFIXTURE/RoleScopePipelineE2E/OriginOnlyFlow"
_KEN = "country/KEN"
_NOR = "country/NOR"
_FRA = "country/FRA"
_DEU = "country/DEU"
_CAN = "country/CAN"
_JPN = "country/JPN"


def _facet(*, obs_count: int = 3, facet_id: str = "f1", latest_date: str = "2020") -> Facet:
    return Facet(
        earliest_date="2015",
        latest_date=latest_date,
        obs_count=obs_count,
        dates=[latest_date],
        values=[0.0],
        facet_id=facet_id,
    )


def _declaring_arcs(*, observation_properties: list[str]) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in observation_properties]},
    }


_UNBOUND_DONOR = "TESTFIXTURE/RoleScopePipelineE2E/UnboundDonor"
_UNBOUND_RECIPIENT = "TESTFIXTURE/RoleScopePipelineE2E/UnboundRecipient"


def _coverage_seed_for(
    role_name: str,
    members: list[str],
    svs: list[str],
    *,
    filters: dict[str, list[str]] | None = None,
    time_period: list[str] | None = None,
    facet_id: str = "f1",
) -> tuple[list[CoverageRow], dict[str, RoleNames]]:
    """Seed rows for each SV/member pair and opposite-role filter.

    Unbound axes use a placeholder dcid; years default to 2020."""
    discover_axis = axis_for_role(role_name)
    assert discover_axis is not None, f"{role_name!r} has no flow axis"
    opposite_role_name, opposite_dcids = next(iter(filters.items())) if filters else (None, None)
    opposite_dcids = opposite_dcids or [
        _UNBOUND_DONOR if discover_axis == "destination" else _UNBOUND_RECIPIENT
    ]
    years = [int(y) for y in time_period] if time_period else [2020]
    rows: list[CoverageRow] = [
        (sv, facet_id, *((opp, member) if discover_axis == "destination" else (member, opp)), years)
        for sv in svs
        for member in members
        for opp in opposite_dcids
    ]
    role_names = {
        sv: (
            RoleNames(donor=opposite_role_name or "creditor", recipient=role_name)
            if discover_axis == "destination"
            else RoleNames(donor=role_name, recipient=opposite_role_name or "debtor")
        )
        for sv in svs
    }
    return rows, role_names


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _graph_for(
    *,
    sv_dcid: str,
    sv_arcs: dict,
    resolve: dict[str, str],
    role_query: str,
    sv_score: float,
    provenance_labels: dict[str, str] | None = None,
    place_graph_labels: dict[str, str] | None = None,
) -> FakeGraph:
    return FakeGraph(
        nodes={
            **{
                dcid: {
                    "label": (place_graph_labels or {}).get(dcid, name),
                    "type": "Country",
                }
                for name, dcid in resolve.items()
            },
            sv_dcid: {"label": "Test flow", "arcs": sv_arcs},
            **{dcid: {"label": label} for dcid, label in (provenance_labels or {}).items()},
        },
        detect={role_query: {"svs": [sv_dcid], "cosine_scores": [sv_score]}},
        resolve=resolve,
        obs={},
    )


def _run(
    *,
    entities: list[str],
    sv_dcid: str,
    sv_arcs: dict,
    resolve: dict[str, str],
    sv_score: float,
    role_query: str,
    sdmx: FakeSdmxClient,
    date_request: DateRequest | None = None,
    include_observations: bool = False,
    unit_text: str | None = None,
    provenance_labels: dict[str, str] | None = None,
    place_graph_labels: dict[str, str] | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    coverage_rows: list[CoverageRow] | None = None,
    coverage_role_names: dict[str, RoleNames] | None = None,
    coverage_facet_metadata: dict[str, FacetMetadata] | None = None,
):
    coverage_snapshot = empty_coverage_snapshot()
    if coverage_rows is not None:
        assert monkeypatch is not None, "coverage_rows needs monkeypatch to seed it"
        _coverage_index = patch_coverage(
            monkeypatch,
            coverage_rows,
            role_names=coverage_role_names or {},
            facet_metadata=coverage_facet_metadata,
        )
        coverage_snapshot = FakeCoverageService(_coverage_index).snapshot()
    graph = _graph_for(
        sv_dcid=sv_dcid,
        sv_arcs=sv_arcs,
        resolve=resolve,
        role_query=role_query,
        sv_score=sv_score,
        provenance_labels=provenance_labels,
        place_graph_labels=place_graph_labels,
    )
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=entities,
            date_request=date_request,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, entities),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            unit_text=unit_text,
            coverage=coverage_snapshot,
            options=EngineOptions(include_observations=include_observations),
        )
    )


def _run_streamed(
    *,
    entities: list[str],
    sv_dcid: str,
    sv_arcs: dict,
    resolve: dict[str, str],
    sv_score: float,
    role_query: str,
    sdmx: FakeSdmxClient,
    date_request: DateRequest | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    coverage_rows: list[CoverageRow] | None = None,
    coverage_role_names: dict[str, RoleNames] | None = None,
    coverage_facet_metadata: dict[str, FacetMetadata] | None = None,
):
    coverage_snapshot = empty_coverage_snapshot()
    if coverage_rows is not None:
        assert monkeypatch is not None, "coverage_rows needs monkeypatch to seed it"
        _coverage_index = patch_coverage(
            monkeypatch,
            coverage_rows,
            role_names=coverage_role_names or {},
            facet_metadata=coverage_facet_metadata,
        )
        coverage_snapshot = FakeCoverageService(_coverage_index).snapshot()
    """Run the streamed resolve and await its observations task in the same event loop.

    Opens request-scoped concurrency because this direct call bypasses core.resolve_events (#624)."""
    graph = _graph_for(
        sv_dcid=sv_dcid, sv_arcs=sv_arcs, resolve=resolve, role_query=role_query, sv_score=sv_score
    )

    async def _body():
        async with scoped_concurrency():
            result = await resolve_variable(
                "aid",
                entities=entities,
                date_request=date_request,
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
                options=EngineOptions(stream_observations=True),
            )
            events, warns = (
                await result.observations_task if result.observations_task is not None else ((), ())
            )
            return result, events, warns

    return asyncio.run(_body())


# ---------------------------------------------------------------------------
# Discovery on the destination axis: neither entity named at all
# ---------------------------------------------------------------------------


def test_no_entity_named_discovers_the_recipient_scope(monkeypatch):
    rows, role_names = _coverage_seed_for("debtor", [_KEN, _NOR], [_SV_DCID])
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("debtor", _NOR),): [_facet()],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert spec.resolved.scope is not None
    assert spec.resolved.scope.axis == "destination"
    assert sorted(m.ref for m in spec.resolved.scope.members) == [_KEN, _NOR]
    assert spec.resolved.entities == []
    assert spec.definition.identity.entities == {}
    assert isinstance(spec.resolved.coverage, CoverageBreadth)
    dims = {d.label: d.count for d in spec.resolved.coverage.dimensions}
    assert dims["recipients"] == 2
    # Discovery enumerates destination; the unbound origin also gets a default warning (#577).
    role_scope_warnings = [w for w in result.warnings if w.code == "role_scope_defaulted"]
    assert len(role_scope_warnings) == 2
    messages = {w.message for w in role_scope_warnings}
    assert (
        "no recipient was named; the answer was resolved across the 2 recipients with data."
        in messages
    )
    assert "no provider was named; the answer was resolved across every provider with data." in (
        messages
    )
    # The plan serves each pair's 2020 point, so the StatVar is dated 2020 to 2020.
    assert spec.resolved.members[0].data_date_range == DateRange(start="2020", end="2020")


def test_origin_bound_and_recipient_open_discovers_the_recipient_scope(monkeypatch):
    filters = {"creditor": [_FRA]}
    rows, role_names = _coverage_seed_for("debtor", [_KEN], [_SV_DCID], filters=filters)
    observations = {_SV_DCID: {(("creditor", _FRA), ("debtor", _KEN)): [_facet()]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _run(
        entities=["France"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"France": _FRA},
        sv_score=0.9,
        role_query="aid from France",
        sdmx=sdmx,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert [e.ref for e in spec.resolved.entities] == [_FRA]
    assert spec.definition.identity.entities.get("origin") == [_FRA]
    assert spec.resolved.scope is not None
    assert spec.resolved.scope.axis == "destination"
    assert [m.ref for m in spec.resolved.scope.members] == [_KEN]
    assert spec.resolved.role_axis_facts is not None
    assert [fact.model_dump(mode="json") for fact in spec.resolved.role_axis_facts] == [
        {"role": "destination", "member_count": 1, "members_available": True}
    ]
    # This is the exact buffered wire fragment consumed by the where editor:
    # no entity_ref key means the recipient axis is open, while role names the
    # identity.entities key a selected recipient must populate on resubmit.
    entity_options = spec.model_dump(mode="json", exclude_none=True)["resolved"]["entity_options"]
    assert entity_options == [
        {"role": "destination", "options": [], "truncated": True, "source": "observed"}
    ]
    role_scope_warnings = [w for w in result.warnings if w.code == "role_scope_defaulted"]
    assert len(role_scope_warnings) == 1
    assert (
        "no recipient was named; the answer was resolved across the 1 recipient with data "
        "from France." in role_scope_warnings[0].message
    )


def test_resubmit_selected_open_destination_uses_destination_identity_role(monkeypatch):
    """A recipient selected from an origin-bound/open-destination result is
    promoted under identity.entities.destination, not observationAbout.

    This exercises the API-produced wire role through the real resubmit
    promotion path. A role-incompatible failure here would reproduce the
    client-routing regression this contract guards.

    The fresh call below reaches discover_role_scope (destination open),
    still role_facet_index-backed (its own observations fixture), and now
    also seeds its own coverage row for the discovery reads
    (open_axis/filter_candidates). The resubmit call re-patches coverage
    with the [2015, 2020] row below before binding the destination role
    explicitly (identity.entities.destination = [_KEN]), reaching
    graph_confirm_resolve's coverage-backed named-destination confirmation
    instead.
    """
    filters = {"creditor": [_FRA]}
    rows, role_names = _coverage_seed_for("debtor", [_KEN], [_SV_DCID], filters=filters)
    observations = {_SV_DCID: {(("creditor", _FRA), ("debtor", _KEN)): [_facet()]}}
    fresh = _run(
        entities=["France"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"France": _FRA},
        sv_score=0.9,
        role_query="aid from France",
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )
    assert fresh.status == "definite", fresh
    spec = fresh.specs[0]

    edited_identity_payload = spec.definition.identity.model_dump(mode="json")
    edited_identity_payload["entities"]["destination"] = [_KEN]
    edited_identity = type(spec.definition.identity).model_validate(edited_identity_payload)
    graph = _graph_for(
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"France": _FRA, "Kenya": _KEN},
        role_query="unused",
        sv_score=0.9,
    )
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _FRA, _KEN, [2015, 2020])],
        role_names={_SV_DCID: RoleNames(donor="creditor", recipient="debtor")},
    )
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=edited_identity,
        ),
        graph=graph,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        options=walk_options(graph),
    )

    assert region.status == "definite", region
    assert region.specs[0].definition.identity.entities == {
        "origin": [_FRA],
        "destination": [_KEN],
    }


def test_discovered_scope_warning_preserves_the_accepted_origin_name(monkeypatch):
    filters = {"creditor": [_FRA]}
    rows, role_names = _coverage_seed_for("debtor", [_KEN], [_SV_DCID], filters=filters)
    observations = {_SV_DCID: {(("creditor", _FRA), ("debtor", _KEN)): [_facet()]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _run(
        entities=["French Republic"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"French Republic": _FRA},
        sv_score=0.9,
        role_query="aid from French Republic",
        sdmx=sdmx,
        place_graph_labels={_FRA: "France"},
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    role_scope_warnings = [w for w in result.warnings if w.code == "role_scope_defaulted"]
    assert len(role_scope_warnings) == 1
    assert "from French Republic." in role_scope_warnings[0].message


# (origins in query order, footnote naming them). Both doors name origins sorted by dcid,
# the order the identity stores them in, whatever order the query named them in.
_MULTI_ORIGIN_CASES = [
    pytest.param([("France", _FRA), ("Germany", _DEU)], "Germany or France", id="two-origins"),
    pytest.param(
        [("France", _FRA), ("Germany", _DEU), ("Canada", _CAN)],
        "Canada, Germany, or France",
        id="three-origins",
    ),
    pytest.param(
        [("France", _FRA), ("Germany", _DEU), ("Canada", _CAN), ("Japan", _JPN)],
        "any of the 4 providers",
        id="four-origins",
    ),
]


def _multi_origin_fixture(origins: list[tuple[str, str]]):
    """(role_query, resolve, coverage_rows, coverage_role_names, observations)
    for a query naming every origin in `origins` and no recipient. Discovery
    enumerates the one recipient (Kenya) that has data from the whole origin
    union. `bound`'s own frozenset makes the coverage lookup order-invariant
    (unlike the deleted availability fixture, which needed a second entry
    for the resubmit path's own dcid-sorted filter), so one seed answers the
    fresh path and the resubmit path alike."""
    origin_dcids = [dcid for _, dcid in origins]
    resolve = {**dict(origins), "Kenya": _KEN}
    role_query = f"aid from {', '.join(name for name, _ in origins)}"
    rows, role_names = _coverage_seed_for(
        "debtor", [_KEN], [_SV_DCID], filters={"creditor": origin_dcids}
    )
    observations = {
        _SV_DCID: {(("creditor", dcid), ("debtor", _KEN)): [_facet()] for dcid in origin_dcids}
    }
    return role_query, resolve, rows, role_names, observations


@pytest.mark.parametrize("reverse", [False, True], ids=["as-named", "reversed"])
@pytest.mark.parametrize(("origins", "expected_named"), _MULTI_ORIGIN_CASES)
def test_several_bound_origins_are_all_named_in_the_role_scope_warning(
    origins: list[tuple[str, str]], expected_named: str, reverse: bool, monkeypatch
):
    """Every bound origin reaches the footnote, in dcid order whatever order the query
    named them in. Two and three names are spelled out, and four or more collapse to a
    count."""
    if reverse:
        origins = origins[::-1]
    role_query, resolve, rows, role_names, observations = _multi_origin_fixture(origins)

    result = _run(
        entities=[name for name, _ in origins],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve=resolve,
        sv_score=0.9,
        role_query=role_query,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert result.status == "definite", result
    assert [e.ref for e in result.specs[0].resolved.entities] == sorted(dcid for _, dcid in origins)
    role_scope_warnings = [w for w in result.warnings if w.code == "role_scope_defaulted"]
    assert [w.message for w in role_scope_warnings] == [
        "no recipient was named; the answer was resolved across the 1 recipient with data "
        f"from {expected_named}."
    ]


@pytest.mark.parametrize(("origins", "expected_named"), _MULTI_ORIGIN_CASES)
def test_resubmit_names_every_bound_origin_in_the_role_scope_warning(
    origins: list[tuple[str, str]], expected_named: str, monkeypatch
):
    """Resubmission names all bound origins in dcid-sorted order."""
    role_query, resolve, rows, role_names, observations = _multi_origin_fixture(origins)
    sv_arcs = _declaring_arcs(observation_properties=["creditor", "debtor"])
    fresh = _run(
        entities=[name for name, _ in origins],
        sv_dcid=_SV_DCID,
        sv_arcs=sv_arcs,
        resolve=resolve,
        sv_score=0.9,
        role_query=role_query,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )
    assert fresh.status == "definite", fresh
    spec = fresh.specs[0]
    assert spec.definition.identity.entities == {"origin": sorted(dcid for _, dcid in origins)}

    # Same coverage seed still active (monkeypatch reverts at test teardown,
    # not between these two calls): the resubmit's own destination axis is
    # still open, reaching discover_role_scope again over the same rows --
    # re-wrapped as its own CoverageSnapshot, since resolve_spec_resubmit
    # takes one as a required parameter now rather than reading the patched
    # get_coverage_service singleton itself.
    coverage_snapshot = FakeCoverageService(
        patch_coverage(monkeypatch, rows, role_names=role_names)
    ).snapshot()
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=spec.definition.identity,
        ),
        graph=_graph_for(
            sv_dcid=_SV_DCID,
            sv_arcs=sv_arcs,
            resolve=resolve,
            role_query="unused",
            sv_score=0.9,
        ),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=coverage_snapshot,
        options=walk_options(
            _graph_for(
                sv_dcid=_SV_DCID,
                sv_arcs=sv_arcs,
                resolve=resolve,
                role_query="unused",
                sv_score=0.9,
            )
        ),
    )

    assert region.status == "definite", region
    role_scope_warnings = [w for w in region.warnings if w.code == "role_scope_defaulted"]
    assert [w.message for w in role_scope_warnings] == [
        "no recipient was named; the answer was resolved across the 1 recipient with data "
        f"from {expected_named}."
    ]


def _record_footnote_label_lookups(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every dcid list the fresh and resubmit paths' role_scope_defaulted
    footnote asks its label_of to label, in call order. The graph is read for
    origin labels on other steps as well, so the footnote's own lookup is
    observed at the label_of the one builder
    (grounding.role_scope_defaulted_warnings, which both doors call) is
    handed."""
    lookups: list[list[str]] = []

    def spying(build):
        def wrapper(mat, *, boundary_roles, label_of, **kwargs):
            def recording(dcids):
                lookups.append(list(dcids))
                return label_of(dcids)

            return build(mat, boundary_roles=boundary_roles, label_of=recording, **kwargs)

        return wrapper

    # Both region doors' own _phase_materialise call the same
    # interpret_materialised (region/materialise_step.py), which is the one
    # place that reads this name -- patching it there covers the fresh and
    # resubmit paths alike, with no separate patch per door.
    monkeypatch.setattr(
        materialise_step_module,
        "role_scope_defaulted_warnings",
        spying(materialise_step_module.role_scope_defaulted_warnings),
    )
    return lookups


@pytest.mark.parametrize(
    ("origins", "expects_lookup"),
    [
        pytest.param(
            [("France", _FRA), ("Germany", _DEU), ("Canada", _CAN)], True, id="three-origins"
        ),
        pytest.param(
            [("France", _FRA), ("Germany", _DEU), ("Canada", _CAN), ("Japan", _JPN)],
            False,
            id="four-origins",
        ),
    ],
)
def test_role_scope_footnote_looks_up_labels_only_for_the_named_origins(
    monkeypatch: pytest.MonkeyPatch, origins: list[tuple[str, str]], expects_lookup: bool
):
    """Three origins get labels; four are represented by a count without label reads."""
    role_query, resolve, rows, role_names, observations = _multi_origin_fixture(origins)
    sv_arcs = _declaring_arcs(observation_properties=["creditor", "debtor"])
    lookups = _record_footnote_label_lookups(monkeypatch)

    fresh = _run(
        entities=[name for name, _ in origins],
        sv_dcid=_SV_DCID,
        sv_arcs=sv_arcs,
        resolve=resolve,
        sv_score=0.9,
        role_query=role_query,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )
    assert fresh.status == "definite", fresh
    assert lookups == ([sorted(dcid for _, dcid in origins)] if expects_lookup else [])

    lookups.clear()
    spec = fresh.specs[0]
    coverage_snapshot = FakeCoverageService(
        patch_coverage(monkeypatch, rows, role_names=role_names)
    ).snapshot()
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=spec.definition.identity,
        ),
        graph=_graph_for(
            sv_dcid=_SV_DCID,
            sv_arcs=sv_arcs,
            resolve=resolve,
            role_query="unused",
            sv_score=0.9,
        ),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=coverage_snapshot,
        options=walk_options(
            _graph_for(
                sv_dcid=_SV_DCID,
                sv_arcs=sv_arcs,
                resolve=resolve,
                role_query="unused",
                sv_score=0.9,
            )
        ),
    )
    assert region.status == "definite", region
    assert lookups == ([sorted(dcid for _, dcid in origins)] if expects_lookup else [])


def test_origin_only_signature_discovers_the_provider_scope(monkeypatch):
    shape_arcs = _declaring_arcs(observation_properties=["creditor"])
    rows, role_names = _coverage_seed_for("creditor", [_DEU, _FRA], [_ORIGIN_ONLY_SV_DCID])
    observations = {
        _ORIGIN_ONLY_SV_DCID: {
            (("creditor", _FRA),): [_facet()],
            (("creditor", _DEU),): [_facet()],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _run(
        entities=[],
        sv_dcid=_ORIGIN_ONLY_SV_DCID,
        sv_arcs=shape_arcs,
        resolve={"France": _FRA, "Germany": _DEU},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert spec.resolved.scope is not None
    assert spec.resolved.scope.axis == "origin"
    assert sorted(m.ref for m in spec.resolved.scope.members) == [_DEU, _FRA]


# ---------------------------------------------------------------------------
# Windowed / latest requests
# ---------------------------------------------------------------------------


def test_windowed_request_narrows_availability_and_reuses_the_facet_index(
    monkeypatch, search_settings
):
    """The 2018-2020 window filters discovery and the bulk preview to those years.

    The pair cap of 1 forces the bulk branch; later-only facets must be excluded."""
    search_settings(obs_preview_max_pairs=1)
    time_period = ["2018", "2019", "2020"]
    rows, role_names = _coverage_seed_for(
        "debtor", [_KEN, _NOR], [_SV_DCID], time_period=time_period
    )
    observations = {
        _SV_DCID: {
            (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [
                _facet(),
                _facet(facet_id="post-window", latest_date="2023"),
            ],
            (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [
                _facet(),
                _facet(facet_id="post-window", latest_date="2023"),
            ],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        date_request=DateRequest(window=TimeWindow(start_year=2018, end_year=2020), latest=False),
        include_observations=True,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert len(sdmx.series_calls) == 1
    assert sdmx.series_calls[0]["time_period"] == time_period
    assert spec.resolved.coverage.window == TimeWindow(start_year=2018, end_year=2020)
    assert spec.resolved.observations is not None
    assert len(spec.resolved.observations.series) == 2
    for series in spec.resolved.observations.series:
        assert [p.date for p in series.points] == ["2020"]


def test_streamed_windowed_request_reuses_the_discovered_index(monkeypatch, search_settings):
    """Both eager and deferred streamed reads reuse the index and filter to 2018-2020.

    Without the deferred index, streaming emits OBSERVATIONS_PROBE_FAILED and no event."""
    search_settings(obs_preview_max_pairs=1)
    time_period = ["2018", "2019", "2020"]
    rows, role_names = _coverage_seed_for(
        "debtor", [_KEN, _NOR], [_SV_DCID], time_period=time_period
    )
    observations = {
        _SV_DCID: {
            (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [_facet()],
            (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [_facet()],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result, events, warns = _run_streamed(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        date_request=DateRequest(window=TimeWindow(start_year=2018, end_year=2020), latest=False),
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert result.status == "definite", result
    assert warns == ()
    assert len(events) == 1
    # One role_series call from the eager availability fetch, one from the
    # deferred relaunch -- both narrowed to the discovered scope's own
    # explicit year list, never full history.
    assert len(sdmx.series_calls) == 2
    for call in sdmx.series_calls:
        assert call["time_period"] == time_period
    series_list = events[0].observations.series
    assert len(series_list) == 2
    for series in series_list:
        assert [p.date for p in series.points] == ["2020"]


def test_dateless_request_narrows_the_preview_to_full_history(monkeypatch, search_settings):
    """A dateless discovered-scope request (no date_request at all, not even
    a latest ask) resolves no window at discovery time, so the observation
    preview's role_series read narrows to no year at all -- full history --
    over the reused (coverage-built, never SDMX-read) role index's own
    surviving pairs."""
    search_settings(obs_preview_max_pairs=1)
    rows, role_names = _coverage_seed_for("debtor", [_KEN, _NOR], [_SV_DCID])
    observations = {
        _SV_DCID: {
            (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [_facet()],
            (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [_facet()],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        include_observations=True,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert len(sdmx.series_calls) == 1
    assert sdmx.series_calls[0]["time_period"] is None
    assert spec.resolved.observations is not None
    assert len(spec.resolved.observations.series) == 2
    for series in spec.resolved.observations.series:
        assert [p.date for p in series.points] == ["2020"]


def _run_kenya_2020_norway_2021_latest(monkeypatch, *, sdmx):
    rows: list[CoverageRow] = [
        (_SV_DCID, "f1", _UNBOUND_DONOR, _KEN, [2020]),
        (_SV_DCID, "f1", _UNBOUND_DONOR, _NOR, [2021]),
    ]
    role_names = {_SV_DCID: RoleNames(donor="creditor", recipient="debtor")}
    return _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        date_request=DateRequest(window=None, latest=True),
        include_observations=True,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )


_KENYA_2020_NORWAY_2021 = {
    _SV_DCID: {
        (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [_facet(latest_date="2020")],
        (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [_facet(latest_date="2021")],
    }
}


def _served_points(spec) -> list[tuple[str, list[str]]]:
    assert spec.resolved.observations is not None
    return [
        (
            next(e.ref for e in series.entities if e.role == "destination"),
            [point.date for point in series.points],
        )
        for series in spec.resolved.observations.series
    ]


def test_bulk_latest_request_serves_only_the_facets_latest_year(monkeypatch, search_settings):
    """Above the pair cap the preview reads the latest point of the elected facet's pairs
    in the year the variable elects, in one read, and serves that year only. That is the
    headline's year, Norway's 2021, and Kenya's 2020 point is left out. Coverage and the
    `when` slot name that one year."""
    search_settings(obs_preview_max_pairs=1)
    sdmx = FakeSdmxClient(observations=_KENYA_2020_NORWAY_2021, availability={})

    result = _run_kenya_2020_norway_2021_latest(monkeypatch, sdmx=sdmx)

    assert result.status == "definite", result
    spec = result.specs[0]
    served_year = TimeWindow(start_year=2021, end_year=2021)
    assert spec.resolved.coverage.window == served_year
    when_slot = next(slot for slot in spec.resolved.slots if slot.key.axis == "when")
    assert when_slot.binding.value.time_window == served_year
    assert [call["time_period"] for call in sdmx.series_calls] == ["LATEST"]
    assert _served_points(spec) == [(_NOR, ["2021"])]


def test_latest_request_reads_the_recorded_facets_year_past_a_projection_facet(
    monkeypatch, search_settings
):
    """Recorded `f-rec` runs 2010-2024 and the `Estimated` projection `f-est` runs 2025-2032,
    both at Kenya and Norway. Equal pair counts leave the observation count to elect `f-rec`,
    so the preview reads `f-rec` alone, whose pairs end in 2024, the year they elect, and
    serves each pair's 2024 point, and coverage, the `when` slot and the headline name
    2024, the last recorded year."""
    search_settings(obs_preview_max_pairs=1)
    recorded = list(range(2010, 2025))
    projected = list(range(2025, 2033))
    rows: list[CoverageRow] = [
        (_SV_DCID, facet_id, _UNBOUND_DONOR, recipient, years)
        for facet_id, years in (("f-rec", recorded), ("f-est", projected))
        for recipient in (_KEN, _NOR)
    ]
    role_names = {_SV_DCID: RoleNames(donor="creditor", recipient="debtor")}

    def _series(facet_id: str, years: list[int]) -> Facet:
        return Facet(
            earliest_date=str(years[0]),
            latest_date=str(years[-1]),
            obs_count=len(years),
            dates=[str(y) for y in years],
            values=[float(y) for y in years],
            facet_id=facet_id,
        )

    observations = {
        _SV_DCID: {
            (("creditor", _UNBOUND_DONOR), ("debtor", recipient)): [
                _series("f-rec", recorded),
                _series("f-est", projected),
            ]
            for recipient in (_KEN, _NOR)
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        date_request=DateRequest(window=None, latest=True),
        include_observations=True,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
        coverage_facet_metadata={
            "f-rec": FacetMetadata(),
            "f-est": FacetMetadata(measurement_method="Estimated"),
        },
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert [(call["facet_ids"], call["time_period"]) for call in sdmx.series_calls] == [
        ({"f-rec"}, "LATEST")
    ]
    assert spec.resolved.observations is not None
    assert [
        [(point.date, point.value) for point in series.points]
        for series in spec.resolved.observations.series
    ] == [[("2024", 2024.0)], [("2024", 2024.0)]]
    served_year = TimeWindow(start_year=2024, end_year=2024)
    assert spec.resolved.coverage.window == served_year
    when_slot = next(slot for slot in spec.resolved.slots if slot.key.axis == "when")
    assert when_slot.binding.value.time_window == served_year
    assert spec.resolved.headline is not None
    assert spec.resolved.headline.year == 2024


# ---------------------------------------------------------------------------
# Refusal and degradation
# ---------------------------------------------------------------------------


def test_window_too_long_refuses_scope_too_broad(search_settings):
    search_settings(scope_max_window_years=5)
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        date_request=DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
    )

    assert result.status == "unsupported", result
    assert result.unsupported is not None
    assert result.unsupported.code == "scope_too_broad"
    assert result.unsupported.pointer == "/definition/identity/entities/destination"
    assert sdmx.availability_calls == []


def test_a_named_destination_refuses_a_window_too_long_alike(search_settings):
    """The window cap is one rule on every role route (`read_plan.role_years`): a named
    destination refuses the same window as discovery does, before any lookup."""
    search_settings(scope_max_window_years=5)
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _run(
        entities=["Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": "country/KEN"},
        sv_score=0.9,
        role_query="aid disbursed to Kenya",
        sdmx=sdmx,
        date_request=DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
    )

    assert result.status == "unsupported", result
    assert result.unsupported is not None
    assert result.unsupported.code == "scope_too_broad"
    assert sdmx.availability_calls == []


def test_empty_discovery_reports_no_observations_naming_the_recipient_axis():
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
    )

    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"
    assert result.no_data_role_axis == "destination"


class _StrayBindLLM:
    """Elects shape index 0 and drafts `gender` Female, a property the shape has no slot
    for."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        stray = SlotBindingDraft(
            axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
        )
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[stray]), None


def test_empty_discovery_with_a_stray_value_refuses_naming_it(monkeypatch):
    """Discovery finds no recipient with data, which refuses `no_observations` before any
    answer is judged. The bind drafted Female on a slot the elected shape lacks, which the
    elected answer leaves unmet, and `qualifier_not_bound` ranks first, so the part refuses
    naming it (`phase_shared.settle_unjudged_refusal`), with the elected answer as its
    flagged match, whose trace records the term, and keeps the recipient axis discovery
    left open. Nothing departs."""
    monkeypatch.setattr(sys.modules[__name__], "_FixedBindLLM", _StrayBindLLM)

    result = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert (result.status, result.no_data_reason) == ("no_data", "qualifier_not_bound")
    assert [q.text for q in result.unbound_qualifiers] == ["Female"]
    (flagged,) = result.flagged_match or ()
    assert [m.ref.ref for m in flagged.resolved.members] == [_SV_DCID]
    assert list(flagged.resolved.trace.query_qualifiers) == list(result.unbound_qualifiers)
    assert result.no_data_role_axis == "destination"
    assert result.departure_facts == {}


# ---------------------------------------------------------------------------
# resubmit round trip
# ---------------------------------------------------------------------------


def test_resubmit_of_a_discovered_scope_definite_rediscovers_with_the_same_spec_id(monkeypatch):
    rows, role_names = _coverage_seed_for("debtor", [_KEN, _NOR], [_SV_DCID])
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("debtor", _NOR),): [_facet()],
        }
    }

    fresh = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )
    assert fresh.status == "definite", fresh
    spec = fresh.specs[0]
    assert spec.resolved.members[0].data_date_range == DateRange(start="2020", end="2020")

    graph = FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _NOR: {"label": "Norway", "type": "Country"},
            _SV_DCID: {
                "label": "Test flow",
                "arcs": _declaring_arcs(observation_properties=["creditor", "debtor"]),
            },
        },
        detect={},
        resolve={},
        obs={},
    )
    coverage_snapshot = FakeCoverageService(
        patch_coverage(monkeypatch, rows, role_names=role_names)
    ).snapshot()
    echo = ResubmitIdentityPayload(
        identity=spec.definition.identity,
    )
    region = resolve_spec_resubmit(
        inp=echo,
        graph=graph,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=coverage_snapshot,
        options=walk_options(graph),
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].spec_id == spec.spec_id
    assert region.specs[0].resolved.scope is not None
    assert region.specs[0].resolved.scope.axis == "destination"
    assert region.specs[0].resolved.members[0].data_date_range == DateRange(
        start="2020", end="2020"
    )


# ---------------------------------------------------------------------------
# The also_plausible candidates path (region/candidates.py)
# ---------------------------------------------------------------------------

_ALT_STD_SV = "TESTFIXTURE/RoleScopePipelineE2E/AltStandard"


def _alt_std_arcs() -> dict:
    """A plain standard aggregate SV's arcs -- no observationProperties, so
    is_role_bearing is False; no constraint properties, so bind_universal's
    own bindings are legitimately empty for it."""
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "Alt standard"}]},
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


class _RoleBearingCandidateBindLLM:
    """Elects the role-bearing discovered-scope shape (index 0) and flags a
    plain standard shape as also_plausible -- the same also_plausible signal
    test_observations_candidates_fanout.py's _CandidateUnitBindLLM uses,
    applied here to a role-bearing elected shape."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None
        ), None


def test_candidates_path_buffered_discovered_scope_single_survivor_reuses_the_index(
    monkeypatch, search_settings
):
    """A single surviving candidate reuses its discovered index for the buffered preview.

    Without it, a windowed bulk fetch emits OBSERVATIONS_PROBE_FAILED."""
    search_settings(obs_preview_max_pairs=1)
    time_period = ["2018", "2019", "2020"]
    rows, role_names = _coverage_seed_for(
        "debtor", [_KEN, _NOR], [_SV_DCID], time_period=time_period
    )
    _auto_coverage_index_1 = patch_coverage(monkeypatch, rows, role_names=role_names)
    observations = {
        _SV_DCID: {
            (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [_facet()],
            (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [_facet()],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})
    graph = FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _NOR: {"label": "Norway", "type": "Country"},
            _SV_DCID: {
                "label": "Test flow",
                "arcs": _declaring_arcs(observation_properties=["creditor", "debtor"]),
            },
            _ALT_STD_SV: {"label": "Alt standard", "arcs": _alt_std_arcs()},
        },
        detect={
            "aid disbursed": {
                "svs": [_SV_DCID, _ALT_STD_SV],
                # The alt SV's own score must clear SEARCH_UNKNOWN_SCORE_THRESHOLD
                # (0.65): _phase_prune_and_rank drops a non-role-bearing shape
                # scoring below it from eligible_shapes before also_plausible
                # ever gets a chance to nominate it.
                "cosine_scores": [0.9, 0.75],
            }
        },
        resolve={"Kenya": _KEN, "Norway": _NOR},
        obs={},
    )

    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=[],
            date_request=DateRequest(
                window=TimeWindow(start_year=2018, end_year=2020), latest=False
            ),
            detect_query="aid disbursed",
            role_query="aid disbursed",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_RoleBearingCandidateBindLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            options=EngineOptions(include_observations=True),
        )
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert len(sdmx.series_calls) == 1
    assert sdmx.series_calls[0]["time_period"] == time_period
    assert spec.resolved.observations is not None
    assert len(spec.resolved.observations.series) == 2
    for series in spec.resolved.observations.series:
        assert [p.date for p in series.points] == ["2020"]
