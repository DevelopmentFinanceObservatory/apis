"""Pins origin-role placement and resubmission for directional shapes.

Fictional flow IDs isolate role handling; the standard-shape case uses the recorded GDP StatVar.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from search.engine.clients.graph import Facet
from search.engine.config import settings
from search.engine.errors import EngineInputError
from search.engine.identity_edit import refuse_over_place_bound
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_spec_resubmit
from search.engine.region.codes import NO_MEMBER_HAS_DATA_AT_PLACE
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.models import (
    BaseIdentity,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityFacetValueAbsent,
    IdentityWindowRange,
    ResubmitIdentityPayload,
)
from series_coverage.schema import FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._base_identity import (
    GDP_SHAPE,
    make_base,
    origin_entity,
    subject_entity,
)
from tests.engine._binders import elected_reply
from tests.engine._builders import value_slot
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_SV_DCID = "TESTFIXTURE/ResubmitDonorRoleE2E/Flow"
_RECIPIENT_ONLY_SV_DCID = "TESTFIXTURE/ResubmitDonorRoleE2E/RecipientOnlyFlow"
_FRA = "country/FRA"
_KEN = "country/KEN"
_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
# The bilateral flow shape declares creditor/debtor observationProperties
# (_bilateral_arcs), placing creditor on the origin axis and debtor on the
# destination axis (materialise.roles._ROLE_AXES).
_FLOW_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")


def _facet(*, facet_id: str = "f1") -> Facet:
    return Facet(
        earliest_date="2015",
        latest_date="2020",
        obs_count=1,
        dates=["2020"],
        values=[0.0],
        facet_id=facet_id,
    )


def _bilateral_arcs() -> dict:
    """A creditor/debtor flow shape -- a donor lands on the creditor
    (origin) role, discovery runs on the debtor (destination) role when no
    recipient is posted."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in ("creditor", "debtor")]},
    }


def _recipient_only_arcs() -> dict:
    """A debtor-only flow shape: no origin-facing role name exists on this
    shape at all, so a posted origin list has nowhere to bind."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "debtor"}]},
    }


_UNBOUND_DONOR = "TESTFIXTURE/ResubmitDonorRoleE2E/UnboundDonor"


def _coverage_rows_for(
    members: list[str], svs: list[str], *, origin_dcids: list[str] | None = None
) -> list[tuple[str, str, str, str, list[int]]]:
    """Builds destination rows for each member and origin; an open origin uses a placeholder."""
    origins = origin_dcids or [_UNBOUND_DONOR]
    return [
        (sv, f"f-{origin.rsplit('/', 1)[-1]}-{member.rsplit('/', 1)[-1]}", origin, member, [2020])
        for sv in svs
        for member in members
        for origin in origins
    ]


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0), None


def _nodes(*, resolve: dict[str, str], label_dcids: list[str], sv_dcid: str, sv_arcs: dict) -> dict:
    return {
        **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
        sv_dcid: {"label": "Test flow", "arcs": sv_arcs},
        **{dcid: {"label": dcid.rsplit("/", 1)[-1]} for dcid in label_dcids},
    }


def _fresh_discover(
    *,
    sv_dcid: str,
    sv_arcs: dict,
    resolve: dict[str, str],
    sdmx: FakeSdmxClient,
    coverage: CoverageSnapshot | None = None,
):
    """Creates a place-less spec so a posted origin is a new resubmit constraint."""
    graph = FakeGraph(
        nodes=_nodes(resolve=resolve, label_dcids=["source/A"], sv_dcid=sv_dcid, sv_arcs=sv_arcs),
        detect={"aid disbursed": {"svs": [sv_dcid], "cosine_scores": [0.9]}},
        resolve=resolve,
        obs={},
    )
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=[],
            date_request=None,
            detect_query="aid disbursed",
            role_query="aid disbursed",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=coverage if coverage is not None else empty_coverage_snapshot(),
        )
    )


def _resubmit(
    *,
    fresh_spec,
    resolve: dict[str, str],
    sv_dcid: str,
    sv_arcs: dict,
    sdmx: FakeSdmxClient,
    origin_dcids: list[str] | None = None,
    recipient_dcid: str | None = None,
    recipient_dcids: list[str] | None = None,
    include_option_coverage: bool = False,
    coverage: CoverageSnapshot | None = None,
):
    graph = FakeGraph(
        nodes=_nodes(resolve=resolve, label_dcids=["source/A"], sv_dcid=sv_dcid, sv_arcs=sv_arcs),
        detect={},
        resolve={},
        obs={},
    )
    destination_dcids = recipient_dcids or ([recipient_dcid] if recipient_dcid else None)
    identity_entities = dict(fresh_spec.definition.identity.entities)
    if origin_dcids is not None:
        identity_entities["origin"] = origin_dcids
    if destination_dcids is not None:
        identity_entities["destination"] = destination_dcids
    inp = ResubmitIdentityPayload(
        identity=fresh_spec.definition.identity.model_copy(update={"entities": identity_entities}),
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=coverage if coverage is not None else empty_coverage_snapshot(),
        options=walk_options(graph, EngineOptions(include_option_coverage=include_option_coverage)),
    )


# ---------------------------------------------------------------------------
# (a) one origin: lands on the origin role, discovers the recipient axis
# ---------------------------------------------------------------------------


def test_one_origin_lands_on_origin_role_and_discovers_recipient_scope(monkeypatch):
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("creditor", _FRA), ("debtor", _KEN)): [_facet()],
        }
    }
    # One donor-recipient row serves both open- and bound-origin lookups.
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _coverage_rows_for([_KEN], [_SV_DCID], origin_dcids=[_FRA]),
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh_sdmx = FakeSdmxClient(observations=observations, availability={})
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=fresh_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    resubmit_sdmx = FakeSdmxClient(observations=observations, availability={})
    region = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=resubmit_sdmx,
        origin_dcids=[_FRA],
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert [e.ref for e in spec.resolved.entities] == [_FRA]
    assert [e.role for e in spec.resolved.entities] == ["origin"]
    assert spec.definition.identity.entities.get("origin") == [_FRA]
    assert spec.resolved.scope is not None
    assert spec.resolved.scope.axis == "destination"
    assert [m.ref for m in spec.resolved.scope.members] == [_KEN]
    # Destination discovery reads coverage, not SDMX availability.
    assert resubmit_sdmx.availability_calls == []


def test_resubmit_promote_discovered_scope_populates_entity_options(monkeypatch):
    """Standard-promote fills open-axis facts and entity options on resubmit."""
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("creditor", _FRA), ("debtor", _KEN)): [_facet()],
        }
    }
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _coverage_rows_for([_KEN], [_SV_DCID], origin_dcids=[_FRA]),
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh_sdmx = FakeSdmxClient(observations=observations, availability={})
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=fresh_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    resubmit_sdmx = FakeSdmxClient(observations=observations, availability={})
    region = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=resubmit_sdmx,
        origin_dcids=[_FRA],
        include_option_coverage=True,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert spec.resolved.scope is not None
    assert spec.resolved.scope.axis == "destination"
    facts = {f.role: f for f in spec.resolved.role_axis_facts or []}
    assert "destination" in facts
    assert facts["destination"].member_count == 1
    entity_options_by_role = {eo.role: eo for eo in spec.resolved.entity_options}
    assert "destination" in entity_options_by_role
    eo = entity_options_by_role["destination"]
    assert {o.value.ref.ref for o in eo.options if o.value.ref} == {_KEN}
    assert "options" in region.timing_by_step


# ---------------------------------------------------------------------------
# (b) origin_dcids + a posted recipient: both bound, no discovery
# ---------------------------------------------------------------------------


def test_origin_dcids_and_posted_recipient_both_bind_with_no_discovery(monkeypatch):
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("creditor", _FRA), ("debtor", _KEN)): [_facet()],
        }
    }
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _FRA, _KEN, [2020])],
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh_sdmx = FakeSdmxClient(observations=observations, availability={})
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=fresh_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    resubmit_sdmx = FakeSdmxClient(observations=observations, availability={})
    region = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=resubmit_sdmx,
        origin_dcids=[_FRA],
        recipient_dcid=_KEN,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert spec.resolved.scope is None
    roles_by_dcid = {e.ref: e.role for e in spec.resolved.entities}
    assert roles_by_dcid == {_FRA: "origin", _KEN: "destination"}
    assert spec.definition.identity.entities.get("origin") == [_FRA]
    assert spec.definition.identity.entities.get("destination") == [_KEN]
    # Both bound axes use the named-pair coverage lookup.
    assert resubmit_sdmx.availability_calls == []


# ---------------------------------------------------------------------------
# (c) a recipient-only shape: origin_dcids has no origin role to bind under
# ---------------------------------------------------------------------------


def test_origin_dcids_against_a_recipient_only_shape_is_role_incompatible(monkeypatch):
    resolve = {"France": _FRA, "Kenya": _KEN}
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _coverage_rows_for([_KEN], [_RECIPIENT_ONLY_SV_DCID]),
        role_names={_RECIPIENT_ONLY_SV_DCID: RoleNames(donor="creditor", recipient="debtor")},
    )
    observations = {_RECIPIENT_ONLY_SV_DCID: {(("debtor", _KEN),): [_facet()]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})
    fresh = _fresh_discover(
        sv_dcid=_RECIPIENT_ONLY_SV_DCID,
        sv_arcs=_recipient_only_arcs(),
        resolve=resolve,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    region = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_RECIPIENT_ONLY_SV_DCID,
        sv_arcs=_recipient_only_arcs(),
        sdmx=sdmx,
        origin_dcids=[_FRA],
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region.status == "no_data"
    assert region.no_data_reason == "role_incompatible"
    # The refusal comes after the bind phase, ahead of materialisation.
    assert "bind" in region.timing_by_step
    assert "materialise" not in region.timing_by_step


# ---------------------------------------------------------------------------
# (d) a standard shape: origin_dcids refuses role_incompatible (422)
# ---------------------------------------------------------------------------


def _standard_shape_with_origin_inp() -> ResubmitIdentityPayload:
    """A GDP resubmit posting an origin ref, which a standard shape has no role for."""
    entities = [subject_entity("country/IND")]
    activity_source_slot = value_slot(
        "how",
        "activitySource",
        "activity source",
        "GrossDomesticProduction",
        "Gross Domestic Production",
    )
    return ResubmitIdentityPayload(
        identity=make_base(
            GDP_SHAPE,
            slots=[activity_source_slot],
            entities=[*entities, origin_entity(_FRA)],
        )[1],
    )


def test_origin_dcids_against_a_standard_shape_raises_role_incompatible():
    with pytest.raises(EngineInputError) as exc_info:
        resolve_spec_resubmit(
            inp=_standard_shape_with_origin_inp(),
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
    assert exc_info.value.code == "role_incompatible"
    assert exc_info.value.status == 422
    assert exc_info.value.pointer == "/input/identity/entities/origin"
    assert "economicactivity_amount_measuredvalue_nominal" in str(exc_info.value)


def test_a_reversed_requested_window_refuses_before_the_role_check():
    """A reversed requested window answers `request_invalid` at identity
    validation, ahead of the origin ref's `role_incompatible`. The payload model
    rejects a reversed range, so the test builds one without validation."""
    inp = _standard_shape_with_origin_inp()
    reversed_window = IdentityWindowRange.model_construct(start_year=2018, end_year=2015)
    inp = inp.model_copy(
        update={"identity": inp.identity.model_copy(update={"requested_window": reversed_window})}
    )
    with pytest.raises(EngineInputError) as exc_info:
        resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
    assert exc_info.value.code == "request_invalid"
    assert exc_info.value.pointer == "/input/identity"


# ---------------------------------------------------------------------------
# (e) resubmitting a donor-bound discovered answer yields the same spec id
# ---------------------------------------------------------------------------


def test_resubmit_with_the_same_origin_dcids_yields_the_same_spec_id(monkeypatch):
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("creditor", _FRA), ("debtor", _KEN)): [_facet()],
        }
    }
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _coverage_rows_for([_KEN], [_SV_DCID], origin_dcids=[_FRA]),
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh_sdmx = FakeSdmxClient(observations=observations, availability={})
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=fresh_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    region_a = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert region_a.status == "definite", region_a.no_data_reason

    region_e = _resubmit(
        fresh_spec=region_a.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region_e.status == "definite", region_e.no_data_reason
    assert region_e.specs[0].spec_id == region_a.specs[0].spec_id


# ---------------------------------------------------------------------------
# (f) resubmitting a spec with both a bound donor and a bound recipient stays
#     definite: the where-slot's own entity re-sends the base's "recipient"
#     role, never "observationAbout"
# ---------------------------------------------------------------------------


def test_resubmit_of_a_bound_donor_and_recipient_spec_stays_definite(monkeypatch):
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("creditor", _FRA), ("debtor", _KEN)): [_facet()],
        }
    }
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _FRA, _KEN, [2020])],
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh_sdmx = FakeSdmxClient(observations=observations, availability={})
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=fresh_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    region_a = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        recipient_dcid=_KEN,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert region_a.status == "definite", region_a.no_data_reason
    roles_by_dcid = {e.ref: e.role for e in region_a.specs[0].resolved.entities}
    assert roles_by_dcid == {_FRA: "origin", _KEN: "destination"}

    # Repost the emitted definition verbatim. The base's bound recipient
    # must be re-accounted through its where slot, not treated as absent.
    region_b = _resubmit(
        fresh_spec=region_a.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region_b.status == "definite", region_b.no_data_reason
    roles_by_dcid = {e.ref: e.role for e in region_b.specs[0].resolved.entities}
    assert roles_by_dcid == {_FRA: "origin", _KEN: "destination"}


def test_resubmit_preserves_every_posted_destination_ref_on_an_untouched_echo(monkeypatch):
    """A role-bearing where BindingSet is the complete destination contract."""
    resolve = {"France": _FRA, "Kenya": _KEN, "United States": "country/USA"}
    usa = resolve["United States"]
    observations = {
        _SV_DCID: {
            (("creditor", _FRA), ("debtor", _KEN)): [_facet()],
            (("creditor", _FRA), ("debtor", usa)): [_facet(facet_id="f2")],
        }
    }
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f1", _FRA, _KEN, [2020]),
            (_SV_DCID, "f2", _FRA, usa, [2020]),
        ],
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    resolved = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        recipient_dcids=[_KEN, usa],
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert resolved.status == "definite", resolved.no_data_reason
    assert {(entity.ref, entity.role) for entity in resolved.specs[0].resolved.entities} == {
        (_FRA, "origin"),
        (_KEN, "destination"),
        (usa, "destination"),
    }

    echoed = _resubmit(
        fresh_spec=resolved.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert echoed.status == "definite", echoed.no_data_reason
    assert echoed.specs[0].spec_id == resolved.specs[0].spec_id


def test_self_flow_keeps_both_role_identities_and_round_trips(monkeypatch):
    """One DCID can bind both ends of a directional flow."""
    resolve = {"France": _FRA}
    observations = {_SV_DCID: {(("creditor", _FRA), ("debtor", _FRA)): [_facet()]}}
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _FRA, _FRA, [2020])],
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    resolved = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        recipient_dcid=_FRA,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert resolved.status == "definite", resolved.no_data_reason
    spec = resolved.specs[0]
    assert {(entity.ref, entity.role) for entity in spec.resolved.entities} == {
        (_FRA, "origin"),
        (_FRA, "destination"),
    }
    assert spec.definition.identity.entities == {"origin": [_FRA], "destination": [_FRA]}

    echoed = _resubmit(
        fresh_spec=spec,
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert echoed.status == "definite", echoed.no_data_reason
    assert echoed.specs[0].spec_id == spec.spec_id


@pytest.mark.parametrize(
    ("recipient_dcids", "expected_recipients"),
    [([_KEN], {_KEN}), ([_KEN, "country/SEN"], {_KEN, "country/SEN"})],
)
def test_plural_donors_round_trip_with_every_destination(
    monkeypatch, recipient_dcids: list[str], expected_recipients: set[str]
):
    """An emitted role-bearing spec preserves every provider and recipient."""
    usa = "country/USA"
    senegal = "country/SEN"
    resolve = {"France": _FRA, "United States": usa, "Kenya": _KEN, "Senegal": senegal}
    observations = {
        _SV_DCID: {
            (provider, recipient): [_facet(facet_id=f"{provider[1]}-{recipient[1]}")]
            for provider in (("creditor", _FRA), ("creditor", usa))
            for recipient in (("debtor", dcid) for dcid in recipient_dcids)
        }
    }
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, f"{provider}-{recipient}", provider, recipient, [2020])
            for provider in (_FRA, usa)
            for recipient in recipient_dcids
        ],
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    resolved = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA, usa],
        recipient_dcids=recipient_dcids,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert resolved.status == "definite", resolved.no_data_reason
    spec = resolved.specs[0]
    assert {(entity.ref, entity.role) for entity in spec.resolved.entities} == {
        (_FRA, "origin"),
        (usa, "origin"),
        *{(dcid, "destination") for dcid in expected_recipients},
    }
    assert spec.definition.identity.entities == {
        "origin": [_FRA, usa],
        "destination": sorted(expected_recipients),
    }

    echoed = _resubmit(
        fresh_spec=spec,
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=spec.definition.identity.entities["origin"],
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert echoed.status == "definite", echoed.no_data_reason
    assert echoed.specs[0].spec_id == spec.spec_id


def test_missing_origin_refs_refuse_at_the_role_confirmation_boundary(monkeypatch):
    """Origin input identifies its invalid wire field."""
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {_SV_DCID: {(("creditor", _FRA), ("debtor", _KEN)): [_facet()]}}
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _coverage_rows_for([_KEN], [_SV_DCID]),
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh

    for kwargs in (
        {"origin_dcids": ["country/MISSING"]},
        {"origin_dcids": [_FRA, "country/MISSING"]},
    ):
        with pytest.raises(EngineInputError) as exc_info:
            _resubmit(
                fresh_spec=fresh.specs[0],
                resolve=resolve,
                sv_dcid=_SV_DCID,
                sv_arcs=_bilateral_arcs(),
                sdmx=FakeSdmxClient(observations=observations, availability={}),
                recipient_dcid=_KEN,
                **kwargs,
            )
        assert exc_info.value.code == "invalid_role_ref"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/input/identity/entities/origin"


def test_origin_role_is_carried_by_the_nested_identity_entities_map():
    inp = ResubmitIdentityPayload(identity=make_base(GDP_SHAPE, entities=[origin_entity(_FRA)])[1])
    assert inp.identity.entities == {"origin": [_FRA]}


@pytest.mark.parametrize(
    ("counts", "pointer", "named"),
    [
        pytest.param({"origin": 300}, "/input/identity/entities/origin", {300, 256}, id="one-role"),
        pytest.param(
            {"observationAbout": 300},
            "/input/identity/entities/observationAbout",
            {300, 256},
            id="subject-role",
        ),
        pytest.param(
            {"destination": 200, "origin": 100},
            "/input/identity/entities",
            {300, 2, 256},
            id="sum-of-two-roles",
        ),
        pytest.param(
            {"destination": 200, "origin": 200},
            "/input/identity/entities",
            {400, 2, 256},
            id="same-refs-on-two-roles",
        ),
    ],
)
def test_place_bound_counts_distinct_refs_per_role_before_graph_work(counts, pointer, named):
    """An identity binding more places than a Spec holds is rejected before graph access,
    at the role's path when one role exceeds alone, at the entity map when only the sum
    does. The message names the count and the bound."""

    class _NoGraph:
        def __getattr__(self, name):
            raise AssertionError(f"graph must not be read for the place bound: {name}")

    assert settings().max_child_places == 256
    inp = ResubmitIdentityPayload(
        identity=BaseIdentity.model_validate(
            {
                "shape": GDP_SHAPE,
                "slots": [],
                "entities": {
                    role: [f"country/P{i}" for i in range(count)] for role, count in counts.items()
                },
                "requested_window": {"kind": "all"},
            }
        ),
    )
    with pytest.raises(EngineInputError) as exc_info:
        resolve_spec_resubmit(
            inp=inp,
            graph=_NoGraph(),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_NoGraph()),
        )
    assert exc_info.value.code == "request_invalid"
    assert exc_info.value.status == 400
    assert exc_info.value.pointer == pointer
    assert {int(number) for number in re.findall(r"\d+", str(exc_info.value))} == named


def test_place_bound_counts_a_repeated_ref_once():
    """Direct validation preserves duplicate refs from an incoming payload, and the bound
    counts each once."""
    duplicate_heavy = ResubmitIdentityPayload(
        identity=BaseIdentity.model_validate(
            {
                "shape": GDP_SHAPE,
                "slots": [],
                "entities": {"origin": ["country/FRA"] * 300},
                "requested_window": {"kind": "all"},
            }
        ),
    )
    assert duplicate_heavy.identity.entities["origin"] == ["country/FRA"] * 300
    refuse_over_place_bound(duplicate_heavy.identity.entities)


def test_origin_role_is_nested_in_the_public_request_schema():
    properties = ResubmitIdentityPayload.model_json_schema()["properties"]
    assert "origin_dcids" not in properties
    assert properties["identity"]["$ref"] == "#/$defs/BaseIdentity"
    entities_schema = ResubmitIdentityPayload.model_json_schema()["$defs"]["BaseIdentity"][
        "properties"
    ]["entities"]
    assert entities_schema["additionalProperties"]["items"]["type"] == "string"


# ---------------------------------------------------------------------------
# (f) a bound origin and recipient with no observation: the resubmit promote
# path returns an editable flagged_match
# ---------------------------------------------------------------------------


def test_zero_observation_bilateral_resubmit_carries_a_flagged_match(monkeypatch):
    """A confirmed bilateral identity with no pair observations returns a flagged match.

    The fresh request has data; resubmit uses an empty role-keyed observation store.
    """
    resolve = {"France": _FRA, "Kenya": _KEN}
    role_tuple = (("creditor", _FRA), ("debtor", _KEN))
    # Coverage supports discovery; the resubmit confirms the bound pair against an empty store.
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _coverage_rows_for([_KEN], [_SV_DCID]),
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=FakeSdmxClient(observations={_SV_DCID: {role_tuple: [_facet()]}}, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh.no_data_reason

    region = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        origin_dcids=[_FRA],
        recipient_dcid=_KEN,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region.status == "no_data", region
    assert region.no_data_reason == "no_observations"
    assert region.gate == NO_MEMBER_HAS_DATA_AT_PLACE
    assert region.flagged_match is not None
    assert len(region.flagged_match) == 1
    assert region.flagged_match[0].resolved.available_at_place is False
    assert [sv.ref.ref for sv in region.flagged_match[0].resolved.members] == [_SV_DCID]


def test_bound_recipient_open_donor_resubmit_warns_about_the_open_origin_axis(monkeypatch):
    """An open donor axis emits a warning; posting both ends suppresses it."""
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {_SV_DCID: {(("creditor", _FRA), ("debtor", _KEN)): [_facet()]}}
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _FRA, _KEN, [2020])],
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert fresh.status == "definite", fresh.no_data_reason

    control = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        recipient_dcid=_KEN,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert control.status == "definite", control.no_data_reason
    assert [w for w in control.warnings if w.code == "role_scope_defaulted"] == []

    region = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        recipient_dcid=_KEN,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert region.status == "definite", region.no_data_reason
    defaulted = [w for w in region.warnings if w.code == "role_scope_defaulted"]
    assert [w.message for w in defaulted] == [
        "no provider was named; the answer was resolved across every provider with data."
    ]
    facts = region.specs[0].resolved.role_axis_facts
    assert facts is not None
    assert [f.model_dump(mode="json") for f in facts] == [
        {"role": "origin", "member_count": 1, "members_available": True}
    ]


def _pinned_resubmit(monkeypatch, axis: str, binding: IdentityFacetBinding):
    """Resolve France to Kenya over facet `f1` (Recorded, US dollars), then resubmit that
    definition with `binding` on `axis`."""
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("creditor", _FRA), ("debtor", _KEN)): [_facet()],
        }
    }
    index = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _FRA, _KEN, [2020])],
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
        facet_metadata={
            "f1": FacetMetadata(
                provenance_id="source/A", measurement_method="Recorded", unit="CurrencyUSD"
            )
        },
    )
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(index).snapshot(),
    )
    named = _resubmit(
        fresh_spec=fresh.specs[0],
        resolve=resolve,
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        origin_dcids=[_FRA],
        recipient_dcid=_KEN,
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert named.status == "definite", named.no_data_reason
    spec = named.specs[0]
    identity = spec.definition.identity
    inp = ResubmitIdentityPayload(
        identity=identity.model_copy(update={"facets": {**(identity.facets or {}), axis: binding}}),
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(
            nodes=_nodes(
                resolve=resolve,
                label_dcids=["source/A"],
                sv_dcid=_SV_DCID,
                sv_arcs=_bilateral_arcs(),
            ),
            detect={},
            resolve={},
            obs={},
        ),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(index).snapshot(),
        options=walk_options(
            FakeGraph(
                nodes=_nodes(
                    resolve=resolve,
                    label_dcids=["source/A"],
                    sv_dcid=_SV_DCID,
                    sv_arcs=_bilateral_arcs(),
                ),
                detect={},
                resolve={},
                obs={},
            ),
            EngineOptions(),
        ),
    )


@pytest.mark.parametrize(
    ("axis", "binding"),
    [
        ("measurementMethod", IdentityBindingValue(ref="Estimated")),
        ("unit", IdentityBindingValue(ref="CurrencyEUR")),
        ("measurementMethod", IdentityFacetValueAbsent(ref=None)),
    ],
    ids=["method", "unit", "null_method"],
)
def test_a_named_role_resubmit_pinning_a_value_no_series_carries_refuses(
    monkeypatch, axis: str, binding: IdentityFacetBinding
):
    """France to Kenya serves only `f1`, Recorded, in US dollars. A pin no series carries is
    refused before any read, by the same check a facet option is judged by."""
    with pytest.raises(EngineInputError) as excinfo:
        _pinned_resubmit(monkeypatch, axis, binding)
    assert excinfo.value.code == "facet_not_available"


def test_a_named_role_resubmit_pinning_the_carried_value_stays_definite(monkeypatch):
    region = _pinned_resubmit(
        monkeypatch, "measurementMethod", IdentityBindingValue(ref="Recorded")
    )
    assert region.status == "definite", region.no_data_reason


@pytest.mark.parametrize("recipient", [_KEN, None], ids=["named", "discovered"])
def test_a_role_resubmit_whose_years_hold_no_point_names_the_span(monkeypatch, recipient) -> None:
    """France funds Kenya from 2015 to 2020. A resubmit asking for 2023 to 2025, with Kenya
    named or discovered, answers for the window and names the span the series runs."""
    resolve = {"France": _FRA, "Kenya": _KEN}
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet()],
            (("creditor", _FRA), ("debtor", _KEN)): [_facet()],
        }
    }
    index = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _FRA, _KEN, list(range(2015, 2021)))],
        role_names={_SV_DCID: _FLOW_ROLE_NAMES},
    )
    fresh = _fresh_discover(
        sv_dcid=_SV_DCID,
        sv_arcs=_bilateral_arcs(),
        resolve=resolve,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(index).snapshot(),
    )
    spec = fresh.specs[0]
    windowed = spec.model_copy(
        update={
            "definition": spec.definition.model_copy(
                update={
                    "identity": spec.definition.identity.model_copy(
                        update={
                            "requested_window": IdentityWindowRange(start_year=2023, end_year=2025)
                        }
                    )
                }
            )
        }
    )
    identity = windowed.definition.identity
    entities = {**dict(identity.entities), "origin": [_FRA]}
    if recipient is not None:
        entities["destination"] = [recipient]
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=identity.model_copy(update={"entities": entities}),
        ),
        graph=FakeGraph(
            nodes=_nodes(
                resolve=resolve,
                label_dcids=["source/A"],
                sv_dcid=_SV_DCID,
                sv_arcs=_bilateral_arcs(),
            ),
            detect={},
            resolve={},
            obs={},
        ),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=FakeCoverageService(index).snapshot(),
        options=walk_options(
            FakeGraph(
                nodes=_nodes(
                    resolve=resolve,
                    label_dcids=["source/A"],
                    sv_dcid=_SV_DCID,
                    sv_arcs=_bilateral_arcs(),
                ),
                detect={},
                resolve={},
                obs={},
            ),
            EngineOptions(),
        ),
    )

    assert region.no_data_reason == "no_observations"
    assert region.gate is None
    assert region.window_gap_fact is not None
    assert (region.window_gap_fact.asked, region.window_gap_fact.data_years) == (
        "2023 to 2025",
        (2015, 2020),
    )
