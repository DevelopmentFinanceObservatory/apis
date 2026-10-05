"""Resubmission preserves posted place sets, expands regions for probes, and reapplies both caps.

An over-limit set is refused because canonicalized identities carry no naming order.
"""

from __future__ import annotations

import pytest

from search.engine.errors import EngineInputError
from search.engine.region import (
    MULTI_RECIPIENT_TRUNCATED,
    STANDARD_PROBE_PLACES_CLAMPED,
    resolve_spec_resubmit,
)
from search.models import (
    BindingSet,
    BindingValue,
    CoverageExact,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from tests.engine._base_identity import (
    PERSON_COUNT_SHAPE,
    make_base,
    make_identity,
    subject_entity,
)
from tests.engine._builders import gref as _gref
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

pytestmark = pytest.mark.usefixtures("request_log")

# Count_Person has no constraintProperties. These tests exercise the where axis.
_SV = "Count_Person"

_NODES: dict = {
    _SV: {
        "label": "Count Person",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Count Person"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/GTM": {"label": "Guatemala", "type": "Country"},
    "country/SLV": {"label": "El Salvador", "type": "Country"},
    "country/HND": {"label": "Honduras", "type": "Country"},
    # Boreal is a Continent. The single-region case tests its expansion.
    "boreal": {"label": "Boreal", "type": "Continent"},
    "country/B1": {"label": "B1", "type": "Country"},
    "country/B2": {"label": "B2", "type": "Country"},
}
_CHILDREN: dict = {
    "boreal": {"Country": ["country/B1", "country/B2"]},
}


# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_OBS_PROVENANCE = "dc/base/WorldDevelopmentIndicators"


def _where_value_slot(dcid: str, label: str) -> Slot:
    return Slot(
        key=SlotKey(axis="where", property=None, label="place"),
        binding=BindingValue(value=SlotValue(ref=_gref(dcid, label), value_kind="entity")),
    )


def _where_set_slot(pairs: list[tuple[str, str]]) -> Slot:
    return Slot(
        key=SlotKey(axis="where", property=None, label="place"),
        binding=BindingSet(
            values=[SlotValue(ref=_gref(dcid, label), value_kind="entity") for dcid, label in pairs]
        ),
    )


def _graph(obs: dict) -> FakeGraph:
    return FakeGraph(nodes=_NODES, obs=obs, children=_CHILDREN)


def _where_slot(region):
    return next(s for s in region.specs[0].resolved.slots if s.key.axis == "where")


def _resubmit(where_slot: Slot, obs: dict):
    base_spec_id, base_identity = make_base(PERSON_COUNT_SHAPE)
    binding = where_slot.binding
    values = [binding.value] if isinstance(binding, BindingValue) else binding.values
    inp = ResubmitIdentityPayload(
        identity=make_identity(
            PERSON_COUNT_SHAPE,
            entities=[subject_entity(value.ref.dcid, value.ref.label) for value in values],
        ),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[_SV],
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=_graph(obs),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )


# Posted place sets retain their values and roles.


def test_posted_two_place_set_survives_as_binding_set_with_both_roles():
    """A posted 2-place set retains both entities with observationAbout roles."""
    obs = {
        f"{_SV}|country/GTM": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
        f"{_SV}|country/SLV": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot([("country/GTM", "Guatemala"), ("country/SLV", "El Salvador")])
    region = _resubmit(where_slot, obs)
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    spec = region.specs[0]
    where = _where_slot(region)
    assert isinstance(where.binding, BindingSet)
    assert [v.ref.dcid for v in where.binding.values] == ["country/GTM", "country/SLV"]
    assert {e.ref for e in spec.resolved.entities} == {"country/GTM", "country/SLV"}


def test_posted_set_whose_data_sits_at_its_second_place_answers():
    """Guatemala holds no data and El Salvador does: the probe reads both places of the
    posted set, so the resubmit answers with El Salvador's point."""
    obs = {
        f"{_SV}|country/SLV": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot([("country/GTM", "Guatemala"), ("country/SLV", "El Salvador")])
    region = _resubmit(where_slot, obs)
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    coverage = region.specs[0].resolved.coverage
    assert isinstance(coverage, CoverageExact)
    assert coverage.has_data is True
    assert coverage.observation_count == 1


# Posted regions expand for the observation probe.


def test_plural_probe_reaches_materialise_for_region_member():
    """The posted set keeps Guatemala and Boreal. The probe expands Boreal's members."""
    # B1 alone has data, so definite status proves the probe expands Boreal.
    obs = {
        f"{_SV}|country/B1": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot([("country/GTM", "Guatemala"), ("boreal", "Boreal")])
    region = _resubmit(where_slot, obs)
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    spec = region.specs[0]
    where = _where_slot(region)
    assert isinstance(where.binding, BindingSet)
    # #572 pins canonical identity order, with boreal before country/GTM.
    assert [v.ref.dcid for v in where.binding.values] == ["boreal", "country/GTM"]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.has_data is True


def test_single_region_resubmit_re_expands_to_member_countries():
    """A single-region value re-expands to member countries for probing."""
    obs = {
        f"{_SV}|country/B1": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_value_slot("boreal", "Boreal")
    region = _resubmit(where_slot, obs)
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    spec = region.specs[0]
    where = _where_slot(region)
    assert isinstance(where.binding, BindingValue)
    assert where.binding.value.ref.dcid == "boreal"
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.has_data is True


# A posted where set takes precedence over a base entity.


def test_unresolvable_base_entity_alongside_valid_where_set_is_inert():
    """The posted where set supplies the entity anchor when the base contains country/ZZZ.

    Resolution uses the set's first member.
    """
    obs = {
        f"{_SV}|country/GTM": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
        f"{_SV}|country/SLV": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot([("country/GTM", "Guatemala"), ("country/SLV", "El Salvador")])
    base_spec_id, base_identity = make_base(
        PERSON_COUNT_SHAPE,
        entities=[subject_entity("country/ZZZ")],  # Absent from the graph.
    )
    inp = ResubmitIdentityPayload(
        identity=make_identity(
            PERSON_COUNT_SHAPE,
            entities=[
                subject_entity(value.ref.dcid, value.ref.label)
                for value in where_slot.binding.values
            ],
        ),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[_SV],
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=_graph(obs),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )
    assert region.status == "definite"
    where = _where_slot(region)
    assert isinstance(where.binding, BindingSet)
    assert {v.ref.dcid for v in where.binding.values} == {"country/GTM", "country/SLV"}


# Reapply the named-place and probe caps on resubmit.


def test_named_place_cap_overflow_refuses(search_settings):
    """An over-limit set returns request_invalid because canonical identities lack query order."""
    search_settings(max_standard_places=2)
    obs = {
        f"{_SV}|country/GTM": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot(
        [
            ("country/GTM", "Guatemala"),
            ("country/SLV", "El Salvador"),
            ("country/HND", "Honduras"),
        ]
    )
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(where_slot, obs)
    assert exc_info.value.code == "request_invalid"
    assert exc_info.value.status == 400
    assert "3 places" in str(exc_info.value)
    assert "2-place limit" in str(exc_info.value)
    assert exc_info.value.pointer == "/input/identity/entities/observationAbout"


def test_probe_union_cap_overflow_fires_standard_probe_places_clamped(search_settings):
    """The probe-place cap is reapplied to posted sets."""
    search_settings(max_standard_probe_places=1)
    obs = {
        f"{_SV}|country/GTM": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_set_slot([("country/GTM", "Guatemala"), ("country/SLV", "El Salvador")])
    region = _resubmit(where_slot, obs)
    clamped = [w for w in region.warnings if w.code == STANDARD_PROBE_PLACES_CLAMPED]
    assert len(clamped) == 1
    assert region.status == "definite"  # GTM (kept) still has data


# Single-value resubmits keep their scalar binding.


def test_posted_single_value_where_unchanged():
    """A single-place resubmit keeps its value binding without cap warnings."""
    obs = {
        f"{_SV}|country/GTM": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": _OBS_PROVENANCE,
            },
        ],
    }
    where_slot = _where_value_slot("country/GTM", "Guatemala")
    region = _resubmit(where_slot, obs)
    assert region.status == "definite"
    where = _where_slot(region)
    assert isinstance(where.binding, BindingValue)
    assert where.binding.value.ref.dcid == "country/GTM"
    assert not any(w.code == MULTI_RECIPIENT_TRUNCATED for w in region.warnings)
    assert not any(w.code == STANDARD_PROBE_PLACES_CLAMPED for w in region.warnings)
